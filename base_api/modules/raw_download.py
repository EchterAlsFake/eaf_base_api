"""Resumable RAW transfers for BaseCore, including persisted multipart ranges."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from pathlib import Path

from .errors import DownloadCancelled, NetworkRequestError
from .progress_bars import Callback

_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


class _Checkpoint:
    def __init__(self, path, source_url, output, mode, size, chunk_size, headers):
        self.path = Path(path)
        self.lock = asyncio.Lock()
        self.restored = False
        count = math.ceil(size / chunk_size) if mode == "multipart" else 1
        self.data = {
            "version": 1, "source": hashlib.sha256(source_url.encode()).hexdigest(),
            "output": str(Path(output).resolve()), "mode": mode, "size": size,
            "chunk_size": chunk_size, "progress": [0] * count,
            "etag": headers.get("ETag"), "last_modified": headers.get("Last-Modified"),
        }
        try:
            saved = json.loads(self.path.read_text())
            if not isinstance(saved, dict):
                return
            if any(saved.get(key) != self.data[key] for key in ("version", "source", "output", "mode", "size")):
                return
            for name in ("etag", "last_modified"):
                if saved.get(name) and saved[name] != self.data[name]:
                    return
            width = saved["chunk_size"]
            progress = saved["progress"]
            if mode == "multipart":
                if not isinstance(width, int) or width <= 0 or len(progress) != math.ceil(size / width):
                    return
                limits = [min(width, size - index * width) for index in range(len(progress))]
            else:
                if len(progress) != 1:
                    return
                limits = [size or Path(output).stat().st_size]
            if any(not isinstance(value, int) or not 0 <= value <= limit
                   for value, limit in zip(progress, limits)):
                return
            length = Path(output).stat().st_size
            if (mode == "multipart" and length != size) or (mode == "linear" and length < progress[0]):
                return
            self.data = saved
            self.restored = True
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
            pass

    async def save(self, progress):
        async with self.lock:
            payload = dict(self.data, progress=list(progress))
            await asyncio.to_thread(self._write, payload)

    def _write(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)

    def remove(self):
        self.path.unlink(missing_ok=True)
        self.path.with_name(self.path.name + ".tmp").unlink(missing_ok=True)


def _stopped(event):
    return event is not None and event.is_set()


async def _retry_delay(event, seconds):
    if event is None:
        await asyncio.sleep(seconds)
        return
    try:
        await asyncio.wait_for(event.wait(), seconds)
    except TimeoutError:
        return
    raise DownloadCancelled("Download cancelled.")


async def _close_response(response):
    if response is None:
        return
    # curl-cffi's aclose() waits for the streaming task; signal its supported
    # quit event first so an interrupted response stops receiving its body.
    quit_event = getattr(response, "quit_now", None)
    if quit_event is not None:
        quit_event.set()
    await response.aclose()


def _range(response, start, end=None, total=None):
    match = _RANGE.fullmatch(response.headers.get("Content-Range", ""))
    if (response.status_code != 206 or match is None or int(match[1]) != start
            or not start <= int(match[2]) < int(match[3])
            or (end is not None and int(match[2]) != end)
            or (total is not None and int(match[3]) != total)):
        raise NetworkRequestError("Server did not return the requested byte range")
    return int(match[3])


async def download_raw(core, url, configuration):
    """Download through the core's real session; checkpoints never infer sparse-file progress."""
    output = Path(configuration.path)
    checkpoint_path = configuration.resume_state_path or f"{output}.raw-state.json"
    event = configuration.stop_event
    if _stopped(event):
        raise DownloadCancelled("Download cancelled.")
    core.initialize_session()
    session = core.session
    output.parent.mkdir(parents=True, exist_ok=True)
    original_url = url
    size, ranges, headers = 0, False, {}
    response = None
    try:
        response = await session.head(url, timeout=configuration.read_timeout,
                                      headers={"Accept-Encoding": "identity"}, allow_redirects=True)
        if response.status_code == 405:
            await _close_response(response)
            response = await session.request("GET", url, stream=True, timeout=configuration.read_timeout,
                                             headers={"Accept-Encoding": "identity"}, allow_redirects=True)
        response.raise_for_status()
        headers = {name: response.headers.get(name) for name in ("ETag", "Last-Modified")}
        size = int(response.headers.get("Content-Length", 0))
        ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
        if response.url:
            url = str(response.url)
    except DownloadCancelled:
        raise
    except Exception:
        # An outage must not turn a persisted partial download into a fresh file.
        if Path(checkpoint_path).exists():
            raise
        core.logger.warning("RAW metadata unavailable; falling back to linear streaming", exc_info=True)
    finally:
        await _close_response(response)
    mode = "multipart" if configuration.allow_multipart and ranges and size > 0 else "linear"
    width = max(configuration.chunk_size, min(10 * 1024 * 1024, max(1, size // 10)))
    state = _Checkpoint(checkpoint_path, original_url, output, mode, size, width, headers)
    progress = list(state.data["progress"])
    if not state.restored:
        # Create the zero-progress checkpoint before preallocation. An existing
        # full-size sparse file without a valid checkpoint is never considered complete.
        await state.save(progress)
        def allocate():
            with output.open("wb") as file:
                if mode == "multipart":
                    file.truncate(size)
        await asyncio.to_thread(allocate)
    elif mode == "linear":
        with output.open("r+b") as file:
            file.truncate(progress[0])
    core.logger.info("RAW %s: mode=%s retained_bytes=%s total=%s", output, mode, sum(progress), size)
    bar = Callback() if configuration.callback is None else None

    def notify(position, total):
        if configuration.callback:
            configuration.callback(position, total)
        elif bar:
            bar.text_progress_bar(downloaded=position, total=total)

    if sum(progress):
        notify(sum(progress), size)
    validator = state.data.get("etag")
    if not validator or validator.startswith("W/"):
        validator = state.data.get("last_modified")

    if mode == "multipart":
        width = state.data["chunk_size"]
        durable = list(progress)
        semaphore = asyncio.Semaphore(max(1, configuration.max_workers))

        async def chunk(index):
            start = index * width
            end = min(start + width, size) - 1
            for attempt in range(configuration.max_retries + 1):
                if _stopped(event):
                    return False
                offset = start + progress[index]
                if offset > end:
                    return True
                response = None
                try:
                    async with semaphore:
                        # Workers queued before a stop must not issue requests afterwards.
                        if _stopped(event):
                            return False
                        request_headers = {"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"}
                        if validator:
                            request_headers["If-Range"] = validator
                        response = await session.request("GET", url, headers=request_headers, stream=True,
                                                         timeout=configuration.read_timeout, allow_redirects=True)
                        response.raise_for_status()
                        _range(response, offset, end, size)
                        file = await asyncio.to_thread(output.open, "r+b")
                        try:
                            await asyncio.to_thread(file.seek, offset)
                            async for data in response.aiter_content():
                                if _stopped(event):
                                    return False
                                if not data:
                                    continue
                                if start + progress[index] + len(data) > end + 1:
                                    raise NetworkRequestError("Server exceeded the requested byte range")
                                await asyncio.to_thread(file.write, data)
                                progress[index] += len(data)
                                notify(sum(progress), size)
                        finally:
                            await asyncio.to_thread(file.close)
                        if start + progress[index] != end + 1:
                            raise NetworkRequestError("Byte range ended before all requested bytes arrived")
                        # Only closed/flushed ranges enter checkpoints while other
                        # workers are active. Partial ranges are saved after all exit.
                        durable[index] = progress[index]
                        await state.save(durable)
                        return True
                except DownloadCancelled:
                    return False
                except Exception:
                    if attempt >= configuration.max_retries:
                        raise
                    core.logger.warning("Retrying RAW range %s at byte %s", index, start + progress[index],
                                        exc_info=True)
                finally:
                    await _close_response(response)
                await _retry_delay(event, min(2 ** attempt, 30))
            return False

        tasks = [asyncio.create_task(chunk(index)) for index in range(len(progress))]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await state.save(progress)
        if _stopped(event):
            raise DownloadCancelled("Download cancelled.")
        if not all(results) or sum(progress) != size:
            raise NetworkRequestError("RAW download did not complete every byte range")
    else:
        for attempt in range(configuration.max_retries + 1):
            if _stopped(event):
                raise DownloadCancelled("Download cancelled.")
            offset = progress[0]
            if size and offset == size:
                break
            response = None
            request_headers = {"Accept-Encoding": "identity"}
            if offset:
                request_headers["Range"] = f"bytes={offset}-"
                if validator:
                    request_headers["If-Range"] = validator
            try:
                response = await session.request("GET", url, headers=request_headers, stream=True,
                                                 timeout=configuration.read_timeout, allow_redirects=True)
                response.raise_for_status()
                if offset and response.status_code == 200:
                    core.logger.warning("Server ignored RAW resume range; restarting from byte zero")
                    progress[0] = 0
                elif offset:
                    size = _range(response, offset, total=size or None)
                else:
                    size = int(response.headers.get("Content-Length", 0))
                state.data["size"] = size
                file = await asyncio.to_thread(output.open, "ab" if progress[0] else "wb")
                try:
                    async for data in response.aiter_content():
                        if _stopped(event):
                            raise DownloadCancelled("Download cancelled.")
                        if data:
                            await asyncio.to_thread(file.write, data)
                            progress[0] += len(data)
                            notify(progress[0], size)
                finally:
                    await asyncio.to_thread(file.close)
                    await state.save(progress)
                if size and progress[0] != size:
                    raise NetworkRequestError("RAW stream ended before its reported length")
                break
            except DownloadCancelled:
                raise
            except Exception:
                if attempt >= configuration.max_retries:
                    raise
                core.logger.warning("Retrying RAW stream from byte %s", progress[0], exc_info=True)
            finally:
                await _close_response(response)
            await _retry_delay(event, min(2 ** attempt, 30))
    if _stopped(event):
        raise DownloadCancelled("Download cancelled.")
    state.remove()
    core.logger.info("RAW download complete: path=%s bytes=%s", output, sum(progress))
    return True
