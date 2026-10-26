# Downloads: pause, resume and cleanup

All downloads use async provider/core methods and an `asyncio.Event` for
cooperative stopping. Set the event, then await the active download before
resuming or deleting its files. Use a new unset event for the next invocation.

## HLS streams

```python
import asyncio
from base_api import BaseCore, DownloadConfigHLS

async def main(master_url):
    stop = asyncio.Event()
    configuration = DownloadConfigHLS(
        quality="720", path="video.mp4", m3u8_base_url=master_url,
        segment_state_path="video.segment-state.json",
        segment_dir="video.segments", keep_segment_dir=True,
        cleanup_on_stop=False, return_report=True, remux=True,
        stop_event=stop,
    )
    async with BaseCore() as core:
        # Another async task or UI callback can call stop.set().
        report = await core.download(configuration)
        if report.status == "cancelled":
            configuration.stop_event = asyncio.Event()
            report = await core.download(configuration)
```

Reusing the output path, segment directory and state file restores saved
segments. A cancelled HLS download raises `DownloadCancelled` unless
`return_report=True`. Successful downloads remove the state file; retained
segment directories can then be purged by the application.

## Direct files (RAW)

```python
import asyncio
from base_api import BaseCore, DownloadConfigRAW
from base_api.modules.errors import DownloadCancelled

async def main(direct_url):
    stop = asyncio.Event()
    configuration = DownloadConfigRAW(
        quality="720", path="video.mp4", no_title=True, stop_event=stop,
        # Optional; the default is video.mp4.raw-state.json.
        resume_state_path="video.raw-state.json",
    )
    async with BaseCore() as core:
        try:
            # Another async task or UI callback can call stop.set().
            await core.legacy_download(direct_url, configuration)
        except DownloadCancelled:
            configuration.stop_event = asyncio.Event()
            await core.legacy_download(direct_url, configuration)
```

The RAW downloader now keeps a JSON checkpoint beside the output by default.
Keep both files when pausing. Provider clients pass the same
`DownloadConfigRAW` through to this implementation.

- Multipart mode records the byte count written in each range, including partial
  ranges when stopping. Resume requests only the missing tails, reports existing
  progress immediately, and supports changing worker count between invocations.
- Preallocated file size is **never** treated as downloaded progress. A file from
  an older version without a valid checkpoint is downloaded afresh.
- Linear mode also persists its byte offset. If the server ignores a resume
  Range and returns HTTP 200, linear mode restarts safely from zero.
- Resumed multipart ranges must return HTTP 206 and matching Content-Range
  offsets/total before data is written. Truncated ranges are retried from their
  written offset; queued workers check the stop event after acquiring a slot.
- The checkpoint ties the output to its original source URL, transfer size and
  available ETag/Last-Modified validators. Incompatible checkpoints are reset.
  Temporary signed source URLs that change between invocations may therefore
  require a fresh download.
- A metadata request failure preserves an existing checkpoint and propagates
  the error, rather than truncating the user's partial file.
- Workers close their files and streaming responses before stop returns.
  Checkpoints are written atomically. Successful downloads remove their
  checkpoint automatically.

For **cleanup on stop**, first set the event and await `DownloadCancelled`, then
remove the output, its `resume_state_path` (or the default sidecar), and any
checkpoint `.tmp` file. Removing files while download workers still run can
leave files recreated by those workers.

`allow_multipart=True` uses concurrent ranges when the server advertises support.
`allow_multipart=False` uses a linear stream. Resume with the same transfer mode;
changing mode invalidates the previous checkpoint.
