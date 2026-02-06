# Downloads: resume and cancel

This guide shows how to stop a download and pick it back up later. It focuses on usage, not internals.

## Threaded HLS downloads (m3u8)

Use `BaseCore.download(...)` for HLS streams. Enable resume by providing a state file path:

```python
import threading
from base_api import BaseCore
from base_api.modules.errors import DownloadCancelled

core = BaseCore()
stop_event = threading.Event()

try:
    core.download(
        video=video,
        quality="720",
        path="video.mp4",
        segment_state_path="video.segment-state.json",
        keep_segment_dir=True,
        stop_event=stop_event,
    )
except DownloadCancelled:
    print("Cancelled. Run again to resume.")
```

- Cancel: call `stop_event.set()` from your UI or another thread.
- Resume: call `download(...)` again with the same `segment_state_path` and `path`.

Notes:
- If you do not pass `segment_dir`, it defaults to `path + ".segments"`.
- `keep_segment_dir=True` keeps already-downloaded segments so resume is fast.
- Cancelled downloads raise `DownloadCancelled` unless you set `return_report=True`.
- On success, the state file is removed automatically.

## Direct file downloads (legacy_download)

For a single file URL, use `legacy_download(...)`:

```python
import threading
from base_api import BaseCore
from base_api.modules.errors import DownloadCancelled

core = BaseCore()
stop_event = threading.Event()

try:
    core.legacy_download(path="video.mp4", url=direct_url, stop_event=stop_event)
except DownloadCancelled:
    print("Cancelled. Run again to resume.")
```

- Resume: if `video.mp4` already exists, the downloader continues from where it left off.
- Cancel: call `stop_event.set()` (or stop the call). The partial file remains, so running it again resumes.

If the server does not support ranged downloads, the download restarts automatically.

## Using the threaded downloader directly

If you call `core.threaded(...)` yourself, pass the same `segment_state_path`, `stop_event`, and cleanup options. The
behavior is identical to `download(...)`.
