# 4.3.0

Release 4.3.0

Update pagination handling, validation, and documentation.

Changes since v4.2.6:

```text
README.md                    | 27 +++++++++++++++++++++++++++
 base_api/__init__.py         | 11 +++++++++++
 base_api/base.py             |  5 +++++
 base_api/modules/provider.py |  8 ++++++++
 base_api/modules/urls.py     |  4 +++-
 pyproject.toml               |  2 +-
 uv.lock                      |  2 +-
 7 files changed, 56 insertions(+), 3 deletions(-)
```

# Changelog

## 4.2.6 — 2026-11-05

- Launch FFmpeg with explicit null standard input and captured output during remuxing and metadata writing. Windows GUI launches no longer depend on an inherited console input handle.
- Keep FFmpeg subprocesses noninteractive and prevent console windows from appearing on Windows.

## 4.2.5 — 2026-10-27

- Keep URLs containing percent escapes from breaking contextual log formatting.
- Sanitize automatic download filenames and use a safe fallback when a title is missing.
- Include the resumable RAW downloader updates previously present only in the source checkout.
