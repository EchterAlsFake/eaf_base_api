# Changelog

## 4.2.6 — Unreleased

- Launch FFmpeg with explicit null standard input and captured output during remuxing and metadata writing. Windows GUI launches no longer depend on an inherited console input handle.
- Keep FFmpeg subprocesses noninteractive and prevent console windows from appearing on Windows.

## 4.2.5 — 2026-10-27

- Keep URLs containing percent escapes from breaking contextual log formatting.
- Sanitize automatic download filenames and use a safe fallback when a title is missing.
- Include the resumable RAW downloader updates previously present only in the source checkout.
