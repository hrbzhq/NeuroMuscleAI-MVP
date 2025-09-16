Network fallback and history-cleanup utilities
=============================================

This folder contains helper scripts for two purposes:

- network fallback: create `git bundle` or `format-patch` artifacts when HTTPS push is unavailable;
- history cleanup: safely remove large or sensitive files from repository history using `git-filter-repo`.

WARNING: History cleanup is destructive for published repos (it rewrites commits). Only run `purge_history.ps1` after making and verifying backups and with repository owner permission. The script below is designed to operate on a mirror clone and push the cleaned mirror back to the origin.

Files
-----
- `make_bundle.ps1` — create `repo.bundle` to transfer the full repo by file.
- `export_patches.ps1` — export local commits as patches (format-patch) for manual application elsewhere.
- `purge_history.ps1` — perform a safe history purge using a mirror clone and `git-filter-repo`.

Prerequisites
-------------
- PowerShell (Windows)
- Python available for `git-filter-repo` (installed via pip)
- Network access to GitHub for pushing the cleaned mirror (or use a relay)

Usage examples and notes are in the scripts. Read them before running.
