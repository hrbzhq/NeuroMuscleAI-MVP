Network fallback and history-cleanup utilities
=============================================

This folder contains helper scripts for two purposes:

- network fallback: create `git bundle` or `format-patch` artifacts when HTTPS push is unavailable;
- history cleanup: safely remove large or sensitive files from repository history using `git-filter-repo`.

WARNING: History cleanup is destructive for published repos (it rewrites commits). Only run `purge_history.ps1` after making and verifying backups and with repository owner permission. The script below is designed to operate on a mirror clone and push the cleaned mirror back to the origin.

Running the history purge (offline steps)
----------------------------------------

Note: The history purge must be run in a cloned *mirror* repository and requires network access to push the cleaned mirror back to the remote. If your environment currently cannot reach GitHub (HTTPS blocked), perform the mirror clone on a machine that can access GitHub.

Recommended steps (run on a machine with good network access):

1. Create a mirror clone (outside your working repo):

```powershell
git clone --mirror https://github.com/<owner>/<repo>.git repo-mirror.git
cd repo-mirror.git
```

2. Run the purge script inside the mirror clone (example removes `data` and `.git_backup_20250915144245`):

```powershell
powershell -ExecutionPolicy Bypass -File ..\tools\network_fallback\purge_history.ps1 -PathsToRemove "data",".git_backup_20250915144245"
```

3. Inspect the mirror repository and verify the history has been cleaned.

4. Push the cleaned mirror back to origin (force):

```powershell
git push --force --all
git push --force --tags
```

5. Notify collaborators: after the force-push, all collaborators must re-clone the repository.

If you cannot execute the above due to network restrictions, create a `repo.bundle` (using `make_bundle.ps1`) on the machine that has the up-to-date local commits and transfer the bundle to a machine with GitHub access; then unbundle and push from there.

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
