# Network fallback & help template

If you cannot push to GitHub due to local network restrictions, follow these steps or paste the help request below into the project's chat/issue board to find a colleague with external access.

1) Create a git bundle locally (recommended):

```powershell
cd /path/to/repo
scripts\create_bundle.ps1
```

2) Verify and share the file `repo.bundle` (checksum printed by the script). Transfer `repo.bundle` to a colleague who has external network access.

3) Colleague side (after receiving `repo.bundle`):

```powershell
# create a clone from bundle
git clone repo.bundle repo-from-bundle
cd repo-from-bundle
# add remote and push
git remote add origin https://github.com/hrbzhq/NeuroMuscleAI-MVP.git
git push origin --all
git push origin --tags
```

Help request template to paste in team chat / issue tracker:

```
Hi team — I can't push to GitHub from my network (HTTPS port 443 blocked). I've prepared a `repo.bundle` with all local commits. Could someone with external network access push it to `origin/master` for me?

Steps I took:
- Created `repo.bundle` (SHA256: <paste hash here>) using `scripts/create_bundle.ps1`.
- Verified locally.

What I need from you:
- Download `repo.bundle` from my shared location (or I can send it to you).
- Run:
  git clone repo.bundle repo-from-bundle
  cd repo-from-bundle
  git remote add origin https://github.com/hrbzhq/NeuroMuscleAI-MVP.git
  git push origin --all
  git push origin --tags

Thanks — I will confirm once remote shows the new commits.
```
