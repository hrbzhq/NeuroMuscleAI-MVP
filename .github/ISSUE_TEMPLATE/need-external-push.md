---
name: Need external push
about: Request a colleague to push a provided repo.bundle or patches to the remote when local network blocks pushing
title: '[REQUEST] External push assistance needed'
labels: ['help wanted', 'ops']
assignees: []
---

Please attach or link to the `repo.bundle` or `patches.zip` and include the SHA256 checksum. Follow the instructions in `docs/network_fallback.md`.

Steps already performed:

- Created `repo.bundle` using `scripts/create_bundle.ps1`.
- Verified bundle checksum.

What I need from volunteers:

- Download `repo.bundle`
- Run:

```powershell
git clone repo.bundle repo-from-bundle
cd repo-from-bundle
git remote add origin https://github.com/hrbzhq/NeuroMuscleAI-MVP.git
git push origin --all
git push origin --tags
```

Add a comment here once you have pushed so I can verify.
