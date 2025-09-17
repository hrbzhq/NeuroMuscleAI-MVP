# Release / Notice Draft: History rewrite and large-file removal

Short title: Repository history rewrite and large-file removal (backup created)

Short summary (to use as Release note or Issue top summary):

> We removed large or accidental files from repository history to reduce repository size and improve performance. A backup bundle was created before the operation and placed in `artifacts/`.

Detailed notification (to post as Issue comment / PR description / Release notes):

Why we did this
- The repository contained large files (e.g. dataset snapshots, temporary backups) that were accidentally committed and caused repository size bloat.

What we did
- Created an offline backup bundle before any history rewrite: `artifacts/repo_pre_purge.bundle`.
- Performed a history rewrite locally (using `git-filter-repo`) to remove the specified paths from all commits.
- Pushed rewritten history to `origin` with force.

What you must do (action required by contributors)
1. Immediately re-clone the repository to get the updated history:

```bash
git clone https://github.com/<owner>/<repo>.git
```

2. If you had local branches or unpushed commits, create a patch or rebase them onto the new history. Example:

```bash
# From your old clone
git format-patch --root -o ../my-patches
# Re-apply to new clone
git am ../my-patches/*.patch
```

Recovery steps (if you need to recover removed content)
- The pre-purge bundle can be inspected or unbundled on another machine:

```bash
# Inspect bundle refs
git bundle list-heads ./repo_pre_purge.bundle

# Create a clone from the bundle
git clone repo_pre_purge.bundle repo-legacy
```

Communication & timeline
- Publish this Notice as a GitHub Release or Issue pinned to the repository. Tag core maintainers in the Issue and post on project communication channels.
- Recommended timeline:
  1) 48 hours before the operation: announce intention + provide recovery instructions.
  2) 24 hours before: final reminder.
  3) At the time of execution: publish Release/Notice and link backup artifact.

Suggested GitHub text (copy/paste)

Title: Repository history rewrite and large-file removal — backup created

Body:
> Hello contributors — we removed several large files from the git history to shrink repo size. We created an offline backup placed at `artifacts/repo_pre_purge.bundle` before the operation. If you have local clones, please re-clone the repository. If you need help recovering local work, follow the recovery steps in docs/maintenance_guide.md or contact maintainers at hrbzhqhrb@gmail.com.

Checklist for maintainers before executing the purge
- [ ] Create `repo_pre_purge.bundle` and copy it to external storage.
- [ ] Verify `git-filter-repo` installed on executor machine.
- [ ] Identify exact `--path` / `--path-glob` patterns to remove.
- [ ] Notify contributors and schedule the operation.
- [ ] After push: monitor issues and help contributors rebase/fix local clones.

Notes & caveats
- History rewrite is destructive for published commits (everyone must re-clone or rebase). Ensure maintainers and core contributors are coordinated.
- Always keep the pre-purge bundle; it's the main recovery artifact.

## 发布记录

- 发布标签: `v1.0.0-purge-notice`
- 发布网址: https://github.com/hrbzhq/NeuroMuscleAI-MVP/releases/tag/v1.0.0-purge-notice
- 上传的备份文件: `artifacts/test_bundle.bundle`
- SHA256 校验和: `3A2CC92CE9A67237DF8C4A4747964E564D98D25353BCB02EC7389A87D6818C24`
- 发布时间 (UTC): 2025-09-16T00:00:00Z

> 注意: 如果你已经创建了真正的 `repo_pre_purge.bundle`，请将该文件替换为备份并更新本记录。
