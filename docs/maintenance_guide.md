# 维护与回滚指南（Maintenance & Recovery Guide）

此文档列出维护仓库、创建离线备份、进行历史清理（git-filter-repo）以及在 GitHub 上协调强制推送的步骤与建议文案。所有步骤以最小破坏为目标，并尽量使用 dry-run / 备份策略。

## 快速概览
- 在任何破坏性操作之前，先创建 `git bundle` 备份并把 bundle 复制到外部安全位置。
- 使用 `tools/maintenance/auto_purge_and_push.ps1` 的 dry-run 模式预览计划（默认 dry-run）。
- 仅在网络恢复并且你确认时，使用 `-Confirm` 执行历史重写并强推（force push）。

## 主要脚本（位置）
- `tools/maintenance/health_check.ps1` — 健康检查脚本，探测 HTTP endpoint 并支持 JSON 输出。
- `tools/maintenance/make_backup.ps1` — 在仓库根创建 `git bundle` 备份。
- `tools/maintenance/auto_purge_and_push.ps1` — 包装脚本：支持 `-Backup`、默认 dry-run，需要 `-Confirm` 才会运行 `git-filter-repo` 并强推。慎用。

## 操作步骤（本地执行示例）
1. 在仓库根创建并验证备份：

```powershell
.\tools\maintenance\make_backup.ps1 -Output '.\artifacts\repo_pre_purge.bundle'
# 手动复制 .\artifacts\repo_pre_purge.bundle 到外部存储或另一台机器
```

2. 在本地或镜像仓库上 dry-run（确认将要运行的 filter-repo 参数与计划）：

```powershell
.\tools\maintenance\auto_purge_and_push.ps1
# 输出示例会显示 Filter-repo args 和计划的远程推送
```

3. 如需执行（危险操作 — 仅在你确认并通知协作者后执行）：

```powershell
.\tools\maintenance\auto_purge_and_push.ps1 -Backup -Confirm
```

4. 若需要在远程强推后通知协作者，可参考下方“建议 GitHub 文案”。

## 安全检查表（Checklist）
- [ ] 已创建并离线保存 `repo_pre_purge.bundle`。
- [ ] 通知并获得核心贡献者/维护者的书面同意（Slack/邮件/Issue 评论）。
- [ ] 在另一台可访问 GitHub 的机器上预演 force-push（或由可信维护者代为执行）。
- [ ] 执行后在仓库主页与 README 通告变更并提供恢复步骤。

## 我在实现这些脚本中学到的技能（可在 GitHub 上写进维护说明）
1. 安全的历史重写流程：先 `git bundle` 备份 → 在镜像 clone（mirror）上运行 `git-filter-repo` → 本地验证变更 → 强制推送回 origin。
2. 干运行（dry-run）与显式确认门控：所有破坏性脚本默认 dry-run，必须传入明确的 `-Confirm` 标志方可执行。
3. 备份与回滚策略：在每次历史变更前生成 bundle 并将其保管在外部位置作为回滚点。
4. 健康检查与自动重试模式：探测服务可用性、重试策略（最大尝试次数、延迟）、在自动化脚本中加入日志与退出码以便 CI/ops 监控。
5. PowerShell 脚本的可移植性注意事项：避免复杂的字符串插值模式（会触发静态检查器的误报），用明确的字符串构造或格式函数代替，给脚本添加 dry-run 与日志功能。

## 建议发布到 GitHub 的短文案（供 Release Notes / README 更新使用）
标题：Repository history rewrite and large-file removal (backup created)

正文（样例）：
> We have removed large/accidental files from repository history to reduce repository size and improve performance. A backup bundle was created and uploaded to `artifacts/` before the operation. If you have local clones, please re-clone the repository. For recovery or questions, contact maintainers at hrbzhqhrb@gmail.com.

可选更详细通知（流程与恢复步骤）：
> 1) We've created an offline backup: `artifacts/repo_pre_purge.bundle`.
> 2) Please re-clone: `git clone https://github.com/<owner>/<repo>.git`.
> 3) If you need to restore old history, use:
> ```bash
> git clone <repo_url> repo-old
> git bundle unbundle ./repo_pre_purge.bundle
> ```

## 联系与后续
如需我（或维护团队）代为在网络恢复时执行强推，请在 Issue 中 @ 指定维护人员并附上确认信息；脚本位于 `tools/maintenance/`，并配有说明文档。
