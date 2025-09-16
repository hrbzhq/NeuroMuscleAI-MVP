# 参考项目维护模式分析报告

来源：`tools/reference_artifacts/study`、`tools/reference_artifacts/allmcp`、`tools/reference_artifacts/ecplpgy`

目的：从三个示例项目中提取常见且可迁移的维护脚本、CI 配置与流程，并为 `NeuroMuscleAI-MVP` 制定可执行的维护工具清单。

摘要发现（高层）

- 健康检查：所有项目都有健康检查脚本（HTTP 健康探针，返回 JSON，并打印可用服务/工具列表）。示例：`allmcp/scripts/health_check.sh`。
- 启动/部署脚本：有统一的启动脚本（包含虚拟环境、依赖安装、端口检查）与部署脚本（支持 Docker 与本地部署、带环境变量校验与回退）。示例：`allmcp/scripts/start.sh`, `ecplpgy/scripts/deploy-production.ps1`。
- 自动化与重试：部署脚本包含重试与等待逻辑（健康检查重试循环），并在多次失败后退出；这对自动化 purge/push 脚本很有借鉴意义。
- 备份与 bundle：存在 `git bundle` 或类似备份流程的建议（我们已在 `tools/network_fallback` 中实现）。
- 通知与模板：项目包含通知模板与监控入口（`ecplpgy` 的 `notifications.json`、`study` 的 PR 模板），可用于自动通知协作者。
- CI/Release：`.github/workflows` 在样本中广泛存在，包含 test/build/deploy 等 job，可参考其中的 job 结构与 problem matchers。

可迁移的维护功能清单（建议优先级）

1. 健康检查工具（高优先级）
   - 统一的 `tools/maintenance/health_check.ps1`（PowerShell & bash wrappers）用于检测本地服务/端口、返回 JSON 并支持重试。

2. 备份/打包（高优先级）
   - `tools/maintenance/make_backup.ps1`：创建 `git bundle`、验证大小并保存到 `artifacts/`。

3. 自动重试与 purge 脚本（中高优先级）
   - `tools/maintenance/auto_purge_and_push.ps1`：在网络恢复后自动运行（dry-run 模式、日志、退避重试、备份创建、`git-filter-repo` 检查），按授权后执行强推。

4. 部署/健康/启动脚本模板（中优先级）
   - 参考 `start.sh` 与 `deploy-production.ps1` 的模式，生成可配置的启动脚本（检查依赖、激活 venv、端口检测、日志目录）。

5. 通知模板与协作者指南（中优先级）
   - 已创建 `docs/recovery_notification.md`，可加入自动化发送占位（require external service）。

6. CI 工作流示例（低中优先级）
   - 参考 `.github/workflows` 中的 job 定义，添加一个 `maintenance` workflow（定期运行健康检查、备份 bundle 并上传到 artifacts）。

实现风险与注意事项

- 凡涉及 `git push --force` 的操作必须在团队授权和通知后进行。脚本应实现 dry-run 并要求明确的 `--confirm` 标志。 
- 不要把敏感凭据或大文件（如原始 bundle）提交到主仓库历史；使用 `.gitignore`、`artifacts/` 或 Releases。 
- `git-filter-repo` 需要在目标环境中安装；脚本需检测并友好提示。

下一步建议

1. 我可以实现上述功能中的 1、2、3（health_check、make_backup、auto_purge_and_push 的改进版），并把它们放在 `tools/maintenance/`，先实现 dry-run 并提交。请确认。
2. 我可以再创建一个 `docs/maintenance.md` 来说明如何使用这些工具和在何种情况下授权强推。
