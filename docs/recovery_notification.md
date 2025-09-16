# 恢复通知模板（邮件 / Slack）

以下为可直接复制粘贴的通知模板。请在计划强制推送（force-push）前 24 小时通知团队，并在完成后再次发送完成通知。

---

主题（邮件）/ 首行（Slack）：[仓库维护通知] 计划的历史重写（强制推送） — YYYY-MM-DD HH:MM UTC

正文（可直接复制）：

```
各位好，

我们计划于 YYYY-MM-DD HH:MM UTC 对 `hrbzhq/NeuroMuscleAI-MVP` 仓库执行一次历史重写并强制推送（`git push --force`）。此操作会修改主分支历史，所有本地克隆的提交 ID 将发生变化。

影响范围：
- 主分支（master）将被重写，相关的分支可能需要重新同步。

你需要做的操作（请在维护开始前完成备份）：
1) 如果你没有未提交改动或本地提交不打算保留，请在维护后运行：

```
git fetch origin
git checkout master
git reset --hard origin/master
git clean -fd
```

2) 如果你有本地未推送的提交并想保留，请在维护前导出补丁或在维护后按照 `docs/recovery_after_force_push.md` 中的说明应用补丁恢复你的提交。

遇到问题请联系：@维护人 或 email@example.com（请替换为实际联系人）。

谢谢配合。

维护团队
```

完成通知（维护完成后发）：

```
维护已完成。
远端历史已重写并推送。请尽快按照以下命令同步本地仓库：

git fetch origin
git checkout master
git reset --hard origin/master
git clean -fd

如遇问题请联系：@维护人 或 email@example.com
```

---

我可以按需把通知模板替换为你团队的实际联系人和时间格式（本地时间或 UTC）。是否要我把示例联系人替换成具体的名字/邮箱？
