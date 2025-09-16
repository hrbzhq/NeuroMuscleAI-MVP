# 恢复指南：当远端被强制推送（history rewrite）后

此文档为协作者在仓库历史被重写并且远端被 `git push --force` 后的恢复步骤提供清晰、可复制的命令（PowerShell 风格），并说明不同情况下的处理方法。

重要说明

- 重写历史后，原有分支的提交 ID 可能改变，老的本地分支将与远端分叉。请在操作前备份本地改动（saving local work）。
- 如果你不确定自己的本地改动是否重要，请先把当前工作区打包（`git status`, `git stash`, 或创建补丁）。

通用准备（先执行）

```powershell
# 检查当前状态
git status --porcelain
git branch --show-current

# 备份当前未提交改动为 stash（可选）
git stash push -m "pre-force-push-backup"

# 或者导出未提交改动为补丁文件
git diff > ../local_uncommitted_changes.patch
```

情形 A：你的本地没有未提交改动（推荐流程）

如果你本地的分支是干净的（`git status` 显示无改动），按以下步骤操作：

```powershell
# 把远端当前状态抓取到本地
git fetch origin

# 强制让本地分支指向远端最新（注意：这会丢弃本地与远端不同的提交）
git reset --hard origin/master

# 可选：清理本地无用引用与未跟踪文件
git clean -fd
```

情形 B：你有本地提交但尚未推送（你想保留这些提交）

步骤：把你的提交保存为补丁或临时分支，然后把它们重新应用到新的远端历史上。

```powershell
# 假设你当前分支是 feature/my-work
git checkout -b tmp/my-work-backup

# 导出本地提交作为补丁（自从分叉点之后的提交）
git format-patch origin/master..tmp/my-work-backup -o ../my_patches

# 切回 master 并硬重置为远端（使用 master 或主要分支名）
git checkout master
git fetch origin
git reset --hard origin/master

# 重新应用补丁（按顺序）
git am ../my_patches/*.patch

# 或者把备份分支变基到最新地址
git checkout tmp/my-work-backup
git rebase --onto origin/master <old-base> tmp/my-work-backup

# 检查并解决可能出现的冲突，之后再提交/推送
```

情形 C：你本地有未提交改动并且想保留

```powershell
# 保存当前未提交改动为 stash
git stash push -m "save-before-reset"

# 重置本地主分支到远端最新
git fetch origin
git checkout master
git reset --hard origin/master

# 恢复你的未提交改动
git stash pop

# 处理冲突并提交
```

管理员建议（在你执行强制推送前）

- 在重写历史并强推前，仓库管理员应创建一个远端备份（例如创建一个 tag 或 release，或在另一处保存 `repo.bundle`）：

```powershell
git fetch --all
git bundle create ../pre_purge_backup.bundle --all
```

- 通知所有协作者做好准备（建议至少 24 小时提前通知并给出恢复说明）。

发布后常见问题

- 我把本地改动覆盖了怎么办？
  - 如果你之前 `git stash` 或导出了补丁，可以从 stash 或补丁恢复。若没有备份，恢复会很困难。

- 我看到合并冲突或 am/rebase 阶段失败
  - 按照 Git 的提示解决冲突，逐个文件处理并完成 `git rebase --continue` 或 `git am --continue`。

参考命令（快速粘贴）

```powershell
# 清理并同步到远端
git fetch origin
git checkout master
git reset --hard origin/master
git clean -fd

# 保存未提交改动
git stash push -m "pre-force-push-backup"

# 导出本地提交为补丁
git format-patch origin/master..HEAD -o ../my_patches

# 从补丁恢复
git apply ../local_uncommitted_changes.patch
git am ../my_patches/*.patch
```

如果你需要我为团队定制一份短邮件/Slack 通知模板和恢复命令清单，我可以把它添加到本文件末尾或生成独立的 `docs/recovery_notification.md`。
