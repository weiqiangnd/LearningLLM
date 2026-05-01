#!/bin/bash
set -euo pipefail

# 仅在 Claude Code on the web 的远端 session 中执行；本地 CLI 不动
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# 拉最新远端引用，--prune 清理已删除的 remote-tracking 分支
git fetch origin --prune

# 切到 main（如果当前在 claude/* 工作分支上，这一步会切走，从而允许下面删它）
# 已经在 main 上时这步是 no-op
git checkout main

# 把本地 main 强制对齐到 origin/main，丢弃任何本地差异
git reset --hard origin/main

# 删除本地遗留的 claude/* 工作分支（harness 每次 session 都会新建一个，不清理会越积越多）
for branch in $(git for-each-ref --format='%(refname:short)' refs/heads/claude/ 2>/dev/null || true); do
  git branch -D "$branch" || true
done
