#!/usr/bin/env bash
# 定期清理 git 历史里 assets/ 与 dist/ 的旧版本，保留当前快照。
#
# 背景：assets/ 下的 PNG/SVG 和 dist/ 下的 PDF/HTML 都会频繁重新生成，
# 每次几 MB 的二进制 blob 都会在 git pack file 里留下来。本脚本用
# git filter-repo 把所有历史里的 assets/ 与 dist/ 路径整段抹掉，再把
# 当前 assets/ 与 dist/ 作为一个新 commit 提交进去。
#
# ⚠️  破坏性操作：这会重写 main 的全部历史
#     - 旧 commit hash 全部变了
#     - 任何 clone 过这个仓库的本地副本都会和远端对不上，需要重新 clone
#     - 单人仓库可以放心用；多人协作前务必先约定
#
# 用法：
#     ./scripts/prune-generated-history.sh         # 交互式确认后执行
#     ./scripts/prune-generated-history.sh -y      # 跳过确认（脚本化场景）
#
# 执行完不会自动 push。脚本最后会提示你检查 git log，确认无误后再手动
# git push --force-with-lease origin main。

set -euo pipefail

#####################################################################
# 0. 解析参数
#####################################################################
AUTO_YES=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes) AUTO_YES=1 ;;
    -h|--help)
      awk 'NR==1{next} /^#/{sub(/^# ?/, ""); print; next} {exit}' "$0"
      exit 0
      ;;
    *)
      echo "未知参数：$arg" >&2
      exit 2
      ;;
  esac
done

#####################################################################
# 1. 前置检查
#####################################################################
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$REPO_ROOT" ]; then
  echo "ERROR: 不在 git 仓库内。" >&2
  exit 1
fi
cd "$REPO_ROOT"

if ! command -v git-filter-repo >/dev/null 2>&1 && ! git filter-repo -h >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: 没找到 git filter-repo。安装方式：

  pip install git-filter-repo
  # 或者 mac:    brew install git-filter-repo
  # 或者 ubuntu: apt install git-filter-repo

EOF
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: 工作区不干净，请先 commit 或 stash。" >&2
  git status --short >&2
  exit 1
fi

CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CUR_BRANCH" != "main" ]; then
  echo "ERROR: 当前不在 main 分支（在 $CUR_BRANCH）。" >&2
  echo "       本脚本只重写 main 的历史，请先 git checkout main。" >&2
  exit 1
fi

if [ ! -d assets ]; then
  echo "ERROR: 仓库里没有 assets/ 目录，没东西要清理。" >&2
  exit 1
fi

if [ ! -d dist ]; then
  echo "ERROR: 仓库里没有 dist/ 目录，没东西要清理。" >&2
  exit 1
fi

#####################################################################
# 2. 体积对比（清理前）
#####################################################################
SIZE_BEFORE=$(du -sh .git | awk '{print $1}')
PACK_BEFORE=$(du -sh .git/objects/pack 2>/dev/null | awk '{print $1}' || echo "?")

echo "==> 清理前 .git 大小：$SIZE_BEFORE（其中 pack：$PACK_BEFORE）"
echo "==> 当前 assets/ 文件数：$(find assets -type f | wc -l | tr -d ' ')"
echo "==> 当前 dist/ 文件数：$(find dist -type f | wc -l | tr -d ' ')"
echo

#####################################################################
# 3. 确认
#####################################################################
cat <<'EOF'
即将执行：
  1. 把当前 assets/ 与 dist/ 备份到仓库父目录
  2. git filter-repo --path assets/ --path dist/ --invert-paths --force
     —— 这会重写 main 的全部历史，删除所有历史 commit 里对 assets/ 与 dist/ 的引用
  3. 把备份的 assets/ 与 dist/ 还原，作为一个新 commit 提交
  4. 提示你手动 git push --force-with-lease

提醒：
  - 远端在你 force-push 后旧 blob 才真正变 unreachable，需要等 GitHub
    自己 GC（通常几天到几周）才会真正释放配额
  - 本地 .git 在脚本末尾会自动 git gc --prune=now 立即回收
EOF

if [ "$AUTO_YES" -ne 1 ]; then
  echo
  read -r -p "继续？输入 yes 确认：" REPLY
  if [ "$REPLY" != "yes" ]; then
    echo "已取消。"
    exit 0
  fi
fi

#####################################################################
# 4. 备份当前 assets/ 与 dist/
#####################################################################
# 不用 /tmp：很多发行版（Fedora/Arch/部分 systemd 默认）把 /tmp 挂成 tmpfs，
# 一旦脚本运行途中机器崩溃/断电，备份会随 RAM 一起丢失，而此时 filter-repo
# 可能已经把 assets/ 与 dist/ 从历史里抹掉 —— 这两个目录就真的没了。
# 改放到仓库父目录（同一持久化磁盘，且不受 tmpfs 容量上限影响）。
BACKUP_DIR=$(mktemp -d -p "$(dirname "$REPO_ROOT")" .prune-generated-backup-XXXXXX)

# 异常退出时把备份路径打到 stderr，避免半途崩溃后用户找不到备份。
# 正常路径走到末尾会主动删掉备份目录并解除 trap。
trap 'echo "⚠️  脚本异常退出，assets/ 与 dist/ 备份仍在：$BACKUP_DIR" >&2' ERR

echo
echo "==> 备份当前 assets/ → $BACKUP_DIR/assets"
cp -a assets "$BACKUP_DIR/assets"
echo "==> 备份当前 dist/ → $BACKUP_DIR/dist"
cp -a dist "$BACKUP_DIR/dist"

#####################################################################
# 5. 保存 origin URL（filter-repo 会出于安全考虑把 origin 删掉）
#####################################################################
ORIGIN_URL=$(git remote get-url origin 2>/dev/null || true)

#####################################################################
# 6. 跑 filter-repo
#####################################################################
echo "==> git filter-repo --path assets/ --path dist/ --invert-paths --force"
git filter-repo --path assets/ --path dist/ --invert-paths --force

#####################################################################
# 7. 恢复 origin
#####################################################################
if [ -n "$ORIGIN_URL" ]; then
  if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin "$ORIGIN_URL"
    echo "==> 已恢复 origin: $ORIGIN_URL"
  fi
fi

#####################################################################
# 8. 还原 assets/ 与 dist/ 并 commit
#####################################################################
echo "==> 还原 assets/ 与 dist/ 并提交快照"
cp -a "$BACKUP_DIR/assets" assets
cp -a "$BACKUP_DIR/dist" dist
git add assets dist
git commit --author="weiqiang <weiqiangnd@gmail.com>" -m "build: assets 与 dist 快照（历史已清理）

定期执行 scripts/prune-generated-history.sh 后的当前 assets/ 与 dist/ 快照。
所有历史 assets/ 与 dist/ blob 已通过 git filter-repo 从 main 历史中移除。"

#####################################################################
# 9. 校验备份与还原一致，一致则删除备份
#####################################################################
echo "==> 校验备份 $BACKUP_DIR 与仓库 assets//dist/ 是否一致"
if diff -rq "$BACKUP_DIR/assets" assets >/dev/null && diff -rq "$BACKUP_DIR/dist" dist >/dev/null; then
  echo "    一致，删除备份目录"
  rm -rf "$BACKUP_DIR"
  BACKUP_DIR=""   # 让后续提示知道备份已清理
else
  echo "⚠️  备份与还原内容不一致，保留备份目录供人工排查：$BACKUP_DIR" >&2
fi

# 走到这里说明主要步骤都成功了，解除 ERR trap，避免后续无关报错触发误导提示。
trap - ERR

#####################################################################
# 10. 本地 GC
#####################################################################
echo "==> 本地 git gc --prune=now --aggressive"
git reflog expire --expire=now --all >/dev/null 2>&1 || true
git gc --prune=now --aggressive >/dev/null 2>&1 || true

#####################################################################
# 11. 体积对比 + 后续提示
#####################################################################
SIZE_AFTER=$(du -sh .git | awk '{print $1}')
PACK_AFTER=$(du -sh .git/objects/pack 2>/dev/null | awk '{print $1}' || echo "?")

echo
echo "✅ 完成。"
echo "    .git 大小：$SIZE_BEFORE → $SIZE_AFTER"
echo "    pack 大小：$PACK_BEFORE → $PACK_AFTER"
if [ -n "$BACKUP_DIR" ]; then
  echo "    备份目录：$BACKUP_DIR（一致性校验未通过，已保留，确认无误后可手动 rm -rf）"
else
  echo "    备份目录：已自动清理"
fi
echo
echo "下一步——先 fetch 让 force-with-lease 的 lease 新鲜，再强推："
echo
echo "    git log --oneline -5                       # 先肉眼检查一遍"
echo "    git fetch origin main                      # ⚠ 关键：刷新 lease，紧贴 push 前"
echo "    git push --force-with-lease origin main"
echo
echo "fetch 和 push 之间不要做别的操作，否则 lease 会再次失效报 (stale info)。"
echo
echo "推完之后，团队里其他人（如果有）同步方式："
echo "  - 本地 main 没有未推送的改动：不必重新 clone，执行"
echo "        git fetch origin"
echo "        git checkout main"
echo "        git reset --hard origin/main"
echo "  - 本地 main 有未推送的改动 / 有基于旧 main 的工作分支：建议重新 clone，"
echo "    或者手动 git rebase --onto origin/main <旧 main tip> <work-branch>"
echo "    把本地工作嫁接到新历史上。"
