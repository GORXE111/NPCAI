---
name: 常用命令速查
description: SSH/Codex/AtomicRunner/Unity/测试的完整命令速查（2026-03-23更新）
type: reference
---

## SSH
```bash
ssh linyang@100.110.20.12  # Mac Mini M4 via Tailscale
```

## Atomic Runner（主引擎）
```bash
# 启动服务（持续运行+Web面板:7788）
cd ~/ai-unity/orchestrator
nohup python3.11 -u atomic_runner.py serve > ~/ai-unity/atomic_serve.log 2>&1 &

# 停止
pkill -9 -f 'atomic_runner'; pkill -9 -f 'codex'; lsof -ti :7788 | xargs kill -9 2>/dev/null

# 提交任务（自动Plan+Run+Test）
curl -X POST http://localhost:7788/api/task -H 'Content-Type: application/json' -d '{"goal":"..."}'

# 仅测试
curl -X POST http://localhost:7788/api/test

# 重试失败
curl -X POST http://localhost:7788/api/retry

# 手动跑已有tasks（跳过Plan，不需要API）
python3.11 atomic_runner.py run --resume --web

# 查状态
curl http://localhost:7788/api/progress
```

## Play Mode 测试（零API依赖）
```bash
# 执行测试（读 test_plan.json，文件通信）
cd ~/ai-unity/orchestrator && python3.11 play_test.py

# 手动进/出 Play Mode
touch ~/ai-unity/gtagame/.playmode_enter  # 等8秒
touch ~/ai-unity/gtagame/.playmode_exit   # 等3秒

# 手动发 TestBridge 命令
echo "query:scene" > ~/ai-unity/gtagame/.test_command
sleep 2 && cat ~/ai-unity/gtagame/.test_result
```

## Unity MCP（仅 Edit Mode 可靠）
```bash
cd ~/ai-unity/orchestrator && python3.11 -c '
from unity_mcp import UnityMCP
m = UnityMCP(); m.connect()
print(m.recompile().get("message",""))
print(len(m.get_errors()), "errors")
m.close()'
```

## 编译检查
```bash
# 查 Editor 编译错误
grep 'error CS' ~/Library/Logs/Unity/Editor.log | tail -10

# 通过 verify.py
python3.11 -c 'from verify import check_compile; ok,msg=check_compile(); print(ok,msg)'
```

## Codex CLI
```bash
export PATH=/opt/homebrew/bin:$PATH
codex exec --full-auto --skip-git-repo-check -C ~/ai-unity/gtagame '任务'
```

## Sprite 生成
```
Unity菜单: FreeLive > Generate UI Sprites
或: touch ~/ai-unity/gtagame/.run_sprite_gen 然后等 AutoRefresh 触发
```

## Git
```bash
cd ~/ai-unity/gtagame
git log --oneline | head -10
git reset --hard HEAD~1  # 回退
```
