---
name: GTA5 Unity AI 自动化项目（完整工程信息）
description: Mac Mini M4 上用 AI 全自动化开发 GTA5 风格 Unity 游戏的完整信息，包括连接方式、路径、工具链、编排器、已知问题
type: project
---

## 连接 Mac Mini
```bash
ssh linyang@100.110.20.12    # Tailscale 免密连接
```
- Mac 用户: linyang
- Mac 主机名: linyangdeMac-mini
- Windows Tailscale IP: 100.86.242.89
- Windows 有系统代理 127.0.0.1:7897（Python 需设 NO_PROXY）

## 关键路径（Mac）
```
~/ai-unity/                          ← AI 工程根目录
├── AGENTS.md                        ← 主控文件（架构+规则+命令）
├── CONSTITUTION.md                  ← 开发宪法（8条硬规则）
├── docs/
│   ├── ARCHITECTURE.md              ← 分层架构
│   ├── golden-principles.md         ← 5条核心原则
│   ├── PROJECT_ROADMAP.md           ← 项目路线图（5个Phase）
│   └── exec-plans/{active,completed}/
├── roles/                           ← 7个 Agent 角色定义
│   ├── planner.md                   ← 规划师（read-only sandbox）
│   ├── client_developer.md          ← 客户端开发者
│   ├── server_developer.md          ← 服务器开发者
│   ├── reviewer.md                  ← 审查员（苛刻评分）
│   ├── test_writer.md               ← 测试编写员
│   ├── tester.md                    ← 测试执行员
│   └── content_creator.md           ← 内容创建者
├── skills/                          ← 13个技能文件
│   ├── codex-dispatch.md            ← Codex 任务派发方法
│   ├── unity-mcp-operations.md      ← MCP 操作 Unity
│   ├── orchestrator-usage.md        ← 编排器使用
│   ├── unity-test-automation.md     ← 自动化测试（state+visual）
│   ├── unity-test-patterns.md       ← 测试模式选择
│   ├── lowpoly-generation.md        ← Low Poly 生成（已废弃，改用其他方案）
│   ├── voxel-generation.md          ← 体素生成（已废弃）
│   ├── gta5-code-research.md        ← GTA5 源码研究（树状原子化）
│   └── ...其他
├── reference/gta5-source/           ← GTA5 参考源码 (16GB)
│   ├── INDEX.md                     ← 按任务查找系统
│   └── src/dev_ng/game/             ← 18个系统，每个有 AGENTS.md + docs/
│       ├── Vehicles/docs/ (6个子文档)
│       ├── Peds/docs/ (7)
│       ├── ai/docs/ (6)
│       └── ...共 111 个子文档
├── orchestrator/                    ← Python 编排系统
│   ├── orchestrator.py              ← 确定性状态机（默认走 pipeline）
│   ├── agents.py                    ← Agent 角色配置
│   ├── codex_runner.py              ← Codex exec 执行器（需要 PATH 环境变量）
│   ├── unity_tools.py               ← 编译检查 + 测试（MCP 方式）
│   ├── web.py                       ← Web 面板（ThreadingHTTPServer）
│   └── run.sh
├── mcp-unity-server/                ← Unity MCP 服务器
│   └── Server~/build/index.js
└── gtagame/                         ← Unity 6000.3.11f1 项目
    └── Assets/
        ├── Scripts/{Core,Vehicle,Character,World,Network,UI}/AGENTS.md
        ├── Scripts/Core/Testing/     ← 测试框架（6个C#文件）
        └── Prefabs,Scenes,Materials/AGENTS.md
```

## 工具链
- Unity: 6000.3.11f1 路径: /Applications/Unity/Hub/Editor/6000.3.11f1/Unity.app/Contents/MacOS/Unity
- Codex CLI: 0.115.0 配置: ~/.codex/config.toml + auth.json
- Codex provider: aixj (https://aixj.vip), model: gpt-5.4
- MCP: mcp-unity → ~/ai-unity/mcp-unity-server/Server~/build/index.js
- Python: /opt/homebrew/bin/python3.11
- Node: /opt/homebrew/bin/node (v25.8.1)
- PATH 重要: subprocess 调用 codex 必须设 env PATH 包含 /opt/homebrew/bin

## 编排器操作
```bash
# 启动
ssh linyang@100.110.20.12 "cd ~/ai-unity/orchestrator && nohup /opt/homebrew/bin/python3.11 orchestrator.py --web > ~/ai-unity/orchestrator.log 2>&1 &"

# Web 面板
http://localhost:7788  (Mac本地)
http://100.110.20.12:7788  (Windows远程)

# 提交任务（通过API）
curl -s -X POST http://localhost:7788/api/task -H 'Content-Type: application/json' -d '{"description":"任务描述"}'

# 直通模式（跳过流水线）
前缀 /direct: {"description":"/direct 直接执行的任务"}

# 停止
pkill -9 -f 'python3.11 orchestrator'; lsof -ti :7788 | xargs kill -9
```

## 流水线
```
PLAN(Planner, read-only) → DEV_LOOP(Developer→Compile→Review ×5, 前3轮强制) → TEST → FINAL
```

## Codex 直接派发
```bash
ssh linyang@100.110.20.12 "export PATH=/opt/homebrew/bin:\$PATH && codex exec --full-auto --skip-git-repo-check -C ~/ai-unity/gtagame '任务描述'"
```

## 已知问题与经验
1. codex_runner.py 的 subprocess 必须设 env PATH（否则 node not found）
2. Planner 必须用 --sandbox read-only（否则它会执行工具而不输出JSON计划）
3. compile_check_mcp: 无 CS error = 通过，异常也默认通过（宁漏勿误）
4. Developer prompt 必须强调"实际创建文件"（否则只输出文字）
5. Reviewer 第一轮容易给高分，需要强制前3轮迭代
6. Web server 必须用 ThreadingHTTPServer（SSE 会阻塞单线程服务器）
7. Ollama Qwen3.5 有 tool parsing bug（用 Qwen3:8b 或远端模型避免）
8. Mac 系统 bash 版本 3.x 不支持 declare -A（用兼容语法）
9. 体素系统编译不过已删除，Low Poly 方案也废弃
10. API provider(aixj.vip) 偶尔 502/503，需要重试
