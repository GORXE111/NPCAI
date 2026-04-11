---
name: Unity 自动编译不需要 batchmode
description: Unity 已通过 MCP 开着，保存文件后自动编译，不要启动第二个 Unity 实例
type: feedback
---

Unity 项目已经通过 MCP 打开着，.cs 文件保存后自动编译，不需要手动跑 batchmode 编译。

**Why:** 启动第二个 Unity 实例会报 "Another Unity instance is running" 错误。编译检查应通过 MCP 的 get_console_logs 查看有没有 CS error。

**How to apply:**
- 不要用 batchmode Unity 做编译检查
- 编译验证用 MCP get_console_logs（无 CS error = 通过）
- play_test.py 也是通过文件通信和 MCP 跑，不需要另起 Unity
