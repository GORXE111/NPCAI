---
name: Agent Development Feedback
description: 开发 mini-claude-code agent 过程中的经验教训
type: feedback
---

本地 9B 模型做 agent 的问题总结：

1. Ollama 0.18.x 对 Qwen3.5 有 tool call parsing bug，即使不传 tools 参数也会崩（500 错误）
2. MLX-LM 对 Qwen3.5 也有同样的 tool parser bug（qwen3_coder.py 崩溃）
3. Qwen3 8B 没有这个 bug，纯文本 tool calling 模式可用
4. 9B 模型生成速度约 13.5 tok/s（Ollama），写大文件需要几分钟容易超时
5. Web server 必须用 ThreadingHTTPServer，否则 SSE 连接会阻塞其他请求
6. Agent loop 必须在独立线程运行，不能阻塞 web server

**Why:** 踩了很多坑，未来如果用户再问本地 agent 开发，直接避开这些问题。

**How to apply:** 推荐用远端大模型（Codex/Claude API）做复杂 agent 任务，本地模型只适合简单批量任务。
