---
name: 测试验收标准需要提高
description: Play Mode测试不能只验证元素存在，需要截图AI审查+UI模式验证+功能正确性检查
type: feedback
---

当前测试只验证"按钮存在且能点击"，不够。需要三层验证：

**Level 1: 功能性**（当前已有）
- 元素存在、可点击、页面打开

**Level 2: UI 正确性**（需要增加）
- Settings 是否在手机框内（APP页面模式 1.7x）
- Camera 是否全屏 Overlay（独立 Canvas）
- 颜色/布局是否符合规范

**Level 3: 截图 AI 审查**（需要增加）
- 测试截图发给 AI 判断 UI 质量
- 对标 GTA5 风格
- 发现视觉 bug（如闪光效果不对、文案没更新）

**Why:** Settings 测试全部"通过"但实际 UI 模式不对（应该在手机框内但变成了全屏），这种问题纯按钮测试发现不了。

**How to apply:**
- 工作流应增加"截图审查"步骤
- test_plan.json 需要检查 UI 层级（如 verify SettingsAppPage is child of PhonePanel）
- 拍照后的文案变化等功能正确性也要验证
- Claude Code 负责优化这套工作流，不负责直接修 bug
