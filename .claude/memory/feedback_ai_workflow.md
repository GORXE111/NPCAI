---
name: AI 自动化工作流经验总结
description: Codex自动开发+测试+修bug完整经验，含编排器优化、UI模式、Fallback路径、TestBridge踩坑
type: feedback
---

## 工作模式选择
| 场景 | 方式 |
|------|------|
| 新功能（多文件） | Atomic Runner POST /api/task（走Plan） |
| Plan超时（API不稳定） | 直接 codex exec 跳过Plan |
| 单文件修复 | 直接 codex exec |
| 测试 | play_test.py（手动写test_plan.json） |

## 编排器优化记录（2026-03-26）
- **3轮 Test→Fix 循环**（之前只有1轮）
- **测试计划模板**: wait 10s + 双热身query:scene + 每步5s
- **Fix agent 增强**: 读AGENTS.md + 常见修复模式速查
- **自动清理flag**: 每轮fix后清理.playmode_enter等
- atomic_runner.py 备份在 atomic_runner.py.bak

## UI 开发四种模式
### 模式1: APP页面（手机内居中1.7x）
- Bank, Contacts, Messages, Settings
- PhoneManager.FindAppPage + ResolvePageName 路由
- **新APP必须在PhoneManager添加路由**

### 模式2: 全屏Overlay（独立Canvas）
- PhoneCallOverlay(100), SmsNotificationPopup(90), CameraOverlay(105), WebBrowserOverlay(110)

### 模式3: 手机内覆盖层
- InCallOverlay 在 HomeScreen 上方

### 模式4: 全屏浏览器（WebBrowserOverlay）
- **关键：内容通过 Fallback 路径创建**
- WebBrowserOverlay.Fallback.cs 根据 URL 判断: dynasty8/superautos/lifeinvader/eyefind
- 详情页在 Fallback.Detail.cs / Fallback.EmailDetail.cs 中
- **不是 PageController.RuntimeUi.cs！**

## 运行时 UI 创建规则
| 规则 | 原因 |
|------|------|
| Text color = Color.white 或深色 | 默认黑色不可见 |
| sizeDelta 明确（宽>200,高>25） | 为0不可见 |
| font = LegacyRuntime.ttf | Unity 6000 移除 Arial |
| ScrollView 需要 RectMask2D | 否则溢出 |
| Button raycastTarget=true | 否则点不了 |
| 背景 raycastTarget=false | 否则挡按钮 |
| Button onClick 运行时绑定 | Prefab不能序列化lambda |

## TestBridge 通信经验
- Play Mode 进入需要 **45秒**
- 前几个命令 **常超时**，测试必须: wait 10s → query:scene × 2
- 截图偶发超时是正常的
- ui:list 搜索所有Canvas的Button
- 卡片需要命名 ItemCard_N / EmailRow_N / PropertyCard_N

## 失败模式速查
| 失败 | 解决 |
|------|------|
| Plan超时 | 直接codex exec跳过 |
| Play Mode进不去 | 增加超时到45s，清理flag |
| 浏览器内容空白 | 检查Fallback路径 |
| APP打开无反应 | PhoneManager路由 + HomeScreenInstaller |
| 运行时UI不可见 | sizeDelta/Color/font检查 |
| 卡片不可点击 | Fallback添加Button组件 |
| Codex超时(19chars) | API不稳定，重试或换时间 |
