---
name: Atomic Runner 原子化自动开发引擎
description: 完整AI自动化管线：AVO Loop持续迭代，10个APP全部达标，手机UI接近GTA5风格
type: project
---

## 当前版本（2026-03-30）
- **10 个 APP 全部完成，UI 品质 9/10**
- 手机位置：右下角 GTA5 风格（anchor 1,0 pos -36,36）
- 图标：70x70，矢量程序化图标，y=110 位置
- X 按钮：红色正确（CloseButton 跳过 NormalizeAppButton）
- IconSpriteCache：Awake 时清除避免跨 PlayMode 缓存污染

## 管线状态（2026-03-30）
- play_test.py: timeout=25s + 3x重试，成功率 ~90%
- TestBridge: launch:dynasty8/autos/lifeinvader/email 命令可用
- dispatch.sh: ~/ai-unity/orchestrator/dispatch.sh（日志统一到 orchestrator）
- API: aixj.vip 偶尔 502，不稳定时直接修代码
- AVOLoop.app: macOS 桌面双击打开

## APP 质量评分
| APP | 评分 | 备注 |
|-----|------|------|
| 主屏 | 9/10 | GTA5风格右下角，矢量图标 |
| Bank | 8/10 | $符号图标，8条交易，部分布局空白 |
| Contacts | 8.5/10 | 10联系人，人形图标 |
| Messages | 9/10 | 7条不重复短信，详情页可访问 |
| Settings | 8/10 | 6个设置项完整 |
| Dynasty8 | 9/10 | 房产列表+地址+价格 |
| SuperAutos | 10/10 | 车辆轮廓图标+分类筛选 |
| LifeInvader | 9.5/10 | 社交帖子+点赞+时间戳 |
| Email | 10/10 | 完整邮箱+分类标签 |
| Camera | 8/10 | Snapmatic全屏Overlay |
| PhoneCall | 9.5/10 | 通话界面+计时器+字幕 |

## 已知问题
1. Bank 交易只显示 3 条（渲染限制，数据库有 9 条）
2. Contacts 图标形状略像桶（倒梯形改了但视觉仍不清晰）
3. X按钮数字2 — 已修（CloseButton跳过+缓存清除）

## 关键文件路径（Mac）
- Unity 项目: ~/ai-unity/gtagame
- Prefabs: Assets/Resources/Prefabs/Phone/
- 主要 Prefab: PhoneUI.prefab（手机主框架）
- 编排器: ~/ai-unity/orchestrator/
- 测试: ~/ai-unity/orchestrator/play_test.py

## AVO Loop 工具
- dispatch.sh "任务" → Codex 执行，日志到 orchestrator/codex_*.log
- launch:dynasty8/autos/lifeinvader/email → TestBridge 直接启动浏览器APP
- CallButton_{Name} → 联系人 Call 按钮名称格式
- MessageItem_{N} → Messages 列表 item 按钮名称格式
