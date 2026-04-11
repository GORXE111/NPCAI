---
name: Unity UI 开发工作流（Prefab 优先）
description: UI用Prefab不用代码创建，Codex用MCP构建Prefab，用户可手动美化，运行时Resources.Load加载
type: reference
---

## 核心原则
**Prefab = 视觉层（可手动编辑），C# = 逻辑层（AI 写）**

## 三步流程

### Step 1: Codex 写 C# 逻辑脚本
- 控制器（事件绑定、数据填充、动画）
- 数据类（BankAccount 等）
- **不在代码中创建 Image/Text/Button 等 UI 组件**
- 用 `[SerializeField]` 或 `transform.Find()` 获取 UI 引用

### Step 2: Codex 用 MCP 构建 Prefab
```
MCP batch_execute:
  update_gameobject → 创建层级
  update_component → 添加 Image/Text/Button/LayoutGroup + 设样式
  update_component → 挂 C# 脚本
  create_prefab → 保存到 Assets/Resources/Prefabs/Phone/
```

### Step 3: 用户手动美化（可选）
打开 Prefab → 双击进编辑模式 → 调颜色/间距/字体 → Ctrl+S

## Prefab 目录
```
Assets/Resources/Prefabs/Phone/
  PhoneUI.prefab          ← 手机主体
  BankAppPage.prefab      ← 银行APP
  ContactsAppPage.prefab  ← 通讯录
  ...
```

## Sprite 资源
- RoundedRect_White.png（9-slice 圆角矩形）
- Circle_White.png（圆形）
- Gradient_Vertical.png（渐变）
- Shadow_Soft.png（阴影）
- 生成: Unity 菜单 `FreeLive > Generate UI Sprites`

## 颜色体系 (PhoneColors.cs)
| 名称 | 值 | 用途 |
|------|-----|------|
| PhoneBlack | #1C1C1E | 手机框/NavBar |
| BankGreen | #34C759 | 银行APP |
| ContactsOrange | #FF9500 | 通讯录 |
| MapBlue | #007AFF | 地图 |
| SettingsGray | #8E8E93 | 设置 |
| CloseRed | #FF3B30 | 关闭按钮 |

## 动画
全部用协程 + Time.unscaledDeltaTime:
- PhoneAnimator: 开关(0.3s EaseOutBack) + 全屏(0.35s EaseOutCubic)
- PhoneAppTransition: 页面切换(0.25s slide)

## TestBridge 不受影响
Prefab 加载后的运行时 UI 和代码创建的完全一样，TestBridge 通过名字查找，不关心来源。

## 当前状态（待迁移）
现有手机 UI 仍然是代码创建的，需要迁移到 Prefab 架构。
迁移步骤: 用 MCP 在 Editor 中构建 UI 层级 → 保存 Prefab → 删除代码中的 UI 创建逻辑 → 改为 Resources.Load
