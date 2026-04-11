---
name: Unity 6000 + Prefab UI 踩坑记录
description: Unity 6000 已知陷阱 + Prefab UI 架构必须知道的规则
type: feedback
---

## Unity 6000 已移除资源
| 旧写法 | 替代 |
|--------|------|
| `Font("Arial")` | `Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf")` |
| `GetBuiltinResource<Sprite>("UI/Skin/Knob.psd")` | `sprite = null` 或自生成 PNG |
| `GetBuiltinResource<Sprite>("UI/Skin/UISprite.psd")` | 同上 |

## Object 歧义
始终写 `UnityEngine.Object.FindFirstObjectByType<T>()`

## MCP 限制
- `update_component` 在 Play Mode 中不能修改运行时字段
- Play Mode 测试用文件通信（.test_command/.test_result）

## 核心架构原则（最重要，已验证有效）
**Prefab 是 UI 布局的真理来源，代码不应在运行时覆盖 Prefab 设置的位置。**
- UI 位置/大小/锚点 → Unity 编辑器改 Prefab，不写代码动态覆盖
- 新增图标 → 改 Prefab 里的 AppGrid 槽位，不写代码运行时添加
- 代码职责：逻辑（GridLayout格子、ScrollView滚动、事件绑定）
- 发现 `anchorMin/anchoredPosition/sizeDelta.x` 赋值：大概率是 bug，检查是否应该移除

**具体案例（PhoneHomeScreenGridExpander.cs）：**
- ❌ 错误：`appGridRect.anchorMin = new Vector2(0f, 1f)` → 覆盖 Prefab 锚点
- ❌ 错误：`appGridRect.anchoredPosition = new Vector2(horizontalInset, 110f)` → 覆盖 Prefab 位置
- ❌ 错误：`padding.top = AppGridTopInset` → 强制顶部偏移
- ❌ 错误：`appGridRect.sizeDelta = new Vector2(contentWidth, contentHeight)` → 覆盖宽度
- ✅ 正确：`appGridRect.sizeDelta = new Vector2(appGridRect.sizeDelta.x, contentHeight)` → 只更新高度供 ScrollRect 使用

**Why:** Codex 多次写代码强制覆盖 appGridRect.anchoredPosition 等，导致运行时位置不对，左右偏移、顶部 80px 空白都是这样引入的。

## Prefab UI 必须遵守的规则

### Button onClick 必须运行时绑定
**Prefab 不能序列化 lambda/delegate。** 所有 Button 的 onClick 事件必须在代码中运行时绑定。
```csharp
// PhoneManager.BindAppIconClicks() 中
btn.onClick.AddListener(() => OpenApp(captured));
```
**Why:** Prefab 保存时 onClick 列表中的运行时引用会丢失。
**How to apply:** 任何新增的 Prefab Button 都需要对应的运行时绑定代码。

### 不要重复 Instantiate Prefab
如果 PhoneManager.FindAppPage 已经加载了 APP Prefab，APP 的 Controller 脚本（如 BankAppController）不应该再自己加载一次。否则会出现套娃（大界面里嵌小界面）。
**规则:** Controller 只负责绑定引用和事件，不负责加载 Prefab。

### APP 页面要 stretch 填满手机
FindAppPage instantiate 后设 `rt.anchorMin=0, anchorMax=1, offset=0` 让 APP 页面填满 PhonePanel。

### 居中放大保持比例
AnimateToFullscreen 不能用 stretch anchor，要用固定 anchor(0.5,0.5) + sizeDelta 放大。当前 scale=1.7x（400x700 → 680x1190）。

### 初始状态
PhoneManager.ApplyInitialState 必须 `phonePanel.SetActive(false)` 强制隐藏，因为 Prefab 中 PhonePanel 可能默认 active。

## Editor.log 路径
`~/Library/Logs/Unity/Editor.log`（不在项目 Logs/ 下）

## EventSystem
UGUI 必须有 EventSystem + StandaloneInputModule 才能接收鼠标/触屏输入。

## verify.py 搜索范围
grep 搜 `Assets/` 全目录，不能只搜 `Assets/Scripts/`。
