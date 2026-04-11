---
name: APP页面模式的正常样子
description: APP打开后手机面板1.7x放大到接近全屏，没有手机壳边框是正常行为
type: feedback
---

APP 页面模式（Bank/Contacts/Messages/Settings）打开后，PhonePanel 会 AnimateToFullscreen 1.7x 放大到接近全屏大面板。这是正常行为，不是 bug。

**Why:** 之前误判 Settings "不在手机框内"，花了 5 轮修复才确认其实和 Bank 一样——都是大面板模式。
**How to apply:** 截图审查时，APP 页面显示为居中大面板是正确的。只有在主屏（HomeScreen）时才能看到手机壳边框。手机壳默认在右下角位置。
