---
name: AVO Loop 实战教训
description: 3轮修复失败的教训：先研究再修、小步迭代、修测试格式、TestBridge优化
type: feedback
---

## 教训 1：先研究再修代码
3轮修 Settings 手机框全失败，因为没理解 Bank/Contacts 为什么能在框内。
**Why:** 盲猜修改方向容易错，改坏系统还要回滚浪费时间。
**How to apply:** 遇到 3 轮内修不好的 bug，先派 Codex 做只读研究（读代码写分析报告），理解根因再动手。

## 教训 2：一次只改一个文件
Round 2 改了 5 个文件导致整个手机崩溃，回滚后很难定位是哪个改动出问题。
**Why:** 大改动 blast radius 大，出错后无法定位。
**How to apply:** Codex prompt 中明确限制 "只改 XXX.cs 一个文件"，且列出"不要修改"的文件。

## 教训 3：先修测试再跑测试
test_plan.json 的 expect 格式不匹配（`found` vs `"message":"found"`），导致实际通过的功能被标为失败。
**Why:** 误报干扰判断，浪费时间分析"假失败"。
**How to apply:** 新一轮 Loop 开始前先修复 test_plan 的 expect 格式。

## 教训 4：TestBridge 热身需要更长
热身步骤 wait 10s + 2x query:scene 不够，前几步经常 no response。
**Why:** Unity Play Mode 启动后 TestBridge 初始化需要更长时间。
**How to apply:** 热身改为 wait 15s + 3x query:scene（每次 wait 8s）。
