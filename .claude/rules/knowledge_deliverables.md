# 论文产出清单与进度

定义"什么是完成"。每个声明对应可交付的具体 artifact，否则就是空气。

最后更新：2026-04-24（Phase 1 Unity 重构 + Stage 1 训练 已开始）

---

## 1. 论文 Claim → Artifact 映射表

| 论文 Claim | LLM 层产出 | Agent 层产出 | 验证方法 |
|-----------|-----------|------------|---------|
| "0.8B 能扮演好 Kim" | Stage 1 LoRA + DEBench persona 分数 | Unity demo 视频 | 多 judge panel 5 维评分 |
| "0.8B 能调用工具" | Stage 2 LoRA + BFCL/DEBench 工具分数 | Unity 工具调用可见执行 | Tool selection accuracy |
| "DPO 提升忠实度" | Stage 3 LoRA + ablation 表 | 同场景 Stage 2 vs 3 对比 | A/B 盲评 |
| "在 M4 Mac 上运行" | 实测延迟数字 | M4 上完整可玩 demo | 重复 100 turn 测平均延迟 |
| "对标 CPDC 14B" | CPDC 2025 跑分 | 同 prompt demo 对比截图 | 4-axis LLM-Judge 同协议 |

**任何一行的 artifact 缺失，对应 claim 就不能写入论文。**

---

## 2. LLM 层 — 11 个必交付物

### A. 模型权重（核心）

| ID | 产出 | 状态 | 备注 |
|----|------|:----:|------|
| **L1** | `kim-q35-08b-stage1.lora` | ✅ **完成 (v2)** | Val 0.9445, persona 测试通过 |
| **L2** | `kim-q35-08b-stage2.lora` | ✅ **完成** | Val 0.7246, JSON 100% valid, tool selection ~25% (待 Stage 3 修) |
| **L3** | `kim-q35-08b-stage3.lora` | ⚪ 待做 | DPO on tool faithfulness |

### B. 训练资产

| ID | 产出 | 状态 |
|----|------|:----:|
| **L4** | `kim_train.jsonl` (1,127 v2 cleaned SFT) | ✅ 完成 |
| **L5** | Stage 2 工具 SFT 数据集 | ⚪ 待做 |
| **L6** | Stage 3 DPO 偏好对 (3-5K) | ⚪ 待做 |
| **L7** | 训练脚本 (3 stage 各一) | 🟡 1/3 完成 |
| **L8** | `qwen35_mps_fix.py` | ✅ 完成 |

### C. 评估资产（学术贡献）

| ID | 产出 | 状态 |
|----|------|:----:|
| **L9** | DEBench 数据集（80 conversation × 3 tasks） | ⚪ 待做 |
| **L10** | DEBench 评分脚本（多 judge panel） | ⚪ 待做 |
| **L11** | CPDC 2025 + BFCL V4 跑分结果 | ⚪ 待做 |

---

## 3. Agent 层 — 8 个必交付物

### A. Unity 系统

| ID | 产出 | 状态 | 备注 |
|----|------|:----:|------|
| **A1** | Unity VN 环境（DE 风格 UI） | 🟡 代码完成，待场景文件 | 12 个新 .cs 文件已 commit |
| **A2** | 18 工具 API 实现 | 🟡 9/18 完成 | Phase 2 加 24 DE 技能 + 剩余 |
| **A3** | Kim 角色资产（5 表情 + 2 背景 + 2 BGM） | ⚪ 待做 | AI 生成或 placeholder 色块 |
| **A4** | 5 分钟 demo 场景剧本 | ⚪ 待做 | Whirling-in-Rags 旅馆开场 |

### B. Agent 代码

| ID | 产出 | 状态 |
|----|------|:----:|
| **A5** | `NPCAgent` (PromptBuilder + VisualNovelManager) | ✅ 完成 |
| **A6** | ToolRegistry + ToolExecutor | ✅ 完成 |
| **A7** | OllamaClient JSON 模式 + schema validation | 🟡 部分（has thinking 关闭，缺 JSON enforce） |

### C. Demo 资产（论文配图）

| ID | 产出 | 状态 |
|----|------|:----:|
| **A8** | Demo 视频 / GIF（30秒 + 2-3 张定格图） | ⚪ 待做 |

---

## 4. 整体进度

```
LLM 层:   ██████░░░░░░░░░░░░░  5/11  (L1, L4, L7-stage1, L8 + 部分 L7)
Agent 层: ██████░░░░░░░░░░░░░  4/8   (A5, A6, 部分 A1, 部分 A7)
─────────────────────────────────────
总进度:   ██████████░░░░░░░░░  9/19  (~47%)
```

里程碑达成: **Stage 1 LoRA 通过 persona 测试**（2026-04-28），可进 Phase 2。

---

## 5. 4 阶段执行计划

### Phase 1（本周）: 最小闭环
**目标**: 玩家选项 → LLM → Unity 立绘+选项变化

- [x] L1 Stage 1 LoRA 训练（Mac PID 66280 跑中）
- [ ] L4-7 文档化训练流程
- [x] A1 Unity VN 代码（12 个 .cs 文件已写）
- [ ] A1 Unity 场景文件 + prefab + Canvas 布局
- [x] A6 ToolRegistry + 9 个核心工具
- [ ] A7 OllamaClient JSON enforce
- [ ] A3 占位资源（5 立绘色块 + 2 背景 + 2 BGM）

**Phase 1 交付**: "问候 Kim → Kim 微笑回应 + 给 3 个选项" 最小可玩 demo

### Phase 2（下周）: 工具增强 + 数据扩展
- [ ] L5 解析 `[Action/Check]` 转 Stage 2 训练数据
- [ ] L2 Stage 2 训练（tool-augmented SFT）
- [ ] A2 实现剩余 9 个工具（含 6 个核心 DE 技能）
- [ ] A4 设计 5 分钟 demo 剧本

### Phase 3（第 3 周）: DPO + Benchmark
- [ ] L6 生成 DPO 偏好对（3-5K，60% 合成 + 30% 自采样 + 10% 对抗）
- [ ] L3 Stage 3 DPO 训练（β=0.1, LR=5e-7, 1 epoch）
- [ ] L9-10 构建 DEBench
- [ ] L11 跑 CPDC 2025 + BFCL V4

### Phase 4（第 4 周）: 论文 + Demo 抛光
- [ ] 填充论文 §7 全部 TODO（数字 + ablation + qualitative）
- [ ] 录 demo 视频（A8）
- [ ] 准备图表

---

## 6. 论文存活的最低门槛

如果时间不够，**至少**要拿到：

1. ✅ Phase 1 完成（end-to-end 闭环 demo）
2. ✅ Stage 1 LoRA + DEBench persona 任务跑通（证明扮演能力）
3. ✅ Stage 2 LoRA + 工具调用 baseline（证明能调工具）
4. ✅ CPDC 2025 zero-shot 基线（证明对标位置）
5. ⚠️ Stage 3 DPO **强烈建议**有，否则降级 claim："小模型能 SFT 出工具调用能力"

不可少的核心数字：
- DEBench persona overall ≥ 3.5/5
- DEBench tool selection accuracy ≥ 65%（仅 Stage 2）/ 80%（含 Stage 3）
- Schema validity rate ≥ 90%
- 平均推理延迟 < 5s on M4

---

## 7. Stage 3 DPO 子计划（重点章节）

详见 `knowledge_methodology.md` 的 §3 Stage 3 章节。简要：

### Phase A: 偏好对生成（1 周）
- Day 1-2: 合成扰动器（8 种 perturbation） → ~3,000 对
- Day 3-4: Stage 2 自采样 + Claude judge → ~1,500 对
- Day 5: 手工对抗样本 → ~500 对
- **合计 ~5,000 对**

### Phase B: DPO 训练（1 周）
- trl 库 DPOTrainer，MPS 兼容
- Stage 2 LoRA 作为 init 和 ref
- β=0.1, LR=5e-7, 1 epoch
- 内存占用：ref (frozen) + policy ≈ 10GB on M4
- 训练时长：3-4 小时

### Phase C: 评估和迭代（1 周）
- DEBench 全套（persona / tool / game-tool）
- Stage 2 vs Stage 3 在 5 维度上对比
- 不达标 → 分析失败 case → 补偏好对 → 再训
- 最多 3 轮迭代

---

## 8. 风险登记表

| 风险 | 严重度 | 概率 | 缓解 |
|------|:------:|:----:|------|
| Stage 1 LoRA 又过拟合 | 🔴 | 中 | 已用保守超参 + early stop（v2 教训） |
| Stage 2 提取的工具数据噪声 | 🔴 | 中 | 加 schema 验证 + 5% 人工抽检 |
| Stage 3 DPO collapse | 🔴 | 中 | β=0.1 + 1 epoch + LR 极小 |
| DEBench 自建被审稿质疑 | 🟠 | 中 | 多 judge panel + Claude vs qwen3.5 比 |
| Unity demo 不专业 | 🟡 | 低 | AI 生成立绘 / placeholder 色块也能发 |
| CPDC 2025 跑分差 | 🟠 | 中 | 即使差也诚实报告，"小模型 Pareto"故事仍成立 |
| Mac 训练中断 | 🟡 | 高 | nohup + 增加 checkpoint 频率 |

---

## 9. 已废弃的产出（v0/v2 历史）

记录避免回到老路：

- ❌ Memory Prefix Injection encoder（v0，benchmark 显示无明显增益）
- ❌ Emotion Head 分类器（v0，对生成无影响）
- ❌ 81K mixed 训练数据（v2 Stage 1，质量差）
- ❌ 8K curated 数据 + LR 2e-4（v2 Stage 1，过拟合）
- ❌ qwen3.5:9b 单 judge benchmark（同族偏差）
- ❌ 10 NPC 通用市场城镇（v0 v2，已归档到 `_Archive/MarketTown/`）
- ❌ HSR 数据集方向（v3 早期，DMCA 风险）

详见 `knowledge_paper_evolution.md`。

---

## 10. Next Action（持续更新）

**当前 (2026-04-24 evening)**:
- ⏳ 等 Stage 1 训练完成（PID 66280, Mac, ~30分钟）
- 然后立即：
  1. 看 Stage 1 final Val Loss
  2. 拷贝 LoRA 到本地
  3. 部署到 Ollama（或直接 PyTorch）
  4. 在 Unity 端做 minimal demo（接 OllamaClient）

**下一个里程碑**: Phase 1 闭环 demo（玩家选项 → Kim 回应 + 工具触发）。
