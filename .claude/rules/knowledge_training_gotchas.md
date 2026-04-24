# 训练踩坑合集

记录所有项目中遇到的训练/推理技术坑。下次直接查表。

---

## Qwen3.5 系列在 Apple MPS 上

### 坑 1: GatedDeltaNet dtype 冲突
- **症状**: `MPSNDArrayMatrixMultiplication.mm:4140 failed assertion: Destination NDArray and Accumulator NDArray cannot have different datatype`
- **根因**: `Qwen3_5GatedDeltaNet.forward` 内部有 `g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`，产生混合 float16/float32
- **解法**: `model/qwen35_mps_fix.py` 的 `patch_qwen35_for_mps()`
  - 强制整个模型 `model.float()`
  - 包装每个 `Qwen3_5GatedDeltaNet` 的 forward，入口 cast float32、出口 cast 回原 dtype
  - 强制该层内部所有 params/buffers 都是 float32
- **验证**: Qwen3.5-0.8B 和 Qwen3.5-2B 均在 MPS 上跑通训练和推理

### 坑 2: Ollama Qwen3.5 thinking mode 污染 response
- **症状**: API 调用 `response` 字段是空字符串
- **根因**: Qwen3.5 默认开 thinking，内容跑到 `thinking` 字段
- **解法**:
  ```python
  requests.post(url, json={
      "model": "qwen3.5:9b",
      "prompt": ...,
      "think": False,           # 关闭 thinking
      "options": {"num_predict": 150}
  })
  # fallback: text = data.get("response") or data.get("thinking") or ""
  ```

---

## M4 Mac 16GB 训练内存实况

### 内存上限（float32 训练）
| 模型 | 模型权重 | 前向激活 | 能否 SFT | 能否 SFT + forward（Stage 2 MPI） |
|------|---------|---------|---------|----------------------------------|
| Qwen2.5-0.5B | 2 GB | ~1 GB | ✅ 很稳 | ✅ 稳 |
| Qwen3.5-0.8B | 3.2 GB | ~2 GB | ✅ 稳 | ✅ 可以 |
| Qwen2.5-3B | 12 GB | ~4 GB | ⚠️ swap | ❌ OOM/极慢 |
| Qwen3.5-2B | 8 GB | ~4 GB | ✅ 稳（float32） | ⚠️ 紧张但能跑 |

### 内存上限（float16 训练）
- **float16 不要用于 LoRA merge**：peft merge_and_unload 会把 LoRA float32 塞进 base float16 → 混合 dtype → forward NaN
- 纯 float16 基座 + float16 LoRA + 推理：OK
- 训练 Memory Encoder / Emotion Head 于 float16：MPS 数值溢出 → NaN loss

### 规则
1. M4 16GB 上**只训 0.8B / 2B + float32**
2. 绝不做 float16 LoRA merge
3. 内存监控：free pages < 100k (1.6GB) 就要警戒
4. Ollama 自动启动会占 CPU，训练前 `launchctl unload` 或 `kill -9`（不影响后续推理）

---

## LoRA 过拟合诊断

### 症状（0.8B Stage 1 v1 血泪教训）
- Val loss 从 2.5 降到 1.89（看起来降了）
- 但 benchmark 结果 0% 合格（60/40 数据中，LoRA config overall 2.53/5）
- 表现：响应重复相同句子 ≥2 次（"I am X. I am X. I am X."）

### 原因
- 学习率太高（2e-4）
- Dropout 太低（0.05）
- Epochs 太多（5）
- 数据 8K 相对 0.8B 模型还是太多、同时样本质量不够严格

### v2 解法（已跑，效果待 benchmark）
```python
LR = 5e-5        # 降 4×
DROPOUT = 0.15   # 升 3×
LORA_R = 16      # 升 2×（大容量 + 强正则）
EPOCHS = 3       # 降
WEIGHT_DECAY = 0.05  # 升 5×
# mid-epoch eval + patience=2 early stopping
```

### 更根本的结论
- 0.8B 可能不适合 NPC LoRA SFT（容量不够 + base 已很强）
- 同样训法在 Qwen3.5-2B 上 val 0.344（正常）
- **Scale of model matters for LoRA adaptation**

---

## Memory Prefix Injection 训练

### 关键点
- Memory Encoder 参数要和 base 同 dtype（float32）
- **inputs_embeds 路径**：绕过 input_ids tokenizer，手动拼 prefix + input embed
- Labels 拼接时前 N 位（prefix 位）标 -100（ignore）
- Gate 初始化为 0，Sigmoid(0)=0.5，避免一开始破坏 base

### Gate 学习曲线（参考）
| 实验 | 初始 | 最终 |
|------|------|------|
| Qwen2.5-0.5B + 10K | 0.500 | 0.423 |
| Qwen3.5-0.8B v1 + 10K | 0.500 | 0.758 |
| Qwen3.5-0.8B v2 (保守 LoRA) + 10K | 0.500 | 0.539 |

**观察**: Gate 变化幅度反映 base 对记忆的"需要程度"。v2 保守 LoRA 后 base 能力保留更多，Gate 涨幅小（模型不那么需要靠 MPI 补救）。

### 坑
- 不要合并 Stage 1 LoRA 再训 Stage 2 → dtype 混合 NaN
- Stage 2 训练用原始 base + inputs_embeds

---

## 数据质量教训

### 量化证据（Val Loss on Qwen3.5-2B）
| 数据 | 样本量 | Val Loss |
|------|-------|---------|
| 278 手写精标 | 278 | **0.344** |
| 8K curated (LIGHT + amaydle 过滤) | 8,131 | 1.757 |
| 81K mixed | 81,036 | 2.011 |

### 过滤策略
- LIGHT dataset 用关键词过滤中世纪奇幻风格
- amaydle 情感数据用关键词置信度验证（≥0.6 才保留）
- 舍弃：WoW 任务文本、Persona-Chat（现代）、RolePlay-NPCv2（质量参差）

### 情感数据（Stage 3）
- 全部数据 12K → Val Acc 29.7%（通用对话拉低 NPC 能力）
- 清洗后 2.6K（amaydle + 手写 + 高置信度 EmotionDialogue）→ Val Acc 72.0%

---

## LLM-as-Judge benchmark

### 坑
1. **同族偏差**: qwen3.5:9b 判断 qwen3.5 输出时 C=D 给不同分（应相同）
2. **Thinking mode**: 评分 JSON 被写进 thinking 字段
3. **格式解析**: 必须让 judge 只输出 JSON，用 regex 提取 `\{[^}]+\}`

### 最佳实践
- 用不同家族的 judge（Claude 评 Qwen，反之亦然）
- **盲评**（shuffle + 匿名化）消除顺序和身份偏差
- 多 judge 面板取平均
- C=D 身份一致性测试（相同响应文本应得相同分数）作为 judge noise 校准

---

## 断点续训

- **Stage 2 Memory Encoder**: `torch.save(state_dict)` + `load_state_dict` 即可
- **Stage 1 LoRA**: peft 的 `save_pretrained` / `from_pretrained`
- **记得保存 best_val 当前值**，不然 resumed 后又从 inf 开始覆盖好 checkpoint
- 用 `nohup ... < /dev/null &` 防 SSH 断开，重定向 stdout 到日志文件

---

## Ollama 在 Mac 上

### 自启动
- Ollama 被 launchctl 管理，kill 后自动重启
- 关闭: `launchctl unload /opt/homebrew/opt/ollama/homebrew.mxcl.ollama.plist`
- 但不关也不占 GPU（除非被调用）

### thinking mode
- 新版 Qwen3.5 系列默认开 thinking
- API 请求加 `"think": false` 关闭
- 某些旧客户端不支持这个字段 → 需要 fallback 读 thinking

### 并发
- `OLLAMA_NUM_PARALLEL=4` 支持 4 并发请求
- 实测 M4 上 3 并发最稳定

---

## SCP / SSH 传大文件

- 50MB+ 的 checkpoint SCP 不要用交互式 bash，用 `scp -r` 或 `rsync`
- Windows bash 的 heredoc `<< 'EOF'` 里有双引号会被吃掉 → 写到 Python 文件再 `scp` 过去
- SSH 断连会杀掉前台训练进程 → 用 `nohup ... &`
