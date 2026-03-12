# Skill Optimization 方法详解

> 结合 Pipeline 完整流程，以 Physics query + Math skill 为例
> 覆盖两种梯度方法: DTO (Logits) 和 Soft Prompt (Embedding)

---

## 1. 共同的 Pipeline 流程

```
用户调用:
  python scripts/example_physics.py --optimization_mode dto       # 方法一
  python scripts/example_physics.py --optimization_mode soft_prompt  # 方法二

Pipeline 执行:
  TTSOPipeline.run()
    → TTSODecoding._create_optimizer()  # 根据 optimization_mode 创建对应 Trainer
    → TTSODecoding.run_iterative()      # 外层 skill↔response 交替循环
        → Trainer.optimize()            # 内层梯度优化 (核心区别在这里)
```

---

## 2. 共同的输入

```
Query:  "A 2 kg block slides down a frictionless inclined plane
         that makes a 30° angle with the horizontal.
         What is the acceleration of the block?"

Skill (Math, 非 Physics):
  "# Mathematical Problem-Solving Skill
   ## Workflow
   1. Understand the problem
   2. List known quantities
   3. Define variables
   4. Formulate equations
   5. Apply mathematical rules
   6. Solve step-by-step
   7. Verify the solution"
```

---

## 3. 共同的 Outer Loop: `run_iterative()`

```
Round 0 (初始):
  Math Skill → 生成 response_0 → RM = 22.25

Round 1:
  optimize(query, response_0, Math Skill) → Optimized Skill_1
  Optimized Skill_1 → 生成 response_1 → RM = ?

Round 2:
  optimize(query, response_1, Optimized Skill_1) → Optimized Skill_2
  Optimized Skill_2 → 生成 response_2 → RM = ?

最终: 取 RM 最高的那一轮
```

两种方法的 outer loop 完全一样，区别在 `optimize()` 内部。

---

## 4. 方法一: DTO (Differentiable Token Optimization)

### 4.1 核心思想

在 **词表维度** 上优化。每个 token 位置维护一个 152064 维的 logits 向量，
通过 softmax 转成概率分布，再加权词表 embedding 得到可微分的表示。

```
优化变量: skill_logits [1, 180, 152064]   ← 每个位置是一个词表分布
```

### 4.2 初始化: Token IDs → One-hot Logits

```python
skill_token_ids = tokenize("# Mathematical Problem-Solving Skill...")
# → [835, 92731, 22079, 12, 50, 4552, ...]   共 180 个 token

# 构造初始 logits: one-hot × init_scale
skill_logits = torch.zeros(1, 180, 152064)
for i, token_id in enumerate(skill_token_ids):
    skill_logits[0, i, token_id] = init_scale  # 如 3.0

# 变成可优化参数
skill_logits = nn.Parameter(skill_logits)
```

此时每个位置的 softmax 分布是高度集中的:
```
位置 0: softmax(logits) ≈ [0, ..., 0, 0.98, 0, ..., 0]
                                          ↑ token "#"
位置 1: softmax(logits) ≈ [0, ..., 0, 0.98, 0, ..., 0]
                                          ↑ token "Mathematical"
...
```

`init_scale` 的影响:
```
init_scale = 10.0:  softmax 极度集中 → 几乎 one-hot → token 锁死, 很难改变
init_scale = 3.0:   softmax 较集中   → 给优化留一些空间
init_scale = 1.0:   softmax 较平坦   → token 容易变但容易乱
init_scale = 0.0:   完全均匀分布     → 纯噪声, 无法使用
```

### 4.3 Forward: STE (Straight-Through Estimator)

```python
# 1. softmax 得到概率分布
soft_probs = softmax(skill_logits)   # [1, 180, 152064]

# 2. 取 argmax 得到 one-hot (前向)
hard_one_hot = one_hot(argmax(skill_logits))  # [1, 180, 152064]

# 3. STE 技巧: 前向用 hard, 反向用 soft
#    (y_hard - y_soft).detach() + y_soft
ste_probs = (hard_one_hot - soft_probs).detach() + soft_probs

# 4. 加权 embedding table 得到 token embedding
skill_embeds = ste_probs @ embedding_table   # [1, 180, 3584]
#    前向: = one_hot @ embedding_table = 精确的 token embedding
#    反向: 梯度穿过 soft_probs 回到 skill_logits
```

为什么需要 STE?
```
直接 argmax:    不可微 → 梯度断开 → 无法优化
纯 softmax:     可微但输出是加权平均 → LM 看到"模糊"的 embedding → 质量差
STE:            前向用精确 token (质量好) + 反向用 softmax (梯度通)
```

### 4.4 构建模板 (SkillGenerationTemplate)

```
LM 输入 (embedding 层面):

[prefix_embeds]              [skill_embeds via STE]        [suffix_embeds]
"user: Use the following     skill 的 180 个 embedding     "\n\nProblem: A 2 kg block...
 skill to solve the problem: (STE → 精确 token embedding)   \nassistant: Let's solve..."
"
 ← frozen →                  ← 通过 STE 可微分 →            ← frozen →
```

### 4.5 梯度优化循环 (100 步)

```python
for it in range(100):
    # 1. STE forward: logits → soft probs → hard one-hot → embedding
    soft_probs = softmax(skill_logits)
    hard = one_hot(argmax(skill_logits))
    ste_probs = (hard - soft_probs).detach() + soft_probs
    skill_embeds = ste_probs @ embedding_table   # [1, 180, 3584]

    # 2. 拼接模板并 LM forward
    full_embeds = [prefix_embeds | skill_embeds | suffix_embeds]
    lm_outputs = lm_model(inputs_embeds=full_embeds)
    logits = lm_outputs.logits  # [1, total_len, vocab_size]

    # 3. Response NLL loss
    pred_logits = logits[:, resp_start-1 : resp_start+resp_len-1, :]
    response_nll = -CrossEntropy(pred_logits, response_token_ids)

    # 4. Skill Fluency 正则化 (LM 自回归概率)
    #    "这段 skill token 序列, LM 认为它多'通顺'?"
    skill_pred = logits[:, skill_start-1 : skill_start+skill_len-1, :]
    skill_fluency = -CrossEntropy(skill_pred, current_skill_token_ids)

    # 5. RM Reward
    rm_embeds = ste_probs @ rm_embedding_table  # 同样通过 STE
    rm_full = [rm_prefix | rm_embeds | rm_suffix]
    reward = rm_model(inputs_embeds=rm_full).logits[0][0]

    # 6. 总 loss
    loss = -(response_nll_coeff × response_nll)
         - (skill_fluency_coeff × skill_fluency)
         - (reward_coeff × reward)

    # 7. 反向传播
    loss.backward()
    #    梯度路径:
    #    loss → logits → pred_logits → full_embeds → skill_embeds
    #         → ste_probs (这里 STE 让梯度穿过 soft_probs)
    #         → skill_logits [1, 180, 152064]
    #
    #    问题: 梯度被 152064 维稀释!
    #    grad shape: [1, 180, 152064]
    #    grad_max ≈ 9e-6  (几乎为零)

    optimizer.step()
```

### 4.6 STE 梯度瓶颈 (核心问题)

```
要让位置 5 从 "Apply" 变成 "Decompose", 需要:
  logits[5]["Decompose"] > logits[5]["Apply"]

初始状态:
  logits[5]["Apply"]     = 3.0   (init_scale)
  logits[5]["Decompose"] = 0.0

gap = 3.0, 需要填补这个差距。

实际梯度:
  grad_max ≈ 9e-6
  每步更新: lr × grad = 0.01 × 9e-6 = 9e-8

需要的步数: 3.0 / 9e-8 = 3300万步
实际只跑 100 步 → 100 × 9e-8 = 9e-6  (远不够!)
```

为什么梯度这么小?
```
softmax 反向传播:
  ∂softmax/∂logits = diag(p) - p·p^T

当 p ≈ one-hot 时 (init_scale=3.0):
  p_target ≈ 0.98, 其他 p_i ≈ 0.98/152063 ≈ 6e-6
  对非 target token 的梯度 ≈ p_target × p_i ≈ 0.98 × 6e-6 ≈ 6e-6

→ 梯度天然被 vocab_size=152064 稀释到 ~1e-6 量级
→ 这不是 bug, 是 softmax + 大词表的数学必然
```

### 4.7 解码回文本

```python
# 直接取 argmax
optimized_token_ids = argmax(skill_logits, dim=-1)  # [180]
optimized_text = tokenizer.decode(optimized_token_ids)
```

### 4.8 Fluency 正则化的作用

```
skill_fluency_coeff = 0 (无约束):
  token 可以自由翻转到任意词
  → 产出 "Carson İnt-S佳 Skill mixin冻" 乱码
  → RM 分数可能偶然升高 (adversarial)

skill_fluency_coeff = 0.01~0.1 (适中):
  LM 的自回归概率约束 token 序列的连贯性
  → 翻转后的 token 需要和上下文语义相容
  → 理论上能保持通顺, 但因为梯度太小, 实际无法翻转

skill_fluency_coeff = 1.0 (太强):
  token 被强烈约束在当前位置
  → skill 完全不变
```

### 4.9 DTO 实验结论

无论如何调参, 都无法同时满足两个条件:
1. Token 能发生有意义的变化 (需要大梯度)
2. 变化后的 skill 保持语义连贯 (需要约束)

```
                    token 不变 ←──────────────────→ token 乱变
                    (梯度太小)                       (无约束)
init_scale=10       ████████░░░░░░░░░░░░░░░░░░░░
init_scale=3        ██████░░░░░░░░░░░░░░░░░░░░░░
init_scale=1        ░░░░░░░░░░░░░░░░░░░░░░░████
fluency=0           ░░░░░░░░░░░░░░░░░░░░████████
fluency=0.1         ██████░░░░░░░░░░░░░░░░░░░░░░

没有中间地带 — 这是 STE + 大词表的根本性限制。
```

---

## 5. 方法二: Soft Prompt Optimization

### 5.1 核心思想

在 **embedding 维度** 上优化。每个 token 位置维护一个 3584 维的连续向量，
梯度直接更新 embedding，不经过 softmax/STE。优化后再投影回最近的 token。

```
优化变量: skill_embeds [1, 180, 3584]   ← 每个位置是一个连续向量
```

### 5.2 初始化: Token IDs → Embedding 向量

```python
skill_token_ids = tokenize("# Mathematical Problem-Solving Skill...")
# → [835, 92731, 22079, 12, 50, 4552, ...]   共 180 个 token

# 从 LM 的 embedding table 查出初始向量
skill_embeds = embedding_table[skill_token_ids]
# → shape: [1, 180, 3584]   (Qwen2.5-7B hidden_dim = 3584)
# → 每行是一个 3584 维的连续向量

# 变成可优化参数
skill_embeds = nn.Parameter(skill_embeds)

# 同时保存初始状态, 用于 drift 正则化
init_embeds = skill_embeds.detach().clone()
```

此时每个 embedding 恰好对应一个真实 token:
```
位置 0: embed = embedding_table["#"]         → 3584维向量
位置 1: embed = embedding_table["Mathematical"] → 3584维向量
位置 2: embed = embedding_table["Problem"]    → 3584维向量
...
```

### 5.3 构建模板 (复用 SkillGenerationTemplate)

```
LM 输入 (embedding 层面):

[prefix_embeds]              [soft_skill_embeds]           [suffix_embeds]
"user: Use the following     skill 的 180 个 embedding     "\n\nProblem: A 2 kg block...
 skill to solve the problem: 向量 (可优化!)                 \nassistant: Let's solve..."
"
 ← frozen →                  ← nn.Parameter →              ← frozen →
```

### 5.4 梯度优化循环 (100 步)

每一步做的事:

```python
for it in range(100):
    # 1. 拼接 embedding (直接用, 无需 STE!)
    full_embeds = [prefix_embeds | skill_embeds | suffix_embeds]
    #              frozen          可优化        frozen

    # 2. LM forward pass
    lm_outputs = lm_model(inputs_embeds=full_embeds)
    logits = lm_outputs.logits  # [1, total_len, vocab_size]

    # 3. 计算 Response NLL loss
    #    "给定这个 skill embedding, LM 预测 response tokens 的概率"
    pred_logits = logits[:, resp_start-1 : resp_start+resp_len-1, :]
    response_nll = -CrossEntropy(pred_logits, response_token_ids)
    #    response_nll 越高 → skill 越能引导 LM 产出这个 response

    # 4. 计算 Drift 正则化
    drift = MSE(skill_embeds, init_embeds) × embed_drift_coeff
    #    drift 越低 → embedding 越接近原始 skill

    # 5. 计算 RM Reward (可微分路径)
    #    注意: RM 可能用不同的 embedding table
    #    需要把 LM 空间的 embedding 转到 RM 空间:
    soft_weights = softmax(cosine_sim(skill_embeds, lm_embed_table) / 0.1)
    #    → [180, 152064]  每个位置对所有 token 的软权重
    rm_embeds = soft_weights @ rm_embed_table
    #    → [180, rm_hidden_dim]  可微分的 RM embedding
    rm_full = [rm_prefix | rm_embeds | rm_suffix]
    reward = rm_model(inputs_embeds=rm_full).logits[0][0]

    # 6. 总 loss
    loss = -(response_nll_coeff × response_nll) - (reward_coeff × reward) + drift
    #         ↑ 最大化 response 概率        ↑ 最大化 RM 分数      ↑ 防止偏离太远

    # 7. 反向传播 + 更新
    loss.backward()
    #    梯度直接流到 skill_embeds! 无 STE 瓶颈!
    #    grad shape: [1, 180, 3584]  每个 embedding 位置都有 3584 维梯度
    optimizer.step()
    #    每个 embedding 向量在 3584 维空间里微移
```

### 5.5 为什么梯度不再是瓶颈

```
DTO 的梯度路径:
  loss → logits [1, 180, 152064] → STE softmax → 梯度被 152064 维稀释
  → grad_max ≈ 9e-6 (几乎为零)

Soft Prompt 的梯度路径:
  loss → LM attention layers → embedding 层 → 直达 skill_embeds [1, 180, 3584]
  → grad_norm 应该 >> 9e-6 (梯度信号强得多)
```

因为:
- **维度更小**: 3584 vs 152064, 梯度不被稀释
- **无 softmax 瓶颈**: 不需要穿过 softmax 的平坦区域
- **连续空间**: embedding 可以平滑移动, 不需要"翻转 argmax"的离散跳跃

### 5.6 Embedding 在 3584 维空间里怎么移动

```
初始状态 (step 0):
  位置 5 的 embedding = embedding_table["Apply"]
  → 恰好在 "Apply" 这个词对应的点上

优化过程中 (step 50):
  位置 5 的 embedding 偏移了
  → 不再恰好对应任何词, 在 "Apply" 和 "Use" 之间的某处
  → 但 LM 看到的是一个有意义的混合 embedding

优化结束 (step 100):
  位置 5 的 embedding 又偏移了
  → 最近邻: "Decompose" (比 "Apply" 更适合 physics 问题)
```

### 5.7 投影回 Tokens (Nearest Neighbor)

优化后每个位置的 embedding 不再对应某个 token, 需要找最近的:

```python
for i in range(180):
    # 计算当前 embedding 和所有 152064 个 token embedding 的余弦相似度
    similarity = cosine_sim(skill_embeds[i], embedding_table)
    # → [152064]

    # 取最高的
    best_token_id = argmax(similarity)
    # 比如: "Apply" → "Decompose"
```

最终得到 180 个新 token IDs → decode 成文本:

```
原始: "# Mathematical Problem-Solving Skill
       ## Workflow
       1. Understand the problem
       2. List known quantities
       ...
       5. Apply mathematical rules
       ..."

优化后 (理想情况):
      "# Physics Problem-Solving Skill
       ## Workflow
       1. Understand the problem
       2. List known quantities
       ...
       5. Decompose forces along axes
       ..."
```

### 5.8 Drift 正则化的作用

```
embed_drift_coeff = 0 (无约束):
  embedding 自由移动 → 可能飘到词表外的"真空地带"
  → nearest neighbor 投影回来可能是无关的词
  → 类似 DTO 的乱码问题

embed_drift_coeff = 0.01 (适中):
  embedding 被轻轻拉回原始位置
  → 每个位置只能偏移有限距离
  → 投影回来的 token 通常和原始 token 语义相近

embed_drift_coeff = 1.0 (太强):
  embedding 几乎不能移动
  → 和 DTO 里 fluency 太高一样, skill 不变
```

### 5.9 RM Embedding 的可微分处理

当 LM 和 RM 用不同模型时 (如 Qwen2.5-7B + Skywork-Reward-4B):

```
LM embedding space (3584 维)  ≠  RM embedding space (2560 维)

不能直接把 LM 的 soft embedding 给 RM 用。

解决方案: 可微分的软投影

  1. 计算 skill_embeds 和 LM 词表中所有 token 的余弦相似度
     sim = cosine(skill_embeds, lm_embed_table)  → [180, 152064]

  2. 用 softmax(sim / temperature) 得到 soft weights
     weights = softmax(sim / 0.1)  → [180, 152064]
     (温度 0.1 让 weights 比较尖锐, 接近 one-hot)

  3. 用 soft weights 加权 RM 的 embedding table
     rm_embeds = weights @ rm_embed_table  → [180, 2560]

  梯度可以通过 softmax → cosine → skill_embeds 回传
  (不像 project_to_token_ids() 用 argmax 断开了梯度)
```

---

## 6. 两种方法的完整对比

### 6.1 核心机制对比

| 维度 | DTO (Logits) | Soft Prompt (Embedding) |
|------|-------------|------------------------|
| **优化变量** | `skill_logits [1, 180, 152064]` | `skill_embeds [1, 180, 3584]` |
| **参数量** | 180 × 152064 = **2740万** | 180 × 3584 = **64万** |
| **初始化** | one-hot × init_scale | `embedding_table[token_ids]` |
| **forward** | STE: softmax → argmax → embed lookup | 直接用 embedding |
| **梯度路径** | loss → STE softmax → logits | loss → LM layers → 直达 embeds |
| **梯度大小** | ~1e-6 (瓶颈) | 预期大几个数量级 |
| **正则化** | skill_fluency (LM 自回归概率) | embed_drift (L2 距离) |
| **解码回文本** | `argmax(logits)` | cosine nearest neighbor |
| **变化方式** | token 离散翻转 (全有或全无) | embedding 连续移动 (渐进) |
| **RM 处理** | 通过同一个 STE 路径 | 可微分软投影 (cosine → softmax → matmul) |

### 6.2 优化空间对比

```
DTO:
  搜索空间 = 152064^180 种离散组合
  但每步只能改变 1 个 token (因为梯度太小, 连 1 个都改不了)
  优化是"翻硬币": 要么不翻, 要么翻到随机位置

Soft Prompt:
  搜索空间 = R^(180×3584) 连续空间
  每步所有位置的 embedding 同时微移
  优化是"拧旋钮": 每个维度都可以精细调整
```

### 6.3 正则化机制对比

```
DTO - Skill Fluency:
  衡量方式: LM 给 token 序列的自回归概率 (perplexity)
  直觉: "这段文本读起来通不通顺?"
  问题: 和优化目标冲突 — 改变 token 必然降低 fluency

Soft Prompt - Embed Drift:
  衡量方式: L2(当前 embedding, 初始 embedding)
  直觉: "每个位置偏移了多远?"
  优势: 不阻止语义变化, 只限制变化幅度
         embedding 可以在语义相近的词之间平滑移动
```

### 6.4 失败模式对比

```
DTO 的失败模式:
  1. 梯度消失: init_scale 高 → softmax 接近 one-hot → 梯度 ~1e-6 → 100步无变化
  2. 乱码输出: init_scale 低/fluency=0 → token 随机翻转 → "Carson İnt-S佳 Skill"
  3. 缓存死锁: token 不变 → gradient cache 不更新 → 梯度更弱 → 恶性循环
  ⚠️ 根本原因: STE + vocab_size=152K 的数学限制, 无法通过调参解决

Soft Prompt 的潜在失败模式:
  1. 投影损失: embedding 优化得很好但投影回 token 后语义丢失
  2. 真空地带: embedding 飘到离所有真实 token 都很远的区域
  3. RM 近似误差: 软投影到 RM 空间可能引入噪声
  ✅ 这些问题可以通过 drift_coeff 调参缓解
```

### 6.5 计算开销对比

```
DTO:
  每步 forward: LM forward (embedding 通过 STE 计算)
  每步 backward: 梯度传到 [1, 180, 152064] 的 logits
  显存: skill_logits + grad = 2 × 180 × 152064 × 4B ≈ 220MB

Soft Prompt:
  每步 forward: LM forward (直接传 embedding)
  每步 backward: 梯度传到 [1, 180, 3584] 的 embedding
  显存: skill_embeds + grad = 2 × 180 × 3584 × 4B ≈ 5MB
  额外: RM 软投影需要计算 [180, 152064] 的相似度矩阵 (临时)
```

### 6.6 一句话总结

```
DTO:  在 152064 维离散空间里找最优 token 组合 — 搜索空间太大, 梯度太小, 走不动
Soft: 在 3584 维连续空间里微调 embedding — 搜索空间合理, 梯度充足, 再投影回 token
```

---

## 7. 关键代码位置

| 组件 | 文件 | 关键函数 |
|------|------|---------|
| **DTO** | | |
| Logits 模块 | `src/skill_embedder.py` | `DiffSkillLogitsToEmbedding` |
| STE forward | `src/skill_embedder.py` | `forward()` |
| 解码 argmax | `src/skill_embedder.py` | `project_to_token_ids()` |
| Fluency 正则 | `src/skill_trainer.py` | `optimize()` 内 |
| DTO 优化主循环 | `src/skill_trainer.py` | `optimize()` |
| **Soft Prompt** | | |
| Soft embedding 模块 | `src/soft_prompt_trainer.py:28` | `SoftPromptEmbedding` |
| 初始化 | `src/soft_prompt_trainer.py:70` | `initialize(skill_token_ids)` |
| Forward (含 RM 投影) | `src/soft_prompt_trainer.py:91` | `forward()` |
| 投影回 tokens | `src/soft_prompt_trainer.py:117` | `project_to_token_ids()` |
| Drift 正则 | `src/soft_prompt_trainer.py:140` | `drift_loss()` |
| Soft 优化主循环 | `src/soft_prompt_trainer.py:218` | `optimize()` |
| **共用** | | |
| Factory 路由 | `src/ttso.py:141` | `_create_optimizer()` |
| 模板拼接 | `src/skill_template.py:142` | `SkillGenerationTemplate.apply()` |
| Outer loop | `src/ttso.py` | `run_iterative()` |
