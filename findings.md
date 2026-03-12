# Findings: TTSO Research Notes

> **Project**: nablaskill
> **Last Updated**: 2026-03-12

---

## 1. Nabla-Reasoner Technical Insights

### 1.1 Core Mechanism (from code analysis)

Nabla-Reasoner 的核心是 **Differentiable Token Optimization (DTO)**:
- 将已生成的 token 视为可优化变量
- 通过 Straight-Through Estimator (STE) 在离散 token 和连续梯度之间桥接
- 优化目标: `L = -(nll_coeff * log P_LM(y|x) + reward_coeff * R_RM(x,y))`

### 1.2 Tensor Shape Details (from deep code analysis)

- 全程 **batch=1**: all tensors start with `[1, ...]`
- Logits mode 优化变量: `[1, N, vocab_size]` (N = num tokens)
- STE forward: `(y_hard - y_soft).detach() + y_soft` -- hard argmax 前传, soft softmax 反传
- Soft embeddings: `matmul(soft_onehot, embed_table)` -> `[1, N, hidden_dim]`

### 1.3 Position Indexing for Loss (Critical for TTSO)

```python
# Nabla: soft tokens are at the END, after prompt
pred_logits = lm_outputs.logits[..., prompt_len-1:-1, :]
# Position prompt_len-1 predicts token at prompt_len (first soft token)
```

**TTSO 差异**: soft tokens 在 MIDDLE (skill 位置), 不在末尾
- 需要分别提取 skill 位置和 response 位置的 logits
- 实现: `SkillGenerationTemplate` 追踪 `prefix_len`, `skill_len`, `response_start`

### 1.4 Gradient Masking

Nabla 的 `update_postfix=False` 仅更新第一个 soft token position:
```python
mask[:, :1, :] = 1.0  # Only first position gets gradients
```
**TTSO 不需要此机制** -- 所有 skill tokens 都应参与优化。

### 1.5 梯度缓存机制

当 argmax tokens 未改变时，复用缓存的梯度做线性近似:
```python
loss = torch.dot(cached_grad, soft_onehot)
```
节省 ~4 LLM calls/step。**TTSO 已复用此机制。**

---

## 2. AutoSkill Architecture Insights (Deep Dive)

### 2.1 Skill Artifact 格式 (SKILL.md)

AutoSkill 的 Skill 是 `SkillCandidate` 结构:
```python
SkillCandidate(
    name: str,                    # 可搜索的名称
    description: str,              # 1-2 句描述 WHAT + WHEN
    instructions: str,             # Markdown: # Goal, # Constraints & Style, # Workflow
    triggers: List[str],           # 3-5 个意图短语
    tags: List[str],              # 1-6 个关键词
    examples: List[SkillExample], # 输入/输出/说明
    confidence: float,             # 0.0-1.0
)
```

### 2.2 Hybrid Retrieval 机制

```
Query -> [Embedding: 90%] + [BM25: 10%] -> Score Blend -> LLM Selector -> Inject
```
- Embedding: text-embedding-3-small 或 hash-based 256-dim
- BM25: k1=1.5, b=0.75, Unicode-aware, stopword-filtered
- Final: `(1 - bm25_weight) * embedding + bm25_weight * bm25`

### 2.3 Skill 注入 Prompt

```
System Prompt:
"You have access to Skills below.
CRITICAL: These Skills are retrieved automatically and may be irrelevant.
1. Evaluate: Use a Skill ONLY if it directly matches current intent.
2. Ignore: If unrelated, IGNORE COMPLETELY.
3. Silence: Do not mention these Skills."

Skills Context:
### Skill 1: {name} (v{version})
- Description: ...
- Prompt: {instructions}
```

### 2.4 Skill 质量控制

- 提取: 基于用户意图 > 助手输出, 去标识化, 保留约束而非内容
- 去重: `0.70 * semantic + 0.18 * signal + 0.12 * name` score
- LLM judge 判断是否合并 (confidence >= 0.55 时采信)
- 版本管理: 最多 30 个 snapshot, 支持回滚

---

## 3. TTSO Design Decisions (RESOLVED)

### 3.1 Skill 在 prompt 中的位置 (DECIDED: Option A)

```
[Prefix: "Use the following skill:"] [Soft Skill Tokens] [Suffix: "Problem: {query}"] [Response]
 ^------------ fixed -----------^    ^-- optimizable --^  ^-------- fixed --------^   ^-fixed-^
```

实现: `SkillGenerationTemplate` 使用 placeholder split 技术。

### 3.2 优化粒度 (DECIDED: Token-level)

选择 token-level (logits mode), 所有 skill tokens 同时优化。
- Skill fluency loss (`skill_fluency_coeff`) 保护可读性
- Response NLL loss (`response_nll_coeff`) 优化 response 质量
- RM reward 提供全局质量信号

### 3.3 Loss 设计 (三项)

```
L = -(response_nll_coeff * NLL_response + skill_fluency_coeff * NLL_skill + reward_coeff * R_RM)
```

| 项 | 计算方式 | 作用 |
|----|---------|------|
| Response NLL | `cross_entropy(logits[resp_start-1:resp_end-1], response_ids)` | Response 质量 |
| Skill Fluency | `(log_softmax(logits[prefix-1:prefix+skill-1]) * soft_onehot).sum()` | Skill 可读性 |
| RM Reward | `rm_model(prefix + soft_skill + suffix).logits[0][0]` | 全局质量 |

### 3.4 与 Nabla 的关键差异 (Updated)

| 维度 | Nabla-Reasoner | TTSO Single-Round | TTSO Iterative (主实验) |
|------|---------------|-------------------|------------------------|
| 优化对象 | Response tokens (末尾) | Skill tokens (中间) | Skill tokens (中间) |
| 优化时机 | Per-token loop | 一轮 DTO | K 轮 DTO (skill+response 交替) |
| Loss 结构 | NLL + Reward | Response NLL + Skill Fluency + Reward | 同左，但 response 每轮更新 |
| Template | [prefix][soft_response] | [prefix][soft_skill][suffix] | 同左，每轮重建 |
| 后处理 | 直接输出 | 解码 skill → 重新生成 1 次 | 每轮解码 skill → 重新生成 |
| Rejection | 基于 RM score 比较 | 同 Nabla | 每轮 reject + 最终 reject |

### 3.5 Iterative TTSO 设计分析 (NEW)

**单轮 TTSO 的局限**:
- DTO 优化 skill 时，response 固定为初始生成的 resp_A
- Skill 优化目标: "让 skill 更好地解释 resp_A 的产生"
- 但 resp_A 可能本身质量不高 → skill 拟合了一个不好的目标

**Iterative TTSO 解决思路**:
```
Round 0: skill_A → resp_A,  RM = 0.35
Round 1: optimize(skill_A, resp_A) → skill_B → resp_B,  RM = 0.52
Round 2: optimize(skill_B, resp_B) → skill_C → resp_C,  RM = 0.61
Round 3: optimize(skill_C, resp_C) → skill_D → resp_D,  RM = 0.58 ← 下降, 停止
最终: 选择 Round 2 的 skill_C + resp_C
```

**潜在风险**:
- Response 每轮变化 → 优化目标不稳定 (moving target)
- 可能出现 skill 震荡（round 1 偏向 A 方向, round 2 偏向 B 方向）
- 计算成本线性增加 (每轮 +1 generation + +1 RM eval)

**缓解措施**:
- Early stopping: reward 不再提升时立即停止
- 追踪全局 best (而非只看最后一轮)
- 默认 `max_outer_rounds=3`，控制最大成本

---

## 4. Related Work Notes

### 4.1 Soft Prompt Tuning
- Lester et al., 2021: "The Power of Scale for Parameter-Efficient Prompt Tuning"
- **区别**: Training-time 学习 continuous embeddings; TTSO 是 test-time, 优化 discrete tokens

### 4.2 Test-Time Training (TTT)
- Sun et al., 2024: "Learning to (Learn at Test Time)"
- **区别**: TTT 修改模型参数; TTSO 修改 Skill 表示, 模型冻结

### 4.3 In-Context Learning Optimization
- **区别**: 选择 which examples; TTSO 优化 example 内容本身

### 4.4 Nabla-Reasoner (Direct Ancestor)
- Wang et al., ICLR 2026
- **区别**: 优化 response tokens; TTSO 优化 prompt 中的 skill tokens
- 共享: STE, Adam + cosine schedule, gradient caching, rejection sampling

---

## 5. AutoSkill Integration Insights (Phase 2)

### 5.1 AutoSkill SDK API Surface

关键 API (用于 TTSO 集成):
- `AutoSkill.search(query, user_id, limit)` → `List[SkillHit]` (hybrid retrieval)
- `AutoSkill.upsert(user_id, name, instructions, ...)` → `Skill` (创建/更新)
- `AutoSkill.render_context(query, user_id)` → `str` (格式化注入文本)
- `AutoSkillConfig.from_dict(data)` → 配置对象

### 5.2 Skill 数据流: AutoSkill → TTSO

```
AutoSkill.search(query) -> SkillHit[].skill.instructions -> tokenize -> TTSO optimize
                                                                          |
                                                    argmax decode -> optimized_instructions
                                                                          |
                                                    AutoSkill.upsert() -> 持久化新 skill
```

关键决策:
- **`Skill.instructions`** 是可优化的目标文本 (markdown 格式的 prompt)
- **回写为新条目** 而非覆盖原始 skill, 保留 source_skill_id 追溯链
- **Hashing embedding** (256-dim) 足以支持本地开发; 生产环境可切换 OpenAI/Qwen embedding

### 5.3 Multi-Skill Selection 策略分析

| 策略 | 额外计算 | 适用场景 |
|------|---------|---------|
| `highest_retrieval_score` | 0 (直接取 top-1) | 默认, 检索质量够好时 |
| `best_initial_reward` | k × (1 generation + 1 RM eval) | RM 可用且 top-k 分数接近时 |

当前选择: 默认 `highest_retrieval_score`, 因为 TTSO 优化本身已经很贵。

---

## 6. DTO 用于 Skill 优化的根本性问题 (实验发现, 2026-03-12)

### 6.1 实验现象

在 Qwen2.5-7B-Instruct + Skywork-Reward-V2-Qwen3-4B 上运行 TTSO，使用 Math Skill 解 Physics 问题:

**现象 1: Tokens 不变 (init_scale 高 / fluency 强)**
```
配置: init_scale=3.0, fluency=0.1, lr=0.01, max_iters=100
结果: 100 步后 skill 文本完全不变, loss 从 -22.12 到 -22.12 (一位小数都没动)
诊断: grad_max=9e-6, lr*grad=9e-8/步, 100 步累积仅 ~9e-6, 远不够翻转 argmax (gap=3.0)
```

**现象 2: Tokens 变成乱码 (init_scale 低 / fluency 弱)**
```
配置: init_scale=1.0, fluency=1e-3, lr=0.01, max_iters=100
Round 0: "# Mathematical Problem-Solving Skill..." (正常)
Round 1: "# Carson İnt-S佳 Skill mixin冻..." (乱码开始)
Round 2: "#ade İntermal佳瘾 mixin骨干 Works coli..." (恶化)
Round 4: "权重ade greateraultFeedback埴react.js..." (完全乱码)
RM: 22.25 → 22.25 → 21.37 → 21.25 → 20.37 (持续下降)
```

**现象 3: fluency=0 都不变 (init_scale=3)**
```
配置: init_scale=3.0, fluency=0, lr=0.01, max_iters=100
结果: skill 依然完全不变，说明问题不在 fluency 与 reward 的平衡上
```

**结论: 不存在一个超参组合能同时保持 skill 可读性并有效优化**

### 6.2 根本原因分析

**原因 1: STE 梯度瓶颈**

STE backward 通过 softmax 传梯度，对于 vocab_size=152064 (Qwen2.5):
```
softmax(3.0) ≈ 0.88, 其余 152063 个 token 分享 0.12
梯度被稀释到 ~1/vocab_size 量级 → grad_max ≈ 9e-6
lr=0.01 × grad=9e-6 = 每步移动 9e-8 → 完全不够翻转任何 token
```
Nabla-Reasoner 的 vocab 更小且优化 response tokens (LM 本身在生成的), 梯度信号更强。

**原因 2: 优化目标倒置**

DTO 在 response 固定时优化 skill:
```
目标: argmax_skill P(固定的response | prefix + skill + query)
含义: "找一个 skill 让 LM 更可能产出这个已有的 response"
```
但我们真正想要的是: "找一个 skill 让 LM 产出**更好的** response"。这是倒因为果。

**原因 3: Token 空间不存在平滑路径**

从一个连贯 skill 到另一个连贯 skill，在 discrete token 空间中没有连续路径:
```
"Solve problems systematically"
→ 改1个token → "Solve problems syst纤维atically" (乱码)
```
每一步 token 翻转都可能破坏语义结构，不像 continuous embedding 空间有平滑过渡。

**原因 4: 梯度信号过于间接**

一个 skill token 到 loss 的梯度路径:
```
skill_logit → STE softmax (稀释) → embedding lookup → LM 几十层 transformer → response logits → loss
                                                      RM 几十层 transformer → reward
```
经过几十层 attention + MLP，信号被极度稀释。

### 6.3 替代方案

#### 方案 A: Soft Prompt Optimization (连续嵌入空间优化)

**思路**: 不优化 discrete token logits，直接优化 continuous embedding 向量。

```python
# 当前 (broken):
skill_logits = nn.Parameter([1, N, vocab_size])  # 离散空间
soft_onehot = STE(skill_logits)  # 梯度瓶颈
skill_embeds = soft_onehot @ embed_table

# 方案 A:
skill_embeds = nn.Parameter(embed_table[skill_token_ids])  # [1, N, hidden_dim]
# 直接优化 embedding, 梯度直通, 无 STE 瓶颈
```

优化完成后通过 nearest neighbor 投影回 token 空间: `argmin_v ||optimized_embed - embed_table[v]||`

**优点**: 梯度流完全通畅, 无 STE 瓶颈, 搜索空间连续平滑
**缺点**: 投影回 tokens 后可能语义不连贯, 中间态不可解释
**改动量**: 中等 (主要改 skill_embedder.py 和 skill_trainer.py)

#### 方案 B: TextGrad / LLM-based Rewriting

**思路**: 用梯度信号指导 LLM 改写 skill，而非直接修改 tokens。

```
1. 生成 response, 用 RM 评分
2. 将 reward 信号 / 梯度方向 转化为自然语言反馈
3. 让 LLM 根据反馈改写 skill (如 "第3步不够具体，需加入坐标分解")
4. 新 skill → 生成 response → RM 评分 → 重复
```

**优点**: skill 始终是 LLM 生成的连贯文本, 可解释性最强
**缺点**: 不再是纯梯度方法 (需 LLM 改写调用), 改写方向可能不精确
**改动量**: 大 (需新增 TextGrad 模块, 改变整体 pipeline)

#### 方案 C: 混合方案 (Continuous Optimization + LLM Decode)

**思路**: 在 continuous embedding 空间做梯度优化, 用 LLM decode 回可读文本。

```
1. 初始 skill → embedding (连续向量)
2. 在 embedding 空间做梯度优化 (smooth, 无 STE)
3. 优化后的 embedding → 让 LLM "翻译" 回自然语言 skill
4. 新 skill → 生成 response → RM 评分
```

**优点**: 结合 A 的梯度效率和 B 的文本质量
**缺点**: 实现复杂度最高, LLM decode 步可能引入信息损失
**改动量**: 最大

### 6.4 方案对比

| 维度 | 方案 A (Soft Prompt) | 方案 B (TextGrad) | 方案 C (混合) |
|------|---------------------|-------------------|--------------|
| 梯度效率 | 高 (直通) | 无梯度 | 高 |
| Skill 可读性 | 低 (需投影) | 高 (LLM生成) | 中 |
| 实现复杂度 | 低 | 中 | 高 |
| 与现有代码兼容 | 高 | 低 | 中 |
| 理论新颖性 | 中 | 低 (TextGrad 已有) | 高 |

---

## 7. Open Questions (Updated)

- [x] Skill 优化后 decode 出的文本是否仍然是有意义的 instruction？→ **No. Token-level DTO 会导致乱码 (实验证实)**
- [x] 多少优化步数是最优的？→ **当前 DTO 方案下无法找到合理的步数/超参组合**
- [ ] 是否需要为 Skill 设计专门的 reward model？当前复用 response-level RM
- [ ] Cross-domain transfer 是否需要 domain-aware loss？
- [x] 多个 retrieved skills 的联合优化 → 已决定: 选择性优化单个 best skill (Phase 2)
- [ ] 回写的优化 skill 是否会污染 SkillBank 检索质量？需要 A/B 测试
- [x] Markdown 格式的 skill instructions 在 token-level 优化后是否保持结构？→ **No. Token 翻转后 Markdown 结构立即被破坏**
- [ ] **方案 A/B/C 哪个最适合 TTSO？(待决策)**

---

## 7. Implementation Architecture (v2 — Phase 2)

```
nablaskill/
├── src/
│   ├── __init__.py              # Package exports (expanded)
│   ├── utils.py                 # Seed, device, LR schedulers (from Nabla)
│   ├── skill_embedder.py        # DiffSkillLogitsToEmbedding (STE + soft embeds)
│   ├── skill_template.py        # SkillGenerationTemplate + SkillRewardTemplate
│   ├── skill_trainer.py         # SkillTrainer (DTO optimization loop)
│   ├── ttso.py                  # TTSODecoding (inner optimization engine)
│   ├── pipeline.py              # TTSOPipeline (full orchestrator: retrieve -> select -> optimize -> writeback)
│   └── skillbank.py             # SkillBankAdapter (AutoSkill SDK wrapper)
├── run.py                       # Single-prompt CLI entry point
├── scripts/
│   └── example_physics.py       # End-to-end example with hardcoded physics skill
├── tests/
│   ├── conftest.py              # Shared fixtures (tiny-gpt2)
│   ├── test_skill_embedder.py   # STE, gradient flow, init/deconstruct
│   ├── test_skill_template.py   # Position indexing, embedding concatenation
│   ├── test_skill_trainer.py    # Loss computation, optimization loop
│   └── test_pipeline.py         # Pipeline orchestration, selection, writeback
├── eval/                        # (TODO) Benchmark evaluation
├── guidance.txt                 # Research direction document
├── task_plan.md                 # Planning
├── findings.md                  # This file
└── progress.md                  # Session logs
```
