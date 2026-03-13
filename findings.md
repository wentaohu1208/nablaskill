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

#### 方案 B: Sequential DTO (逐 token 优化, 参考 Nabla-Reasoner)

**思路**: 不同时优化所有 skill tokens，而是逐个 token 优化并提交。

```
1. 从 skill token 0 开始
2. 优化当前 token + 后续所有 ahead tokens (联合优化, 类似 Nabla lookahead)
3. Commit 当前 token (argmax), 前进到下一个
4. 每步 rollout 到 response 计算 loss (NLL + RM reward)
```

**优点**: 缩小每步搜索空间, 渐进式优化更稳定
**缺点**: 计算成本 = N × max_iters (N = skill token 数), 串行执行
**改动量**: 中等 (新增 sequential_trainer.py)

### 6.4 方案对比

| 维度 | 全局 DTO | Soft Prompt | Sequential DTO |
|------|---------|-------------|----------------|
| 优化空间 | logits [N, V] | embeddings [N, D] | logits [ahead, V] 逐步 |
| 梯度效率 | 低 (STE 瓶颈) | 高 (直通) | 中 (STE 但搜索空间小) |
| Skill 可读性 | 低 (乱码) | 低 (需投影) | 中 (渐进式) |
| 计算成本 | max_iters | max_iters | N × max_iters |
| 理论新颖性 | 低 | 中 | 高 (Nabla 启发) |

---

## 8. Nabla-Reasoner Sequential Decoding 深入分析 (2026-03-13)

### 8.1 核心架构: GenerationStates

Nabla-Reasoner 并非逐 token 独立优化，而是维护一个 **lookahead buffer** 联合优化后逐个提交:

```python
class GenerationStates:
    past_token_ids: List[int]   # 已提交的 tokens (frozen)
    ahead_token_ids: List[int]  # lookahead buffer (jointly optimized)
    kv_cache: past key-value cache for frozen prefix
```

### 8.2 优化循环

```
for each position:
    1. sample_token() → 采样一个新 token 加入 ahead buffer
    2. move_to_next_optimizable_token() → selector 判断是否需要优化
       - EntropySelector: 高 entropy 位置需要优化
       - ConfidenceSelector: 低 confidence 位置需要优化
       - GradientSelector: 高梯度 norm 位置需要优化
    3. optimize_ahead_latents() → 联合优化整个 ahead buffer
       - LatentTrainer.optimize(): Adam + cosine LR, max_iters 步
       - Loss: NLL + RM reward (同 TTSO)
    4. commit_n_tokens(1) → 提交第一个 ahead token 到 past
       - 更新 kv_cache
       - 缩短 ahead buffer
```

### 8.3 关键洞察: 联合优化 vs 逐个提交

Nabla **不是** 逐个 token 独立优化的:
- 每次 `optimize_ahead_latents()` 同时优化 **所有 ahead tokens** (长度=lookahead_len)
- 但每次只 **提交 1 个 token** (最前面的)
- 类似 MPC (Model Predictive Control): plan ahead, commit one step

### 8.4 适配到 Skill Token 优化的关键差异

| 维度 | Nabla (Response Tokens) | TTSO Sequential (Skill Tokens) |
|------|------------------------|-------------------------------|
| 优化位置 | 末尾 (after prompt) | 中间 (between prefix and response) |
| Past/Ahead 管理 | past_token_ids 单调增长 | past_skill_tokens 单调增长 |
| 采样 | 需要 `sample_token()` 产生初始值 | 已有初始 skill tokens |
| Response | 每步可能变化 | 非优化变量, 但每步 rollout 经过 response 计算 loss |
| KV Cache | 对 past tokens 缓存 | 对 prefix + past_skill_tokens 缓存 |
| Selector | entropy/confidence/gradient | 无 (已删除, one-hot 初始化下 entropy/confidence 退化) |

### 8.5 适配设计 (for Skill)

由于 skill tokens 数量有限 (通常 50-200 tokens)，可简化 sampling，但保留 **rollout 到最后** 的核心:

- **Selector 已删除**: entropy/confidence selector 在 one-hot 初始化下退化 (所有位置 softmax 分布相同 → 全 skip 或全 optimize), 仅 gradient selector 理论可行但性价比不足
- **无 sampling**: 初始 skill tokens 已知 (来自原始 skill text)
- **Lookahead = 剩余所有 skill tokens**: 每次优化当前位置到末尾的所有 skill tokens
- **提交 1 token, 滑动窗口**: commit 后缩短 ahead, prefix 增长
- **Rollout 经过 response**: 每步 forward 都经过 `[prefix + committed_skills + ahead_skills + response]`，response 不是优化变量但参与 loss 计算 (NLL + RM reward)，与 Nabla-Reasoner 一致
- **Trajectory-based rejection sampling**: 参考 Nabla 的 `acceptance_criteria`，每次 commit 前通过完整 trajectory 比较 RM reward，只有更好才接受。维护动态 `reward_old` baseline。

```
Step 0: [prefix] [skill_0(opt) skill_1..N(ahead)] → optimize → rejection → commit skill_0
Step 1: [prefix skill_0(frozen)] [skill_1(opt) skill_2..N(ahead)] → optimize → rejection → commit skill_1
...
Step N: [prefix skill_0..N-1(frozen)] [skill_N(opt)] → optimize → rejection → commit skill_N

Trajectory-Based Rejection Sampling:
  opt_token = argmax(optimized ahead_logits[0])
  orig_token = original_skill[position]
  if opt_token ≠ orig_token:
    new_skill = decode(past + [opt_token] + argmax(ahead[1:]))
    new_response = generate(query, new_skill)
    reward_new = RM(query + new_skill, new_response)
    if reward_new > reward_old:
      commit(opt_token); reward_old = reward_new  # 动态提升门槛
    else:
      commit(orig_token)  # 保留原始

为什么只比较 RM reward:
  - ahead tokens 在旧 context 下优化, fluency 天然偏向旧 token → 不公平
  - 完整 trajectory (generate + RM) 是最可靠的评估方式
```

---

## 7. Open Questions (Updated)

- [x] Skill 优化后 decode 出的文本是否仍然是有意义的 instruction？→ **No. Token-level DTO 会导致乱码 (实验证实)**
- [x] 多少优化步数是最优的？→ **当前 DTO 方案下无法找到合理的步数/超参组合**
- [ ] 是否需要为 Skill 设计专门的 reward model？当前复用 response-level RM
- [ ] Cross-domain transfer 是否需要 domain-aware loss？
- [x] 多个 retrieved skills 的联合优化 → 已决定: 选择性优化单个 best skill (Phase 2)
- [ ] 回写的优化 skill 是否会污染 SkillBank 检索质量？需要 A/B 测试
- [x] Markdown 格式的 skill instructions 在 token-level 优化后是否保持结构？→ **No. Token 翻转后 Markdown 结构立即被破坏**
- [x] **方案 A/B/C 哪个最适合 TTSO？** → dto/soft_prompt/sequential_dto 三种 (TextGrad 已删除)
- [ ] **Sequential DTO 能否解决全局 DTO 的 STE 瓶颈？** — 逐 token commit 缩小搜索空间, 但 STE 问题本质不变
- [ ] **Sequential 方案 vs Soft Prompt 方案对比**: 哪个更适合 skill optimization?

---

## 9. Implementation Architecture (v3 — Phase 2 Complete)

```
nablaskill/
├── src/
│   ├── __init__.py              # Package exports (expanded)
│   ├── utils.py                 # Seed, device, LR schedulers (from Nabla)
│   ├── skill_embedder.py        # DiffSkillLogitsToEmbedding (STE + soft embeds)
│   ├── skill_template.py        # SkillGenerationTemplate + SkillRewardTemplate
│   ├── skill_trainer.py         # SkillTrainer (DTO optimization loop)
│   ├── sequential_states.py     # SequentialSkillStates (past/ahead state management)
│   ├── sequential_trainer.py    # SequentialSkillTrainer (token-by-token DTO)
│   ├── soft_prompt_trainer.py   # SoftPromptTrainer (continuous embedding optimization)
│   ├── ttso.py                  # TTSODecoding (inner engine + factory router)
│   ├── pipeline.py              # TTSOPipeline (full orchestrator)
│   └── skillbank.py             # SkillBankAdapter (AutoSkill SDK wrapper)
├── run.py                       # Single-prompt CLI entry point
├── scripts/
│   ├── example_physics.py       # End-to-end example with hardcoded skill
│   ├── sweep_hyperparams.sh     # 1000-config DTO hyperparameter sweep
│   └── sweep_sequential_dto.sh # 214-config Sequential DTO hyperparameter sweep
├── tests/
│   ├── conftest.py              # Shared fixtures (tiny-gpt2)
│   ├── test_skill_embedder.py   # STE, gradient flow, init/deconstruct
│   ├── test_skill_template.py   # Position indexing, embedding concatenation
│   ├── test_skill_trainer.py    # Loss computation, optimization loop
│   └── test_pipeline.py         # Pipeline orchestration, selection, writeback
├── eval/                        # (TODO) Benchmark evaluation
├── guidance.txt                 # Research direction document
├── skill_optimization.md        # DTO + Soft Prompt 技术文档
├── task_plan.md                 # Planning
├── findings.md                  # This file
└── progress.md                  # Session logs
```
