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

| 维度 | Nabla-Reasoner | TTSO (Implemented) |
|------|---------------|-------------------|
| 优化对象 | Response tokens (末尾) | Skill tokens (中间) |
| 优化时机 | Per-token loop (多轮) | Per-query (一轮 DTO) |
| Loss 结构 | NLL + Reward | Response NLL + Skill Fluency + Reward |
| Template | [prefix][soft_response] | [prefix][soft_skill][suffix] |
| 后处理 | 直接输出 | 解码 skill text -> 重新生成 response |
| Rejection | 基于 RM score 比较 | 基于 RM score 比较 (同 Nabla) |

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

## 6. Open Questions

- [ ] Skill 优化后 decode 出的文本是否仍然是有意义的 instruction？(需要实验验证 skill_fluency_coeff 的效果)
- [ ] 多少优化步数是最优的？当前默认 20 步, 是否足够？
- [ ] 是否需要为 Skill 设计专门的 reward model？当前复用 response-level RM
- [ ] Cross-domain transfer 是否需要 domain-aware loss？
- [x] 多个 retrieved skills 的联合优化 → 已决定: 选择性优化单个 best skill (Phase 2)
- [ ] 回写的优化 skill 是否会污染 SkillBank 检索质量？需要 A/B 测试
- [ ] Markdown 格式的 skill instructions 在 token-level 优化后是否保持结构？

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
