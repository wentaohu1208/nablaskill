# TTSO Algorithm: Test-Time Skill Optimization

> **Project**: nablaskill
> **Last Updated**: 2026-03-12

---

## Notation

| Symbol | Definition |
|--------|-----------|
| $s$ | Skill token sequence $s = (s_1, s_2, \ldots, s_N)$ |
| $q$ | Query (user input, fixed) |
| $y$ | Response token sequence $y = (y_1, y_2, \ldots, y_M)$ |
| $\phi$ | Skill logits, the optimizable variable: $\phi \in \mathbb{R}^{N \times V}$ |
| $V$ | Vocabulary size |
| $\text{LM}$ | Frozen language model (parameters fixed) |
| $\text{RM}$ | Frozen reward model (parameters fixed) |
| $E_{\text{LM}}$ | LM embedding table: $\mathbb{R}^{V \times d}$ |
| $\sigma_\tau(\cdot)$ | Softmax with temperature $\tau$ |
| $\text{STE}(\cdot)$ | Straight-Through Estimator |

---

## Algorithm 1: Inner Loop — Differentiable Token Optimization (DTO) for Skill

**Input**: Query $q$, response $y$ (fixed), initial skill text $s^{(0)}$, LM, RM, hyperparameters $(T, \eta, \lambda_{\text{nll}}, \lambda_{\text{flu}}, \lambda_{\text{rm}})$

**Output**: Optimized skill text $s^*$

```
1.  Tokenize skill: s_ids = Tokenize(s⁰)                          // [N] token IDs
2.  Initialize logits: φ ∈ ℝ^{N×V}, φ[i, s_ids[i]] = 10, else 0  // scaled one-hot
3.  Tokenize response: y_ids = Tokenize(y)                         // [M] token IDs
4.  Build template:
      prefix_emb = Embed(LM, "Use the following skill:\n")         // fixed
      suffix_emb = Embed(LM, "\nProblem: q" + y)                   // fixed
      // Layout: [prefix_emb | SOFT_SKILL | suffix_emb]
      // Track positions: prefix_len, skill_len, query_suffix_len, resp_start

5.  optimizer = Adam({φ}, lr=η)
6.  cached_grad = None

7.  For t = 1 to T:
      optimizer.zero_grad()

      // --- STE Forward ---
8.    p = σ_τ(φ)                                     // soft probabilities [N, V]
9.    ŝ = one_hot(argmax(p))                          // hard one-hot [N, V]
10.   z = STE(p, ŝ) = ŝ - sg(p) + p                  // forward: ŝ, backward: ∂/∂p
          // sg(·) = stop_gradient

      // --- Gradient Caching Shortcut ---
11.   If cached_grad ≠ None AND argmax(φ) unchanged:
12.     loss = ⟨cached_grad, z⟩                       // linear approx, skip LM forward
13.     Go to step 22

      // --- Soft Embedding ---
14.   e_skill = z · E_LM                              // [N, d] soft skill embeddings

      // --- LM Forward ---
15.   input_emb = Concat(prefix_emb, e_skill, suffix_emb)   // [L, d], L = total length
16.   logits_all = LM(input_emb)                             // [L, V]

      // --- Loss 1: Response NLL ---
17.   logits_resp = logits_all[resp_start-1 : resp_start+M-1]   // [M, V]
18.   L_nll = -λ_nll · CrossEntropy(logits_resp, y_ids)
          // Measures: how well does this skill make LM "want to" produce response y

      // --- Loss 2: Skill Fluency ---
19.   logits_skill = logits_all[prefix_len-1 : prefix_len+N-1]  // [N, V]
20.   L_flu = λ_flu · Σ_i log_softmax(logits_skill[i]) · sg(z[i])
          // Measures: how likely is this skill under the LM (readability)
          // Note: z is detached as target to avoid moving-target gradient bias

      // --- Loss 3: Reward Model ---
21.   If RM available:
        e_skill_rm = z · E_RM                                    // RM embeddings
        rm_input = Concat(rm_prefix_emb, e_skill_rm, rm_suffix_emb)
        L_rm = λ_rm · RM(rm_input)
      Else:
        L_rm = 0

      // --- Combined Loss (minimize) ---
22.   loss = -(L_nll + L_flu) - L_rm
          // Equivalent to: maximize (response_quality + skill_readability + reward)

23.   loss.backward()                                 // gradient flows through STE
24.   cached_grad = ∂loss/∂z                          // cache for next iteration
25.   optimizer.step()
26.   lr_scheduler.step()

      // --- Check token change ---
27.   If argmax(φ) changed:
28.     cached_grad = None                             // invalidate cache

29. s* = Decode(argmax(φ))                             // final optimized skill text
30. Return s*
```

---

## Algorithm 2: Single-Round TTSO (`run`)

**Input**: Query $q$, initial skill text $s_0$, frozen LM, frozen RM

**Output**: Final skill $s_{\text{final}}$, final response $y_{\text{final}}$, final reward $r_{\text{final}}$

```
===================== Phase A: 用原始 skill 生成并评分 =====================

1.  构造 prompt_0 = "Use the following skill:\n{s₀}\n\nProblem: {q}"
2.  y₀ = LM.generate(prompt_0)
      // LM 自回归生成, 产出完整 response 文本
      // 例: y₀ = "The block accelerates at g·sin(30°) = 4.9 m/s²"
3.  r₀ = RM(q, s₀, y₀)
      // RM 对 (query+skill, response) 整体打分
      // 例: r₀ = 0.35 (不太好)

===================== Phase B: 判断是否值得优化 ============================

4.  If r₀ ≥ min_reward_threshold:
5.    // 初始 response 已经够好, 不需要优化 skill
6.    Return (s₀, y₀, r₀)

===================== Phase C: DTO 优化 skill (核心) ======================

7.  调用 Algorithm 1: s* = DTO(q, y₀, s₀, LM, RM)
      //
      // 关键: 整个 DTO 过程中, y₀ 是 **固定的常数**, 不会变
      //
      // DTO 内部做了什么 (T 步梯度下降):
      //   - 把 s₀ 的每个 token 变成可优化的 logits φ
      //   - 构造输入: [prefix_emb | soft_skill(φ) | suffix_emb(q + y₀)]
      //                                                    ↑ y₀ 在这里, 作为固定 label
      //   - 每步计算 loss:
      //       "这个 skill 能多好地让 LM 输出 y₀?" (Response NLL)
      //       + "这个 skill 读起来像人话吗?"       (Skill Fluency)
      //       + "RM 觉得这个组合好吗?"             (RM Reward)
      //   - 梯度只更新 φ (skill logits), 其他一切不动
      //   - T 步后, argmax(φ) → decode → 得到优化后的 skill 文本 s*
      //
      // 例: s₀ = "1. Draw diagram 2. Apply F=ma 3. Solve"
      //     s* = "1. Decompose gravity along incline 2. Apply F=ma on slope axis 3. Use g·sin(θ)"
      //     (梯度把 skill 推向了更适合这道斜面题的方向)

===================== Phase D: 用优化后的 skill 重新生成 ===================

8.  构造 prompt* = "Use the following skill:\n{s*}\n\nProblem: {q}"
9.  y* = LM.generate(prompt*)
      // 注意: 这是一次全新的自回归生成, 不是 y₀
      // LM 看到了优化后的 skill, 可能生成完全不同的 response
      // 例: y* = "沿斜面方向, F = mg·sin(30°) = 2×9.8×0.5 = 9.8N, a = F/m = 4.9 m/s²"
10. r* = RM(q, s*, y*)
      // 例: r* = 0.72 (比 r₀=0.35 好很多)

===================== Phase E: Rejection Sampling ==========================

11. If r* > r₀ + δ:                    // δ = reward_improvement_threshold (默认 0)
12.   // 优化有效, 接受
13.   Return (s*, y*, r*)
14. Else:
15.   // 优化后反而更差或没提升, 拒绝, 保留原始版本
16.   Return (s₀, y₀, r₀)
```

**为什么需要 Phase D (重新生成)?**
- DTO 优化 skill 时, response 是固定的 y₀
- 但我们真正关心的是: 新 skill 能不能引导 LM 生成更好的 response
- 所以必须用 s* 重新生成一个 y*, 然后让 RM 独立评估

**为什么需要 Phase E (rejection sampling)?**
- DTO 的优化方向是 "让 skill 更好地解释 y₀"
- 但 y₀ 可能本身不好, 优化方向可能偏了
- 重新生成的 y* 未必比 y₀ 好 → 需要 RM 兜底

---

## Algorithm 3: Iterative TTSO (`run_iterative`) — Main Experiment

**Input**: Query $q$, initial skill text $s^{(0)}$, frozen LM, frozen RM, max rounds $K$

**Output**: Best skill $s_{\text{best}}$, best response $y_{\text{best}}$, best reward $r_{\text{best}}$

**动机**: Algorithm 2 的问题是 DTO 始终拟合初始 response $y_0$, 如果 $y_0$ 质量差,
skill 的优化方向可能偏。Iterative TTSO 让 skill 和 response 交替提升。

```
===================== 初始化 ================================================

1.  构造 prompt₀ = "Use the following skill:\n{s⁰}\n\nProblem: {q}"
2.  y⁰ = LM.generate(prompt₀)
      // 例: y⁰ = "acceleration is g sin theta = 4.9"  (简略, 质量一般)
3.  r⁰ = RM(q, s⁰, y⁰)
      // 例: r⁰ = 0.35

4.  记录: best_skill = s⁰, best_response = y⁰, best_reward = r⁰
5.  记录: current_skill = s⁰, current_response = y⁰

6.  If r⁰ ≥ min_reward_threshold:
7.    Return (s⁰, y⁰, r⁰)                  // 已经够好

===================== 外层循环: 交替优化 =====================================

8.  For k = 1, 2, ..., K:

    //
    // -------- Step A: 优化 skill (response 固定为上一轮的) --------
    //

9.    s^k = DTO(q, current_response, current_skill, LM, RM)
        //
        // 调用 Algorithm 1, 关键参数:
        //   - response = current_response (固定, 不参与梯度)
        //   - initial_skill = current_skill (起点)
        //
        // Round 1: DTO(q, y⁰, s⁰) → 优化 skill 使其更好地解释 y⁰
        // Round 2: DTO(q, y¹, s¹) → 优化 skill 使其更好地解释 y¹ (更好的 response)
        // Round 3: DTO(q, y², s²) → 优化 skill 使其更好地解释 y² (更更好的 response)
        //
        // 每一轮的优化目标都在变 (因为 response 在变), 这就是 "交替优化"

    //
    // -------- Step B: 用新 skill 重新生成 response --------
    //

10.   构造 prompt_k = "Use the following skill:\n{s^k}\n\nProblem: {q}"
11.   y^k = LM.generate(prompt_k)
        // 全新的自回归生成, LM 看到了更好的 skill

    //
    // -------- Step C: 评分 --------
    //

12.   r^k = RM(q, s^k, y^k)

    //
    // -------- Step D: 接受或提前停止 --------
    //

13.   If r^k > best_reward:
14.     best_skill = s^k
15.     best_response = y^k
16.     best_reward = r^k
        // 这一轮比之前都好, 记录为新的 best
17.   Else:
18.     Break
        // 这一轮没有提升, 说明已经收敛 (或者开始震荡), 提前停止
        // 不继续浪费计算了

    //
    // -------- Step E: 准备下一轮 --------
    //

19.   current_skill = s^k
20.   current_response = y^k
        // 下一轮的 DTO 将以这一轮的 response 作为优化目标

===================== 最终 Rejection Sampling ================================

21. improvement = best_reward - r⁰
22. If improvement > δ:                        // δ = reward_improvement_threshold
23.   Return (best_skill, best_response, best_reward)
24. Else:
25.   // 所有轮次的优化都没能显著超过原始版本
26.   Return (s⁰, y⁰, r⁰)
```

### 具体例子: 3 轮迭代

```
Round 0 (初始):
  skill:    "1. Draw diagram 2. Apply F=ma 3. Solve"
  response: "acceleration is g sin theta = 4.9"            (简略)
  reward:   0.35

Round 1:
  DTO 优化 skill (拟合目标 = Round 0 的 response):
    skill → "1. Decompose gravity along incline 2. Apply F=ma on slope 3. Compute"
  重新生成 response:
    response: "Along the incline, F = mg sin30° = 9.8N, a = F/m = 4.9 m/s²"
  reward: 0.58  ✓ 比 0.35 好, 接受, 继续

Round 2:
  DTO 优化 skill (拟合目标 = Round 1 的 response, 比 Round 0 的更好):
    skill → "1. Set up coordinate along incline 2. Net force = mg sin θ 3. a = g sin θ"
  重新生成 response:
    response: "Taking the incline as x-axis: F_net = mg sin(30°) = 2×9.8×0.5 = 9.8N.
               By Newton's second law, a = F/m = 9.8/2 = 4.9 m/s²."
  reward: 0.72  ✓ 比 0.58 好, 接受, 继续

Round 3:
  DTO 优化 skill (拟合目标 = Round 2 的 response):
    skill → "1. Choose axis parallel to inclined surface 2. mg sin θ = ma 3. a = g sin θ"
  重新生成 response:
    response: (类似 Round 2, 略有不同)
  reward: 0.70  ✗ 比 0.72 差, 停止

最终: 选择 Round 2 的结果 (reward = 0.72)
最终 rejection: 0.72 - 0.35 = +0.37 > δ, 接受
```

### 为什么每一轮的 DTO 目标不同?

```
Round 1 的 DTO 问的是:
  "什么样的 skill 能让 LM 自然地输出 'acceleration is g sin theta = 4.9' ?"
  → skill 被推向 "分解重力沿斜面" 方向

Round 2 的 DTO 问的是:
  "什么样的 skill 能让 LM 自然地输出 'Along the incline, F = mg sin30°...' ?"
  → skill 被推向 "建立坐标系, 列方程" 方向 (更精确, 因为 response 更详细了)

Round 3 的 DTO 问的是:
  "什么样的 skill 能让 LM 自然地输出 Round 2 那种详细推导?"
  → skill 已经很好了, 优化空间不大, reward 不再提升
```

**核心洞察**: 每一轮, DTO 的优化目标 (response) 质量越来越高, 所以 skill 被推向越来越精确的方向。这就是 "skill 和 response 协同进化" 的含义。

**Computational cost per query**:
- Single-round (Alg. 2): $T$ LM forwards (DTO) + 2 generations + 2 RM evals
- Iterative (Alg. 3): $K \times (T$ LM forwards + 1 generation + 1 RM eval$)$ + 1 initial generation + 1 initial RM eval
- Gradient caching reduces effective LM forwards: when argmax tokens unchanged, DTO step costs $O(1)$ instead of $O(L)$
- Early stopping 使实际 $K$ 通常小于 max_outer_rounds

---

## Key Properties

### What is optimized
- **Only** the skill logits $\phi \in \mathbb{R}^{N \times V}$ (one `nn.Parameter`)
- LM, RM, prefix, suffix, response — all frozen

### STE gradient flow
```
Forward:  φ → softmax → argmax → one-hot (discrete)
Backward: φ ← ∂softmax/∂φ ← identity ← ∂loss/∂z (continuous)
```
The hard one-hot goes forward (so LM sees discrete tokens), but the gradient pretends it was soft (so Adam can update φ continuously).

### Position indexing (critical correctness detail)
```
Sequence:  [p₁ ... p_P | s₁ ... s_N | q₁ ... q_Q | y₁ ... y_M]
Positions:  0      P-1   P     P+N-1  P+N    ...    resp_start

LM logits[i] predicts token at position i+1, therefore:
  - Skill fluency:  logits[P-1 : P+N-1]  predicts  s₁ ... s_N
  - Response NLL:   logits[resp_start-1 : resp_start+M-1]  predicts  y₁ ... y_M
  - resp_start = P + N + Q
```

### Three loss terms and their roles

| Term | Formula | Gradient pushes skill toward |
|------|---------|------------------------------|
| Response NLL | $-\lambda_1 \cdot \text{CE}(\text{logits}_y, y)$ | Skill that makes response more likely under LM |
| Skill Fluency | $\lambda_2 \cdot \sum \log P_\text{LM}(s_i \mid \text{prefix})$ | Skill that is readable / natural language |
| RM Reward | $\lambda_3 \cdot \text{RM}(\text{query+skill, response})$ | Skill that yields higher quality score |

### Iterative vs Single-Round

| | Single-Round | Iterative |
|--|--|--|
| Skill optimization target | Fixed resp $y^0$ (may be low quality) | Updated resp $y^{k-1}$ each round |
| Risk | Skill overfits to bad initial response | Moving target may cause oscillation |
| Mitigation | Rejection sampling at end | Early stopping + global best tracking |
| Cost | $O(T)$ | $O(K \cdot T)$, but K typically small (2-3) |
