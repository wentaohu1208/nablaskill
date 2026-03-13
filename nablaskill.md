# NablaSkill: Test-Time Skill Optimization via Sequential DTO

## 算法概述

将 Nabla-Reasoner 的逐 token 优化思想应用于 Skill prompt 优化。
给定一个 query 和一个可能不完全匹配的 skill，通过梯度下降逐 token 调整 skill 文本，使其更好地引导 LM 产出高质量 response。

---

## 例子设定

**Query**: "A 2kg block slides down a 30-degree frictionless incline. What is the acceleration?"

**原始 Skill**（Math skill，不是 Physics skill，共 30 tokens）:

> 1. Understand the problem
> 2. List known quantities
> 3. Apply mathematical rules
> 4. Solve step-by-step
> 5. Verify the solution

**目标**: 通过优化，让这个 Math skill 更适合解 Physics 问题。

---

## 外层：Round 0（初始评估）

用原始 skill 生成 response:

> Let me solve this. Given mass = 2kg, angle = 30 degrees. Using basic math: 2 times 9.8 times 0.5 = 9.8. The answer is 9.8.

RM 评分: **reward_old = 22.25**（答案对了但推理过程粗糙，没有物理分析）

---

## 外层：Round 1 → 进入内层 Sequential DTO

输入: 原始 skill（30 tokens），response_0（固定），reward_old = 22.25

---

### 内层 Step 0（优化全部 30 个 token）

将 30 个 skill tokens 全部初始化为可优化的 logits，用 Adam 跑 20 步梯度下降。

每步 forward 把 skill 的 soft embeddings 和 response_0 拼在一起，计算 loss（response NLL + skill fluency + RM reward）。

20 步后，检查第 1 个 token 的 argmax:

- 优化后: "#" → 还是 "#"
- 和原来一样 → **直接 commit，不触发 rejection sampling**

进入 Step 1。

---

### 内层 Step 1-7（前 8 个 token 都没变）

优化后第一个位置的 argmax 每次都和 original 一样（STE 梯度太小，没翻转）。

全部直接 commit，零额外成本。

此时 past = ["#", "Under", "stand", "the", "problem", "\n", "2", "."]，共 8 个 committed tokens（全是原始的）。

---

### 内层 Step 8（第一个真正的变化）

**Past**: 8 个已 committed 的原始 tokens（frozen）

**Ahead**: 从 original tokens 8-29 重新初始化，共 22 个 logits

20 步梯度优化后，第一个位置的 argmax:

- 原始: "Apply"
- 优化后: "Decompose"
- 不同! → **触发 rejection sampling**

**Rejection sampling 过程**:

Step A — 构建 new trajectory 的完整 skill:

> 把 past（8 个 committed）+ "Decompose" + 剩余 21 个 optimized tokens 的 argmax 拼起来，decode 成文本:
>
> 1. Understand the problem
> 2. List known quantities
> 3. **Decompose** mathematical rules  ← 这里变了
> 4. Solve step-by-step
> 5. Verify the solution

Step B — 用这个 new skill 生成一个完整的 new response:

> Let me decompose the forces on the block. On a 30-degree incline, the component of gravity along the surface is mg sin(30) = 2 times 9.8 times 0.5 = 9.8 N. By Newton's second law, a = F/m = 9.8/2 = 4.9 m/s squared.

Step C — RM 评分:

> reward_new = RM(query + new_skill, new_response) = **23.10**

Step D — 和 old trajectory 比较:

> reward_new(23.10) > reward_old(22.25)? → **YES，接受!**

**Commit "Decompose"**。

---

### 关键：rejection 之后下一步怎么走

Accept 了 "Decompose" 之后：

- past 变成 9 个 tokens: [..., "Decompose"]
- **reward_old 更新为 23.10**（new trajectory 的 reward 变成了新的 baseline）
- 下一步 ahead 从 original tokens 9-29 **重新初始化**（MPC 思想：旧的 ahead 是在 "Apply" 的 context 下优化的，现在 context 变成了 "Decompose"，重新规划更可靠）

如果刚才 reject 了呢？

- past 变成 9 个 tokens: [..., "Apply"]（保留原始）
- **reward_old 保持 22.25 不变**
- 下一步同样从 original tokens 9-29 重新初始化

无论 accept 还是reject，下一步的处理方式相同：重新初始化 ahead，继续优化。唯一的区别是 committed 的 token 不同，以及 reward_old 是否更新。

---

### 内层 Step 9-19（又一段没变化的区间）

每步优化后第一个 token 都和 original 一样 → 直接 commit。

---

### 内层 Step 20（第二个变化）

Past: 20 个 committed tokens（其中位置 8 是 "Decompose"，其余是原始）

优化后第一个位置: "step" → "force"

触发 rejection sampling:

- 构建 new skill: "...4. Solve **force**-by-step..."（语法不通）
- 生成 new response: LM 被乱 skill 误导，回答质量下降
- reward_new = 21.50
- 21.50 > 23.10 (当前 reward_old)? → **NO，拒绝!**

Commit 原始的 "step"。reward_old 保持 23.10。

---

### 内层 Step 21-26（继续）

都没变 → 直接 commit。

---

### 内层 Step 27（第三个变化）

优化后: "solution" → "result"

Rejection sampling:

- New skill: "...5. Verify the **result**"
- 生成 new response，推理清晰
- reward_new = 23.40
- 23.40 > 23.10? → **YES，接受!**

Commit "result"。reward_old 更新为 23.40。

---

### 内层 Step 28-29（最后两步）

没变 → commit → 内层结束。

---

### 内层结果

最终 skill_1:

> 1. Understand the problem
> 2. List known quantities
> 3. **Decompose** mathematical rules  ← accepted (Step 8)
> 4. Solve step-by-step               ← "force" rejected，保留原始 (Step 20)
> 5. Verify the **result**             ← accepted (Step 27)

统计: 30 tokens 中 3 个触发 rejection sampling，2 个 accepted，1 个 rejected。
生成次数: 3 次（只有变化的 token 才需要生成）。

---

## 回到外层

用 skill_1 正式生成 response_1:

> The block experiences gravitational force mg along the incline surface, which equals mg sin(30). Decompose: F = 2 × 9.8 × sin(30°) = 9.8 N. By Newton's second law, a = F/m = 9.8/2 = 4.9 m/s².

reward_1 = RM(query + skill_1, response_1) = **23.50**

reward_1 (23.50) > reward_0 (22.25) → 接受 Round 1，进入 Round 2。

---

## 外层：Round 2

输入: skill_1，response_1，reward_old = 23.50

内层 Sequential DTO 在 skill_1 基础上继续优化:

- 30 个 token 中只有 1 个翻转，rejection sampling 后 reward = 23.20 < 23.50 → rejected
- 其余全部没变

skill_2 ≈ skill_1（几乎没变化）

reward_2 = 23.10 < reward_1 = 23.50 → 停止。

---

## 最终输出

取所有 round 中 reward 最高的:

**Round 1: skill_1 + response_1, RM = 23.50**

---

## 算法特性总结

**逐 token 的保守策略**: 每个 token 的变化都必须经过完整 trajectory 验证（generate + RM），保证 skill 质量单调不降。

**成本可控**: 大部分 token STE 梯度不够翻转 argmax，直接 commit。只有少数变化的 token 触发 generate + RM，实际生成次数远小于 token 总数。

**reward_old 动态更新**: 每次 accept 后 baseline 升高，后续变化必须超过新 baseline 才能被接受。标准越来越严格，防止后续 token 退化。

**每步从 original 重新初始化 ahead**: 类似 MPC（Model Predictive Control），commit 一步后重新规划。因为 context 已变（committed 的 token 可能和 original 不同），旧规划不再可靠。

**双层循环分工**: 内层负责微调 skill tokens（response 固定）。外层负责刷新 response（用最新 skill 重新生成），让 skill 和 response 交替进化。
