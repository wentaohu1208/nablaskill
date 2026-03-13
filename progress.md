# Progress Log: TTSO Project

> **Project**: nablaskill
> **Started**: 2026-03-12

---

## Session: 2026-03-12 (Session 1 - Planning)

### Completed
- [x] 完成研究方向梳理 (guidance.txt)
- [x] 深入分析 Nabla-Reasoner 代码库 (decoding.py, optimize.py, templates.py, utils.py)
- [x] 分析 AutoSkill 的 Skill 表示和检索机制
- [x] 创建项目 planning 文件 (task_plan.md, findings.md, progress.md)
- [x] 识别核心技术组件和设计决策点

### Key Insights
1. Nabla 的 `latents` mode (hidden state optimization) 比 `logits` mode 更适合 Skill optimization
2. Skill 的结构化特性 (steps, constraints) 可能需要 segment-level optimization 而非 token-level
3. 梯度缓存机制可直接复用，显著降低计算成本
4. Skill selection criteria 应从 token-level 提升到 Skill-level

---

## Session: 2026-03-12 (Session 2 - Implementation)

### Completed
- [x] **Phase 0 Complete**: 深入研究 AutoSkill 和 Nabla-Reasoner 代码
  - AutoSkill: SKILL.md artifact 格式, hybrid retrieval (90% embedding + 10% BM25), LLM selector
  - Nabla: batch=1 全程, STE hard=True, position indexing for loss, gradient masking
- [x] **Phase 1 Core Engine v1**:
  - `src/utils.py` — 种子管理, 设备推断, LR schedulers (from Nabla)
  - `src/skill_embedder.py` — `DiffSkillLogitsToEmbedding` with STE
  - `src/skill_template.py` — `SkillGenerationTemplate` + `SkillRewardTemplate` (placeholder split)
  - `src/skill_trainer.py` — `SkillTrainer` with 3-component loss (response_nll + skill_fluency + reward)
  - `src/ttso.py` — `TTSODecoding` orchestrator (generate -> optimize -> accept/reject)
  - `run.py` — CLI entry point with full parameter set

### Key Design Decisions
1. **Skill template 使用 placeholder split**: `[prefix][soft_skill][suffix]`, skill 在中间位置
2. **三项 loss**: response NLL (让 response 更好), skill fluency (保持 skill 可读), RM reward (全局质量)
3. **Rejection sampling**: 比较 optimized vs original RM score, 仅在 improvement > threshold 时接受
4. **Gradient caching**: 复用 Nabla 机制, argmax tokens 不变时跳过完整 forward+backward
5. **vLLM + HF dual backend**: vLLM for fast generation, HF for gradient optimization (同 Nabla)

### Architecture Difference from Nabla
- Nabla: 优化 response tokens (在 prompt 之后)
- TTSO: 优化 skill tokens (在 prompt 中间, query 之前)
- Nabla: per-token iterative loop
- TTSO: per-query single optimization -> regenerate response

### Next Steps
1. [x] 运行 code review 检查实现质量
2. [x] 编写单元测试验证 template position 计算和 loss 正确性
3. [ ] 在实际模型上运行 prototype 测试
4. [x] Phase 2: 集成 AutoSkill 的 SkillBank 检索

---

## Session: 2026-03-12 (Session 3 - Code Review Fixes + Phase 2)

### Code Review Fixes (Phase 1 Post)
- [x] `skill_trainer.py`: `.detach()` on skill fluency loss target (moving-target gradient bias)
- [x] `skill_trainer.py`: `print()` → `logger.info()`, autocast device-aware
- [x] `skill_template.py`: tokenization boundary validation with warning
- [x] `utils.py`: `except Exception` → `except AttributeError`
- [x] `ttso.py`: 消除 `should_optimize()` 中重复 RM scoring
- [x] `ttso.py`: 移除未使用的 imports (`F`, `Dict`)
- [x] `ttso.py`: `print()` → `logger.info()` (全部 `_print` → `_log`)

### Unit Tests Created
- [x] `tests/conftest.py` — 共享 fixtures (tiny-gpt2 session-scoped)
- [x] `tests/test_skill_embedder.py` — STE soft/hard 模式, 梯度流, 初始化/销毁
- [x] `tests/test_skill_template.py` — 位置索引, embedding 拼接顺序, 无 response 模式
- [x] `tests/test_skill_trainer.py` — 三项 loss 计算, 优化循环, embedder 清理

### Phase 2 Complete: Skill Retrieval Pipeline Integration
- [x] **2.1 TTSO Pipeline** → `src/pipeline.py` (260 lines)
  - `TTSOPipeline`: 完整编排器 (retrieve → select → optimize → writeback)
  - 支持 direct skill 模式 (跳过检索) 和 SkillBank 检索模式
- [x] **2.2 AutoSkill SkillBank 集成** → `src/skillbank.py` (175 lines)
  - `SkillBankAdapter`: AutoSkill SDK 薄包装层
  - `SkillCandidate`: frozen dataclass 用于 pipeline 传递
  - 优雅降级: 未安装 AutoSkill 时抛出清晰 ImportError
- [x] **2.3 Multi-skill selection** → `src/pipeline.py:_select_skill()`
  - `highest_retrieval_score`: 零额外计算, 默认策略
  - `best_initial_reward`: 每个 candidate 生成 + RM 评分
- [x] **2.4 Writeback 机制** → `src/skillbank.py:writeback()`
  - 创建新 skill 条目 (不覆盖原始), 携带 source_skill_id 元数据
  - 双重门控: writeback_enabled + writeback_min_improvement
- [x] **2.5 Example script** → `scripts/example_physics.py` (195 lines)
  - 硬编码 Newtonian Mechanics 7步 workflow skill
  - 支持 tiny-gpt2 (CPU) 和真实模型 (GPU)
- [x] **Pipeline tests** → `tests/test_pipeline.py` (200 lines)
  - direct skill 模式, selection 策略, writeback 门控逻辑

### Key Design Decisions (Phase 2)
1. **SkillBank 隔离**: AutoSkill SDK 通过 adapter 模式隔离，optional dependency
2. **回写为新条目**: 不覆盖原始 skill, 保留 source → optimized 追溯链
3. **默认 highest_retrieval_score**: 因为 TTSO 优化本身已贵, 选择阶段不再增加 LLM calls
4. **Pipeline 支持双模式**: 有 SkillBank 时自动检索, 无 SkillBank 时接受 direct skill_text

---

## Session: 2026-03-12 (Session 4 - Iterative TTSO Variant)

### Completed
- [x] **2.6 Iterative TTSO (主实验变体)** → `src/ttso.py:run_iterative()`
  - Skill 和 Response 交替优化 (EM-style outer loop)
  - 每轮: DTO 优化 skill → regenerate response → RM 评分 → accept/reject
  - Early stopping: reward 不再提升时停止，追踪全局 best
  - `TTSOResult` 新增 `round_history` 和 `num_outer_rounds` 字段
  - `TTSOConfig` 新增 `max_outer_rounds` (默认 1 = 退化为单轮)
  - Pipeline 自动路由: `max_outer_rounds > 1` → `run_iterative()`
  - `run.py` 和 `example_physics.py` 均支持 `--max_outer_rounds`
  - `tests/test_iterative.py` 验证 round history, early stopping, 累积统计

### Key Design Decision
> **Iterative TTSO 作为主实验, 单轮作为 ablation baseline**
> - 动机: 单轮 skill 只拟合初始 (可能低质量的) response
> - 多轮让 skill 和 response 协同进化
> - 权衡: 每轮 +1 generation + +1 RM eval, 但 early stopping 控制成本
> - 默认 `max_outer_rounds=3`

### Next Steps
1. [x] 运行 `python scripts/example_physics.py` 验证 end-to-end pipeline
2. [ ] Phase 3: Cross-Domain Skill Transfer
3. [ ] Phase 4: Evaluation & Experiments

---

## Session: 2026-03-12 (Session 5 - DTO 实验 & 方案验证失败)

### 实验过程

1. **初次运行**: init_scale=10, grad_caching=True, max_iters=100
   - 结果: Grad Steps=1, skill 完全不变
   - 原因: cache 死锁 (tokens 不变 → cache 不失效 → 永远用旧梯度)

2. **修复 cache 死锁**: 加入 `cache_refresh_interval` 周期性强制 full forward
   - 仍然不变: init_scale=10 太高, softmax 太尖锐

3. **降低 init_scale=1, fluency=1e-3**:
   - 结果: tokens 翻转了, 但变成乱码
   - Round 1: "Carson İnt-S佳 Skill mixin冻"
   - Round 4: "权重ade greateraultFeedback埴react.js"
   - RM 持续下降: 22.25 → 20.37

4. **提高 fluency=0.1**:
   - 结果: tokens 完全不变 (fluency 把 tokens 锁死)
   - RM 波动 21.87~23.12 纯粹是 response 采样方差

5. **极端测试 fluency=0, init_scale=3**:
   - 结果: 依然不变
   - 诊断: grad_max=9e-6, 100步累积 ~9e-6, 远不够翻转 (gap=3.0)

### 根本结论

**Token-level DTO (STE) 用于 Skill 优化根本不可行**:
- STE 梯度瓶颈: vocab=152K 稀释梯度到 ~1e-6
- 不存在超参 sweet spot: tokens 要么不变要么乱码
- 优化目标倒置: 拟合固定 response 而非产出更好 response
- Token 空间无平滑路径: 一个 token 翻转就破坏语义

### 提出三个替代方案

1. **方案 A: Soft Prompt Optimization** — 直接优化 continuous embeddings, 绕过 STE
2. **方案 B: TextGrad / LLM Rewriting** — 用梯度/reward 信号指导 LLM 改写 skill
3. **方案 C: 混合方案** — continuous embedding 优化 + LLM decode

### 待决策
> ~~选择方案 A/B/C 之一推进实现 (Phase 2.7)~~
> **决策**: 方案 A (Soft Prompt) 和方案 B (TextGrad) 都实现, 通过 `--optimization_mode` 切换

---

## Session: 2026-03-12 (Session 6 - Phase 2.7 Implementation)

### Completed
- [x] **方案 A: Soft Prompt Optimization** → `src/soft_prompt_trainer.py`
  - `SoftPromptEmbedding`: 直接优化 continuous embeddings [1, N, hidden_dim]
  - 梯度直通无 STE 瓶颈
  - L2 drift 正则化 (`embed_drift_coeff`) 防止 embedding 偏离太远
  - cosine similarity nearest neighbor 投影回 tokens
- [x] **方案 B: TextGrad / LLM Rewriting** → `src/textgrad_trainer.py`
  - 纯推理方法, num_grad_steps 恒为 0
  - Feedback 生成 + Skill 改写 + RM 评分 循环
  - 结构化 prompt 模板保证输出质量
- [x] **Factory Pattern 路由** → `src/ttso.py:_create_optimizer()`
  - `TTSOConfig.optimization_mode`: "dto" | "soft_prompt" | "textgrad"
  - `run()` 和 `run_iterative()` 零改动, 统一接口
- [x] **CLI 支持** → `run.py` + `scripts/example_physics.py`
  - `--optimization_mode`, `--embed_drift_coeff`, `--textgrad_max_rewrites`
- [x] **Package exports** → `src/__init__.py` 新增 SoftPromptTrainer, SoftPromptEmbedding, TextGradTrainer

### Key Design Decision
> **三种方案共存, factory pattern 路由**:
> - 原始 DTO: `skill_trainer.py` 不动
> - Soft Prompt: `soft_prompt_trainer.py` 新文件
> - TextGrad: `textgrad_trainer.py` 新文件
> - 统一接口: `optimize()` + `get_reward_for_text()`
> - 路由: `TTSODecoding._create_optimizer()` 根据 config 创建对应 trainer

### Next Steps
1. [ ] 运行 Soft Prompt 模式验证 (应该能看到梯度有效流动)
2. [ ] 运行 TextGrad 模式验证 (应该能看到 skill 被 LLM 有意义地改写)
3. [ ] 对比三种方案的 RM score 和 skill 质量

---

## Session: 2026-03-12 (Session 7 - Code Hardening & Hyperparameter Tooling)

### Completed
- [x] **Code Review**: 全项目 code review, 发现硬编码超参、align_vocab 边界风险、测试覆盖不足
- [x] **消除硬编码超参数**:
  - `soft_prompt_trainer.py`: RM projection temperature `/0.1` → `/self.rm_projection_temperature`
  - `skill_trainer.py`: `init_logits.scatter_(2, ..., 10)` → `self.init_logit_scale`
  - `ttso.py` TTSOConfig 新增字段, factory 传递参数
- [x] **加固 align_vocab()**: pad_token_id fallback, src_idx 边界检查, 日志告警
- [x] **暴露 loss 配比为 CLI 参数**: `example_physics.py` 新增 5 个 CLI args
- [x] **技术文档**: 创建 `skill_optimization.md` (合并 DTO + Soft Prompt 详解 + 方案对比)
- [x] **超参扫描脚本**: `scripts/sweep_hyperparams.sh` — 1000 个 DTO 超参组合, 5 tier 覆盖

### Modified Files
- `src/soft_prompt_trainer.py` — rm_projection_temperature 参数化
- `src/skill_trainer.py` — init_logit_scale 参数化
- `src/ttso.py` — TTSOConfig 新增 2 字段, factory 传参
- `src/utils.py` — align_vocab 加固
- `run.py` — 新增 CLI args
- `scripts/example_physics.py` — 新增 loss 配比 + 超参 CLI args
- `scripts/sweep_hyperparams.sh` — **新建** (1000 configs sweep)
- `skill_optimization.md` — **新建** (技术文档)

---

## Session: 2026-03-13 (Session 8 - Sequential Optimization Research)

### Completed
- [x] **Nabla-Reasoner 逐 token 优化架构研究**
  - 分析 `decoding.py`: `GenerationStates`, `move_to_next_optimizable_token()`, `commit_n_tokens()`
  - 分析 `optimize.py`: `LatentTrainer.optimize()`, `DiffLogitsToEmbedding`
  - 关键洞察: Nabla 联合优化所有 ahead tokens, 但逐个提交 (类似 MPC)
- [x] **确定适配方案**: skill tokens 在中间 (非末尾), 无需 selector 和 sampling, lookahead = 剩余 skill tokens

### Key Design Decision
> **Sequential Skill Token Optimization 设计方案**:
> - 新增 `--optimization_mode sequential_dto`
> - 从 token 0 开始, 优化当前 + 后续所有 skill tokens (lookahead)
> - 每次提交 1 个 skill token, 前进到下一个
> - Response 始终固定, 仅提供 loss 信号
> - 简化版 (无 selector, 无 sampling), 适合 skill tokens 数量有限的场景

### Next Steps
1. [x] 实现 `SequentialSkillTrainer` → `src/sequential_trainer.py`
2. [x] 实现 `SequentialSkillStates` (past/ahead skill token 管理)
3. [x] 集成到 factory pattern
4. [ ] 端到端验证 (sequential_dto vs dto vs soft_prompt)
5. [ ] Phase 3: Cross-Domain Skill Transfer

---

## Session: 2026-03-13 (Session 9 - Sequential DTO Implementation + TextGrad Removal)

### Completed
- [x] **删除 TextGrad 方案**: 移除 `src/textgrad_trainer.py` 及所有引用
  - `src/__init__.py`: 移除 import 和 __all__ 条目
  - `src/ttso.py`: factory 分支从 "textgrad" 替换为 "sequential_dto", TTSOConfig 移除 textgrad_max_rewrites
  - `run.py`: CLI choices 和 config 传参更新
  - `scripts/example_physics.py`: 同步更新
- [x] **实现 Sequential DTO** → `src/sequential_trainer.py` (~370 lines)
  - `SequentialSkillStates`: 管理 past_ids (committed) + ahead (to optimize)
    - `init_ahead_logits()`: 为剩余 token 创建 scaled one-hot logits
    - `commit(n, ahead_logits)`: argmax 提交 n 个 token 到 past
    - `get_past_embeds()`: 查 embedding table 获取 frozen past embeddings
  - `SequentialSkillTrainer`: 逐 token 优化主循环
    - `_build_full_embeds()`: [prefix] + [past_skill_frozen] + [ahead_skill_soft] + [suffix+response]
    - `_build_rm_full_embeds()`: RM 版本 (past token embeds + ahead soft onehot × rm_embed_table)
    - `_optimize_position()`: 对当前位置的 ahead tokens 运行 max_iters 步 Adam 优化
    - `optimize()`: 外层循环遍历所有 skill token 位置
    - Loss: response_nll + skill_fluency (仅 ahead 部分) + RM reward
  - 集成到 factory: `TTSOConfig.sequential_commit_every`, `--optimization_mode sequential_dto`
- [x] **更新所有 markdown**: task_plan.md, findings.md, progress.md

### Key Design Decisions
> **每步 rollout through response**: 与 Nabla-Reasoner 一致, response 非优化变量但每步 forward pass 都经过 response 计算 NLL + RM reward
> **Skill fluency 仅对 ahead tokens**: past tokens 已 committed 不需要 fluency 约束
> **commit_every 可配置**: 默认 1 (逐个), 可设大以加速 (牺牲精度)

### Modified Files
- `src/textgrad_trainer.py` — **删除**
- `src/sequential_trainer.py` — **新建**
- `src/__init__.py` — 替换 TextGradTrainer → SequentialSkillTrainer
- `src/ttso.py` — factory 分支 + TTSOConfig 更新
- `run.py` — CLI args 更新
- `scripts/example_physics.py` — CLI args 更新

### Next Steps
1. [ ] Code review 修复 (等待 review 结果)
2. [ ] 端到端验证 (`--optimization_mode sequential_dto`)
3. [ ] Phase 3: Cross-Domain Skill Transfer

---

## Session: 2026-03-13 (Session 10 - Per-Token Rejection Sampling)

### Completed
- [x] **Per-token rejection sampling** → `src/sequential_trainer.py`
  - `_evaluate_token_choice()`: 对候选 token 做 no-grad forward pass, 计算 NLL + RM loss
  - `commit_token_ids()`: 新增方法支持直接提交指定 token IDs (rejection sampling 使用)
  - `optimize()` 主循环: 优化后对每个 changed token 比较 optimized vs original loss
    - 同 token → 直接提交 (无额外 forward pass)
    - 不同 token → 2× forward pass 比较, 更好才接受
  - 返回 dict 新增 `rejection_accepted` 和 `rejection_rejected` 统计
  - 日志输出 rejection sampling 统计 (accepted/rejected 比例)
- [x] **更新 markdown 文件**
  - `skill_optimization.md` §6.7: 文档化 rejection sampling 机制和伪代码
  - `skill_optimization.md` §7.1: 对比表新增 Rejection 行
  - `skill_optimization.md` §8: 代码位置表新增 rejection sampling 条目
  - `task_plan.md` §2.9.6: 标记 rejection sampling 完成
  - `findings.md` §8.5: 适配设计新增 rejection sampling 说明和流程图

### Key Design Decision
> **Per-token rejection sampling 保证质量单调不降**:
> - 参考 Nabla-Reasoner 的 `acceptance_criteria()`
> - 每个 token 变化都经过验证: optimized loss ≤ original loss 才接受
> - 防止 STE 梯度噪声导致的退化 (argmax 翻转但方向错误)
> - 额外成本: 每个 changed token +2 forward passes (无梯度, 较轻量)

### Next Steps
1. [ ] 端到端验证 (`--optimization_mode sequential_dto`)
2. [ ] Phase 3: Cross-Domain Skill Transfer

---

## Session: 2026-03-13 (Session 11 - Trajectory-Based Rejection Sampling)

### Completed
- [x] **Trajectory-based rejection sampling 重写** → `src/sequential_trainer.py`
  - 移除 `_evaluate_token_choice()` (embedding-space full loss 比较)
  - 移除 `_evaluate_token_reward()` (embedding-space RM 比较)
  - 新增 `_evaluate_trajectory_reward()`: 解码完整 skill → generate response → RM score
  - `optimize()` 重写 rejection 逻辑:
    - 同 token → 直接 commit (零额外成本)
    - 不同 token → trajectory-based rejection: reward_new > reward_old 才 accept
    - 动态 reward_old: accept 后更新为 reward_new，门槛越来越严格
  - 新增参数: `reward_old`, `seed` 传入 `optimize()`
  - 返回 dict 新增 `reward_old` (final baseline value)
- [x] **移除 `sequential_rejection_criterion` 配置项**
  - `ttso.py` TTSOConfig: 移除 `sequential_rejection_criterion` 字段
  - Factory: 传递 `response_generator=self._generator` 替代 `rejection_criterion`
  - `run.py`: 移除 `--sequential_rejection_criterion` CLI arg
  - `scripts/example_physics.py`: 同步移除
- [x] **更新所有 markdown 文件**
  - `skill_optimization.md` §6.7: 改为 trajectory-based rejection 文档
  - `skill_optimization.md` §7.1: 对比表更新 Rejection 行
  - `skill_optimization.md` §8: 代码位置表更新为 `_evaluate_trajectory_reward()`
  - `findings.md` §8.5: 更新 rejection sampling 流程图和说明
  - `task_plan.md` §2.9.6: 更新为 trajectory-based rejection 描述
  - `nablaskill.md`: 已在上一 session 创建，算法描述与代码一致

### Key Design Decision
> **Trajectory-based rejection 是唯一模式** (移除 loss/reward 二选一):
> - embedding-space RM 比较有 fluency 偏差 (ahead tokens 在旧 context 下优化)
> - trajectory-based (decode → generate → RM) 更公平、更原则性
> - 动态 reward_old 保证 skill 质量单调不降 (参考 Nabla-Reasoner)
> - 额外成本: 每个 changed token +1 generate +1 RM eval, 但大部分 token 不变 (STE 梯度太小)

### Next Steps
1. [ ] 端到端验证 (`--optimization_mode sequential_dto`)
2. [ ] Phase 3: Cross-Domain Skill Transfer
