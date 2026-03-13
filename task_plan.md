# Task Plan: Test-Time Skill Optimization (TTSO)

> **Project**: nablaskill
> **Created**: 2026-03-12
> **Status**: Phase 2 Fully Complete (2.8 hardening + 2.9 sequential design), Phase 3 Next
> **Goal**: 将 Nabla-Reasoner 的 test-time gradient optimization 思想应用于 Skill-based Agent 系统，实现推理时动态优化 Skill

---

## Executive Summary

将 AutoSkill 的 Skill Library 检索机制与 Nabla-Reasoner 的 Test-Time Optimization (TTO) 结合，提出 **Test-Time Skill Optimization (TTSO)**：在推理阶段通过梯度下降动态调整 Skill 表示，使静态 Skill prompt 转变为可优化的 reasoning operator。

---

## Phase 0: Research Foundation & Feasibility Analysis
**Status**: [x] Complete
**Priority**: Critical
**Estimated Scope**: Research only

### Tasks
- [x] 0.1 深入分析 Nabla-Reasoner 的 DTO 机制，确定可迁移的核心组件
  - Nabla 核心: `optimize.py` 中 `LatentTrainer` + `DiffLogitsToEmbedding` / `DiffLatentsToEmbedding`
  - 关键洞察: Skill text -> soft token embeddings -> gradient optimization -> updated Skill
  - **关键发现**: Nabla 全程 batch=1, STE 默认 hard=True, 梯度缓存在 argmax tokens 不变时复用
- [x] 0.2 分析 AutoSkill 的 Skill 表示格式 (SKILL.md artifact) 和检索流程
  - AutoSkill 核心: `autoskill/management/` (Skill CRUD) + `autoskill/embeddings/` (检索)
  - Skill 格式: SKILL.md artifact，含 name/description/instructions/triggers/tags/examples
  - 检索: Hybrid (90% embedding + 10% BM25) -> LLM selector -> 注入 system prompt
  - 注入方式: "You have access to Skills below. Use ONLY if relevant." + rendered skills
- [x] 0.3 确定 Skill 的可优化表示形式
  - **Option A (CHOSEN)**: Skill text -> tokenize -> soft logits (类似 Nabla logits mode)
  - 初始化: one-hot * 10.0 scale, 优化 [1, num_skill_tokens, vocab_size] logit tensor
  - STE: hard forward (argmax one-hot), soft backward (softmax gradient)
- [x] 0.4 定义评估指标和 baseline
  - Baseline 1: Static Skill (AutoSkill 原始流程)
  - Baseline 2: Best-of-N with Skill (多次采样选最优)
  - Baseline 3: Nabla-Reasoner without Skill
  - 指标: avg@k, pass@k, RM score delta, LLM calls count

### Key Decision (RESOLVED)
> **Decided**: Skill logits mode (Option A)
> - Skill text -> tokenize -> soft logits [1, N, V]
> - 通过 STE 桥接离散/连续
> - 优化后 argmax decode 得到 optimized skill text

---

## Phase 1: Core TTSO Engine
**Status**: [x] v1 Implementation Complete
**Priority**: Critical
**Dependencies**: Phase 0

### Tasks
- [x] 1.1 设计 SkillEmbedder 模块 -> `src/skill_embedder.py`
  - `DiffSkillLogitsToEmbedding`: 维护 `skill_logits` nn.Parameter [1, N, V]
  - forward() -> soft_onehot (STE), lm_embeds, rm_embeds
  - decode_text() -> argmax decode 回可读 skill text
- [x] 1.2 实现 Skill Optimization Loop -> `src/skill_trainer.py`
  - `SkillTrainer.optimize(query, response_text, skill_text)`
  - 三项 loss: response_nll + skill_fluency + reward_coeff * reward
  - Adam + cosine LR schedule, gradient caching support
- [x] 1.3 实现 Skill Selection Criteria -> `src/ttso.py:should_optimize()`
  - 基于 min_reward_threshold: 初始 RM score 已高则跳过
- [x] 1.4 实现 Skill Rejection Sampling -> `src/ttso.py:run()`
  - 比较 optimized vs original RM score, 支持 reward_improvement_threshold
- [x] 1.5 实现 Prompt Template 系统 -> `src/skill_template.py`
  - `SkillGenerationTemplate`: [prefix][soft_skill][suffix(query+response)]
  - `SkillRewardTemplate`: RM version with placeholder trick
  - 正确追踪 response_start position 用于 loss 计算
- [x] 1.6 CLI 入口 -> `run.py`
  - 完整参数体系, 支持 vLLM 和 HuggingFace 两种 backend

### Architecture Sketch
```
SkillOptimizer (core engine)
  |-- SkillEmbedder (DiffSkillToEmbedding)
  |     |-- initialize(skill_text) -> soft_tokens
  |     |-- forward() -> soft_embeddings for LM and RM
  |     +-- decode() -> optimized skill text
  |
  |-- SkillTrainer (optimization loop)
  |     |-- optimize(query, skill, response) -> optimized_skill
  |     +-- compute_loss(soft_embeds, lm_template, rm_template)
  |
  +-- SkillSelector (when to optimize)
        +-- should_optimize(query, skill, initial_response) -> bool
```

---

## Phase 2: Integration with Skill Retrieval Pipeline
**Status**: [x] Complete
**Priority**: High
**Dependencies**: Phase 1

### Tasks
- [x] 2.1 设计 TTSO Pipeline -> `src/pipeline.py`
  ```
  Query -> Skill Retrieval -> Skill Selection -> TTSO Optimization -> Writeback
  ```
  - `TTSOPipeline` 编排器支持两种模式: direct skill (跳过检索) 和 SkillBank retrieval
  - `PipelineConfig` 控制所有 pipeline 行为
  - `PipelineResult` 包含完整中间结果和最终输出
- [x] 2.2 集成 AutoSkill 的 SkillBank 检索 -> `src/skillbank.py`
  - `SkillBankAdapter`: 薄包装层，隔离 AutoSkill SDK 依赖
  - `SkillCandidate` frozen dataclass: skill_id, name, instructions, score
  - `retrieve()`: 调用 AutoSkill.search() 返回 top-k SkillCandidate
  - 优雅降级: AutoSkill 未安装时抛出清晰的 ImportError
- [x] 2.3 实现 multi-skill optimization -> `src/pipeline.py:_select_skill()`
  - 策略 1: `highest_retrieval_score` (默认, 零额外计算)
  - 策略 2: `best_initial_reward` (为每个 candidate 生成 response + RM 评分, 选最优)
  - 仅优化选中的单个 skill (不同时优化多个, 控制计算成本)
- [x] 2.4 实现 optimized skill 的回写机制 -> `src/skillbank.py:writeback()`
  - 决定: **创建新条目** (不覆盖原始 skill), 携带 source_skill_id 元数据
  - `writeback_enabled` + `writeback_min_improvement` 双重门控
  - 元数据包含: source_skill_id, optimization_query, reward_delta, optimized_at
- [x] 2.5 Example script -> `scripts/example_physics.py`
  - 硬编码 Newtonian Mechanics 物理 skill (7步 workflow)
  - 支持 tiny-gpt2 (CPU 测试) 和真实模型 (GPU)
  - 完整输出: original/optimized skill text, responses, RM scores, stats
- [x] 2.6 **Iterative TTSO 变体 (主实验)** -> `src/ttso.py:run_iterative()`
  - Skill 和 Response 交替优化: `skill_A + resp_A → skill_B + resp_B → skill_C + resp_C ...`
  - 每轮: DTO 优化 skill → 重新生成 response → RM 评分 → 接受/拒绝
  - Early stopping: reward 不再提升时停止
  - 追踪完整 `round_history` (每轮 skill/response/reward)
  - `max_outer_rounds` 配置控制, 默认 3 (example), 1 = 退化为原始单轮
  - Pipeline 自动路由: `max_outer_rounds > 1` → `run_iterative()`, 否则 → `run()`

### Key Decision (Phase 2.6)
> **Iterative TTSO 作为主实验**:
> - 单轮 `run()` 保留作为 ablation baseline
> - 多轮 `run_iterative()` 作为主实验方法
> - 原因: 避免 skill 只拟合一个可能不好的初始 response (moving-target 问题)
> - 权衡: 每轮多一次 generation + RM eval, 但 skill 和 response 协同进化

### Architecture (Phase 2)
```
TTSOPipeline (orchestrator)
  |-- SkillBankAdapter (AutoSkill SDK wrapper)
  |     |-- retrieve(query) -> List[SkillCandidate]
  |     |-- writeback(candidate, optimized_text) -> skill_id
  |     +-- render_context(query) -> formatted text
  |
  |-- _select_skill() (multi-skill selection)
  |     |-- highest_retrieval_score (top-1)
  |     +-- best_initial_reward (generate + RM score each)
  |
  +-- TTSODecoding (Phase 1 engine)
        |-- run(query, skill_text) -> TTSOResult           [单轮 baseline]
        +-- run_iterative(query, skill_text) -> TTSOResult [多轮主实验]
```

### New Files
- `src/skillbank.py` (175 lines) — SkillBankAdapter, SkillCandidate, SkillBankConfig
- `src/pipeline.py` (260 lines) — TTSOPipeline, PipelineConfig, PipelineResult
- `scripts/example_physics.py` (195 lines) — End-to-end example
- `tests/test_pipeline.py` (200 lines) — Pipeline unit tests

---

## Phase 2.7: DTO 方案验证失败 & 方案重新设计
**Status**: [x] Implementation Complete
**Priority**: Critical
**Dependencies**: Phase 2

### 问题: Token-level DTO 用于 Skill 优化根本不可行

在 Qwen2.5-7B + Skywork-Reward-V2-Qwen3-4B 上实验发现:
- **超参不存在 sweet spot**: init_scale 高 → tokens 不变; init_scale 低 → tokens 变乱码
- **STE 梯度瓶颈**: vocab=152K 导致 grad_max ≈ 9e-6，lr=0.01 下每步只移动 9e-8
- **根因**: 优化目标倒置 (拟合已有 response 而非产出更好的) + token 空间无平滑路径

详细分析见 findings.md §6

### 实现: 三种优化模式共存, 通过 `--optimization_mode` 切换

- [x] **方案 A: Soft Prompt Optimization** → `src/soft_prompt_trainer.py`
  - `SoftPromptEmbedding`: 直接优化 `nn.Parameter([1, N, hidden_dim])` continuous embeddings
  - 初始化: 从 skill token IDs 查 LM embedding table 获取初始向量
  - 梯度直通: 无 STE 瓶颈, 梯度直接流到 embedding 参数
  - Loss: response_nll + embed_drift (L2 正则, 防止离原始 embedding 太远) + RM reward
  - 投影回 tokens: cosine similarity nearest neighbor
  - RM 处理: 投影到最近 LM token → 查 RM embedding table
- [x] **原始 DTO** 保留在 `src/skill_trainer.py`, 不做修改
- ~~方案 B: TextGrad~~ — 已删除 (2026-03-13)

### 切换方式

```bash
# 原始 DTO (默认)
python scripts/example_physics.py --optimization_mode dto

# Soft Prompt (Approach A)
python scripts/example_physics.py --optimization_mode soft_prompt --lr 0.1

# Sequential DTO (逐 token 优化)
python scripts/example_physics.py --optimization_mode sequential_dto --max_iters 20
```

### 架构: Factory Pattern 路由

```
TTSOConfig.optimization_mode → TTSODecoding._create_optimizer()
  |-- "dto"             → SkillTrainer (全局 DTO)
  |-- "soft_prompt"     → SoftPromptTrainer (连续 embedding)
  +-- "sequential_dto"  → SequentialSkillTrainer (逐 token DTO)

三者共享接口:
  optimize(query, response_text, skill_text, system_prompt, **kwargs) → Dict
  get_reward_for_text(query, skill_text, response) → float

SequentialSkillTrainer 内部组件:
  SequentialSkillStates (src/sequential_states.py) — past/ahead token 管理
  _evaluate_trajectory_reward() — trajectory-based rejection sampling
```

### Files
- `src/soft_prompt_trainer.py` (~300 lines) — SoftPromptEmbedding + SoftPromptTrainer
- ~~`src/textgrad_trainer.py`~~ — 已删除

---

## Phase 2.8: Code Hardening & Hyperparameter Tooling
**Status**: [x] Complete
**Priority**: Medium
**Dependencies**: Phase 2.7

### Tasks
- [x] 2.8.1 消除硬编码超参数
  - `soft_prompt_trainer.py`: RM 投影 softmax temperature `/ 0.1` → `/ self.rm_projection_temperature` (可配置)
  - `skill_trainer.py`: `init_logits.scatter_(2, ..., 10)` → `self.init_logit_scale` (可配置)
  - `ttso.py`: TTSOConfig 新增 `rm_projection_temperature` 和 `init_logit_scale` 字段
  - Factory `_create_optimizer()` 传递新参数到对应 trainer
- [x] 2.8.2 加固 `align_vocab()` 边界检查
  - `pad_token_id is None` fallback (默认 0 + warning)
  - `src_idx < src_vocab_size` 边界检查防止 out-of-bounds
  - 新增 unmapped tokens 计数 warning 和 alignment result info 日志
- [x] 2.8.3 暴露 loss 配比为 CLI 参数
  - `example_physics.py` 新增: `--response_nll_coeff`, `--skill_fluency_coeff`, `--reward_coeff`
  - `run.py` 新增: `--rm_projection_temperature`, `--init_logit_scale`
- [x] 2.8.4 创建 DTO 超参数扫描脚本 → `scripts/sweep_hyperparams.sh`
  - 1000 个超参数组合, 分 5 tier 覆盖:
    - Tier 1: Core grid (lr × init_scale × max_iters) = 256 configs
    - Tier 2a: sweep response_nll_coeff = 160 configs
    - Tier 2b: sweep skill_fluency_coeff = 180 configs
    - Tier 2c: sweep reward_coeff = 160 configs
    - Tier 3: combined extreme configs = 244 configs
  - 每个 config 结果保存到 `outputs/sweep_YYYYMMDD_HHMMSS/{config_name}.txt`
  - 自动打印 summary table (RM_orig, RM_final, Delta)
- [x] 2.8.5 技术文档 → `skill_optimization.md`
  - 合并 DTO + Soft Prompt 方法的完整技术文档
  - 覆盖 pipeline 流程、loss 细节、STE/nearest neighbor 投影、方案对比

---

## Phase 2.9: Sequential Token-by-Token Skill Optimization (设计)
**Status**: [x] Research & Design Complete, Implementation Pending
**Priority**: High
**Dependencies**: Phase 2.7

### 动机
参考 Nabla-Reasoner 的逐 token 优化思想，应用于 skill tokens:
- 原始 DTO 同时优化所有 skill tokens → STE 梯度瓶颈
- **新方案**: 逐个 skill token 优化 (token 0 → commit → token 1 → commit → ...)
- 仅优化 skill tokens，response tokens 固定

### Nabla-Reasoner 架构研究 (已完成)
- `GenerationStates`: 管理 `past_token_ids` (已提交) + `ahead_token_ids` (待优化 lookahead buffer)
- `move_to_next_optimizable_token()`: entropy/confidence/gradient selector 选择需优化的 position
- `optimize_ahead_latents()`: 联合优化所有 ahead tokens (非逐个), 通过 `LatentTrainer.optimize()`
- `commit_n_tokens()`: 将优化后的 tokens 从 ahead 移到 past
- `sample_token()` + `acceptance_criteria()`: rejection sampling 确保 token 质量

### 适配设计要点
1. **位置差异**: Nabla 的 soft tokens 在末尾; TTSO 的 skill tokens 在中间 (prefix+skill+response)
2. **优化范围**: 每次优化当前 token + 后续 lookahead buffer 的 skill tokens
3. **提交策略**: 优化完成后 commit 当前 token, 移动到下一个 skill token
4. **Response 处理**: response 不是优化变量 (不对 response tokens 做梯度更新), 但每次 rollout 都经过 response 计算 loss — 与 Nabla-Reasoner 一致, 每步都 rollout 到最后
5. **新 optimization_mode**: `"sequential_dto"` 加入 factory pattern

### Tasks
- [x] 2.9.1 研究 Nabla-Reasoner sequential decoding 架构
- [x] 2.9.2 确定适配方案 (skill-specific 差异分析)
- [x] 2.9.3 实现 `SequentialSkillTrainer` → `src/sequential_trainer.py`
- [x] 2.9.4 实现 `SequentialSkillStates` (past/ahead skill token 管理)
- [x] 2.9.5 集成到 factory pattern (`--optimization_mode sequential_dto`)
- [x] 2.9.6 Trajectory-based rejection sampling: 优化后 token 变化时，解码完整 skill → 生成 response → RM 评分 → 与 reward_old 比较
  - `_evaluate_trajectory_reward()`: 解码 skill → generate → RM score
  - `commit_token_ids()`: 支持直接提交指定 token IDs
  - 动态 reward_old: accept 后提升门槛，保证 skill 质量单调不降
  - MPC 重初始化: 每步从 original tokens 重初始化 ahead logits
  - 移除旧的 `_evaluate_token_choice()` 和 `_evaluate_token_reward()` (embedding-space 比较)
  - 移除 `--sequential_rejection_criterion` 配置项 (trajectory-based 为唯一模式)
- [x] ~~2.9.7 Token Selector~~ — **已删除**: entropy/confidence selector 在 one-hot 初始化下退化 (所有位置 softmax 分布相同), 仅 gradient selector 有效但性价比不够
  - `SequentialSkillStates` 提取为独立文件 `src/sequential_states.py` (保留)
- [ ] 2.9.8 单元测试 + 端到端验证（本地环境若是不行就算了）

---

## Phase 3: Cross-Domain Skill Transfer
**Status**: [ ] Not Started (blocked by Phase 2.7)
**Priority**: Medium
**Dependencies**: Phase 2.7

### Tasks
- [ ] 3.1 设计 cross-domain transfer 实验
  - Physics skill -> Math problem
  - Coding skill -> Logic problem
  - 测量优化前后的 performance gap
- [ ] 3.2 实现 Skill Morphing 机制
  - 通过梯度优化将 domain-specific skill 适配到新 domain
  - 保留 skill 的 structural reasoning pattern，调整 domain-specific content
- [ ] 3.3 分析 transfer 的粒度
  - 哪些 skill 组件容易迁移（high-level strategy vs low-level steps）
  - 优化步数与迁移效果的关系

---

## Phase 4: Evaluation & Experiments
**Status**: [ ] Not Started
**Priority**: High
**Dependencies**: Phase 2

### Tasks
- [ ] 4.1 构建评估数据集
  - 复用 Nabla-Reasoner 的 MATH-500, AIME-2024, AIME-2025, AMC
  - 增加 cross-domain 评测 (e.g., physics -> math transfer)
  - 增加 skill-specific benchmarks
- [ ] 4.2 实现评估脚本
  - avg@k, pass@k (复用 `eval/eval_outputs.py`)
  - Skill 优化前后 RM score 对比
  - 计算成本 (LLM calls, optimization steps)
- [ ] 4.3 Ablation studies
  - w/ vs w/o skill optimization
  - logits vs latents mode for skill optimization
  - 不同 selection criteria 的效果
  - reward_coeff 敏感性分析
- [ ] 4.4 Baseline 对比实验
  - Static Skill (AutoSkill)
  - Best-of-N + Skill
  - Nabla-Reasoner (no skill)
  - TTSO (ours)

---

## Phase 5: Paper Writing
**Status**: [ ] Not Started
**Priority**: Medium
**Dependencies**: Phase 4

### Tasks
- [ ] 5.1 确定论文定位和 venue
- [ ] 5.2 撰写 method section
- [ ] 5.3 撰写 experiment section
- [ ] 5.4 完成 introduction 和 related work

---

## Research Questions Mapping

| RQ | Phase | Key Experiment |
|----|-------|---------------|
| RQ1: TTO 能否提升 skill 在新任务上的效果？ | Phase 2, 4 | TTSO vs Static Skill on MATH/AIME |
| RQ2: Skill 能否通过优化实现跨领域迁移？ | Phase 3, 4 | Physics skill -> Math problem |
| RQ3: 优化 skill 表示能否提升 reasoning performance？ | Phase 4 | Full ablation study |

---

## Technical Risks (Updated)

| Risk | Impact | Status | Mitigation |
|------|--------|--------|------------|
| ~~Skill text 太短，优化空间有限~~ | ~~High~~ | **REALIZED** | Token-level DTO 完全不可行, 需切换方案 |
| ~~优化后 Skill decode 出的文本不可读~~ | ~~Medium~~ | **REALIZED** | Fluency 约束无法解决 STE 梯度瓶颈问题 |
| 计算成本过高 | High | Open | 梯度缓存 + selection criteria 跳过低价值优化 |
| RM 对 Skill quality 的评估不准确 | High | Open | 使用 process reward model 或 task-specific verifier |
| Cross-domain transfer 效果不显著 | Medium | Blocked | 先解决 Phase 2.7 的方案选择 |
| **STE 梯度瓶颈 (NEW)** | **Critical** | **REALIZED** | **切换到 continuous embedding 或 LLM-based 优化** |
