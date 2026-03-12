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
1. [ ] 运行 `python scripts/example_physics.py` 验证 end-to-end pipeline
2. [ ] Phase 3: Cross-Domain Skill Transfer
3. [ ] Phase 4: Evaluation & Experiments
