# program.md

## 项目名称

A Small-Scale Controlled Study of Heterogeneous Preference Noise in Direct Preference Optimization

中文说明：

本项目研究不同结构的偏好噪声如何影响 Direct Preference Optimization，即 DPO。

这是一个深度学习课程作业项目，目标是完成一篇 6–8 页左右的 Word 双栏论文。

---

## 项目总目标

本项目需要完成：

1. 使用 UltraFeedback-derived preference data。
2. 使用小模型进行 DPO 小规模训练。
3. 构造三类偏好噪声：
   - label-flip noise
   - ambiguous preference noise
   - weak-quality preference noise
4. 在固定训练设置下比较不同噪声对 DPO 的影响。
5. 生成实验表格、图表和分析报告。
6. 写出课程论文草稿。
7. 最终输出可用于 Word 双栏排版的论文内容、图表和表格。

---

## 研究问题

主研究问题：

```text
How do different structures of preference noise affect DPO training and alignment behavior?
```

中文解释：

不同类型的偏好噪声是否会以不同方式影响 DPO 训练和对齐效果？

---

## 核心假设

### H1: Label-flip noise 可能最有破坏性

原因：

```text
Label-flip noise 会直接反转 chosen/rejected 的监督方向，使模型学习错误偏好。
```

### H2: Ambiguous preference noise 主要削弱学习信号

原因：

```text
Ambiguous preference noise 中 chosen 和 rejected 差距较小，监督方向可能仍然正确，但训练信号较弱。
```

### H3: Weak-quality preference noise 可能降低输出质量，但不一定完全破坏偏好方向

原因：

```text
Weak-quality preference noise 中 chosen 仍然比 rejected 好，但 chosen 本身质量不高，因此可能影响最终 response quality。
```

---

## 固定约束

### 数据集固定

使用：

```text
UltraFeedback-derived preference data
```

优先数据集：

```text
trl-lib/ultrafeedback_binarized
```

允许使用小子集。

禁止换数据集。

---

### 模型固定

主模型：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

备用模型：

```text
EleutherAI/pythia-410m
```

禁止使用 7B 或更大模型。

---

### 算力策略固定

用户没有本地 GPU。

因此项目分成本地 CPU 阶段和 Colab GPU 阶段。

本地 CPU 负责：

```text
repo setup
data loading
subset creation
noise injection
unit tests
preview reports
analysis scripts
paper skeleton
```

Colab 免费 GPU 负责：

```text
small-scale DPO training
Version A experiments
Version B experiments if resources allow
```

---

## 实验版本设计

### Version A: 保底实验版本

Version A 是最小可交版本。

包含 4 组实验：

```text
clean
label_flip_20
ambiguous_20
weak_quality_20
```

目的：

```text
确保至少有一组 clean baseline 和三种噪声类型的对比结果。
```

如果只完成 Version A，也必须能写出 6–8 页论文。

---

### Version B: 完整实验版本

Version B 是正式目标版本。

包含 10 组实验：

```text
clean

label_flip_10
label_flip_20
label_flip_30

ambiguous_10
ambiguous_20
ambiguous_30

weak_quality_10
weak_quality_20
weak_quality_30
```

目的：

```text
增加 noise rate sensitivity analysis。
```

也就是分析噪声比例从 10% 到 30% 增加时，不同噪声类型对 DPO 的影响趋势。

---

### Version A 和 Version B 的关系

Version A 必须是 Version B 的子集。

所有实验必须 config-driven。

禁止在 Python 脚本中 hard-code noise type 或 noise rate。

---

## 推荐项目结构

```text
dpo-noise-study/
├── agent.md
├── program.md
├── README.md
├── requirements.txt
├── Makefile
├── external-skills/
│   ├── get-available-resources/
│   ├── hypothesis-generation/
│   ├── exploratory-data-analysis/
│   ├── transformers/
│   ├── statistical-analysis/
│   ├── scientific-visualization/
│   ├── scientific-writing/
│   ├── peer-review/
│   ├── docx/
│   └── citation-management/
├── configs/
│   ├── base.yaml
│   ├── clean.yaml
│   ├── label_flip_10.yaml
│   ├── label_flip_20.yaml
│   ├── label_flip_30.yaml
│   ├── ambiguous_10.yaml
│   ├── ambiguous_20.yaml
│   ├── ambiguous_30.yaml
│   ├── weak_quality_10.yaml
│   ├── weak_quality_20.yaml
│   └── weak_quality_30.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── noisy/
├── src/
│   ├── load_data.py
│   ├── inject_noise.py
│   ├── train_dpo.py
│   ├── evaluate.py
│   ├── analyze_results.py
│   └── utils.py
├── tests/
│   ├── test_load_data.py
│   └── test_inject_noise.py
├── scripts/
│   ├── run_cpu_smoke.sh
│   ├── run_version_a.sh
│   ├── run_version_b.sh
│   ├── run_colab_version_a.sh
│   ├── run_colab_version_b.sh
│   └── make_figures.sh
├── notebooks/
│   └── run_dpo_colab.ipynb
├── results/
│   ├── experiments.tsv
│   ├── metrics.csv
│   └── runs/
├── reports/
│   ├── environment.md
│   ├── research_plan.md
│   ├── data_report.md
│   ├── noise_definition.md
│   ├── noise_preview.md
│   ├── cpu_smoke_report.md
│   ├── version_a_report.md
│   ├── version_b_report.md
│   ├── analysis.md
│   ├── reviewer_comments.md
│   ├── required_fixes.md
│   └── final_checklist.md
└── paper/
    ├── paper_outline.md
    ├── draft.md
    ├── draft.docx
    ├── figures/
    └── tables/
```

---

# Phase 0: 环境检查与项目搭建

## 使用 skills

开始前阅读：

```text
external-skills/get-available-resources/SKILL.md
```

## 目标

搭建项目基础结构，并确认本地和 Colab 的运行边界。

## 任务

1. 创建项目目录结构。
2. 创建 `requirements.txt`。
3. 创建 `README.md`。
4. 创建 `Makefile`。
5. 创建 `results/experiments.tsv`。
6. 创建基础 config 文件。
7. 检查当前机器是否有 GPU。
8. 明确本地只能做 CPU 任务。

## 不做什么

禁止：

1. 不要开始 DPO 训练。
2. 不要下载大模型。
3. 不要运行 Version A 或 Version B。
4. 不要写论文正文。
5. 不要引入复杂框架。

## 产出

```text
README.md
requirements.txt
Makefile
results/experiments.tsv
configs/base.yaml
configs/clean.yaml
reports/environment.md
```

## 成功标准

1. 项目结构完整。
2. `make test` 命令存在。
3. `make smoke` 命令存在。
4. `results/experiments.tsv` 有表头。
5. 本地环境检查结果写入 `reports/environment.md`。

## 停止条件

如果 Python 环境无法创建，停止并汇报。

---

# Phase 1: 研究问题和实验计划

## 使用 skills

开始前阅读：

```text
external-skills/hypothesis-generation/SKILL.md
```

## 目标

形成课程论文级别的研究计划。

## 任务

创建：

```text
reports/research_plan.md
```

内容必须包括：

1. 研究问题
2. 三个假设
3. 自变量
4. 因变量
5. 控制变量
6. 噪声类型定义
7. Version A 和 Version B 的实验矩阵
8. 预期风险
9. 论文叙事主线

## 不做什么

禁止：

1. 不要改变 abstract 中的数据集。
2. 不要提出新算法。
3. 不要扩大成多个研究问题。
4. 不要写成正式论文正文。
5. 不要承诺大规模实验。

## 产出

```text
reports/research_plan.md
```

## 成功标准

`research_plan.md` 必须能回答：

```text
What is the research question?
What are the hypotheses?
What variables are controlled?
What exactly changes across experiments?
What evidence will be collected?
```

## 停止条件

如果三类噪声定义无法区分，停止并汇报。

---

# Phase 2: UltraFeedback 数据加载与子集创建

## 使用 skills

开始前阅读：

```text
external-skills/exploratory-data-analysis/SKILL.md
```

## 目标

加载 UltraFeedback-derived preference data，并创建小规模 train/eval 子集。

## 任务

实现：

```text
src/load_data.py
tests/test_load_data.py
```

要求：

1. 加载 `trl-lib/ultrafeedback_binarized`。
2. 提取 prompt/chosen/rejected 字段。
3. 创建 deterministic subset。
4. 支持 train_size 和 eval_size 配置。
5. 保存 processed data。
6. 生成数据检查报告。

推荐默认值：

```text
train_size: 1000
eval_size: 200
preview_size: 100
seed: 42
```

如果 Colab 免费 GPU 不稳定，可以降级：

```text
train_size: 500
eval_size: 100
```

## 不做什么

禁止：

1. 不要换数据集。
2. 不要删除大量样本但不记录原因。
3. 不要修改 raw data。
4. 不要开始训练。
5. 不要将 train/eval 混用。

## 产出

```text
src/load_data.py
tests/test_load_data.py
data/processed/train_clean.jsonl
data/processed/eval_clean.jsonl
reports/data_report.md
```

## 成功标准

1. 数据可以成功加载。
2. 每条样本包含 prompt/chosen/rejected。
3. train/eval split 可复现。
4. 相同 seed 生成相同子集。
5. `pytest tests/test_load_data.py` 通过。

## 停止条件

如果 UltraFeedback 数据无法下载，停止并汇报替代方案，但不要自动换数据集。

---

# Phase 3: 三类偏好噪声注入

## 使用 skills

开始前阅读：

```text
external-skills/exploratory-data-analysis/SKILL.md
external-skills/statistical-analysis/SKILL.md
```

## 目标

实现三类 noise injection，并证明它们确实不同。

## 任务

实现：

```text
src/inject_noise.py
tests/test_inject_noise.py
scripts/preview_noise.py
```

必须支持：

```text
noise_type:
- clean
- label_flip
- ambiguous
- weak_quality

noise_rate:
- 0.0
- 0.1
- 0.2
- 0.3

seed:
- deterministic
```

## label_flip 实现要求

行为：

```text
随机选择 noise_rate 比例的样本，交换 chosen 和 rejected。
```

测试必须验证：

1. 被选中样本 chosen/rejected 真的交换。
2. 未选中样本保持不变。
3. noise_rate 大致正确。
4. seed 可复现。

## ambiguous 实现要求

行为：

```text
构造或选择 chosen/rejected 差距较小的样本。
```

优先方案：

```text
如果数据中有 score 或 rating 字段，使用 score gap。
```

如果没有 score 字段，允许使用 proxy：

```text
response length difference
text similarity
reward proxy
available quality metadata
```

但必须把定义写入：

```text
reports/noise_definition.md
```

测试必须验证：

1. ambiguous 样本不是 label_flip。
2. chosen/rejected 没有被简单交换。
3. ambiguous 的构造逻辑和文档一致。

## weak_quality 实现要求

行为：

```text
chosen 仍然保留为 chosen，但 chosen 的绝对质量较弱。
```

优先方案：

```text
使用 chosen score 较低但仍高于 rejected 的样本。
```

如果没有 score 字段，允许使用 proxy，但必须写入文档。

测试必须验证：

1. chosen/rejected 没有被交换。
2. pair direction 保持。
3. weak_quality 和 ambiguous 的定义不同。

## 不做什么

禁止：

1. 不要把 ambiguous 写成 label_flip。
2. 不要把 weak_quality 写成 label_flip。
3. 不要 hard-code 20%。
4. 不要删除样本但不记录。
5. 不要开始训练。
6. 不要为了结果好看改变噪声定义。

## 产出

```text
src/inject_noise.py
tests/test_inject_noise.py
scripts/preview_noise.py
reports/noise_definition.md
reports/noise_preview.md
data/noisy/*.jsonl
```

## 成功标准

1. `pytest tests/test_inject_noise.py` 通过。
2. `reports/noise_preview.md` 中每种噪声至少有 5 个 before/after example。
3. 相同 seed 生成相同 noisy dataset。
4. 三类噪声定义清楚且可区分。

## 停止条件

如果 ambiguous 和 weak_quality 无法根据数据字段稳定构造，停止并汇报可选 proxy，不要自行乱改。

---

# Phase 4: 本地 CPU Smoke Test

## 使用 skills

开始前阅读：

```text
external-skills/exploratory-data-analysis/SKILL.md
```

## 目标

在不使用 GPU 的情况下验证完整非训练流程。

## 任务

实现：

```text
scripts/run_cpu_smoke.sh
```

该脚本必须完成：

1. 加载小子集。
2. 生成 clean 数据。
3. 生成 label_flip_20 数据。
4. 生成 ambiguous_20 数据。
5. 生成 weak_quality_20 数据。
6. 运行所有 pytest。
7. 生成 preview report。
8. 生成 dummy 或 placeholder metrics schema。

## 不做什么

禁止：

1. 不要在本地 CPU 上跑正式 DPO。
2. 不要下载大模型。
3. 不要生成假实验结果。
4. 不要把 smoke test 结果写成最终论文结果。

## 产出

```text
reports/cpu_smoke_report.md
results/smoke_metrics_schema.csv
```

## 成功标准

1. CPU smoke test 一条命令可运行。
2. 数据处理流程完整。
3. 噪声注入流程完整。
4. tests 全部通过。
5. Colab 训练前的所有数据准备完成。

## 停止条件

如果 smoke test 不通过，不允许进入 Phase 5。

---

# Phase 5: Colab DPO Training - Version A

## 使用 skills

开始前阅读：

```text
external-skills/get-available-resources/SKILL.md
external-skills/transformers/SKILL.md
```

## 目标

在 Google Colab 免费 GPU 上跑通 Version A。

## 任务

实现：

```text
src/train_dpo.py
src/evaluate.py
scripts/run_colab_version_a.sh
notebooks/run_dpo_colab.ipynb
```

Version A 包含：

```text
clean
label_flip_20
ambiguous_20
weak_quality_20
```

每组实验必须独立保存：

```text
results/runs/{experiment_id}/config.yaml
results/runs/{experiment_id}/train.log
results/runs/{experiment_id}/metrics.json
results/runs/{experiment_id}/samples.jsonl
```

每组实验结束后，必须立即 append 到：

```text
results/experiments.tsv
results/metrics.csv
```

推荐训练参数：

```text
model_name: Qwen/Qwen2.5-0.5B-Instruct
train_size: 500 or 1000
eval_size: 100 or 200
max_steps: 100 to 200
batch_size: 1
gradient_accumulation_steps: use if needed
fp16 or bf16: use if supported
seed: 42
```

如果显存不足，按顺序降级：

1. 减小 batch size。
2. 减少 max length。
3. 减少 train_size。
4. 减少 max_steps。
5. 使用 fallback model。
6. 停止并汇报。

## 不做什么

禁止：

1. 不要直接跑 Version B。
2. 不要在 Version A 失败时继续扩展。
3. 不要修改噪声定义。
4. 不要修改评价指标。
5. 不要只保存最终结果而不保存 logs。
6. 不要假装 crash 的实验成功。

## 产出

```text
src/train_dpo.py
src/evaluate.py
scripts/run_colab_version_a.sh
notebooks/run_dpo_colab.ipynb
results/runs/
results/metrics.csv
reports/version_a_report.md
```

## 成功标准

1. clean baseline 成功。
2. 三个 20% noise condition 成功。
3. 每组实验有 metrics。
4. 每组实验有 log。
5. `results/metrics.csv` 至少有 4 行成功实验。
6. Version A 结果足以支持保底论文。

## 停止条件

如果 clean baseline 都无法训练成功，停止并汇报。

---

# Phase 6: Colab DPO Training - Version B

## 使用 skills

开始前阅读：

```text
external-skills/get-available-resources/SKILL.md
external-skills/transformers/SKILL.md
```

## 目标

在 Version A 成功后，扩展到 Version B 十组实验。

## 前置条件

只有在 Phase 5 成功后，才能进入 Phase 6。

## 任务

实现：

```text
scripts/run_colab_version_b.sh
```

Version B 包含：

```text
clean

label_flip_10
label_flip_20
label_flip_30

ambiguous_10
ambiguous_20
ambiguous_30

weak_quality_10
weak_quality_20
weak_quality_30
```

注意：

如果 Version A 已经跑过某些实验，不要重复跑，除非用户明确要求。

脚本必须支持 resume：

```text
如果某个 experiment_id 已经有 metrics.json，则跳过。
如果某个实验 crash，记录 crash 并继续下一个实验。
```

## 不做什么

禁止：

1. 不要修改 Version A 的成功结果。
2. 不要为了跑完 B 而偷偷降低部分实验设置。
3. 不要对不同 noise type 使用不同训练参数。
4. 不要删除失败实验。
5. 不要因为结果不好而重跑直到变好。

## 产出

```text
scripts/run_colab_version_b.sh
results/runs/
results/metrics.csv
reports/version_b_report.md
```

## 成功标准

1. `results/metrics.csv` 至少包含 Version B 的有效实验记录。
2. 每个实验的 config 可追溯。
3. 每个实验的 metric 可追溯。
4. crash 或 skipped 实验有 notes。

## 停止条件

如果 Colab 资源不足，优先保留 Version A，停止并汇报未完成的 B 实验。

---

# Phase 7: 结果分析与可视化

## 使用 skills

开始前阅读：

```text
external-skills/statistical-analysis/SKILL.md
external-skills/scientific-visualization/SKILL.md
```

## 目标

把实验输出转成论文可用的表格、图和分析。

## 任务

实现：

```text
src/analyze_results.py
scripts/make_figures.sh
```

生成：

1. final metrics summary
2. noise type comparison table
3. noise rate sensitivity table
4. training loss curve
5. reward margin curve
6. chosen/rejected gap comparison
7. qualitative examples table

每个结论必须按照以下格式写入：

```text
Observation:
Evidence:
Interpretation:
Limitation:
```

## 不做什么

禁止：

1. 不要编造不存在的指标。
2. 不要把不显著趋势写成强结论。
3. 不要删除异常结果。
4. 不要只挑好看的图。
5. 不要改变 metric 定义。

## 产出

```text
results/summary_metrics.csv
paper/tables/
paper/figures/
reports/analysis.md
```

推荐图表：

```text
paper/figures/loss_curve.pdf
paper/figures/reward_margin_by_noise.pdf
paper/figures/noise_rate_sensitivity.pdf
paper/tables/final_metrics.csv
paper/tables/noise_definition_table.csv
paper/tables/qualitative_examples.csv
```

## 成功标准

1. 所有图表都来自真实 results。
2. 所有图表有标题、坐标轴、legend。
3. 每张图都有 caption draft。
4. `reports/analysis.md` 能直接支持论文 Results 和 Discussion。

## 停止条件

如果 results/metrics.csv 缺失或格式错误，停止并汇报。

---

# Phase 8: 论文初稿写作

## 使用 skills

开始前阅读：

```text
external-skills/scientific-writing/SKILL.md
external-skills/citation-management/SKILL.md
external-skills/docx/SKILL.md
```

## 目标

生成 Word 双栏论文草稿所需内容。

## 任务

创建：

```text
paper/paper_outline.md
paper/draft.md
paper/draft.docx
```

论文结构必须包括：

```text
Abstract
1. Introduction
2. Background
3. Method
4. Experimental Setup
5. Results
6. Discussion
7. Limitations
8. Conclusion
References
```

## 论文写作要求

### Abstract

必须包括：

1. DPO 和 preference data 的背景。
2. 三类噪声。
3. 小规模受控实验。
4. 主要观察结果。
5. 不夸大贡献。

### Introduction

必须说明：

1. DPO 依赖 pairwise preference data。
2. preference data 在现实中可能有噪声。
3. 不同噪声结构可能影响不同。
4. 本项目研究 label-flip、ambiguous、weak-quality 三类噪声。
5. 本项目是 small-scale controlled study。

### Background

必须解释：

1. DPO 的基本思想。
2. chosen/rejected preference pair。
3. preference noise 的问题。
4. 为什么 noise structure 重要。

### Method

必须说明：

1. clean baseline。
2. 三类噪声定义。
3. noise rate。
4. 实验控制变量。
5. 为什么这是 controlled comparison。

### Experimental Setup

必须说明：

1. dataset: UltraFeedback-derived preference pairs。
2. model: Qwen/Qwen2.5-0.5B-Instruct。
3. train/eval subset size。
4. DPO training setup。
5. metrics。
6. compute limitation。

### Results

必须引用：

1. final metrics table。
2. training curve。
3. reward margin 或 equivalent metric。
4. qualitative examples。

### Discussion

必须分析：

1. 哪类 noise 影响最大。
2. 是否符合 H1/H2/H3。
3. 为什么不同 noise 机制不同。
4. small-scale setup 的局限。

### Limitations

必须诚实写：

1. 小模型。
2. 小数据子集。
3. Colab 免费 GPU 限制。
4. seed 数量有限。
5. evaluation 不代表人类全面偏好。
6. 结论不应泛化到所有模型和数据。

### Conclusion

必须总结：

1. 本项目比较了三类偏好噪声。
2. 结果显示不同噪声结构对 DPO 的影响不同。
3. 该发现支持对 preference data quality 做更细粒度分析。
4. 未来工作可以扩大模型、数据和评价方式。

## 不做什么

禁止：

1. 不要编造不存在的实验结果。
2. 不要编造 citation。
3. 不要声称大规模结论。
4. 不要把小项目包装成顶会突破。
5. 不要写没有证据的强结论。
6. 不要改变标题和研究问题，除非用户明确要求。

## 产出

```text
paper/paper_outline.md
paper/draft.md
paper/draft.docx
paper/references.md
```

## 成功标准

1. draft 能支撑 6–8 页双栏论文。
2. 每个实验结论都能对应到 table 或 figure。
3. 每个 citation 都是真实来源。
4. Word 草稿包含图表占位。
5. 语气符合课程论文，不夸大。

## 停止条件

如果缺少真实 results，不允许写 Results 的确定性结论。

---

# Phase 9: Peer Review 和最终修正

## 使用 skills

开始前阅读：

```text
external-skills/peer-review/SKILL.md
external-skills/citation-management/SKILL.md
```

## 目标

按课程 rubric 检查论文质量。

## 任务

创建：

```text
reports/reviewer_comments.md
reports/required_fixes.md
reports/final_checklist.md
```

按照以下维度检查：

1. Novelty & Significance
2. Soundness & Content
3. Clarity & Presentation
4. Review & Positioning
5. Formatting & Compliance

## 审查问题

必须回答：

```text
研究问题是否清楚？
三类噪声定义是否可区分？
实验是否公平？
baseline 是否存在？
变量是否控制？
结果是否支持结论？
是否有过度 claim？
是否有假引用？
图表是否清楚？
论文是否能达到 6–8 页？
是否符合 Word 双栏格式？
```

## 不做什么

禁止：

1. 不要为了让论文看起来更强而编结果。
2. 不要隐藏 limitation。
3. 不要删除负结果。
4. 不要重写研究问题。
5. 不要新增未做过的实验描述。

## 产出

```text
reports/reviewer_comments.md
reports/required_fixes.md
reports/final_checklist.md
```

## 成功标准

1. reviewer comments 具体且可执行。
2. required fixes 能直接修改论文。
3. final checklist 全部确认。
4. 没有明显假引用或无证据 claim。

## 停止条件

如果发现核心实验结果不足以支撑论文，停止并建议回到 Phase 5 或 Phase 6 补实验。

---

# Skill 使用总表

| Phase | 阶段 | 需要阅读的 skill |
|---|---|---|
| Phase 0 | 环境和 repo setup | get-available-resources |
| Phase 1 | 研究计划 | hypothesis-generation |
| Phase 2 | 数据加载和检查 | exploratory-data-analysis |
| Phase 3 | 噪声注入 | exploratory-data-analysis, statistical-analysis |
| Phase 4 | CPU smoke test | exploratory-data-analysis |
| Phase 5 | Colab Version A 训练 | get-available-resources, transformers |
| Phase 6 | Colab Version B 训练 | get-available-resources, transformers |
| Phase 7 | 结果分析和图表 | statistical-analysis, scientific-visualization |
| Phase 8 | 论文写作 | scientific-writing, citation-management, docx |
| Phase 9 | 最终审稿 | peer-review, citation-management |

---

# 实验日志格式

`results/experiments.tsv` 必须包含以下字段：

```text
experiment_id
config_name
version
noise_type
noise_rate
seed
model_name
dataset_name
train_size
eval_size
max_steps
status
primary_metric
notes
```

`results/metrics.csv` 必须至少包含：

```text
experiment_id
config_name
noise_type
noise_rate
seed
train_loss
eval_loss
reward_margin
chosen_logprob
rejected_logprob
win_rate
runtime_minutes
status
```

如果某些 metric 无法获得，必须写：

```text
NA
```

并在 notes 中解释原因。

---

# Config 规则

所有实验必须从 config 文件读取。

禁止在 Python 中 hard-code：

```text
noise_type
noise_rate
train_size
eval_size
max_steps
model_name
seed
```

推荐 `configs/base.yaml`：

```yaml
project_name: dpo_noise_study

dataset:
  name: trl-lib/ultrafeedback_binarized
  train_size: 1000
  eval_size: 200
  seed: 42

model:
  primary: Qwen/Qwen2.5-0.5B-Instruct
  fallback: EleutherAI/pythia-410m

training:
  max_steps: 200
  learning_rate: 0.000005
  batch_size: 1
  gradient_accumulation_steps: 8
  max_length: 512
  max_prompt_length: 256
  seed: 42

evaluation:
  metrics:
    - train_loss
    - eval_loss
    - reward_margin
    - chosen_logprob
    - rejected_logprob

compute:
  local_cpu_only: true
  colab_gpu_training: true
```

推荐单个实验 config：

```yaml
inherits: base.yaml
experiment:
  version: A
  noise_type: label_flip
  noise_rate: 0.2
  seed: 42
```

---

# 降级策略

如果 Colab 免费 GPU 不稳定，按顺序降级：

1. train_size 从 1000 降到 500。
2. eval_size 从 200 降到 100。
3. max_steps 从 200 降到 100。
4. max_length 从 512 降到 384。
5. batch_size 保持 1。
6. 使用 gradient_accumulation。
7. 从 Qwen fallback 到 Pythia。
8. 只完成 Version A。
9. 论文标题保留 Small-Scale。

禁止直接换数据集。

---

# 最终论文最低可交标准

如果只完成 Version A，论文必须至少包含：

1. clean baseline
2. 三类 20% noise 实验
3. 一张 noise definition table
4. 一张 experimental setup table
5. 一张 final metrics table
6. 一张 training curve
7. 一张 noise type comparison figure
8. 若干 qualitative examples
9. limitation section

---

# 最终论文增强标准

如果完成 Version B，论文额外加入：

1. noise rate sensitivity analysis
2. 10/20/30% trend figure
3. 不同 noise type 随比例变化的讨论

---

# Codex 执行总指令

执行本项目时，必须遵循以下顺序：

```text
1. Read agent.md.
2. Read program.md.
3. Before each phase, read required external-skills listed in program.md.
4. Complete only the current phase.
5. Produce the required outputs.
6. Check success criteria.
7. Stop and summarize before moving to the next phase.
```

禁止一次性完成所有 phase。

每完成一个 phase，必须输出：

```text
Completed phase:
Files created/modified:
Tests run:
Results:
Problems:
Next recommended phase:
```

---

# 当前默认执行策略

默认执行路线：

```text
Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4
```

先在本地完成工程、数据、噪声和 smoke test。

然后进入：

```text
Phase 5
```

在 Colab 上跑 Version A。

如果 Version A 成功，再进入：

```text
Phase 6
```

跑 Version B。

最后：

```text
Phase 7 -> Phase 8 -> Phase 9
```

分析、写论文、审稿。
