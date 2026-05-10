---

## Phase 1 — Research Plan and Hypotheses

**Date:** 2026-05-10
**Status:** Completed

### Goal

完成研究问题、三类噪声的可检验定义、核心假设、变量设计和 Version A / B 的实验矩阵。

### Completed Work

- 明确了单一研究问题：不同结构的 preference noise 如何影响 DPO 训练与对齐行为。
- 写出了三个核心假设，分别对应 `label_flip`、`ambiguous` 和 `weak_quality` 三类噪声。
- 列出了自变量、因变量和控制变量，保持数据源、模型和实验框架不变。
- 以操作性定义方式区分了三类噪声，避免把它们混为一类。
- 给出了 Version A 与 Version B 的实验矩阵，并明确 Version B 是 Version A 的超集。
- 记录了预期风险和论文叙事主线。

### Files Created or Modified

- `reports/research_plan.md`
- `reports/project_log.md`

### Commands Run

```bash
Get-Content -Raw 'C:\Users\a1882\.codex\skills\using-superpowers\SKILL.md'
Get-Content -Raw 'C:\Users\a1882\.codex\skills\brainstorming\SKILL.md'
if (Test-Path 'AGENTS.md') { Get-Content -Raw 'AGENTS.md' } elseif (Test-Path 'agent.md') { Get-Content -Raw 'agent.md' } else { Write-Output '__MISSING__' }
Get-Content -Raw 'program.md'
Get-Content -Raw 'reports/project_log.md'
Get-Content -Raw 'external-skills/hypothesis-generation/SKILL.md'
```

### Test Results

- `external-skills/hypothesis-generation/SKILL.md` exists and was read.
- `AGENTS.md` is not present in the repo; `agent.md` was used as the local project instruction file.
- `reports/research_plan.md` was created successfully.

### Problems / Blockers

- No blocking issue for Phase 1.
- The skill text mentions schematic generation, but this phase remains doc-only and does not start implementation or training.

### Decisions Made

- Kept the research scope to one research question.
- Kept the dataset source fixed to UltraFeedback-derived preference data.
- Kept the plan aligned with the later CPU-only local setup and Colab GPU training split.

### Next Step

- Move to Phase 2 only after the user approves or the workflow proceeds to dataset loading.

---

## Phase 2 — UltraFeedback Data Loading and Subset Creation

**Date:** 2026-05-10
**Status:** Completed

### Goal

加载 UltraFeedback-derived preference data，构建可复现的 train/eval 子集，并生成数据报告。

### Completed Work

- 实现了 `src/load_data.py`，支持从本地 JSON/JSONL 或 Hugging Face 数据集读取偏好数据。
- 支持将消息列表格式的 `chosen` / `rejected` 对话规范化为 `prompt`、`chosen`、`rejected` 三字段。
- 实现了确定性子集抽样、JSONL 写出和数据报告生成。
- 新增了 `tests/test_load_data.py`，覆盖字段规范化、确定性拆分、写出和报告生成。
- 成功加载 `trl-lib/ultrafeedback_binarized`，并生成 1000 条 train 子集与 200 条 eval 子集。
- 生成了 `reports/data_report.md`，并保留了 `data/processed/data_report.md` 作为处理产物记录。

### Files Created or Modified

- `requirements.txt`
- `src/__init__.py`
- `src/load_data.py`
- `tests/test_load_data.py`
- `data/processed/train_clean.jsonl`
- `data/processed/eval_clean.jsonl`
- `data/processed/data_report.md`
- `reports/data_report.md`
- `reports/project_log.md`

### Commands Run

```bash
python -m pytest tests/test_load_data.py
python -m pip install pytest
python -m pip install datasets
python -c "from datasets import load_dataset; ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:2]'); print(ds.features); print(ds[0].keys()); print(ds[0])"
python -c "from pathlib import Path; from src.load_data import prepare_from_source; summary = prepare_from_source('trl-lib/ultrafeedback_binarized', output_dir=Path('data/processed'), train_size=1000, eval_size=200, seed=42, split='train'); print(summary)"
```

### Test Results

- `pytest tests/test_load_data.py` 通过，6 个测试全部绿色。
- 真实数据集加载成功，训练集加载到 62,135 条记录，按 seed 42 生成 1000/200 的确定性子集。
- `data/processed/train_clean.jsonl` 含 1000 行。
- `data/processed/eval_clean.jsonl` 含 200 行。
- `reports/data_report.md` 已生成。

### Problems / Blockers

- Hugging Face Hub 发出了未认证请求警告，但不影响本次数据加载。
- Windows 环境提示 Hugging Face 缓存符号链接受限，但缓存仍可用。

### Decisions Made

- 保持数据源不变，继续使用 `trl-lib/ultrafeedback_binarized`。
- 将真实数据处理结果写入 `data/processed/`，并在 `reports/` 下保留一份正式报告。
- 规范化逻辑兼容对话消息列表格式，以匹配真实数据集结构。

### Next Step

- 进入 Phase 3，处理三类 preference noise 的注入与验证。

---

## Phase 3 — Preference Noise Injection

**Date:** 2026-05-10
**Status:** Completed

### Goal

实现三类 preference noise 注入，并生成可复现的 noisy datasets 与预览报告。

### Completed Work

- 实现了 `src/inject_noise.py`，支持 `clean`、`label_flip`、`ambiguous`、`weak_quality` 四种噪声模式。
- `label_flip` 使用 seed 控制的随机采样交换 `chosen` / `rejected`。
- `ambiguous` 按最小 `score_chosen - score_rejected` gap 选择样本，但不交换 pair 方向。
- `weak_quality` 保持 pair 方向不变，但优先选择 `score_chosen` 较低且仍高于 `score_rejected` 的样本。
- 新增了 `tests/test_inject_noise.py`，覆盖三类噪声的核心行为、文件写出和预览报告生成。
- 新增了 `scripts/preview_noise.py`，可以从 `data/processed/train_clean.jsonl` 生成噪声数据和预览报告。
- 生成了 `data/noisy/*.jsonl` 的全套噪声文件，覆盖 `10/20/30` 三个噪声率和三类噪声。
- 生成了 `reports/noise_definition.md` 和 `reports/noise_preview.md`。

### Files Created or Modified

- `src/inject_noise.py`
- `scripts/preview_noise.py`
- `tests/test_inject_noise.py`
- `reports/noise_definition.md`
- `reports/noise_preview.md`
- `data/noisy/clean.jsonl`
- `data/noisy/label_flip_10.jsonl`
- `data/noisy/label_flip_20.jsonl`
- `data/noisy/label_flip_30.jsonl`
- `data/noisy/ambiguous_10.jsonl`
- `data/noisy/ambiguous_20.jsonl`
- `data/noisy/ambiguous_30.jsonl`
- `data/noisy/weak_quality_10.jsonl`
- `data/noisy/weak_quality_20.jsonl`
- `data/noisy/weak_quality_30.jsonl`
- `reports/project_log.md`

### Commands Run

```bash
python -m pytest tests/test_load_data.py tests/test_inject_noise.py
python scripts/preview_noise.py --input data/processed/train_clean.jsonl --output-dir data/noisy --report reports/noise_preview.md --seed 42
```

### Test Results

- `pytest` 通过，11 个测试全部绿色。
- 噪声预览报告生成成功。
- `reports/noise_preview.md` 包含 50 个 example 区块，对应 10 个噪声配置，每个配置预览 5 条样本。
- `ambiguous_20` 和 `weak_quality_20` 的预览显示文本未交换，仅样本选择策略不同。

### Problems / Blockers

- `reports/noise_preview.md` 初版把 metadata 变动也算作文本变动，已修正为只比较 `prompt/chosen/rejected`。
- `scripts/preview_noise.py` 初版缺少项目根目录导入路径，已修正。

### Decisions Made

- 采用 `score_chosen` / `score_rejected` 作为主要噪声代理，优先利用数据集自带信号。
- 保留 `clean` 文件作为噪声基线，并将 Phase 3 的所有噪声矩阵写入 `data/noisy/`。
- 预览报告采用“Before / After”结构，突出 pair 是否交换以及样本选择策略。

### Next Step

- 进入 Phase 4，做本地 CPU smoke test。

---

## Phase 4 - Local CPU Smoke Test

**Date:** 2026-05-10
**Status:** Completed

### Goal

在不使用 GPU 的情况下验证完整的数据与噪声流水线，并保留一个不包含正式训练结果的 smoke 报告和占位指标 schema。

### Completed Work

- 新增 `src/smoke.py`，把本地 smoke 流程封装成可复用入口。
- 新增 `scripts/run_cpu_smoke.sh`，作为 shell 包装入口。
- 将 `Makefile` 的 `test` 和 `smoke` 目标改为真实命令。
- 运行 smoke 流程加载 `data/processed/train_clean.jsonl`，确认子集记录数为 1000。
- 生成 `clean`、`label_flip_20`、`ambiguous_20`、`weak_quality_20` 的 smoke 噪声文件。
- 运行项目测试集 `pytest tests`，并确认全部通过。
- 重新生成 `reports/noise_preview.md`。
- 生成 `reports/cpu_smoke_report.md`。
- 生成 `results/smoke_metrics_schema.csv`，仅包含占位字段，不包含真实训练指标。

### Files Created or Modified

- `src/smoke.py`
- `scripts/run_cpu_smoke.sh`
- `tests/test_smoke.py`
- `Makefile`
- `reports/cpu_smoke_report.md`
- `results/smoke_metrics_schema.csv`
- `reports/project_log.md`

### Commands Run

```bash
python -m pytest tests/test_smoke.py
python -m pytest tests
python -m src.smoke
& 'C:\Program Files\Git\bin\bash.exe' scripts/run_cpu_smoke.sh
```

### Test Results

- `tests/test_smoke.py` 通过。
- `pytest tests` 通过，共 13 个测试通过。
- `python -m src.smoke` 成功生成 smoke 报告和 schema。
- `scripts/run_cpu_smoke.sh` 通过 Git Bash 成功执行。

### Problems / Blockers

- `bash` 指向 Windows 的 WSL launcher，直接调用时不可用。
- 已改用 Git Bash 路径验证 shell 脚本，同时保留 `python -m src.smoke` 作为本地稳定入口。
- 仓库中存在与本项目无关的辅助 skill 测试；Phase 4 的 smoke 现在明确限制在 `tests/` 目录。

### Decisions Made

- 保持本地阶段只做 CPU 数据准备和测试，不进入正式 DPO。
- 使用占位 metrics schema，而不是伪造实验结果。
- 将 smoke 流程集中到一个 Python 入口，shell 脚本只做薄包装。

### Next Step

- 进入 Phase 5，在 Colab 免费 GPU 上开始 Version A DPO 训练。
---

## Phase 5 - Colab Entry Layer and Training Orchestration

**Date:** 2026-05-10
**Status:** In progress

### Goal

将 `main.py` 接入可执行的 DPO 训练调度与按实例记录，让 Version A 能在 Colab 上用单一入口运行。

### Completed Work

- 新增 `src/train_dpo.py`，负责训练配置、数据路径解析和训练执行编排。
- 新增 `src/evaluate.py`，把 TRL 训练日志归一到项目的 metrics 结构。
- 更新 `src/pipeline.py`，支持批量调度、`resume` 跳过和可选执行。
- 更新 `main.py`，让 `train` / `batch` / `resume` 共用同一入口。
- 更新 `requirements.txt`，补入 `torch`、`transformers`、`trl`、`accelerate` 等训练依赖。
- 新增 `scripts/run_colab_version_a.sh` 和 `notebooks/run_dpo_colab.ipynb`。
- 新增 `reports/version_a_report.md` 作为 Version A 结果占位。

### Tests Run

- `python -m pytest tests -q`
- `python main.py --help`
- `python main.py inspect --plan configs/version_a.yaml`

### Results

- 当前共有 27 个测试全部通过。
- `main.py` 已经可以展开 Version A 计划，并作为轻量入口调度后续训练。
- 执行训练的通路已经接好，但这台当前 shell 环境里没有可用的 `conda` 和 TRL/torch 训练栈，因此还不能在这里正式跑 DPO。

### Problems / Blockers

- 当前 PowerShell 环境没有可用的 `conda` 命令。
- 核心训练依赖 `torch`、`accelerate`、`transformers`、`trl` 尚未在这里就绪。

### Next Step

- 在 Colab 或可用的 Anaconda 训练环境里运行 `python main.py batch --plan configs/version_a.yaml`，再把真实结果回填到 `results/` 和 `reports/version_a_report.md`。
### Progress Signal Update

- 新增 live output 和 `results/runs/{experiment_id}/heartbeat.json`，用于判断训练是否仍在继续。
- 训练日志会实时刷到标准输出，不再只在结束后一次性写回。
