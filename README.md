# Compression for Reasoning Language Models (Reasoning-Aware Compression)

This repository contains code on compression for open-source reasoning language models.
It provides instructions for reproducing results and extending the codebase, including:

1. **Collecting reasoning traces from R1-distilled models**
2. **Running pruning and quantization (including on reasoning traces)**
3. **Evaluating models on math and coding benchmarks**

---

### Setup

From the project root:

```bash
cd open-r1-main
pip install -r requirements.txt
```

---

## 1. Collecting reasoning traces from R1-distilled models

The first step is to collect reasoning traces from the dense (i.e. unpruned) models.
The [open-r1 repo](https://github.com/huggingface/open-r1/tree/main) does not provide this functionality directly.

Run the following:

### Code traces

```bash
python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path <PATH_TO_MODEL> \
  --dataset_name <PATH_TO_CODEFORCES_DATASET> \
  --dataset_prompt_column prompt \
  --save_dir <OUTPUT_DIR> \
  --num_generations 2 \
  --beta 0 \
  --report_to none \
  --use_vllm False \
  --max_completion_length 8192 \
  --do_train False \
  --trace_only \
  --trace_tokens 1_000_000
```

### Math traces

```bash
python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path <PATH_TO_MODEL> \
  --dataset_name <PATH_TO_MATH_DATASET> \
  --dataset_prompt_column problem \
  --save_dir <OUTPUT_DIR> \
  --num_generations 2 \
  --beta 0 \
  --report_to none \
  --use_vllm False \
  --max_completion_length 8192 \
  --do_train False \
  --trace_only \
  --trace_tokens 1_000_000
```

**Key arguments:**

* `config`: YAML file for model loading (flash attention, CPU/GPU offload, etc.).
* `model_name_or_path`: Path to the model for trace generation.
* `dataset_name`: Path to dataset directory containing prompts/problems.
* `dataset_prompt_column`: Column used as the prompt.

  * Codeforces → `prompt`
  * Math dataset → `problem`
* `num_generations`: Number of traces per prompt.
* `max_completion_length`: Maximum CoT tokens per prompt (default 8192).
* `do_train`: Set `False` to disable training.
* `trace_only`: Generate traces without training.
* `trace_tokens`: Overall token budget for traces.

---

## 2. Pruning and quantization

After collecting traces, they will be saved under the `save_dir`. For example:

```
<OUTPUT_DIR>/dataset_DeepSeek-R1-Distill-Qwen-14B_trace_codeforces-firstsol
```

### Prompt + CoT pruning

```bash
python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path <PATH_TO_MODEL> \
  --dataset_name <TRACE_DATASET> \
  --dataset_prompt_column prompt \
  --beta 0 \
  --report_to none \
  --use_vllm False \
  --do_train False \
  --prune \
  --pruning_method SparseGPT \
  --prune_sparsity 0.4 \
  --prune_calib_tokens 1_000_000 \
  --push_to_hub False \
  --score_completions False
```

**Notes:**

* Always set `dataset_prompt_column = prompt` for trace datasets.
* `pruning_method`: Options include `SparseGPT`, `WANDA`, `MP`.
* `prune_sparsity`: Layer-wise sparsity level (e.g. `0.4` = 40%).
* `prune_calib_tokens`: Number of calibration tokens for pruning/quantization.

### Prompt-only pruning

Use the original dataset instead of the trace dataset:

**Coding:**

```bash
python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path <PATH_TO_MODEL> \
  --dataset_name <PATH_TO_CODEFORCES_DATASET> \
  --dataset_prompt_column prompt \
  --beta 0 \
  --report_to none \
  --use_vllm False \
  --do_train False \
  --prune \
  --pruning_method SparseGPT \
  --prune_sparsity 0.4 \
  --prune_calib_tokens 1_000_000 \
  --push_to_hub False \
  --score_completions False
```

**Math:**

```bash
python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path <PATH_TO_MODEL> \
  --dataset_name <PATH_TO_MATH_DATASET> \
  --dataset_prompt_column problem \
  --beta 0 \
  --report_to none \
  --use_vllm False \
  --do_train False \
  --prune \
  --pruning_method SparseGPT \
  --prune_sparsity 0.4 \
  --prune_calib_tokens 1_000_000 \
  --push_to_hub False \
  --score_completions False
```

### C4 pruning

```bash
python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path <PATH_TO_MODEL> \
  --dataset_name allenai/c4 \
  --dataset_config_name default \
  --dataset_split train \
  --dataset_prompt_column text \
  --report_to none \
  --use_vllm False \
  --prune \
  --pruning_method SparseGPT \
  --prune_sparsity 0.4 \
  --prune_calib_tokens 1_000_000
```

### Quantization

First, install quantization dependencies:

```bash
pip install -r requirements_quant.txt
```

Example:

```bash
python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path <PATH_TO_MODEL> \
  --dataset_name <TRACE_DATASET> \
  --dataset_prompt_column prompt \
  --use_vllm False \
  --do_train False \
  --quantize \
  --quantize_method W4A16 \
  --quantize_calib_tokens 1000000 \
  --smoothquant_strength 0.8
```

---

## 3. Evaluation

The evaluation pipeline follows the [open-r1 repo](https://github.com/huggingface/open-r1/tree/main).

### Math evaluation

```bash
#!/usr/bin/env bash
set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn

NUM_GPUS=2
MODEL_DIR=<PATH_TO_PRUNED_MODEL>
MODEL_TAG=$(basename "$MODEL_DIR")

MODEL_ARGS="model_name=${MODEL_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
max_model_length=32768,\
gpu_memory_utilization=0.8,\
data_parallel_size=${NUM_GPUS},\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=<OUTPUT_DIR>

lighteval vllm "$MODEL_ARGS" "lighteval|math_500|0|0" \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR" \
  --save-details
```

### Coding evaluation

```bash
#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="<CACHE_DIR>"
mkdir -p "${CACHE_ROOT}/datasets"

export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export HF_HOME="${CACHE_ROOT}"

pip install --user --upgrade "lighteval[extended-tasks]"
pip uninstall -y datasets || true
pip install --user "datasets<3.0.0"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1

MODEL_DIR=<PATH_TO_PRUNED_MODEL>
MODEL_ARGS="model_name=${MODEL_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
max_model_length=32768,\
gpu_memory_utilization=0.8,\
data_parallel_size=${NUM_GPUS},\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=<OUTPUT_DIR>

lighteval vllm "${MODEL_ARGS}" \
        "extended|lcb:codegeneration|0|0" \
        --use-chat-template \
        --output-dir "${OUTPUT_DIR}" \
        --save-details 
```

---
