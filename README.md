# LLM Classifier + Safety Drift (Gemma-2-2B-IT)

This project trains and evaluates **Gemma-2-2B-IT** on a news classification task while measuring its safety alignment drift after post-training.  
It supports three modes:
- **Prompt-only** (no fine-tuning)
- **Full fine-tuning**
- **LoRA fine-tuning (parameter-efficient)**

The script reports **classification accuracy**, **refusal rate** on harmful prompts, and **efficiency metrics** (GPU memory, training time, trainable parameters).

---

## 📦 Setup

1. **Create and activate a Python environment**
python -m venv .venv
source .venv/bin/activate

2. **Install dependencies**
pip install --upgrade pip
pip install torch transformers datasets peft accelerate tensorboard

> Some Gemma models require accepting license terms on Hugging Face.
> ```
> huggingface-cli login
> ```

3. **GPU requirements**  
Training assumes a CUDA GPU (e.g., A100).  
For CPU-only, use `--method prompt` or `--eval_only` (fine-tuning will be slow otherwise).

---

## 🚀 Usage

**Prompt-only (no fine-tuning)**
python train_llm_classifier.py
--method prompt
--model_id google/gemma-2-2b-it
--max_eval_examples 2000
--max_safety_examples 400
--bf16

**Full fine-tuning**
python train_llm_classifier.py
--method full_ft
--model_id google/gemma-2-2b-it
--num_train_epochs 2
--learning_rate 2e-5
--per_device_train_batch_size 2
--gradient_accumulation_steps 4
--max_train_examples 15000
--max_eval_examples 2000
--max_safety_examples 400
--bf16

**LoRA fine-tuning (parameter-efficient)**
python train_llm_classifier.py
--method lora
--model_id google/gemma-2-2b-it
--num_train_epochs 3
--learning_rate 2e-4
--per_device_train_batch_size 2
--gradient_accumulation_steps 4
--max_train_examples 15000
--max_eval_examples 2000
--max_safety_examples 400
--bf16

**Evaluation only (skip training)**
python train_llm_classifier.py --method full_ft --eval_only --bf16

---
## 📂 Outputs

Results and checkpoints are saved under:
runs/<model_name>/<method>/

Example structure:
runs/
gemma-2-2b-it/
lora/
summary_metrics.json
checkpoint-500/

Key outputs:
- `summary_metrics.json` — summarized metrics:
  - classification accuracy
  - refusal rate
  - total/trainable params
  - peak GPU memory (GB)
  - training time (s)
- `trainer_state.json`, model weights, LoRA adapters, TensorBoard logs

---

## ⚙️ Common Flags

| Flag | Description | Default |
|------|--------------|----------|
| `--model_id` | Base model name | `google/gemma-2-2b-it` |
| `--method` | `prompt`, `full_ft`, or `lora` | `prompt` |
| `--num_train_epochs` | Number of epochs | 2 |
| `--learning_rate` | Learning rate | 2e-5 |
| `--per_device_train_batch_size` | Batch size | 2 |
| `--gradient_accumulation_steps` | Steps to accumulate gradients | 4 |
| `--max_train_examples` | Max AG News training samples | 20000 |
| `--max_eval_examples` | Max AG News eval samples | 2000 |
| `--max_safety_examples` | Max DoNA prompts | 500 |
| `--max_seq_length` | Token length limit | 512 |
| `--bf16` | Use bfloat16 on GPU | off |
| `--eval_only` | Skip training | off |
| `--seed` | Random seed | 42 |

---






