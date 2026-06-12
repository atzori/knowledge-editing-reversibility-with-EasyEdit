# Knowledge Editing Reversibility & Butterfly Effect Analysis

This repository contains the experimental framework developed for the Master's Thesis on **reversibility in Knowledge Editing (KE)** for Large Language Models (LLMs).
The project investigates how factual edits applied to a model can be **applied, evaluated, and reverted**, while analyzing **side effects** on unrelated knowledge using Butterfly Effect–style metrics.

The framework is built on top of **EasyEdit** and supports editing methods such as **ROME**, **MEMIT**, and **MEND**, with experiments conducted on autoregressive language models (GPT-2-XL, GPT-J-6B).

---

## Index

* [1. Project Goals](#1-project-goals)
* [2. Background Concepts](#2-background-concepts)

  * [2.1 Knowledge Editing](#21-knowledge-editing)
  * [2.2 ROME (Rank-One Model Editing)](#22-rome-rank-one-model-editing)
  * [2.3 MEMIT (Mass Editing Memory in Transformers)](#23-memit-mass-editing-memory-in-transformers)
  * [2.4 MEND (Model Editor Networks using Gradient Decomposition)](#24-mend-model-editor-networks-using-gradient-decomposition)
  * [2.5 Butterfly Effect in Knowledge Editing](#25-butterfly-effect-in-knowledge-editing)
* [3. Repository Structure](#3-repository-structure)
* [4. Getting Started (Linux / Server)](#4-getting-started-linux--server)

  * [4.1 Install Anaconda / Miniconda](#41-install-anaconda--miniconda)
  * [4.2 Obtain the Repository](#42-obtain-the-repository)
  * [4.3 Create the Python Environment](#43-create-the-python-environment)
  * [4.4 Run ROME or MEMIT experiments](#44-run-rome-or-memit-experiments)
  * [4.5 Logs and Results](#45-logs-and-results)
* [5. MEND — Training and Experiments](#5-mend--training-and-experiments)

  * [5.1 Why MEND requires a separate training phase](#51-why-mend-requires-a-separate-training-phase)
  * [5.2 Hardware requirements](#52-hardware-requirements)
  * [5.3 Download the ZsRE dataset](#53-download-the-zsre-dataset)
  * [5.4 Configuration files](#54-configuration-files)
  * [5.5 Train the MEND hypernetwork](#55-train-the-mend-hypernetwork)
  * [5.6 Monitor training progress](#56-monitor-training-progress)
  * [5.7 Resume training from a checkpoint](#57-resume-training-from-a-checkpoint)
  * [5.8 Run MEND experiments](#58-run-mend-experiments)

---

## 1. Project Goals

The main objectives of this project are:

* Apply **localized factual edits** to pretrained language models
* Measure **edit effectiveness** on the target fact
* Analyze **collateral changes** on unrelated facts (Butterfly Effect)
* Study **edit reversibility**, i.e. whether a sequence of inverse edits can restore the original model behavior
* Provide a **clean, reproducible experimental pipeline** suitable for academic research

---

## 2. Background Concepts

### 2.1 Knowledge Editing

**Knowledge Editing** refers to techniques that modify a pretrained model's internal representations in order to change specific factual knowledge **without retraining the model from scratch**.

Formally, given a model *M* and a factual statement *(s, r, o)*:

* Before editing:
  `M(s, r) → o_old`
* After editing:
  `M'(s, r) → o_new`

The goal is to enforce this change **while preserving all other knowledge**.

---

### 2.2 ROME (Rank-One Model Editing)

ROME edits a model by applying a **rank-one update** to a specific MLP layer.
The update is computed so that the hidden representation corresponding to the edited fact is redirected toward the desired output.

Key properties:

* Single-layer intervention
* Fast and deterministic at inference time — **no prior training required**
* Highly localized, but still prone to side effects

---

### 2.3 MEMIT (Mass Editing Memory in Transformers)

MEMIT generalizes ROME to **multiple edits**, distributing updates across layers to store a set of new facts more robustly.

Compared to ROME:

* Supports batch edits
* Better retention across prompts
* Higher risk of global interference if not controlled
* **No prior training required**

---

### 2.4 MEND (Model Editor Networks using Gradient Decomposition)

MEND is fundamentally different from ROME and MEMIT. Instead of computing a closed-form weight update at inference time, MEND **trains a hypernetwork** that learns — from thousands of examples — how to transform fine-tuning gradients into a good edit.

The key idea: when you want to edit a fact at inference time, MEND computes the gradient of the loss on the new fact, decomposes it using a low-rank factorization, and passes it through the trained hypernetwork, which outputs the actual weight update to apply to the model.

This design has two major consequences:

1. **MEND requires an offline training phase** (described in Section 5) before it can be used for any editing. The training teaches the hypernetwork what a "good" gradient transform looks like for the target model.
2. **At inference time, editing is very fast** — it is a single forward pass through the hypernetwork followed by a direct weight update, with no expensive optimization loop.

The training uses the **ZsRE** (Zero-Shot Relation Extraction) dataset and operates as a **meta-learning** procedure: at each training step, the trainer simulates a full edit (forward + backward through the LM), passes the gradient to the hypernetwork, applies the resulting hypothetical update, and then measures both efficacy (did the edit work?) and locality (did unrelated facts change?). The hypernetwork is updated to minimize the combined loss. Because each step differentiates *through* a simulated optimization step, training is significantly more expensive than standard fine-tuning.

---

### 2.5 Butterfly Effect in Knowledge Editing

In Knowledge Editing, the **Butterfly Effect** refers to unintended changes in model behavior on **unrelated prompts** after a factual edit.

Even when an edit is successful locally, it may:

* Alter probabilities of unrelated tokens
* Change generations for semantically distant facts
* Affect linguistic fluency or coherence

This project explicitly measures these effects using perplexity-based metrics on a held-out text corpus (ME-PPL).

---

## 3. Repository Structure

```text
.
├── easyeditor/                        # EasyEdit core (unmodified)
│   ├── dataset/
│   │   └── zsre.py                    # ZsRE dataset loader used by MEND training
│   └── trainer/
│       ├── BaseTrainer.py             # Training loop (patched for Windows compat.)
│       └── training_hparams/
│           └── mend_training_hparams.py
│
├── data/
│   └── zsre/                          # ZsRE dataset (must be downloaded — see §5.3)
│       ├── zsre_mend_train.json
│       └── zsre_mend_eval.json
│
├── thesis_experiments/
│   ├── checkpoints/
│   │   └── mend/
│   │       └── gpt2-xl                # MEND checkpoint (committed to repo — see §5.7)
│   │
│   ├── configs/
│   │   ├── exp_gpt2xl_rome.yaml       # Experiment config — ROME on GPT-2-XL
│   │   ├── exp_gpt2xl_memit.yaml      # Experiment config — MEMIT on GPT-2-XL
│   │   ├── exp_gpt2xl_mend.yaml       # Experiment config — MEND on GPT-2-XL
│   │   ├── exp_gptj6b_rome.yaml       # Experiment config — ROME on GPT-J-6B
│   │   ├── exp_gptj6b_memit.yaml      # Experiment config — MEMIT on GPT-J-6B
│   │   ├── hparams_rome_gpt2xl.yaml
│   │   ├── hparams_memit_gpt2xl.yaml
│   │   ├── hparams_mend_gpt2xl.yaml   # MEND algorithm hparams (used for both train and inference)
│   │   ├── hparams_rome_gptj6b.yaml
│   │   └── hparams_memit_gptj6b.yaml
│   │
│   ├── data/
│   │   └── counterfact/
│   │       └── counterfact.json       # CounterFact evaluation dataset
│   │
│   └── scripts/
│       ├── ke_core.py                 # Core knowledge editing utilities
│       ├── counterfact_io.py          # CounterFact loading/normalization
│       ├── reverse_on_counterfact_batch.py   # Main experiment runner
│       ├── run_edit_and_rollback_engine_batch.py
│       └── train_mend_gpt2xl.py       # MEND hypernetwork training script
│
└── logs/                              # Experiment outputs (JSONL)
```

---

## 4. Getting Started (Linux / Server)

> **Note for Windows users**
> The scripts run on Windows with minor caveats (Python path quoting). The instructions below use bash syntax. On Windows, replace `python` with the full path to your conda environment's Python executable if needed.

### 4.1 Install Anaconda / Miniconda

* **Recommended**: Miniconda (lighter and easier to manage)

Follow the official instructions:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda --version   # verify
```

---

### 4.2 Obtain the Repository

```bash
git clone https://github.com/emanuele-caddeo/knowledge-editing-reversibility-with-EasyEdit.git
cd knowledge-editing-reversibility-with-EasyEdit
```

---

### 4.3 Create the Python Environment

Python **3.10** is strongly recommended:

```bash
conda create -n easyedit python=3.10
conda activate easyedit
pip install -r requirements.txt
```

---

### 4.4 Run ROME or MEMIT experiments

ROME and MEMIT require no prior training. Run directly:

```bash
# ROME on GPT-2-XL
python -m thesis_experiments.scripts.reverse_on_counterfact_batch \
  --config thesis_experiments/configs/exp_gpt2xl_rome.yaml \
  --alg rome

# MEMIT on GPT-2-XL
python -m thesis_experiments.scripts.reverse_on_counterfact_batch \
  --config thesis_experiments/configs/exp_gpt2xl_memit.yaml \
  --alg memit
```

---

### 4.5 Logs and Results

Results are written as JSONL files to the path defined in each experiment config (`exp_reverse_out_path`), typically under `logs/`.

---

## 5. MEND — Training and Experiments

> **Read this section carefully before running anything.**
> MEND cannot be used out of the box. It requires a dedicated offline training phase that produces a checkpoint. The checkpoint is then loaded at inference time for all editing experiments.

---

### 5.1 Why MEND requires a separate training phase

As described in [§2.4](#24-mend-model-editor-networks-using-gradient-decomposition), MEND works by training a **hypernetwork** that learns to transform LM gradients into weight updates. This hypernetwork is model-specific — a checkpoint trained on GPT-2-XL cannot be used on GPT-J-6B.

The training is a **meta-learning** loop: at each step the trainer:

1. Samples an edit `(subject, new_answer, rephrasing, locality_question)` from ZsRE
2. Computes the gradient of the cross-entropy loss on the new answer w.r.t. the target MLP weights
3. Decomposes the gradient and passes it through the hypernetwork
4. Applies the resulting update to a copy of the model
5. Measures efficacy (does the edited model answer correctly?) and locality (are unrelated answers preserved?)
6. Backpropagates the combined loss back through the hypernetwork

This requires a full forward+backward pass through GPT-2-XL (1.5B parameters) at every step, on top of the hypernetwork forward/backward. **Expect training to take several days on a single GPU.**

The checkpoint committed to this repository (`thesis_experiments/checkpoints/mend/gpt2-xl`) represents the best validation checkpoint reached so far. Training can be resumed from it (see [§5.7](#57-resume-training-from-a-checkpoint)).

---

### 5.2 Hardware requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 10 GB   | 24 GB       |
| RAM      | 32 GB   | 64 GB       |
| Storage  | 20 GB   | 50 GB       |

The training script uses `model_parallel: true`, which distributes the model across all available GPUs. On a single 10 GB GPU (e.g. RTX 3080), memory usage will be at the limit (~9.3 GB allocated). A multi-GPU server will allow significantly larger batch sizes and faster evaluation.

---

### 5.3 Download the ZsRE dataset

MEND training requires two files from the ROME project page:

```bash
mkdir -p data/zsre
wget -P data/zsre https://rome.baulab.info/data/dsets/zsre_mend_train.json
wget -P data/zsre https://rome.baulab.info/data/dsets/zsre_mend_eval.json
```

> These files are **not committed to the repository** (they are several hundred MB each). You must download them manually before starting training.

Expected location after download:

```text
data/
└── zsre/
    ├── zsre_mend_train.json    # ~52k training examples
    └── zsre_mend_eval.json     # ~500 validation examples
```

Each record in these files has the following structure (required by `easyeditor/dataset/zsre.py`):

```json
{
  "src":      "The capital of France is",
  "alt":      "Lyon",
  "answers":  ["Paris"],
  "rephrase": "France's capital city is",
  "loc":      "nq question: what is the capital of Germany?",
  "loc_ans":  "Berlin"
}
```

Fields:
- `src`: the prompt whose completion the edit targets
- `alt`: the new (desired) answer after editing
- `answers`: the original correct answers (list; only the first element is used)
- `rephrase`: a rephrasing of `src` used to test generalization
- `loc`: a locality/neighborhood question — **must contain the prefix `nq question: `** (enforced by an assertion in the dataset loader)
- `loc_ans`: the correct answer to the locality question

Records where `alt` is an empty string are silently skipped during loading.

---

### 5.4 Configuration files

Two YAML files control MEND for GPT-2-XL:

**`thesis_experiments/configs/hparams_mend_gpt2xl.yaml`** — algorithm hyperparameters:

```yaml
alg: "MEND"
model_name: "gpt2-xl"
archive: "thesis_experiments/checkpoints/mend/gpt2-xl"  # path to checkpoint (relative to repo root)

inner_params:                          # which weight matrices MEND edits
  - transformer.h.45.mlp.c_proj.weight
  - transformer.h.45.mlp.c_fc.weight
  - transformer.h.46.mlp.c_proj.weight
  - transformer.h.46.mlp.c_fc.weight
  - transformer.h.47.mlp.c_proj.weight
  - transformer.h.47.mlp.c_fc.weight

rank: 1920                             # low-rank decomposition size for the hypernetwork
lr: 1.0e-6
max_iters: 100000
val_interval: 1000                     # run validation every N training steps
early_stop_patience: 20000             # stop if no improvement for this many steps
```

The `inner_params` list defines which MLP weight matrices the hypernetwork is trained to edit. For GPT-2-XL (48 layers), layers 45–47 (the last three) are used, following the convention established in the MEND paper.

**`thesis_experiments/configs/exp_gpt2xl_mend.yaml`** — experiment runner config:

```yaml
exp_method: "mend"
exp_hparams_path: "thesis_experiments/configs/hparams_mend_gpt2xl.yaml"
exp_counterfact_n_samples: 1000
exp_reverse_out_path: "logs/reverse_counterfact_gpt2xl_mend.jsonl"
```

---

### 5.5 Train the MEND hypernetwork

With the ZsRE files in place, start training from scratch:

```bash
python thesis_experiments/scripts/train_mend_gpt2xl.py
```

The script will:

1. Load `hparams_mend_gpt2xl.yaml`
2. Override `eval_only=False` and `archive=None` (always trains from scratch)
3. Load GPT-2-XL via HuggingFace (downloaded automatically on first run, ~6 GB)
4. Load the ZsRE train and eval sets
5. Run the meta-learning training loop

**Expected training time**: ~1h 20min per 1000 training steps + ~45min for each validation run on a single 10 GB GPU. With `max_iters=100000` and `val_interval=1000`, the full run takes approximately **5–6 days**. On a multi-GPU server this is significantly reduced.

Checkpoints are saved to:

```text
thesis_experiments/checkpoints/mend/gpt2-xl
```

The trainer saves a new checkpoint only when the validation metric (`loss/total_edit_val`) improves. The previous checkpoint is kept as `gpt2-xl.bk`.

---

### 5.6 Monitor training progress

The training loop logs metrics to stdout at every `log_interval` steps (default: 1000). The most relevant metrics are:

| Metric | Meaning |
|--------|---------|
| `loss/total_edit_train` | Combined edit loss on training set — should decrease |
| `edit/acc_train` | Fraction of training edits where the model answers correctly post-edit |
| `loss/total_edit_val` | Combined edit loss on validation set — used for early stopping and checkpoint saving |
| `edit/acc_val` | Validation edit accuracy — the primary quality indicator |
| `acc/pre_val` / `acc/post_val` | Model accuracy before and after edit on locality prompts — should stay close |

A typical healthy run shows:

- `loss/total_edit_train` dropping from ~1.26 at step 1000 to ~0.32 at step 2000 and continuing to decrease
- `edit/acc_val` increasing gradually over thousands of steps
- `acc/pre_val` and `acc/post_val` remaining close (locality preserved)

To check which step the current checkpoint corresponds to:

```python
import torch
ck = torch.load("thesis_experiments/checkpoints/mend/gpt2-xl", map_location="cpu")
print("step:", ck["step"])
```

---

### 5.7 Resume training from a checkpoint

The checkpoint committed to this repository (`thesis_experiments/checkpoints/mend/gpt2-xl`) is the best validation checkpoint reached so far in training. **The training script always starts from scratch** (it sets `archive=None` internally), so to resume from the committed checkpoint you need to modify the training script slightly.

Open `thesis_experiments/scripts/train_mend_gpt2xl.py` and comment out the line that forces `archive=None`:

```python
hparams = MENDTrainingHparams.from_hparams(HPARAMS_PATH)
hparams.eval_only = False    # keep this — enables training
# hparams.archive = None     # comment this out to resume from checkpoint
```

With `archive=None` commented out, the trainer will load the checkpoint from the path defined in `hparams_mend_gpt2xl.yaml` (`thesis_experiments/checkpoints/mend/gpt2-xl`) and resume from the saved step.

> **Note**: The checkpoint in the repo was saved on a Windows machine. It is a standard PyTorch `.pt` file and loads correctly on Linux with `torch.load(..., map_location="cpu")` or `map_location="cuda:0"`.

---

### 5.8 Run MEND experiments

Once a checkpoint exists at `thesis_experiments/checkpoints/mend/gpt2-xl`, run the experiments with:

```bash
python -m thesis_experiments.scripts.reverse_on_counterfact_batch \
  --config thesis_experiments/configs/exp_gpt2xl_mend.yaml \
  --alg mend
```

The experiment runner will:

1. Load the MEND hparams from `hparams_mend_gpt2xl.yaml` (which includes `archive` pointing to the checkpoint)
2. Load GPT-2-XL
3. Load the committed checkpoint into the hypernetwork
4. For each CounterFact sample: apply the MEND edit, evaluate efficacy and locality, then roll back
5. Write results to `logs/reverse_counterfact_gpt2xl_mend.jsonl`

> The `eval_only` flag in the hparams YAML is `true` by default, which is correct for inference. The training script overrides it; the experiment runner does not.

---

This repository is based on [EasyEdit](https://github.com/zjunlp/EasyEdit) (MIT License).
Additional modifications for thesis experiments have been introduced.
