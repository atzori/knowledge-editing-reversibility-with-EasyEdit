# Knowledge Editing Reversibility & Butterfly Effect Analysis

This repository contains the experimental framework developed for the Master's Thesis on **reversibility in Knowledge Editing (KE)** for Large Language Models (LLMs).
The project investigates how factual edits applied to a model can be **applied, evaluated, and reverted**, while analyzing **side effects** on unrelated knowledge using Butterfly Effect–style metrics.

The framework is built on top of **EasyEdit** and supports editing methods such as **ROME** and **MEMIT**, with experiments conducted on autoregressive language models (e.g., GPT-2-XL, GPT-J, LLaMA).

---

## Index

* [1. Project Goals](#1-project-goals)
* [2. Background Concepts](#2-background-concepts)

  * [2.1 Knowledge Editing](#21-knowledge-editing)
  * [2.2 ROME (Rank-One Model Editing)](#22-rome-rank-one-model-editing)
  * [2.3 MEMIT (Mass Editing Memory in Transformers)](#23-memit-mass-editing-memory-in-transformers)
  * [2.4 Butterfly Effect in Knowledge Editing](#24-butterfly-effect-in-knowledge-editing)
* [3. Repository Structure](#3-repository-structure)
* [4. Getting Started (Linux Only)](#4-getting-started-linux-only)

  * [4.1 Install Anaconda / Miniconda](#41-install-anaconda--miniconda)
  * [4.2 Obtain the Repository](#42-obtain-the-repository)
  * [4.3 Create the Python Environment](#43-create-the-python-environment)
  * [4.4 Configure the Experiment (YAML)](#44-configure-the-experiment-yaml)
  * [4.5 Run the Reference Script](#45-run-the-reference-script)
  * [4.6 Logs and Results](#46-logs-and-results)

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

**Knowledge Editing** refers to techniques that modify a pretrained model’s internal representations in order to change specific factual knowledge **without retraining the model from scratch**.

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
* Fast and deterministic
* Highly localized, but still prone to side effects

---

### 2.3 MEMIT (Mass Editing Memory in Transformers)

MEMIT generalizes ROME to **multiple edits**, distributing updates across layers to store a set of new facts more robustly.

Compared to ROME:

* Supports batch edits
* Better retention across prompts
* Higher risk of global interference if not controlled

---

### 2.4 Butterfly Effect in Knowledge Editing

In Knowledge Editing, the **Butterfly Effect** refers to unintended changes in model behavior on **unrelated prompts** after a factual edit.

Even when an edit is successful locally, it may:

* Alter probabilities of unrelated tokens
* Change generations for semantically distant facts
* Affect linguistic fluency or coherence

This project explicitly measures these effects.

---

## 3. Repository Structure

```text
.
├── easyeditor
│   └── <edit mechanisms>
│
├── hparams
│   ├── AlphaEdit
│   ├── FT
│   ├── KN
│   ├── MELO
│   ├── MEMIT
│   ├── MEND
│   └── ROME
│
├── logs
│   └── logs-YYYY-MM-DDTime*
│       ├── run.log
│       └── results.json
│
└── thesis_experiments
    ├── configs
    │   ├── exp_*.yaml
    │   └── hparams_*.yaml
    │
    ├── data
    │   └── counterfact
    │
    ├── notebooks
    │
    ├── scripts
    │   ├── butterfly_effect_ppl.py
    │   ├── ke_core.py
    │   ├── pretty_print_utilities.py
    │   ├── run_single_edit_with_BE.py
    │   └── utils_io.py
    │
    └── stats
        ├── gpt2-xl
        │   └── wikipedia_stats
        └── Llama-3.2-3B
            └── wikipedia_stats
```

---

## 4. Getting Started (Linux Only)

> **Important note**
> This setup guide currently targets **Linux systems only**.
> A dedicated guide for **Windows 11** will be provided in a future update.

### 4.1 Install Anaconda / Miniconda

To run the framework, you need a Conda-based Python distribution.

* **Recommended**: Miniconda (lighter and easier to manage)
* Alternative: Full Anaconda distribution

Follow the official instructions for Linux:

* [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

After installation, ensure that `conda` is available in your shell:

```bash
conda --version
```

---

### 4.2 Obtain the Repository

You have two supported options.

#### Option A — Clone from GitHub (recommended)

```bash
git clone https://github.com/emanuele-caddeo/knowledge-editing-reversibility-with-EasyEdit.git
cd knowledge-editing-reversibility-with-EasyEdit
```

#### Option B — Use a ZIP archive

```bash
unzip knowledge-editing-reversibility-with-EasyEdit.zip
cd knowledge-editing-reversibility-with-EasyEdit
```

---

### 4.3 Create the Python Environment

Create a dedicated Conda environment (Python **3.10 is strongly recommended**):

```bash
conda create -n easyedit python=3.10
conda activate easyedit
```

Install all required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

### 4.4 Configure the Experiment (YAML)

The framework is fully configured through **YAML files** located in:

```text
thesis_experiments/configs/
```

---

### 4.5 Run the Reference Script

```bash
python -m thesis_experiments.scripts.run_edit_and_rollback \
  --config thesis_experiments/configs/exp_gpt2xl_memit.yaml
```

---

### 4.6 Logs and Results

All experiment artifacts are saved under:

```text
logs/
```

Each run creates a timestamped subdirectory containing logs, metrics, and Butterfly Effect reports.

---

This repository is based on EasyEdit (MIT License).
Additional modifications for thesis experiments have been introduced.
