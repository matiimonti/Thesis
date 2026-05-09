# Mapping Reasoning Faithfulness in LLMs for Financial Sentiment Tasks

**MSc Thesis — Data Science & Artificial Intelligence Strategy**  
Matilde Monti

---

## Overview

This repository contains the code and notebooks for the thesis *"Mapping Reasoning Faithfulness in Large Language Models for Financial Sentiment Tasks"*. The project investigates whether LLM-generated explanations for financial sentiment classifications faithfully reflect actual decision processes, or constitute post-hoc rationalization.

Faithfulness is evaluated through four behavioral tests - feature attribution via redaction, counterfactual perturbation, chain-of-thought (CoT) intervention, and prompt stability — applied to three models: **Llama-3.2-3B-Instruct**, **Gemma-2-2b-it**, and **FinGPT** (fingpt-mt_llama-7b_lora). The dataset is the [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) `sentences_allagree` subset (n=2,264).

---

## Notebooks

| Notebook | Description | Open in Colab |
|---|---|---|
| `thesis_analysis.ipynb` | Setup, data loading, baseline classification, and model evaluation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matiimonti/llm-faithfulness-financial-sentiment-analysis/blob/main/thesis_analysis.ipynb) |
| `thesis_experiments.ipynb` | Full faithfulness experiments: redaction, counterfactual, CoT intervention, prompt stability | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matiimonti/llm-faithfulness-financial-sentiment-analysis/blob/main/thesis_experiments.ipynb) |

---

## Setup

### Requirements

- Python 3.10+
- A GPU runtime is strongly recommended (notebooks were developed on Google Colab with a T4 GPU; full experimental runs use an A100)

### Installation

```bash
pip install transformers datasets torch accelerate peft sentencepiece
```

### Model Access

**Llama-3.2-3B-Instruct** and **Gemma-2-2b-it** require accepting their respective license agreements on Hugging Face and authenticating with a token:

```python
from huggingface_hub import login
login(token="your_hf_token")
```

**FinGPT** (fingpt-mt_llama-7b_lora) and **FinBERT** are publicly available without authentication.

### Dataset

The Financial PhraseBank dataset loads directly via Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
```

---

## Repository Structure

```
├── thesis_analysis.ipynb       # Baseline setup and model evaluation
├── thesis_experiments.ipynb    # Faithfulness experiments
└── README.md
```

---

## Research Questions

- **RQ1** — To what extent are LLM-generated explanations for financial sentiment classifications faithful to the model's actual decision process?
- **RQ2** — Does domain specialization (FinGPT) produce different reasoning behavior compared to general-purpose models?
- **RQ3** — How stable are model explanations under prompt variation and sampling randomness?