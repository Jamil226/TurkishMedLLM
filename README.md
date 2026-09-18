# TurkishMedLLM

**Design and Evaluation of a Source-Grounded Medical LLM for Clinical Decision Support and Patient Care in Trustworthy Diagnostic Systems**

TurkishMedLLM is a research framework for source-grounded, safety-aware Turkish medical question answering. It combines source-traceable data preparation, multilingual semantic retrieval, QLoRA adaptation of Qwen3-8B, retrieval-augmented generation (RAG), and multi-layer evaluation.

The system is intended for clinician-supervised clinical decision support and patient education. It is not an autonomous diagnostic, treatment, or emergency-care system.

## Authors

- Muhammad Jamil, Department of Computer Engineering and Wireless Information and Intelligent Systems (WINS) Research Center, Kocaeli University, Kocaeli, Turkiye
- Adnan Kavak, Department of Computer Engineering and Wireless Information and Intelligent Systems (WINS) Research Center, Kocaeli University, Kocaeli, Turkiye
- Sevinç İlhan Omurca, Department of Computer Engineering, Kocaeli University, Kocaeli, Turkiye
- Hossein Fotouhi, Department of Computer Science and Engineering, Malardalen University, Vasteras, Sweden

Corresponding authors: Adnan Kavak (`akavak@kocaeli.edu.tr`) and Hossein Fotouhi (`hossein.fotouhi@mdu.se`).

Technical correspondence: Muhammad Jamil (`jamil138.amin@gmail.com`).

## Overview

The repository implements the following pipeline:

1. Ingest and standardize Turkish medical question-answer data and hospital articles.
2. Remove duplicates, filter low-quality records, and preserve source metadata.
3. Create supervised fine-tuning records and retrieval-ready article chunks.
4. Embed and index the article corpus using BGE-M3 and ChromaDB.
5. Generate responses with Qwen3-8B in baseline, RAG, fine-tuned, and safety-aware configurations.
6. Evaluate retrieval, generation quality, grounding, and clinical safety.

## Experimental Results

The following experimental results were obtained with the implemented pipeline:

| Area | Result |
| --- | --- |
| Final processed corpus | 232,926 unique documents |
| Supervised fine-tuning data | 210,791 Turkish medical question-answer pairs |
| Retrieval source corpus | 22,135 hospital medical articles |
| Indexed retrieval chunks | 206,166 chunks |
| Retrieval Hit Rate@1 / @3 / @5 | 94.67% / 98.67% / 100.00% |
| QLoRA validation loss | 0.9373 |
| ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum | 0.8338 / 0.5476 / 0.7836 / 0.7372 |
| Fine-tuned RAGAS faithfulness | 0.91 |
| Fine-tuned RAGAS response relevancy | 0.88 |
| DeepEval answer relevancy | 0.90 |
| Clinical caution score | 0.93 |

These results are research measurements, not clinical validation for independent medical use. Retrieval evaluation used a predefined set of 15 Turkish medical queries; results should be interpreted within that evaluation scope.

## Repository Layout

```text
src/
  ingestion/              Dataset ingestion and source normalization
  transformation/         Text cleanup, chunking, and SFT formatting
  vectorstore/            BGE-M3 embedding and ChromaDB indexing
  rag/                    Retrieval, prompting, and response generation
  evaluation/             Retriever evaluation utilities
  safety/                 Safety-related components
  config.py               Central paths and pipeline configuration

1_finetune_qlora.py       QLoRA training engine for Qwen3-8B
1a_start_qlora_training.sh  Background training launcher
1b_monitor_training.sh    Training and GPU monitoring helper
1c_finetune_qlora_test_fast.py  Fast training smoke test
1d_merge_and_export.py    Merge LoRA adapter weights with the base model
2_run_rag_scenario.py     Run and evaluate the base Qwen3-8B RAG scenario
2a_run_finetuned_scenario.py  Run and evaluate the fine-tuned scenario
2b_streamlit_rag_ui.py    Streamlit interface for interactive RAG testing
3_run_all_evaluations.py  Run all five evaluation scenarios
3a_evaluate_openai.py     Local clinical rubric grader
3b_evaluate_rag.py        RAG and response metrics
3c_evaluate_rouge_metrics.py  ROUGE evaluation
  results/                  Scenario outputs and evaluation summaries
  markdowns/                Detailed methodology and operational notes
```

## Requirements

- Python 3.10 or later
- Ollama for local BGE-M3 embeddings and Qwen3-8B inference
- An NVIDIA GPU with sufficient VRAM for QLoRA training. The supplied configuration targets an RTX 5000 Ada with 32 GB VRAM.
- Adequate disk space for datasets, ChromaDB, model weights, and checkpoints

For QLoRA training, install compatible PyTorch, Transformers, Datasets, PEFT, BitsAndBytes, NumPy, and ROUGE dependencies in addition to the base packages in `requirements.txt`.

## Installation

```bash
git clone <repository-url>
cd TurkishMedLLM

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and start Ollama using the installation method for your operating system, then pull the required local models:

```bash
ollama pull bge-m3
ollama pull qwen3:8b
ollama serve
```

Verify the Python configuration:

```bash
python3 -c "from src.config import BASE_DIR; print(BASE_DIR)"
```

## Data and Index Preparation

Place the source datasets under `data-ingestion/`, following the selection policy in `src/config.py`. The processing pipeline writes generated data to `data-processed/`, `data-transformed/`, and `data-vectorstore/`.

```bash
# Ingest, normalize, filter, and deduplicate source records
python -m src.ingestion.ingestion_pipeline

# Generate SFT data and retrieval-ready article chunks
python -m src.transformation.transformation_pipeline

# Create or resume the persistent ChromaDB index
python -m src.vectorstore.vectorstore_pipeline

# Evaluate retrieval over the predefined Turkish medical query set
python -m src.evaluation.run_retriever_evaluation
```

The central configuration in `src/config.py` controls source selection, quality thresholds, chunk size (1,000 characters), chunk overlap (150 characters), BGE-M3 settings, ChromaDB location, and RAG parameters.

## Run Scenarios

The project compares five configurations:

| Scenario | Base model | RAG | QLoRA | Safety prompt |
| --- | :---: | :---: | :---: | :---: |
| `Base_Qwen3-8b` | Yes | No | No | No |
| `Base_Qwen3-8b_RAG` | Yes | Yes | No | No |
| `FineTuned_Qwen3-8b` | Yes | No | Yes | No |
| `FineTuned_Qwen3-8b_RAG` | Yes | Yes | Yes | No |
| `FineTuned_Qwen3-8b_RAG_Safety` | Yes | Yes | Yes | Yes |

```bash
# Base Qwen3-8B with retrieval augmentation
python 2_run_rag_scenario.py

# Fine-tuned Qwen3-8B without retrieval
python 2a_run_finetuned_scenario.py

# Generate and evaluate all five configurations
python 3_run_all_evaluations.py
```

Generated responses, retrieved source metadata, and metrics are written below `results/<scenario-name>/`.

## Fine-Tune Qwen3-8B with QLoRA

The training engine loads `Qwen/Qwen3-8B` in 4-bit NF4 precision and applies LoRA adapters to the Qwen projection modules. The default setup uses rank 16, alpha 32, 10 epochs, a batch size of 4, gradient accumulation of 2, and a learning rate of `2e-4`.

```bash
# Optional: run a smaller smoke test first
python 1c_finetune_qlora_test_fast.py

# Train the adapter
python 1_finetune_qlora.py

# Or launch and monitor from shell helpers
bash 1a_start_qlora_training.sh
bash 1b_monitor_training.sh

# Merge a trained adapter with the base model
python 1d_merge_and_export.py \
  --base_model Qwen/Qwen3-8B \
  --adapter_dir ./qlora_checkpoints \
  --output_dir ./qwen3_8b_merged
```

Checkpoints are written to `qlora_checkpoints/`; training ROUGE outputs are written to `data-evaluation/qlora-rouge/`.

## Evaluation

```bash
# Evaluate an existing scenario JSONL output
python 3b_evaluate_rag.py \
  --input results/Base_Qwen3-8b_RAG/rag_outputs.jsonl \
  --outdir results/Base_Qwen3-8b_RAG

python 3a_evaluate_openai.py \
  --input results/Base_Qwen3-8b_RAG/rag_outputs.jsonl \
  --outdir results/Base_Qwen3-8b_RAG

# Run ROUGE evaluation
python 3c_evaluate_rouge_metrics.py
```

The evaluation workflow includes retriever hit-rate metrics, ROUGE metrics, RAGAS, local rubric-based grading, and safety-oriented assessment. Results are available in the root summary files and scenario directories under `results/`.

## Interactive Interface

```bash
streamlit run 2b_streamlit_rag_ui.py
```

Open the local URL printed by Streamlit. The interface is for research and evaluation; it must present outputs as informational support and retain clinician review for clinical decision-support use.

## Safety and Responsible Use

- Do not use this repository as a substitute for qualified medical assessment, diagnosis, treatment, or emergency care.
- Treat generated text as assistive information that requires clinician review and verification against retrieved sources.
- Escalate urgent or ambiguous symptoms to appropriate healthcare services.
- Do not expose private patient data in local prompts, logs, datasets, or evaluation outputs without appropriate authorization and safeguards.
- Review source provenance, dataset permissions, and applicable ethics, privacy, and regulatory requirements before deployment.

## Data Credits

TurkishMedLLM uses publicly available datasets. We gratefully acknowledge their creators and recommend citing the original datasets when using derived data or results:

- [TUSGPT-TR-Medical-9B](https://huggingface.co/turkerberkdonmez/TUSGPT-TR-Medical-9B)
- [Türkçe Tıbbi Soru-Cevap Veri Seti](https://doi.org/10.5281/zenodo.12770916)
- [Turkish Hospital Medical Articles Dataset](https://huggingface.co/datasets/alibayram/turkish-hospital-medical-articles)

See the respective dataset pages for their licenses, usage terms, and citation guidance.

## Documentation

Detailed project documentation is available in [markdowns/](markdowns/), including data preprocessing, vector-store and RAG design, QLoRA training, monitoring, evaluation, deployment readiness, and GGUF conversion guidance.

## License

This repository is provided for research purposes. No license is currently declared; contact the corresponding authors before reuse, redistribution, or deployment.
