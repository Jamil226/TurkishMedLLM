# Centralized Configuration Module
# This file stores directory paths, data selection policies, quality thresholds,
# vector store models, and RAG/SFT parameters to ensure pipeline reproducibility.

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_INGESTION_DIR = BASE_DIR / "data-ingestion"
DATA_PROCESSED_DIR = BASE_DIR / "data-processed"

# Output Directories
FINETUNING_DIR = DATA_PROCESSED_DIR / "finetuning"
RAG_CORPUS_DIR = DATA_PROCESSED_DIR / "rag-corpus"
LOGS_DIR = DATA_PROCESSED_DIR / "logs"

# Output Files
QA_DOCUMENTS_FILE = FINETUNING_DIR / "turkishmedllm_qa_documents.jsonl"
SFT_INSTRUCTION_FILE = FINETUNING_DIR / "turkishmedllm_sft_instruction_format.jsonl"
ARTICLE_DOCUMENTS_FILE = RAG_CORPUS_DIR / "turkishmedllm_article_documents.jsonl"
ALL_DOCUMENTS_FILE = DATA_PROCESSED_DIR / "turkishmedllm_all_ingested_documents.jsonl"
INGESTION_REPORT_FILE = LOGS_DIR / "ingestion_report.json"

# Source Mapping
SOURCE_MAPPING = {
    "source-1": "TUSGPT-TR Medical Dataset",
    "source-2": "Zenodo Turkish Medical Q&A Dataset",
    "source-3": "Turkish Hospital Medical Articles"
}

# Detection Keywords
QUESTION_COLUMNS = ["question", "questions", "soru", "sorular", "input", "prompt", "instruction", "user", "query", "hasta_sorusu", "Question", "Soru", "question_content"]
ANSWER_COLUMNS = ["answer", "answers", "cevap", "cevaplar", "output", "response", "assistant", "completion", "yanit", "yanıt", "Answer", "Cevap", "question_answer"]

# Files to Ignore (Generic)
IGNORE_FILES = ["source.txt", "data-source.txt", "dataset-source.txt", "README.md", ".DS_Store"]

# Dataset Selection Policy
DATASET_SELECTION_POLICY = {
    "source-1": "Use train/test/validation split files only; ignore FULL file to prevent duplicate counting.",
    "source-2": "Use Parquet file only; ignore CSV file because both contain the same records.",
    "source-3": "Use article JSON file."
}

# Source-specific ignore rules
SOURCE_IGNORE_RULES = {
    "source-1": {
        "patterns": ["FULL"],
        "reason": "Duplicate full version"
    },
    "source-2": {
        "extensions": [".csv"],
        "reason": "Duplicate CSV version"
    }
}

# Quality Thresholds
QUALITY_THRESHOLDS = {
    "qa": {
        "min_question_len": 10,
        "min_answer_len": 30,
        "min_combined_len": 50
    },
    "article": {
        "min_article_len": 200
    }
}

# Transformation Directories
DATA_TRANSFORMED_DIR = BASE_DIR / "data-transformed"
TRANSFORMED_FINETUNING_DIR = DATA_TRANSFORMED_DIR / "finetuning"
TRANSFORMED_RAG_DIR = DATA_TRANSFORMED_DIR / "rag-corpus"
TRANSFORMED_LOGS_DIR = DATA_TRANSFORMED_DIR / "logs"

# Transformation Output Files
SFT_CLEANED_FILE = TRANSFORMED_FINETUNING_DIR / "turkishmedllm_sft_cleaned.jsonl"
ARTICLE_CHUNKS_FILE = TRANSFORMED_RAG_DIR / "turkishmedllm_article_chunks.jsonl"
TRANSFORMATION_REPORT_FILE = TRANSFORMED_LOGS_DIR / "transformation_report.json"

# Vectorstore Directories
DATA_VECTORSTORE_DIR = BASE_DIR / "data-vectorstore"
CHROMA_DIR = DATA_VECTORSTORE_DIR / "chroma"
VECTORSTORE_LOGS_DIR = DATA_VECTORSTORE_DIR / "logs"

# Vectorstore Settings
EMBEDDING_BACKEND = "ollama"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"

VECTORSTORE_BACKEND = "chroma"
BATCH_SIZE = 500
CHROMA_PERSIST_DIR = CHROMA_DIR / "turkishmedllm_chroma_db"
VECTORSTORE_REPORT_FILE = VECTORSTORE_LOGS_DIR / "embedding_vectorstore_report.json"
VECTORSTORE_PROGRESS_FILE = VECTORSTORE_LOGS_DIR / "vectorstore_progress.json"

# Article Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    ""
]
CHUNK_MIN_LENGTH = 100

# SFT Instruction Template
SFT_SYSTEM_PROMPT = "Aşağıdaki tıbbi soruyu Türkçe olarak güvenli, anlaşılır ve klinik olarak dikkatli şekilde cevapla."

# Ensure directories exist
for directory in [
    FINETUNING_DIR, RAG_CORPUS_DIR, LOGS_DIR,
    TRANSFORMED_FINETUNING_DIR, TRANSFORMED_RAG_DIR, TRANSFORMED_LOGS_DIR,
    CHROMA_DIR, VECTORSTORE_LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# --- Step 4: Evaluation Configuration ---
EVALUATION_DIR = BASE_DIR / "data-evaluation"
RETRIEVER_EVAL_DIR = EVALUATION_DIR / "retriever"
RETRIEVER_TEST_QUERIES_FILE = RETRIEVER_EVAL_DIR / "retriever_test_queries.json"
RETRIEVER_RESULTS_FILE = RETRIEVER_EVAL_DIR / "retriever_results_top5.jsonl"
RETRIEVER_REPORT_FILE = RETRIEVER_EVAL_DIR / "retriever_evaluation_report.json"

# Create evaluation directories
RETRIEVER_EVAL_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 5: RAG Configuration ---
RAG_DIR = BASE_DIR / "data-rag"
RAG_OUTPUTS_DIR = RAG_DIR / "outputs"
RAG_LOGS_DIR = RAG_DIR / "logs"
RAG_DEMO_OUTPUTS_FILE = RAG_OUTPUTS_DIR / "rag_demo_outputs.jsonl"
RAG_REPORT_FILE = RAG_LOGS_DIR / "rag_run_report.json"

# RAG Model Settings
OLLAMA_CHAT_MODEL = "qwen3:8b"
RAG_TOP_K = 3  # Reduced for faster retrieval
RAG_TEMPERATURE = 0.1  # Very low for consistent fast responses
RAG_TIMEOUT = 60  # 60 second timeout for responses

# Create RAG directories
RAG_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
RAG_LOGS_DIR.mkdir(parents=True, exist_ok=True)
