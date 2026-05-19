# RAG Pipeline Module

## Overview

The RAG (Retrieval-Augmented Generation) pipeline combines semantic search with a language model to answer medical questions using a curated knowledge base. This module handles the complete end-to-end flow from query to response generation with built-in safety constraints.

---

## Architecture

### Pipeline Flow

```
User Query
    ↓
[Retriever] → Fetch top-5 most relevant chunks from ChromaDB
    ↓
[Re-ranker] → Rank retrieved chunks by relevance
    ↓
[Context Builder] → Assemble context window for LLM
    ↓
[Safety Checker] → Validate no unsafe patterns in context
    ↓
[Prompt Template] → Inject context into medical-specific prompt
    ↓
[Generator] → qwen3:8b generates response with constraints
    ↓
[Post-processor] → Add disclaimers, safety warnings
    ↓
Response with Retrieved Sources
```

### Components

#### 1. **Retriever** (`retriever.py`)
Handles semantic search against ChromaDB vector store.

**Key Features:**
- Uses `bge-m3` embeddings (1024-dimensional)
- Returns top-K most similar chunks (default K=5)
- Preserves metadata (source, section, document ID)
- No external API dependencies

**Usage:**
```python
from src.rag.retriever import TurkishMedRetriever

retriever = TurkishMedRetriever()
results = retriever.retrieve(query="Hipertansiyon tedavisi", top_k=5)
# Returns: [{"content": "...", "metadata": {...}, "score": 0.92}, ...]
```

#### 2. **RAG Chain** (`rag_chain.py`)
Orchestrates the full RAG pipeline with safety constraints.

**Key Features:**
- Combines retriever + LLM generator
- Enforces medical safety prompts (no diagnosis, medication dosage, etc.)
- Tracks retrieved source information
- Generates detailed logs with traceability

**Usage:**
```python
from src.rag.rag_chain import TurkishMedRAG

rag = TurkishMedRAG()
response = rag.generate(
    query="Çocuklarda ateşin nedenleri nelerdir?",
    top_k=5,
    allow_partial_index=False
)
# Returns: {"answer": "...", "sources": [...], "safety_check": True}
```

#### 3. **Prompt Templates** (`prompt_templates.py`)
Defines medical-specific system and user prompts.

**System Prompt Focus:**
- Instructs model to provide informative medical content
- Mandates safety disclaimers
- Requires references to sources
- Prevents definitive diagnoses and medication dosages

**Example:**
```
You are a helpful Turkish medical information assistant. Your role is to provide
accurate, evidence-based medical information using the context provided below.

CRITICAL SAFETY REQUIREMENTS:
- DO NOT provide definitive diagnoses
- DO NOT recommend specific medications or dosages
- DO NOT replace professional medical consultation
- ALWAYS include appropriate medical disclaimers
- ALWAYS cite your sources from the provided context
```

#### 4. **Run RAG Demo** (`run_rag_demo.py`)
Executes the RAG pipeline on a predefined set of 6 medical test queries.

**Features:**
- 6 example Turkish medical queries (diverse specialties)
- Detailed logging of each step
- Saves results to JSONL format with full traceability
- Optional flag to allow partial index (<95% complete)

**Usage:**
```bash
python -m src.rag.run_rag_demo
# Or with partial index:
python -m src.rag.run_rag_demo --allow_partial_index
```

**Output:** `data-rag/outputs/rag_demo_outputs.jsonl`

---

## Safety Constraints

The RAG pipeline implements multi-layer safety checks:

### 1. **Prompt-Level Safety**
- System prompt explicitly instructs against diagnosis and dosage recommendations
- Requires inclusion of medical disclaimers in all responses
- Enforces reference to provided sources

### 2. **Content Validation**
- Screens context for harmful medical advice
- Blocks generation if dangerous patterns detected
- Logs all safety violations for audit

### 3. **Post-Processing**
- Appends standardized medical disclaimer to all responses
- Verifies response doesn't make definitive medical claims
- Ensures emergency/professional consultation recommendations

### Example Response with Safety:
```
[Generated Response]
Ateş (hipertermia), vücudun sıcaklık dengesinin bozulması sonucu ortaya çıkar. 
Çocuklarda yaygın nedenleri arasında enfeksiyonlar, aşılar, diş çıkması ve 
otoimmün hastalıklar bulunur.

[Automated Safety Disclaimer]
⚠️ ÖNEMLI: Bu bilgi eğitim amaçlıdır ve profesyonel tıbbi tavsiye değildir. 
Çocuğunuzda ateş varsa, lütfen bir doktor veya sağlık uzmanına danışınız.
```

---

## Configuration

All RAG settings defined in `src/config.py`:

```python
# Retrieval
RAG_TOP_K = 5                           # Number of chunks to retrieve
CHUNK_SIZE = 1000                       # Size of indexed chunks
CHUNK_OVERLAP = 150                     # Overlap between chunks

# Ollama Models
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"
OLLAMA_CHAT_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Vector Store
VECTORSTORE_DB_PATH = Path(BASE_DIR) / "data-vectorstore/chroma/turkishmedllm_chroma_db"
```

---

## Example Queries (from Demo)

The RAG demo includes these 6 medical queries for testing:

1. **Hipertansiyon (Hypertension)**
   - "Yüksek tansiyon nedir ve tedavisi nasıl yapılır?"

2. **Çocuk Sağlığı (Pediatrics)**
   - "Çocuklarda ateşin nedenleri nelerdir?"

3. **Kardiyoloji (Cardiology)**
   - "Kalp krizi belirtileri ve acil müdahale nedir?"

4. **Neuroloji (Neurology)**
   - "Migren nedir ve nasıl tedavi edilir?"

5. **Enfeksiyon Hastalıkları (Infectious Diseases)**
   - "COVID-19 virüsünün belirtileri nelerdir?"

6. **Dermatoloji (Dermatology)**
   - "Ciltte kaşıntı sebepleri neler olabilir?"

---

## Evaluation Metrics

### RAG Performance Metrics

**Retriever Quality:**
- Hit Rate @ 1: ~93% (first result is relevant)
- Hit Rate @ 5: 100% (one of top-5 is relevant)
- Semantic relevance score: 0-1 range

**Generator Quality:**
- Answer relevancy: 0.6591 (avg, based on query-answer similarity)
- Faithfulness: 0.9050 (avg, based on answer-source grounding)
- Response length: avg 2,154 chars per query
- Safety compliance: 100% (all responses include disclaimers)

**Query Examples:**
```json
{
  "query_id": 1,
  "query": "Yüksek tansiyon nedir ve tedavisi nasıl yapılır?",
  "answer": "Hipertansiyon, sistolik basıncın 140 mmHg veya daha yüksek...",
  "retrieved_sources": 5,
  "retrieved_titles": ["Hipertansiyon Tedavisi", "Kardiyoloji Temel Bilgiler", ...],
  "answer_relevancy": 0.568,
  "faithfulness": 0.9014
}
```

---

## Outputs

### RAG Demo Output Format

File: `data-rag/outputs/rag_demo_outputs.jsonl`

**Structure:**
```json
{
  "query_id": 1,
  "query": "Original medical question in Turkish",
  "generated_answer": "Full LLM-generated response with safety disclaimers",
  "retrieved_contexts": [
    "First retrieved context chunk...",
    "Second retrieved context chunk...",
    ...
  ],
  "retrieved_metadata": [
    {
      "source": "TUSGPT_TR_Medical_FULL.json",
      "section": "Kardiyoloji",
      "doc_id": "doc_001"
    },
    ...
  ],
  "model_used": "qwen3:8b",
  "timestamp": "2026-05-11T14:15:30.123456",
  "top_retrieved_title": "Primary source document title"
}
```

### Evaluation Output Files

Related outputs in root directory:
- `rag_results.csv` - Simplified results for analysis
- `rag_evaluation_basic_metrics.csv` - Query/answer statistics
- `rag_evaluation_deepeval.csv` - Semantic metrics (relevancy, faithfulness)
- `rag_evaluation_summary.csv` - Aggregated statistics

---

## Troubleshooting

### Common Issues

#### 1. **"Vector store not found" error**
```
Solution: Run vectorstore pipeline first
python -m src.vectorstore.vectorstore_pipeline
```

#### 2. **"Ollama connection refused"**
```
Solution: Start Ollama server
ollama serve
```

#### 3. **Low answer relevancy scores**
```
Possible causes:
- Retriever missing relevant context (add more sources)
- Query too specific (broaden the question)
- LLM not understanding Turkish medical terminology

Solution: Test with different top_k values or update prompt template
```

#### 4. **Timeout during generation**
```
Solution: Increase timeout in src/config.py
OLLAMA_REQUEST_TIMEOUT = 120  # seconds
```

---

## Performance Monitoring

### Check RAG Status

```bash
# View latest demo output
tail -f data-rag/outputs/rag_demo_outputs.jsonl

# Monitor retrieval performance
cat data-evaluation/retriever/retriever_evaluation_report.json

# Check RAG logs
tail -f data-rag/logs/*.log
```

### Metrics Dashboard

Run Streamlit UI for interactive testing:
```bash
streamlit run streamlit_rag_ui.py
```

---

## Integration with Fine-Tuning

After QLoRA fine-tuning, the generator can be updated:

```python
# Use fine-tuned model (if integrated with Ollama)
rag = TurkishMedRAG(model="qwen3:8b-finetuned")

# Or compare before/after metrics
results_baseline = rag.generate(query)
results_finetuned = rag_finetuned.generate(query)
```

Expected improvements:
- Answer relevancy: +15-25%
- Faithfulness: +10-20%
- Medical accuracy: +20-30% (qualitative)

---

## Key Takeaways

- ✅ **End-to-end RAG** with retriever + generator
- ✅ **Medical safety** constraints at multiple layers
- ✅ **Full traceability** of sources and decisions
- ✅ **Easy evaluation** with built-in metrics
- ✅ **Production-ready** with comprehensive logging

---

**Module Version:** 1.0  
**Last Updated:** May 11, 2026  
**Status:** ✅ Production-Ready
