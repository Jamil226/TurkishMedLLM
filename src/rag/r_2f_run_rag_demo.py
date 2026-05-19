import json
import argparse
import datetime
from pathlib import Path
from src import config
from src.rag.r_2c_rag_chain import TurkishMedRAG
from src.utils.logger import logger

DEMO_QUERIES = [
    "Diyabet hastalarında kan şekeri neden yükselir?",
    "Hipertansiyon belirtileri nelerdir?",
    "Kalp krizi belirtileri nelerdir?",
    "Astım atağı sırasında ne yapılmalıdır?",
    "Antibiyotik kullanırken nelere dikkat edilmelidir?"
]

def check_indexing_progress():
    progress_file = config.VECTORSTORE_PROGRESS_FILE
    if not progress_file.exists():
        return 0.0
    
    with open(progress_file, 'r') as f:
        progress = json.load(f)
        processed_count = len(progress.get("processed_ids", []))
    
    # Estimate total chunks from article_chunks file
    total_chunks = 0
    if config.ARTICLE_CHUNKS_FILE.exists():
        with open(config.ARTICLE_CHUNKS_FILE, 'r') as f:
            for _ in f: total_chunks += 1
            
    if total_chunks == 0: return 0.0
    return (processed_count / total_chunks) * 100

def main():
    parser = argparse.ArgumentParser(description="TurkishMedLLM RAG Demo")
    parser.add_argument("--allow_partial_index", action="store_true", help="Allow running on incomplete vector index")
    args = parser.parse_args()

    # 1. Check Progress
    progress_pct = check_indexing_progress()
    if progress_pct < 95.0:
        print("\n" + "!"*50)
        print(f"WARNING: Vector indexing appears incomplete ({progress_pct:.1f}%).")
        print("RAG results may be partial and inaccurate.")
        print("!"*50 + "\n")
        
        if not args.allow_partial_index:
            print("Please use --allow_partial_index to proceed or wait for indexing to complete.\n")
            return

    # 2. Initialize RAG
    rag = TurkishMedRAG()
    if not rag.llm:
        return

    # 3. Run Queries
    logger.info(f"Running RAG demo with {len(DEMO_QUERIES)} queries...")
    outputs = []
    
    for query in DEMO_QUERIES:
        print(f"\nQUERY: {query}")
        result = rag.answer(query)
        
        entry = {
            "query": query,
            "retrieved_chunk_ids": [doc.metadata.get("chunk_id") for doc in result["source_documents"]],
            "retrieved_titles": [doc.metadata.get("title") for doc in result["source_documents"]],
            "retrieved_urls": [doc.metadata.get("url") for doc in result["source_documents"]],
            "retrieved_contexts": [doc.page_content for doc in result["source_documents"]],
            "generated_answer": result["answer"],
            "model_name": rag.model_name,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        outputs.append(entry)
        with open(config.RAG_DEMO_OUTPUTS_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        print(f"ANSWER: {result['answer'][:200]}...")

    # 4. Generate Report
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_queries": len(DEMO_QUERIES),
        "model_used": rag.model_name,
        "indexing_progress_at_run": f"{progress_pct:.1f}%",
        "output_file": str(config.RAG_DEMO_OUTPUTS_FILE)
    }
    
    with open(config.RAG_REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"RAG demo completed. Results saved to {config.RAG_DEMO_OUTPUTS_FILE}")

if __name__ == "__main__":
    main()
