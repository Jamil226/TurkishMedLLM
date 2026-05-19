import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Fix the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.rag.r_2c_rag_chain import TurkishMedRAG

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

scenario = {
    "name": "FineTuned_Qwen3-8b",
    "use_rag": False,
    "use_safety_prompt": False,
    "use_finetuned": True
}

DEMO_QUERIES = [
    "Diyabet hastalarında kan şekeri neden yükselir?",
    "Hipertansiyon belirtileri nelerdir?",
    "Kalp krizi belirtileri nelerdir?",
    "Astım atağı sırasında ne yapılmalıdır?",
    "Antibiyotik kullanırken nelere dikkat edilmelidir?"
]

def generate_outputs(scenario):
    name = scenario["name"]
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "rag_outputs.jsonl"
    
    print(f"\n=============================================")
    print(f"Processing Scenario: {name}")
    print(f"=============================================")
    
    # Initialize RAG for this scenario
    rag = TurkishMedRAG(
        use_rag=scenario["use_rag"],
        use_safety_prompt=scenario["use_safety_prompt"],
        use_finetuned=scenario["use_finetuned"]
    )
    
    if not rag.llm:
        print(f"Failed to initialize LLM for {name}")
        return False
        
    outputs = []
    
    # Clear previous JSONL
    if out_jsonl.exists():
        out_jsonl.unlink()
        
    for query in DEMO_QUERIES:
        print(f"  Q: {query}")
        result = rag.answer(query)
        
        entry = {
            "query": query,
            "retrieved_chunk_ids": [doc.metadata.get("chunk_id") for doc in result["source_documents"]] if scenario["use_rag"] else [],
            "retrieved_titles": result["retrieved_titles"] if scenario["use_rag"] else [],
            "retrieved_urls": [doc.metadata.get("url") for doc in result["source_documents"]] if scenario["use_rag"] else [],
            "retrieved_contexts": [doc.page_content for doc in result["source_documents"]] if scenario["use_rag"] else [],
            "generated_answer": result["answer"],
            "model_name": name,
            "timestamp": datetime.now().isoformat()
        }
        
        outputs.append(entry)
        with open(out_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        print(f"  A: {result['answer'][:150]}...")
        
    return out_jsonl

def run_evaluations(jsonl_file, out_dir):
    print(f"Running Evaluations for {out_dir.name}...")
    
    # Run RAGAS and DeepEval
    cmd_rag = ["python", str(BASE_DIR / "3b_evaluate_rag.py"), "--input", str(jsonl_file), "--outdir", str(out_dir)]
    subprocess.run(cmd_rag)
    
    # Run OpenAI / Custom Grader (Local Qwen3)
    cmd_local = ["python", str(BASE_DIR / "3a_evaluate_openai.py"), "--input", str(jsonl_file), "--outdir", str(out_dir)]
    subprocess.run(cmd_local)
    
    # Convert original jsonl to rag_results.csv
    import csv
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    if data:
        with open(out_dir / "rag_results.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

if __name__ == "__main__":
    jsonl_file = generate_outputs(scenario)
    if jsonl_file:
        run_evaluations(jsonl_file, RESULTS_DIR / scenario["name"])
    print("Scenario generated and evaluated successfully!")
