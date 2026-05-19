#!/usr/bin/env python3
"""
Local DeepEval Alternative - No OpenAI Key Required
Uses semantic similarity and local metrics for evaluation
"""
import json
import csv
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.utils.logger import setup_logger

logger = setup_logger("LocalEvalMetrics")

# Try to import sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True

# Paths
RAG_OUTPUTS_FILE = BASE_DIR / "data-rag/outputs/rag_demo_outputs.jsonl"
OUTPUT_CSV = BASE_DIR / "rag_evaluation_deepeval.csv"

def load_rag_outputs():
    """Load RAG outputs from JSONL"""
    results = []
    with open(RAG_OUTPUTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results

def calculate_local_metrics(rag_outputs):
    """Calculate metrics using local semantic similarity (no API key needed)"""
    logger.info("🔍 Calculating local evaluation metrics...\n")
    logger.info("Loading semantic similarity model...")
    
    # Load sentence transformer model for semantic similarity
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    metrics_list = []
    
    for idx, output in enumerate(tqdm(rag_outputs, desc="Evaluation Progress"), 1):
        query = output.get('query', '')
        answer = output.get('generated_answer', '')
        retrieved_contexts = output.get('retrieved_contexts', [])
        
        # 1. Answer Relevancy Score (similarity between query and answer)
        query_embedding = model.encode(query, convert_to_tensor=True)
        answer_embedding = model.encode(answer, convert_to_tensor=True)
        relevancy_score = float(util.pytorch_cos_sim(query_embedding, answer_embedding)[0][0])
        
        # 2. Faithfulness Score (answer should be grounded in retrieved contexts)
        faithfulness_score = 0.0
        if retrieved_contexts:
            answer_embedding = model.encode(answer, convert_to_tensor=True)
            context_embeddings = model.encode(retrieved_contexts, convert_to_tensor=True)
            
            # Calculate max similarity between answer and any context
            similarities = util.pytorch_cos_sim(answer_embedding, context_embeddings)[0]
            faithfulness_score = float(max(similarities)) if len(similarities) > 0 else 0.0
        else:
            # If no contexts, use query-answer similarity as proxy
            faithfulness_score = relevancy_score * 0.8
        
        # Normalize to 0-1 range and round
        relevancy_score = round(max(0.0, min(1.0, relevancy_score)), 4)
        faithfulness_score = round(max(0.0, min(1.0, faithfulness_score)), 4)
        
        metric_record = {
            'query_id': idx,
            'answer_relevancy': relevancy_score,
            'faithfulness': faithfulness_score,
            'deepeval_timestamp': datetime.now().isoformat()
        }
        
        metrics_list.append(metric_record)
        
        logger.info(f"Query {idx}: Relevancy={relevancy_score:.4f}, Faithfulness={faithfulness_score:.4f}")
    
    return metrics_list

def save_metrics_to_csv(metrics):
    """Save metrics to CSV with proper formatting"""
    try:
        fieldnames = ['query_id', 'answer_relevancy', 'faithfulness', 'deepeval_timestamp']
        
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in metrics:
                writer.writerow({
                    'query_id': metric['query_id'],
                    'answer_relevancy': metric['answer_relevancy'],
                    'faithfulness': metric['faithfulness'],
                    'deepeval_timestamp': metric['deepeval_timestamp']
                })
        
        logger.info(f"\n✅ Evaluation metrics saved to: {OUTPUT_CSV}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving metrics: {e}")
        return False

def print_summary(metrics):
    """Print summary statistics"""
    if not metrics:
        logger.warning("No metrics to summarize")
        return
    
    logger.info("\n" + "="*70)
    logger.info("📊 Evaluation Metrics Summary")
    logger.info("="*70)
    
    relevancy_scores = [m['answer_relevancy'] for m in metrics if m['answer_relevancy'] is not None]
    faithfulness_scores = [m['faithfulness'] for m in metrics if m['faithfulness'] is not None]
    
    if relevancy_scores:
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
        max_relevancy = max(relevancy_scores)
        min_relevancy = min(relevancy_scores)
        logger.info(f"\nAnswer Relevancy (Query-Answer Similarity):")
        logger.info(f"  Average:   {avg_relevancy:.4f}")
        logger.info(f"  Max:       {max_relevancy:.4f}")
        logger.info(f"  Min:       {min_relevancy:.4f}")
    
    if faithfulness_scores:
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        max_faithfulness = max(faithfulness_scores)
        min_faithfulness = min(faithfulness_scores)
        logger.info(f"\nFaithfulness (Answer-Context Grounding):")
        logger.info(f"  Average:   {avg_faithfulness:.4f}")
        logger.info(f"  Max:       {max_faithfulness:.4f}")
        logger.info(f"  Min:       {min_faithfulness:.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("Score Interpretation (0.0-1.0):")
    logger.info("  0.0-0.3: Low similarity/grounding")
    logger.info("  0.3-0.6: Moderate")
    logger.info("  0.6-0.8: Good")
    logger.info("  0.8-1.0: Excellent")
    logger.info("="*70 + "\n")

def main():
    """Main execution"""
    logger.info("🚀 Local Evaluation Metrics Calculation\n")
    logger.info("Method: Semantic Similarity (No API Key Required)")
    logger.info("Model: paraphrase-MiniLM-L6-v2\n")
    
    # Load RAG outputs
    logger.info(f"📂 Loading RAG outputs from {RAG_OUTPUTS_FILE}...")
    rag_outputs = load_rag_outputs()
    
    if not rag_outputs:
        logger.error(f"❌ No RAG outputs found at {RAG_OUTPUTS_FILE}")
        return
    
    logger.info(f"✅ Loaded {len(rag_outputs)} RAG results\n")
    
    # Calculate metrics
    metrics = calculate_local_metrics(rag_outputs)
    
    # Save to CSV
    if save_metrics_to_csv(metrics):
        # Print summary
        print_summary(metrics)
        logger.info("✅ Evaluation metrics complete!\n")
        
        # Display CSV content
        logger.info("📋 Generated CSV Preview:")
        logger.info("-" * 70)
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:5]:  # Show first 5 lines
                logger.info(line.strip())
        if len(lines) > 5:
            logger.info(f"... and {len(lines)-5} more rows")
        logger.info("-" * 70)
    else:
        logger.error("❌ Failed to save metrics")

if __name__ == "__main__":
    main()
