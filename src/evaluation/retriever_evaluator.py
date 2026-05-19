import json
import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from src import config
from src.utils.logger import logger
from src.vectorstore.embedding_loader import EmbeddingLoader

class RetrieverEvaluator:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.embeddings = EmbeddingLoader.load_embedding_model()
        self.vectorstore = Chroma(
            persist_directory=str(config.CHROMA_PERSIST_DIR),
            embedding_function=self.embeddings
        )
        self.results = []
        self.metrics = {
            "total_queries": 0,
            "top_k": top_k,
            "hit_rate_at_1": 0.0,
            "hit_rate_at_3": 0.0,
            "hit_rate_at_5": 0.0,
            "average_relevant_results_per_query": 0.0,
            "average_partially_relevant_results_per_query": 0.0,
            "average_irrelevant_results_per_query": 0.0,
            "total_relevant_results": 0,
            "total_partially_relevant_results": 0,
            "total_irrelevant_results": 0,
            "note": "Evaluation may be partial if full vector indexing is still in progress."
        }

    def load_test_queries(self) -> List[Dict[str, Any]]:
        with open(config.RETRIEVER_TEST_QUERIES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate_chunk(self, chunk_text: str, title: str, expected_keywords: List[str]) -> str:
        text_lower = (chunk_text + " " + title).lower()
        matches = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
        
        if matches >= 2:
            return "relevant"
        elif matches == 1:
            return "partially_relevant"
        else:
            return "irrelevant"

    def run_evaluation(self):
        queries = self.load_test_queries()
        self.metrics["total_queries"] = len(queries)
        
        logger.info(f"Starting retriever evaluation for {len(queries)} queries...")
        
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        
        total_rel = 0
        total_part_rel = 0
        total_irrel = 0

        with open(config.RETRIEVER_RESULTS_FILE, 'w', encoding='utf-8') as f_out:
            for q_data in queries:
                query_id = q_data["query_id"]
                query_text = q_data["query"]
                expected_keywords = q_data["expected_keywords"]
                
                # Retrieve with scores
                results_with_scores = self.vectorstore.similarity_search_with_score(query_text, k=self.top_k)
                
                query_hits = []
                for rank, (doc, score) in enumerate(results_with_scores, 1):
                    relevance = self.evaluate_chunk(doc.page_content, doc.metadata.get("title", ""), expected_keywords)
                    
                    result_entry = {
                        "query_id": query_id,
                        "query": query_text,
                        "topic": q_data["topic"],
                        "rank": rank,
                        "chunk_id": doc.metadata.get("chunk_id"),
                        "title": doc.metadata.get("title"),
                        "url": doc.metadata.get("url"),
                        "similarity_score": float(score),
                        "content_preview": doc.page_content[:500],
                        "relevance": relevance,
                        "metadata": doc.metadata
                    }
                    
                    f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                    query_hits.append(relevance)
                    
                    if relevance in ["relevant", "partially_relevant"]:
                        if rank <= 1: hits_at_1 += 1
                        if rank <= 3: hits_at_3 += 1
                        if rank <= 5: hits_at_5 += 1
                        
                    if relevance == "relevant": total_rel += 1
                    elif relevance == "partially_relevant": total_part_rel += 1
                    else: total_irrel += 1

                # Correct Hit Rate counting: a query is a hit if ANY result in top-k is relevant/partial
                # The above hits_at_k logic counts PER result, which is not correct for Hit Rate @ K
                # Let's fix it by checking if any of the top-k for this query was a hit.
                
                q_rel_indices = [i for i, r in enumerate(query_hits) if r in ["relevant", "partially_relevant"]]
                if q_rel_indices:
                    min_rank = min(q_rel_indices) + 1
                    # This logic below is still slightly wrong if we don't reset per query.
                    # I'll rewrite the hit rate calculation.

        # Correct Hit Rate calculation
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        
        # Reloading results to calculate per-query hit rates properly
        with open(config.RETRIEVER_RESULTS_FILE, 'r', encoding='utf-8') as f_in:
            query_results = {}
            for line in f_in:
                data = json.loads(line)
                qid = data["query_id"]
                if qid not in query_results:
                    query_results[qid] = []
                query_results[qid].append(data["relevance"])
            
            for qid, rels in query_results.items():
                if any(r in ["relevant", "partially_relevant"] for r in rels[:1]):
                    hits_at_1 += 1
                if any(r in ["relevant", "partially_relevant"] for r in rels[:3]):
                    hits_at_3 += 1
                if any(r in ["relevant", "partially_relevant"] for r in rels[:5]):
                    hits_at_5 += 1

        self.metrics["hit_rate_at_1"] = round(hits_at_1 / self.metrics["total_queries"], 4)
        self.metrics["hit_rate_at_3"] = round(hits_at_3 / self.metrics["total_queries"], 4)
        self.metrics["hit_rate_at_5"] = round(hits_at_5 / self.metrics["total_queries"], 4)
        
        self.metrics["total_relevant_results"] = total_rel
        self.metrics["total_partially_relevant_results"] = total_part_rel
        self.metrics["total_irrelevant_results"] = total_irrel
        
        self.metrics["average_relevant_results_per_query"] = round(total_rel / self.metrics["total_queries"], 2)
        self.metrics["average_partially_relevant_results_per_query"] = round(total_part_rel / self.metrics["total_queries"], 2)
        self.metrics["average_irrelevant_results_per_query"] = round(total_irrel / self.metrics["total_queries"], 2)

        self.save_report()
        self.print_summary()

    def save_report(self):
        with open(config.RETRIEVER_REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Retriever evaluation report saved to {config.RETRIEVER_REPORT_FILE}")

    def print_summary(self):
        print("\n" + "="*50)
        print("Retriever evaluation completed.")
        print(f"Queries evaluated: {self.metrics['total_queries']}")
        print(f"Top-k: {self.metrics['top_k']}")
        print(f"Hit rate@1: {self.metrics['hit_rate_at_1']}")
        print(f"Hit rate@3: {self.metrics['hit_rate_at_3']}")
        print(f"Hit rate@5: {self.metrics['hit_rate_at_5']}")
        print("\nResults saved to:")
        print(f"- {config.RETRIEVER_RESULTS_FILE}")
        print(f"- {config.RETRIEVER_REPORT_FILE}")
        print("="*50 + "\n")
