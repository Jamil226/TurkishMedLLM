import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src import config
from src.utils.logger import logger
from src.vectorstore.embedding_loader import EmbeddingLoader

class VectorstorePipeline:
    def __init__(self, max_chunks=None):
        self.max_chunks = max_chunks
        self.embeddings = EmbeddingLoader.load_embedding_model()
        self.vectorstore = None
        self.progress = self._load_progress()
        self.report = {
            "embedding_backend": config.EMBEDDING_BACKEND,
            "embedding_model": config.OLLAMA_EMBEDDING_MODEL,
            "vectorstore_backend": config.VECTORSTORE_BACKEND,
            "total_chunks_loaded": 0,
            "total_valid_chunks": 0,
            "total_embedded_chunks": 0,
            "skipped_chunks": 0,
            "duplicate_ids_skipped": 0,
            "batch_size": config.BATCH_SIZE,
            "output_path": str(config.CHROMA_PERSIST_DIR),
            "errors": []
        }

    def _load_progress(self):
        if config.VECTORSTORE_PROGRESS_FILE.exists():
            with open(config.VECTORSTORE_PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return {"processed_ids": []}

    def run(self):
        logger.info("Starting Vector Store Pipeline...")
        
        # 1. Initialize Chroma
        self.vectorstore = Chroma(
            persist_directory=str(config.CHROMA_PERSIST_DIR),
            embedding_function=self.embeddings
        )

        # 2. Load and filter chunks
        documents = []
        doc_ids = []
        
        processed_set = set(self.progress["processed_ids"])
        
        print(f"\nExisting processed IDs found: {len(processed_set)}")
        logger.info(f"Loading chunks from {config.ARTICLE_CHUNKS_FILE}")
        
        with open(config.ARTICLE_CHUNKS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if self.max_chunks and self.report["total_chunks_loaded"] >= self.max_chunks:
                    break
                
                self.report["total_chunks_loaded"] += 1
                try:
                    data = json.loads(line)
                    chunk_id = data.get("chunk_id")
                    
                    if chunk_id in processed_set:
                        self.report["duplicate_ids_skipped"] += 1
                        continue
                    
                    if not data.get("page_content"):
                        self.report["skipped_chunks"] += 1
                        continue

                    # Create LangChain Document
                    doc = Document(
                        page_content=data["page_content"],
                        metadata=data["metadata"]
                    )
                    # Ensure chunk_id and parent_document_id are in metadata if not already
                    doc.metadata["chunk_id"] = chunk_id
                    doc.metadata["parent_document_id"] = data.get("parent_document_id")
                    
                    documents.append(doc)
                    doc_ids.append(chunk_id)
                    self.report["total_valid_chunks"] += 1
                    
                except Exception as e:
                    self.report["errors"].append(f"Error loading chunk: {e}")

        remaining_count = len(documents)
        print(f"Remaining chunks to process: {remaining_count}\n")

        if not documents:
            logger.info("No new documents to embed.")
            self.generate_report()
            return

        # 3. Batch insert into Chroma
        logger.info(f"Embedding {remaining_count} new documents in batches of {config.BATCH_SIZE}...")
        
        for i in tqdm(range(0, len(documents), config.BATCH_SIZE)):
            batch_docs = documents[i : i + config.BATCH_SIZE]
            batch_ids = doc_ids[i : i + config.BATCH_SIZE]
            
            try:
                self.vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                self.report["total_embedded_chunks"] += len(batch_docs)
                # Update local progress and save
                self.progress["processed_ids"].extend(batch_ids)
                self._save_progress()
            except Exception as e:
                err_msg = f"Error in batch {i//config.BATCH_SIZE}: {e}"
                logger.error(err_msg)
                self.report["errors"].append(err_msg)

        self.generate_report()
        self.print_summary()

    def generate_report(self):
        with open(config.VECTORSTORE_REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        logger.info(f"Vector store report saved to {config.VECTORSTORE_REPORT_FILE}")

    def print_summary(self):
        print("\n" + "="*50)
        print("Vector store construction completed.")
        print(f"Chunks loaded: {self.report['total_chunks_loaded']}")
        print(f"New chunks embedded: {self.report['total_embedded_chunks']}")
        print(f"Duplicates skipped: {self.report['duplicate_ids_skipped']}")
        print(f"Errors encountered: {len(self.report['errors'])}")
        print(f"ChromaDB location: {config.CHROMA_PERSIST_DIR}")
        print("="*50 + "\n")

    def _save_progress(self):
        """Saves current processed IDs to the progress file."""
        # Ensure uniqueness
        self.progress["processed_ids"] = list(set(self.progress["processed_ids"]))
        with open(config.VECTORSTORE_PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurkishMedLLM Vector Store Pipeline")
    parser.add_argument("--max_chunks", type=int, help="Maximum number of chunks to process (for testing)")
    args = parser.parse_args()

    pipeline = VectorstorePipeline(max_chunks=args.max_chunks)
    pipeline.run()
