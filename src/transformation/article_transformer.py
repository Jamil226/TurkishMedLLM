import json
from src.transformation.text_cleaner import TextCleaner
from src.transformation.text_splitter import TextSplitter
from src.utils.logger import logger
from src import config

class ArticleTransformer:
    def __init__(self):
        self.splitter = TextSplitter()
        self.stats = {
            "articles_processed": 0,
            "total_chunks": 0,
            "skipped_chunks": 0,
            "skipped_empty": 0,
            "skipped_short": 0,
            "skipped_non_alpha": 0,
            "chunks_per_article": [],
            "chunk_lengths": []
        }

    def transform_articles(self, input_file, output_file):
        """Loads, cleans, splits and saves article chunks."""
        logger.info(f"Starting Article Transformation: {input_file}")
        
        all_chunks = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    self.stats["articles_processed"] += 1
                    
                    content = doc.get("page_content", "")
                    metadata = doc.get("metadata", {})
                    
                    # 1. Clean text before splitting
                    clean_content = TextCleaner.clean_text(content, remove_html=True)
                    if not clean_content:
                        continue

                    # 2. Split into chunks
                    raw_chunks = self.splitter.split_text(clean_content)
                    
                    article_chunks_count = 0
                    for i, chunk_text in enumerate(raw_chunks):
                        # 3. Quality Filter
                        is_valid, reason = TextSplitter.is_quality_chunk(chunk_text)
                        if not is_valid:
                            self.stats["skipped_chunks"] += 1
                            if reason == "empty": self.stats["skipped_empty"] += 1
                            elif reason == "short": self.stats["skipped_short"] += 1
                            elif reason == "non_alpha": self.stats["skipped_non_alpha"] += 1
                            continue

                        # 4. Create chunk record
                        chunk_id = f"{metadata.get('source_id', 'unknown')}_{metadata.get('original_file', 'unknown')}_{metadata.get('row_index', '0')}_chunk_{i}"
                        
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            "dataset_type": "article_chunk",
                            "chunk_number": i,
                            "chunk_size": len(chunk_text),
                            "chunk_overlap": config.CHUNK_OVERLAP,
                            "intended_use": "rag_corpus",
                            "transformation_stage": "text_splitting"
                        })

                        chunk_record = {
                            "chunk_id": chunk_id,
                            "parent_document_id": f"{metadata.get('source_id')}_{metadata.get('original_file')}_{metadata.get('row_index')}",
                            "page_content": chunk_text,
                            "metadata": chunk_metadata
                        }
                        
                        all_chunks.append(chunk_record)
                        self.stats["total_chunks"] += 1
                        self.stats["chunk_lengths"].append(len(chunk_text))
                        article_chunks_count += 1
                    
                    self.stats["chunks_per_article"].append(article_chunks_count)
                except Exception as e:
                    logger.error(f"Error processing article record: {e}")

        # Save to output
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Article Transformation complete. Created {self.stats['total_chunks']} chunks from {self.stats['articles_processed']} articles.")
        return all_chunks

    def get_summary(self):
        import numpy as np
        return {
            "total_articles_processed": self.stats["articles_processed"],
            "total_article_chunks": self.stats["total_chunks"],
            "avg_chunks_per_article": float(np.mean(self.stats["chunks_per_article"])) if self.stats["chunks_per_article"] else 0,
            "min_chunks_per_article": int(np.min(self.stats["chunks_per_article"])) if self.stats["chunks_per_article"] else 0,
            "max_chunks_per_article": int(np.max(self.stats["chunks_per_article"])) if self.stats["chunks_per_article"] else 0,
            "avg_chunk_length": float(np.mean(self.stats["chunk_lengths"])) if self.stats["chunk_lengths"] else 0,
            "min_chunk_length": int(np.min(self.stats["chunk_lengths"])) if self.stats["chunk_lengths"] else 0,
            "max_chunk_length": int(np.max(self.stats["chunk_lengths"])) if self.stats["chunk_lengths"] else 0,
            "skipped_chunks": self.stats["skipped_chunks"],
            "skipped_empty": self.stats["skipped_empty"],
            "skipped_short": self.stats["skipped_short"],
            "skipped_non_alpha": self.stats["skipped_non_alpha"]
        }
