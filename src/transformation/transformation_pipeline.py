import json
from datetime import datetime
from src import config
from src.utils.logger import logger
from src.transformation.qa_transformer import QATransformer
from src.transformation.article_transformer import ArticleTransformer

class TransformationPipeline:
    def __init__(self):
        self.qa_transformer = QATransformer()
        self.article_transformer = ArticleTransformer()
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "input_files": {
                "sft": str(config.SFT_INSTRUCTION_FILE),
                "article": str(config.ARTICLE_DOCUMENTS_FILE)
            },
            "output_files": {
                "sft_cleaned": str(config.SFT_CLEANED_FILE),
                "article_chunks": str(config.ARTICLE_CHUNKS_FILE)
            },
            "chunking_configuration": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "min_length": config.CHUNK_MIN_LENGTH
            },
            "qa_stats": {},
            "article_stats": {},
            "errors": []
        }

    def run(self):
        logger.info("Starting Data Transformation and Text Splitting Pipeline...")

        # 1. Transform Q&A / SFT Records
        try:
            self.qa_transformer.transform_sft(
                config.SFT_INSTRUCTION_FILE, 
                config.SFT_CLEANED_FILE
            )
            self.report["qa_stats"] = self.qa_transformer.get_summary()
        except Exception as e:
            err = f"Error in Q&A transformation: {e}"
            logger.error(err)
            self.report["errors"].append(err)

        # 2. Transform Article Documents
        try:
            self.article_transformer.transform_articles(
                config.ARTICLE_DOCUMENTS_FILE,
                config.ARTICLE_CHUNKS_FILE
            )
            self.report["article_stats"] = self.article_transformer.get_summary()
        except Exception as e:
            err = f"Error in Article transformation: {e}"
            logger.error(err)
            self.report["errors"].append(err)

        # 3. Generate Report
        self.generate_report()
        self.print_summary()

    def generate_report(self):
        with open(config.TRANSFORMATION_REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        logger.info(f"Transformation report saved to {config.TRANSFORMATION_REPORT_FILE}")

    def print_summary(self):
        qa = self.report["qa_stats"]
        art = self.report["article_stats"]
        
        print("\n" + "="*50)
        print("Data transformation and text splitting completed.")
        print(f"SFT records cleaned: {qa.get('total_sft_cleaned', 0)}")
        print(f"Article documents processed: {art.get('total_articles_processed', 0)}")
        print(f"Article chunks created: {art.get('total_article_chunks', 0)}")
        print(f"Skipped chunks: {art.get('skipped_chunks', 0)}")
        print("\nOutput saved to:")
        print(f"- {config.SFT_CLEANED_FILE.relative_to(config.BASE_DIR)}")
        print(f"- {config.ARTICLE_CHUNKS_FILE.relative_to(config.BASE_DIR)}")
        print(f"- {config.TRANSFORMATION_REPORT_FILE.relative_to(config.BASE_DIR)}")
        print("="*50 + "\n")

if __name__ == "__main__":
    pipeline = TransformationPipeline()
    pipeline.run()
