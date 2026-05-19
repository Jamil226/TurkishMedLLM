import json
import os
from pathlib import Path
from tqdm import tqdm
from src import config
from src.utils.logger import logger
from src.ingestion.data_loader import DataLoader
from src.ingestion.schema_detector import SchemaDetector
from src.ingestion.quality_checker import QualityChecker
from src.ingestion.document_converter import DocumentConverter

class IngestionPipeline:
    def __init__(self):
        self.qa_documents = []
        self.sft_records = []
        self.article_documents = []
        self.all_documents = []
        
        # Global deduplication sets
        self.processed_qa_hashes = set()
        self.processed_article_hashes = set()
        
        self.report = {
            "dataset_selection_policy": config.DATASET_SELECTION_POLICY,
            "quality_thresholds_used": config.QUALITY_THRESHOLDS,
            "total_files_processed": 0,
            "total_qa_documents": 0,
            "total_article_documents": 0,
            "total_documents": 0,
            "global_duplicate_qa_count": 0,
            "global_duplicate_article_count": 0,
            "low_quality_qa_count": 0,
            "low_quality_article_count": 0,
            "total_skipped_records": 0,
            "duplicate_count": 0,
            "per_source_stats": {},
            "per_file_stats": {},
            "selected_files": [],
            "ignored_files": [],
            "errors": []
        }

    def run(self):
        logger.info("Starting Data Ingestion Pipeline...")
        
        # Collect all files and apply selection policy
        files_to_process = []
        for source_id in ["source-1", "source-2", "source-3"]:
            source_path = config.DATA_INGESTION_DIR / source_id
            if not source_path.exists():
                logger.warning(f"Source path {source_path} does not exist. Skipping.")
                continue
            
            ignore_rules = config.SOURCE_IGNORE_RULES.get(source_id, {})
            
            for file in source_path.iterdir():
                if not file.is_file():
                    continue
                
                # 1. Generic ignore
                if file.name in config.IGNORE_FILES:
                    self.report["ignored_files"].append({
                        "file": str(file.relative_to(config.BASE_DIR)),
                        "reason": "Metadata file"
                    })
                    continue
                
                # 2. Source-specific ignore rules
                is_ignored = False
                for pattern in ignore_rules.get("patterns", []):
                    if pattern in file.name:
                        is_ignored = True
                        break
                
                for ext in ignore_rules.get("extensions", []):
                    if file.suffix.lower() == ext.lower():
                        is_ignored = True
                        break
                
                if is_ignored:
                    self.report["ignored_files"].append({
                        "file": str(file.relative_to(config.BASE_DIR)),
                        "reason": ignore_rules.get("reason", "Policy-based ignore")
                    })
                else:
                    files_to_process.append(file)
                    self.report["selected_files"].append(str(file.relative_to(config.BASE_DIR)))

        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                self.process_file(file_path)
            except Exception as e:
                error_msg = f"Critical error processing {file_path}: {e}"
                logger.error(error_msg)
                self.report["errors"].append(error_msg)

        # Save results
        self.save_outputs()
        self.generate_report()
        self.print_summary()

    def process_file(self, file_path: Path):
        logger.info(f"File started: {file_path.name}")
        
        # 1. Load Data
        df = DataLoader.load(file_path)
        if df is None or df.empty:
            logger.warning(f"Skipping empty or invalid file: {file_path}")
            return

        # 2. Detect Metadata & Schema
        source_id, source_name = SchemaDetector.detect_source(file_path)
        split = SchemaDetector.detect_split(file_path)
        dataset_type = SchemaDetector.detect_dataset_type(source_id)
        
        logger.info(f"Source detected: {source_name} ({source_id})")
        logger.info(f"Split detected: {split}")
        logger.info(f"Dataset type: {dataset_type}")
        logger.info(f"Available columns: {list(df.columns)}")
        
        question_col, answer_col = (None, None)
        text_col, title_col = (None, None)
        detection_method = "direct"

        if dataset_type == "qa":
            question_col, answer_col, detection_method = SchemaDetector.detect_qa_columns(df, df.columns)
            logger.info(f"QA Schema detected ({detection_method}): Q={question_col}, A={answer_col}")
            if not question_col or not answer_col:
                logger.warning(f"Incomplete QA schema for {file_path}.")
        elif dataset_type == "article":
            text_col, title_col = SchemaDetector.detect_article_columns(df.columns)
            logger.info(f"Article Schema detected: Text={text_col}, Title={title_col}")

        # 3. Quality Check & Cleaning (Local deduplication and null checks)
        checker = QualityChecker()
        df = checker.process_df(df, dataset_type, question_col, answer_col, text_col)
        
        # 4. Conversion & Global Deduplication
        file_qa_count = 0
        file_article_count = 0
        file_global_dup_count = 0
        
        for idx, row in df.iterrows():
            metadata_base = {
                "source_id": source_id,
                "source_name": source_name,
                "split": split,
                "original_file": file_path.name,
                "row_index": idx
            }
            
            if dataset_type == "qa":
                q_text = str(row[question_col]).strip().lower()
                a_text = str(row[answer_col]).strip().lower()
                qa_hash = f"{q_text} || {a_text}"
                
                if qa_hash in self.processed_qa_hashes:
                    file_global_dup_count += 1
                    self.report["global_duplicate_qa_count"] += 1
                    continue
                
                self.processed_qa_hashes.add(qa_hash)
                
                doc = DocumentConverter.to_qa_document(row, question_col, answer_col, metadata_base)
                sft = DocumentConverter.to_sft_format(row, question_col, answer_col, metadata_base)
                self.qa_documents.append(doc)
                self.sft_records.append(sft)
                self.all_documents.append(doc)
                file_qa_count += 1
                checker.update_lengths([str(row[question_col]), str(row[answer_col])])
                
            elif dataset_type == "article":
                art_text = str(row[text_col]).strip().lower()
                if art_text in self.processed_article_hashes:
                    file_global_dup_count += 1
                    self.report["global_duplicate_article_count"] += 1
                    continue
                
                self.processed_article_hashes.add(art_text)
                
                doc = DocumentConverter.to_article_document(row, text_col, title_col, metadata_base)
                self.article_documents.append(doc)
                self.all_documents.append(doc)
                file_article_count += 1
                checker.update_lengths([str(row[text_col])])

        # Update stats
        file_stats = checker.get_file_summary()
        file_stats["global_duplicates_skipped"] = file_global_dup_count
        file_stats["detected_schema"] = {
            "question_col": question_col,
            "answer_col": answer_col,
            "text_col": text_col,
            "title_col": title_col,
            "detection_method": detection_method,
            "all_columns": list(df.columns)
        }
        self.report["per_file_stats"][file_path.name] = file_stats
        
        if source_id not in self.report["per_source_stats"]:
            self.report["per_source_stats"][source_id] = {"qa": 0, "article": 0, "skipped": 0, "low_quality": 0}
        self.report["per_source_stats"][source_id]["qa"] += file_qa_count
        self.report["per_source_stats"][source_id]["article"] += file_article_count
        self.report["per_source_stats"][source_id]["skipped"] += file_stats["skipped_records"] + file_global_dup_count
        self.report["per_source_stats"][source_id]["low_quality"] += file_stats["low_quality_records"]

        if dataset_type == "qa":
            self.report["low_quality_qa_count"] += file_stats["low_quality_records"]
        else:
            self.report["low_quality_article_count"] += file_stats["low_quality_records"]

        self.report["total_files_processed"] += 1
        self.report["total_qa_documents"] += file_qa_count
        self.report["total_article_documents"] += file_article_count
        self.report["total_documents"] += (file_qa_count + file_article_count)
        self.report["total_skipped_records"] += file_stats["skipped_records"] + file_global_dup_count
        self.report["duplicate_count"] += file_stats["duplicate_records"]

        logger.info(f"File completed: {file_path.name}. Created {file_qa_count} QA and {file_article_count} Article docs.")

    def save_outputs(self):
        def write_jsonl(data, path):
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    if hasattr(item, 'to_json'): # LangChain Document
                        json.dump({
                            "page_content": item.page_content,
                            "metadata": item.metadata
                        }, f, ensure_ascii=False)
                    else:
                        json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

        logger.info("Saving processed outputs...")
        write_jsonl(self.qa_documents, config.QA_DOCUMENTS_FILE)
        write_jsonl(self.sft_records, config.SFT_INSTRUCTION_FILE)
        write_jsonl(self.article_documents, config.ARTICLE_DOCUMENTS_FILE)
        write_jsonl(self.all_documents, config.ALL_DOCUMENTS_FILE)

    def generate_report(self):
        with open(config.INGESTION_REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        logger.info(f"Ingestion report saved to {config.INGESTION_REPORT_FILE}")

    def print_summary(self):
        print("\n" + "="*50)
        print("Data ingestion completed.")
        print(f"Selected files processed: {self.report['total_files_processed']}")
        print(f"Ignored files: {len(self.report['ignored_files'])}")
        print(f"Q&A documents created: {self.report['total_qa_documents']}")
        print(f"Article documents created: {self.report['total_article_documents']}")
        print(f"Global duplicate Q&A records skipped: {self.report['global_duplicate_qa_count']}")
        print(f"Global duplicate article records skipped: {self.report['global_duplicate_article_count']}")
        print(f"Low quality Q&A records skipped: {self.report['low_quality_qa_count']}")
        print(f"Low quality article records skipped: {self.report['low_quality_article_count']}")
        print(f"Total documents created: {self.report['total_documents']}")
        print(f"SFT instruction records created: {len(self.sft_records)}")
        print("\nOutput saved to:")
        print(f"- {config.QA_DOCUMENTS_FILE}")
        print(f"- {config.SFT_INSTRUCTION_FILE}")
        print(f"- {config.ARTICLE_DOCUMENTS_FILE}")
        print(f"- {config.ALL_DOCUMENTS_FILE}")
        print(f"- {config.INGESTION_REPORT_FILE}")
        print("="*50 + "\n")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
