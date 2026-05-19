import json
from src.transformation.text_cleaner import TextCleaner
from src.utils.logger import logger

class QATransformer:
    def __init__(self):
        self.stats = {
            "total_loaded": 0,
            "total_cleaned": 0,
            "total_skipped": 0,
            "input_lengths": [],
            "output_lengths": []
        }

    def transform_sft(self, input_file, output_file):
        """Loads, cleans, and saves SFT records."""
        logger.info(f"Starting Q&A Transformation: {input_file}")
        
        cleaned_records = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.stats["total_loaded"] += 1
                try:
                    record = json.loads(line)
                    # Extract fields
                    system = record.get("system", "")
                    instruction = record.get("instruction", "")
                    output = record.get("output", "")
                    metadata = record.get("metadata", {})

                    # Clean fields
                    clean_instruction = TextCleaner.clean_text(instruction)
                    clean_output = TextCleaner.clean_text(output)

                    if not clean_instruction or not clean_output:
                        self.stats["total_skipped"] += 1
                        continue

                    # Update record
                    record["instruction"] = clean_instruction
                    record["output"] = clean_output
                    
                    # Update stats
                    self.stats["input_lengths"].append(len(clean_instruction))
                    self.stats["output_lengths"].append(len(clean_output))
                    self.stats["total_cleaned"] += 1
                    
                    cleaned_records.append(record)
                except Exception as e:
                    logger.error(f"Error processing SFT record: {e}")
                    self.stats["total_skipped"] += 1

        # Save to output
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in cleaned_records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Q&A Transformation complete. Cleaned {self.stats['total_cleaned']} records.")
        return cleaned_records

    def get_summary(self):
        import numpy as np
        return {
            "total_sft_loaded": self.stats["total_loaded"],
            "total_sft_cleaned": self.stats["total_cleaned"],
            "skipped_sft_records": self.stats["total_skipped"],
            "avg_input_length": float(np.mean(self.stats["input_lengths"])) if self.stats["input_lengths"] else 0,
            "avg_output_length": float(np.mean(self.stats["output_lengths"])) if self.stats["output_lengths"] else 0,
            "min_output_length": int(np.min(self.stats["output_lengths"])) if self.stats["output_lengths"] else 0,
            "max_output_length": int(np.max(self.stats["output_lengths"])) if self.stats["output_lengths"] else 0
        }
