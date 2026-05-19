import pandas as pd
from src import config
from src.utils.logger import logger

class QualityChecker:
    def __init__(self):
        self.stats = {
            "total_records": 0,
            "valid_records": 0,
            "skipped_records": 0,
            "duplicate_records": 0,
            "low_quality_records": 0,
            "missing_question": 0,
            "missing_answer": 0,
            "missing_article_text": 0,
            "lengths": []
        }

    def process_df(self, df, dataset_type, question_col=None, answer_col=None, text_col=None):
        """Cleans and validates the dataframe based on nulls, duplicates, and quality thresholds."""
        initial_count = len(df)
        self.stats["total_records"] += int(initial_count)
        
        # 1. Remove empty records (completely NaN rows)
        df = df.dropna(how='all')
        
        if dataset_type == "qa":
            if not question_col or not answer_col:
                logger.error(f"Cannot clean QA dataset without detected columns: Q={question_col}, A={answer_col}")
                self.stats["skipped_records"] += int(initial_count)
                return pd.DataFrame()
            
            # Remove records where question or answer is missing
            mask_missing = df[question_col].isna() | df[answer_col].isna()
            self.stats["missing_question"] += int(df[question_col].isna().sum())
            self.stats["missing_answer"] += int(df[answer_col].isna().sum())
            df = df[~mask_missing]
            
            # Local deduplication
            before_dedup = len(df)
            df = df.drop_duplicates(subset=[question_col, answer_col])
            self.stats["duplicate_records"] += int(before_dedup - len(df))
            
            # Quality Filtering (Length Thresholds)
            q_thresh = config.QUALITY_THRESHOLDS["qa"]
            q_len = df[question_col].astype(str).str.len()
            a_len = df[answer_col].astype(str).str.len()
            combined_len = q_len + a_len
            
            mask_low_quality = (
                (q_len < q_thresh["min_question_len"]) |
                (a_len < q_thresh["min_answer_len"]) |
                (combined_len < q_thresh["min_combined_len"])
            )
            self.stats["low_quality_records"] += int(mask_low_quality.sum())
            df = df[~mask_low_quality]

        elif dataset_type == "article":
            if not text_col:
                logger.error(f"Cannot clean Article dataset without detected text column.")
                self.stats["skipped_records"] += int(initial_count)
                return pd.DataFrame()
                
            # Remove records where article text is missing
            mask_missing = df[text_col].isna()
            self.stats["missing_article_text"] += int(mask_missing.sum())
            df = df[~mask_missing]
            
            # Local deduplication
            before_dedup = len(df)
            df = df.drop_duplicates(subset=[text_col])
            self.stats["duplicate_records"] += int(before_dedup - len(df))
            
            # Quality Filtering (Length Thresholds)
            a_thresh = config.QUALITY_THRESHOLDS["article"]
            text_len = df[text_col].astype(str).str.len()
            mask_low_quality = text_len < a_thresh["min_article_len"]
            
            self.stats["low_quality_records"] += int(mask_low_quality.sum())
            df = df[~mask_low_quality]

        final_count = len(df)
        self.stats["valid_records"] += int(final_count)
        self.stats["skipped_records"] = int(initial_count - final_count)
        
        return df

    def update_lengths(self, text_list):
        for text in text_list:
            if isinstance(text, str):
                self.stats["lengths"].append(len(text))

    def get_file_summary(self):
        summary = self.stats.copy()
        if summary["lengths"]:
            summary["avg_length"] = sum(summary["lengths"]) / len(summary["lengths"])
            summary["min_length"] = min(summary["lengths"])
            summary["max_length"] = max(summary["lengths"])
        else:
            summary["avg_length"] = 0
            summary["min_length"] = 0
            summary["max_length"] = 0
        
        del summary["lengths"]
        return summary
