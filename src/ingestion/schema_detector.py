from pathlib import Path
from src import config

class SchemaDetector:
    @staticmethod
    def detect_source(file_path: Path):
        """Detects the source from the folder name."""
        parent_folder = file_path.parent.name
        source_id = parent_folder
        source_name = config.SOURCE_MAPPING.get(source_id, "Unknown Source")
        return source_id, source_name

    @staticmethod
    def detect_split(file_path: Path):
        """Detects the split from the filename."""
        filename = file_path.name.lower()
        if "train" in filename:
            return "train"
        elif "test" in filename:
            return "test"
        elif "validation" in filename:
            return "validation"
        elif "full" in filename:
            return "full"
        return "unknown"

    @staticmethod
    def detect_dataset_type(source_id: str):
        """Source-1 and Source-2 are QA, Source-3 is article."""
        if source_id in ["source-1", "source-2"]:
            return "qa"
        elif source_id == "source-3":
            return "article"
        return "unknown"

    @staticmethod
    def detect_qa_columns(df, df_columns):
        """Detects question and answer columns using keywords or text length heuristic."""
        question_col = None
        answer_col = None
        detection_method = "keyword"
        
        cols_lower = [c.lower() for c in df_columns]
        
        # 1. Look for question using keywords
        for target in config.QUESTION_COLUMNS:
            if target.lower() in cols_lower:
                idx = cols_lower.index(target.lower())
                question_col = df_columns[idx]
                break
        
        # 2. Look for answer using keywords
        for target in config.ANSWER_COLUMNS:
            if target.lower() in cols_lower:
                idx = cols_lower.index(target.lower())
                answer_col = df_columns[idx]
                break
        
        # 3. Heuristic: Shorter vs Longer text columns
        if not question_col or not answer_col:
            # Filter for string/object columns only
            text_candidates = []
            for col in df_columns:
                if df[col].dtype == 'object' or str(df[col].dtype).startswith('string'):
                    text_candidates.append(col)
            
            if len(text_candidates) >= 2:
                # Calculate average length for each candidate
                avg_lengths = {}
                for col in text_candidates:
                    avg_lengths[col] = df[col].astype(str).str.len().mean()
                
                # Sort candidates by length
                sorted_cols = sorted(avg_lengths, key=avg_lengths.get)
                
                if not question_col:
                    question_col = sorted_cols[0]
                    detection_method = "heuristic_length"
                
                if not answer_col:
                    # Use the longest one that isn't the question
                    answer_col = sorted_cols[-1]
                    if answer_col == question_col and len(sorted_cols) > 1:
                        answer_col = sorted_cols[-2]
                    detection_method = "heuristic_length"
                    
        return question_col, answer_col, detection_method

    @staticmethod
    def detect_article_columns(df_columns):
        """Detects article text, title, etc."""
        text_col = None
        title_col = None
        
        cols_lower = [c.lower() for c in df_columns]
        
        # Often 'text', 'content', 'article'
        targets = ['text', 'content', 'article', 'context', 'body']
        for target in targets:
            if target in cols_lower:
                idx = cols_lower.index(target)
                text_col = df_columns[idx]
                break
        
        # Title
        if 'title' in cols_lower:
            title_col = df_columns[cols_lower.index('title')]
        elif 'baslik' in cols_lower:
            title_col = df_columns[cols_lower.index('baslik')]
            
        return text_col, title_col
