import pandas as pd
import json
from pathlib import Path
from src.utils.logger import logger

class DataLoader:
    @staticmethod
    def load(file_path: Path):
        """Loads a file based on its extension."""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # If it's a list of dicts, it's already in the format we want for pandas
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Some JSONs might be a dict with a key containing the list
                    for key in data:
                        if isinstance(data[key], list):
                            return pd.DataFrame(data[key])
                    return pd.DataFrame([data])
            
            elif suffix == '.csv':
                return pd.read_csv(file_path)
            
            elif suffix == '.parquet':
                return pd.read_parquet(file_path)
            
            else:
                logger.warning(f"Unsupported file format: {suffix} for {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
