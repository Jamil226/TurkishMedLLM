import requests
from langchain_ollama import OllamaEmbeddings
from src import config
from src.utils.logger import logger

class EmbeddingLoader:
    @staticmethod
    def load_embedding_model():
        """
        Loads the embedding model based on the configuration.
        Currently supports Ollama with bge-m3.
        """
        if config.EMBEDDING_BACKEND == "ollama":
            # 1. Check if Ollama is running
            try:
                response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                if response.status_code != 200:
                    raise ConnectionError(f"Ollama is not responding at {config.OLLAMA_BASE_URL}")
                
                # 2. Check if the model exists
                models = [m['name'] for m in response.json().get('models', [])]
                # Ollama models might have tags like 'bge-m3:latest'
                model_exists = any(config.OLLAMA_EMBEDDING_MODEL in m for m in models)
                
                if not model_exists:
                    logger.warning(f"Model '{config.OLLAMA_EMBEDDING_MODEL}' not found in Ollama tags: {models}")
                    logger.info(f"Attempting to proceed anyway, but model may need to be pulled.")

                logger.info(f"Loading Ollama embeddings: {config.OLLAMA_EMBEDDING_MODEL}")
                return OllamaEmbeddings(
                    model=config.OLLAMA_EMBEDDING_MODEL,
                    base_url=config.OLLAMA_BASE_URL
                )
                
            except requests.exceptions.ConnectionError:
                err_msg = f"Could not connect to Ollama at {config.OLLAMA_BASE_URL}. Is Ollama Desktop running?"
                logger.error(err_msg)
                raise ConnectionError(err_msg)
            except Exception as e:
                logger.error(f"Error checking Ollama status: {e}")
                raise e
        else:
            raise ValueError(f"Unsupported embedding backend: {config.EMBEDDING_BACKEND}")
