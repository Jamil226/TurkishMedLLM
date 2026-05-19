from langchain_chroma import Chroma
from src import config
from src.vectorstore.embedding_loader import EmbeddingLoader

class TurkishMedRetriever:
    def __init__(self, top_k: int = None):
        self.top_k = top_k or config.RAG_TOP_K
        self.embeddings = EmbeddingLoader.load_embedding_model()
        self.vectorstore = Chroma(
            persist_directory=str(config.CHROMA_PERSIST_DIR),
            embedding_function=self.embeddings
        )

    def get_relevant_documents(self, query: str):
        return self.vectorstore.similarity_search(query, k=self.top_k)

    def get_retriever_as_langchain(self):
        return self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
