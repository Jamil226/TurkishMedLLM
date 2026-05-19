from langchain_chroma import Chroma
from src import config
from src.vectorstore.embedding_loader import EmbeddingLoader
from src.utils.logger import logger

def verify_vectorstore():
    logger.info("Initializing Verification...")
    
    # 1. Load embedding model
    embeddings = EmbeddingLoader.load_embedding_model()
    
    # 2. Load ChromaDB
    logger.info(f"Loading ChromaDB from {config.CHROMA_PERSIST_DIR}")
    vectorstore = Chroma(
        persist_directory=str(config.CHROMA_PERSIST_DIR),
        embedding_function=embeddings
    )
    
    # 3. Define query
    query = "Diyabet hastalarında kan şekeri neden yükselir?"
    logger.info(f"Running test query: {query}")
    
    # 4. Search
    results = vectorstore.similarity_search_with_score(query, k=5)
    
    # 5. Print results
    print("\n" + "="*50)
    print(f"QUERY: {query}")
    print(f"RESULTS FOUND: {len(results)}")
    print("="*50)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n[{i+1}] Score: {score:.4f}")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"Title: {doc.metadata.get('title')}")
        print(f"URL: {doc.metadata.get('url')}")
        print("-" * 20)
        content = doc.page_content[:300].replace('\n', ' ')
        print(f"Content: {content}...")
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    verify_vectorstore()
