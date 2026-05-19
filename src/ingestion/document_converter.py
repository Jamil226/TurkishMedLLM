from langchain_core.documents import Document
from src import config

class DocumentConverter:
    @staticmethod
    def to_qa_document(row, question_col, answer_col, metadata_base):
        """Converts a QA row to a LangChain Document."""
        question = str(row[question_col])
        answer = str(row[answer_col])
        
        content = f"Question:\n{question}\n\nAnswer:\n{answer}"
        
        metadata = metadata_base.copy()
        metadata.update({
            "dataset_type": "qa",
            "intended_use": "fine_tuning",
            "category": row.get("category", row.get("field", "medical"))
        })
        
        return Document(page_content=content, metadata=metadata)

    @staticmethod
    def to_sft_format(row, question_col, answer_col, metadata_base):
        """Converts a QA row to SFT instruction format."""
        return {
            "instruction": config.SFT_SYSTEM_PROMPT,
            "input": f"Hastanın sorusu: {str(row[question_col])}",
            "output": str(row[answer_col]),
            "metadata": metadata_base
        }

    @staticmethod
    def to_article_document(row, text_col, title_col, metadata_base):
        """Converts an article row to a LangChain Document."""
        content = str(row[text_col])
        
        metadata = metadata_base.copy()
        metadata.update({
            "dataset_type": "article",
            "intended_use": "rag_corpus",
            "title": str(row[title_col]) if title_col and title_col in row else None,
            "url": row.get("url", row.get("link")),
            "hospital": row.get("hospital", row.get("source"))
        })
        
        return Document(page_content=content, metadata=metadata)
