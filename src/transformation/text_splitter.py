from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config

class TextSplitter:
    def __init__(self, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=config.CHUNK_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self, text):
        """Splits text into chunks using LangChain splitter."""
        return self.splitter.split_text(text)

    @staticmethod
    def is_quality_chunk(text):
        """
        Checks if a chunk meets quality standards:
        - Not empty
        - At least 100 characters
        - Not just punctuation or numbers
        """
        if not text:
            return False, "empty"
        
        if len(text) < config.CHUNK_MIN_LENGTH:
            return False, "short"
        
        # Check if it contains at least one alphabetic character (Turkish characters included)
        # Regex for letters including Turkish ones: [a-zA-ZÇŞĞÜÖİçşğüöı]
        if not any(c.isalpha() for c in text):
            return False, "non_alpha"
            
        return True, "valid"
