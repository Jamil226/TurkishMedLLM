import re
from bs4 import BeautifulSoup

class TextCleaner:
    @staticmethod
    def clean_text(text, remove_html=False):
        """
        Applies standard cleaning to text:
        - Strip leading/trailing spaces
        - Normalize repeated spaces
        - Normalize repeated newlines
        - Remove HTML tags if requested
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Remove HTML tags if requested
        if remove_html:
            # Use BeautifulSoup for cleaner HTML removal
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator=" ")

        # 2. Normalize repeated newlines
        text = re.sub(r'\n+', '\n', text)
        
        # 3. Normalize repeated spaces
        text = re.sub(r' +', ' ', text)
        
        # 4. Final strip
        return text.strip()

    @staticmethod
    def clean_qa_pair(question, answer):
        """Applies light normalization to Q&A pairs."""
        return (
            TextCleaner.clean_text(question, remove_html=False),
            TextCleaner.clean_text(answer, remove_html=False)
        )
