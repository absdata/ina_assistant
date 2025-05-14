import re
from typing import List, Tuple
import PyPDF2
import docx
from io import BytesIO

class TextChunker:
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on semantic boundaries.
        
        Args:
            text (str): Text to split into chunks
            
        Returns:
            List[str]: List of text chunks
        """
        # Clean and normalize text
        text = self._normalize_text(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Combine sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Add current chunk to results
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = 0
                overlap_chunk = []
                
                for prev_sentence in reversed(current_chunk):
                    if overlap_size + len(prev_sentence) > self.overlap:
                        break
                    overlap_chunk.insert(0, prev_sentence)
                    overlap_size += len(prev_sentence)
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and special characters."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Basic sentence splitting pattern
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def parse_file(self, file_content: bytes, file_type: str) -> Tuple[str, List[str]]:
        """
        Parse different file types and return extracted text and chunks.
        
        Args:
            file_content (bytes): Raw file content
            file_type (str): File type (pdf, docx, txt)
            
        Returns:
            Tuple[str, List[str]]: Tuple of (full text, list of chunks)
        """
        text = ""
        
        if file_type == "pdf":
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = " ".join(page.extract_text() for page in pdf_reader.pages)
            
        elif file_type == "docx":
            doc_file = BytesIO(file_content)
            doc = docx.Document(doc_file)
            text = " ".join(paragraph.text for paragraph in doc.paragraphs)
            
        elif file_type == "txt":
            text = file_content.decode('utf-8')
        
        # Clean and chunk the text
        text = self._normalize_text(text)
        chunks = self.chunk_text(text)
        
        return text, chunks 