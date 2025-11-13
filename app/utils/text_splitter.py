import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking"""
    chunk_size: int = 512  # tokens
    overlap_pct: float = 0.15
    min_chunk_size: int = 50
    max_chunk_size: int = 1024
    encoding_name: str = "cl100k_base"


class TextChunker:
    """Intelligent text chunking with token-aware splitting"""
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.encoding = tiktoken.get_encoding(self.config.encoding_name)
        self.overlap_size = int(self.config.chunk_size * self.config.overlap_pct)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def split_by_tokens(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        """
        Split text into chunks by token count with overlap
        
        Args:
            text: Input text
            chunk_size: Target chunk size in tokens
            overlap: Overlap size in tokens
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.overlap_size
        
        if not text.strip():
            return []
        
        # Encode text to tokens
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Clean up chunk
            chunk_text = chunk_text.strip()
            if chunk_text and self.count_tokens(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - overlap
            
            # Prevent infinite loop
            if start >= len(tokens) - overlap:
                break
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def split_by_sentences(
        self,
        text: str,
        chunk_size: Optional[int] = None
    ) -> List[str]:
        """
        Split text at sentence boundaries while respecting token limits
        
        Args:
            text: Input text
            chunk_size: Target chunk size in tokens
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds limit, split it by tokens
            if sentence_tokens > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence
                sub_chunks = self.split_by_tokens(sentence, chunk_size)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding sentence exceeds limit
            if current_tokens + sentence_tokens > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    
                    # Apply overlap: keep last sentence
                    current_chunk = [current_chunk[-1], sentence]
                    current_tokens = self.count_tokens(' '.join(current_chunk))
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} sentence-based chunks")
        return chunks
    
    def split_by_paragraphs(
        self,
        text: str,
        chunk_size: Optional[int] = None
    ) -> List[str]:
        """
        Split text at paragraph boundaries while respecting token limits
        
        Args:
            text: Input text
            chunk_size: Target chunk size in tokens
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # If single paragraph exceeds limit, split by sentences
            if para_tokens > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph
                sub_chunks = self.split_by_sentences(para, chunk_size)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding paragraph exceeds limit
            if current_tokens + para_tokens > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} paragraph-based chunks")
        return chunks
    
    def smart_split(
        self,
        text: str,
        prefer_paragraphs: bool = True
    ) -> List[str]:
        """
        Intelligently split text using best strategy
        
        Args:
            text: Input text
            prefer_paragraphs: Prefer paragraph boundaries over sentences
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        total_tokens = self.count_tokens(text)
        
        # If text fits in one chunk, return as-is
        if total_tokens <= self.config.chunk_size:
            return [text.strip()]
        
        # Choose splitting strategy
        if prefer_paragraphs and '\n\n' in text:
            return self.split_by_paragraphs(text)
        elif '.' in text:
            return self.split_by_sentences(text)
        else:
            return self.split_by_tokens(text)


def get_text_chunker(config: Optional[ChunkConfig] = None) -> TextChunker:
    """Get TextChunker instance"""
    return TextChunker(config)