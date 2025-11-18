from typing import List, Dict, Any, Optional, BinaryIO
import logging
from pathlib import Path
from datetime import datetime
import uuid

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None

from app.utils.text_splitter import get_text_chunker, ChunkConfig

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process various document types and extract text"""
    
    def __init__(self, chunk_config: Optional[ChunkConfig] = None):
        self.chunker = get_text_chunker(chunk_config)
    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID"""
        return f"chunk_{uuid.uuid4().hex[:12]}"
    
    def _build_base_metadata(
        self,
        document_id: str,
        workspace_id: str,
        document_type: str,
        source_file: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Build base metadata for chunks"""
        return {
            "document_id": document_id,
            "workspace_id": workspace_id,
            "document_type": document_type,
            "source_file": source_file,
            "upload_date": datetime.utcnow().isoformat(),
            **kwargs
        }
    
    def process_pdf(
        self,
        file_path: str,
        document_id: str,
        workspace_id: str,
        **metadata_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process PDF file and return chunks with metadata
        
        Args:
            file_path: Path to PDF file
            document_id: Unique document identifier
            workspace_id: Business identifier
            **metadata_kwargs: Additional metadata
            
        Returns:
            List of chunks with metadata
        """
        if PdfReader is None:
            raise ImportError("PyPDF2 is required for PDF processing")
        
        try:
            reader = PdfReader(file_path)
            all_chunks = []
            chunk_counter = 0
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                if not text.strip():
                    logger.warning(f"Page {page_num} has no text (might be scanned)")
                    continue
                
                # Chunk the page text
                page_chunks = self.chunker.smart_split(text, prefer_paragraphs=True)
                
                for chunk_text in page_chunks:
                    base_metadata = self._build_base_metadata(
                        document_id=document_id,
                        workspace_id=workspace_id,
                        document_type="pdf",
                        source_file=Path(file_path).name,
                        **metadata_kwargs
                    )
                    
                    chunk_data = {
                        "id": self._generate_chunk_id(),
                        "text": chunk_text,
                        "metadata": {
                            **base_metadata,
                            "page_number": page_num,
                            "chunk_index": chunk_counter,
                            "extraction_method": "text"
                        }
                    }
                    all_chunks.append(chunk_data)
                    chunk_counter += 1
            
            # Update total chunks in all metadata
            for chunk in all_chunks:
                chunk["metadata"]["total_chunks"] = len(all_chunks)
            
            logger.info(f"Processed PDF: {len(all_chunks)} chunks from {len(reader.pages)} pages")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def process_image(
        self,
        file_path: str,
        document_id: str,
        workspace_id: str,
        use_ocr: bool = True,
        **metadata_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process image file and extract text via OCR
        
        Args:
            file_path: Path to image file
            document_id: Unique document identifier
            workspace_id: Business identifier
            use_ocr: Whether to use OCR
            **metadata_kwargs: Additional metadata
            
        Returns:
            List of chunks with metadata
        """
        if Image is None or pytesseract is None:
            raise ImportError("Pillow and pytesseract are required for image processing")
        
        try:
            image = Image.open(file_path)
            
            if use_ocr:
                # Extract text using OCR
                ocr_text = pytesseract.image_to_string(image)
                
                if not ocr_text.strip():
                    logger.warning(f"No text extracted from image: {file_path}")
                    return []
                
                # Chunk with smaller size for OCR (less reliable)
                config = ChunkConfig(chunk_size=400, overlap_pct=0.20)
                chunker = get_text_chunker(config)
                text_chunks = chunker.smart_split(ocr_text)
                
                chunks = []
                for idx, chunk_text in enumerate(text_chunks):
                    base_metadata = self._build_base_metadata(
                        document_id=document_id,
                        workspace_id=workspace_id,
                        document_type="image",
                        source_file=Path(file_path).name,
                        **metadata_kwargs
                    )
                    
                    chunk_data = {
                        "id": self._generate_chunk_id(),
                        "text": chunk_text,
                        "metadata": {
                            **base_metadata,
                            "chunk_index": idx,
                            "total_chunks": len(text_chunks),
                            "extraction_method": "ocr",
                            "image_size": f"{image.width}x{image.height}"
                        }
                    }
                    chunks.append(chunk_data)
                
                logger.info(f"Processed image: {len(chunks)} chunks via OCR")
                return chunks
            else:
                # Store image metadata only (for future Vision API processing)
                base_metadata = self._build_base_metadata(
                    document_id=document_id,
                    workspace_id=workspace_id,
                    document_type="image",
                    source_file=Path(file_path).name,
                    **metadata_kwargs
                )
                
                chunk_data = {
                    "id": self._generate_chunk_id(),
                    "text": f"Image: {Path(file_path).name}",
                    "metadata": {
                        **base_metadata,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "extraction_method": "metadata_only",
                        "image_size": f"{image.width}x{image.height}",
                        "requires_vision_api": True
                    }
                }
                
                return [chunk_data]
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    def process_text(
        self,
        text: str,
        document_id: str,
        workspace_id: str,
        document_type: str = "text",
        source_file: str = "text_input",
        **metadata_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process plain text and return chunks
        
        Args:
            text: Input text
            document_id: Unique document identifier
            workspace_id: Business identifier
            document_type: Type of document
            source_file: Source identifier
            **metadata_kwargs: Additional metadata
            
        Returns:
            List of chunks with metadata
        """
        try:
            text_chunks = self.chunker.smart_split(text, prefer_paragraphs=True)
            
            chunks = []
            for idx, chunk_text in enumerate(text_chunks):
                base_metadata = self._build_base_metadata(
                    document_id=document_id,
                    workspace_id=workspace_id,
                    document_type=document_type,
                    source_file=source_file,
                    **metadata_kwargs
                )
                
                chunk_data = {
                    "id": self._generate_chunk_id(),
                    "text": chunk_text,
                    "metadata": {
                        **base_metadata,
                        "chunk_index": idx,
                        "total_chunks": len(text_chunks)
                    }
                }
                chunks.append(chunk_data)
            
            logger.info(f"Processed text: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise
    
    def process_qna(
        self,
        qna_pairs: List[Dict[str, str]],
        document_id: str,
        workspace_id: str,
        **metadata_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process Q&A pairs as atomic chunks
        
        Args:
            qna_pairs: List of {"question": "...", "answer": "..."}
            document_id: Unique document identifier
            workspace_id: Business identifier
            **metadata_kwargs: Additional metadata
            
        Returns:
            List of chunks with metadata
        """
        try:
            chunks = []
            
            for idx, qna in enumerate(qna_pairs):
                question = qna.get("question", "").strip()
                answer = qna.get("answer", "").strip()
                category = qna.get("category", "General")
                
                if not question or not answer:
                    logger.warning(f"Skipping empty Q&A pair at index {idx}")
                    continue
                
                # Combine Q&A
                combined_text = f"Q: {question}\nA: {answer}"
                token_count = self.chunker.count_tokens(combined_text)
                
                # If answer is too long, split it
                if token_count > self.chunker.config.chunk_size:
                    # Split answer into parts
                    answer_chunks = self.chunker.split_by_sentences(
                        answer,
                        chunk_size=self.chunker.config.chunk_size - self.chunker.count_tokens(f"Q: {question}\nA (part /): ")
                    )
                    
                    for part_idx, answer_part in enumerate(answer_chunks, start=1):
                        chunk_text = f"Q: {question}\nA (part {part_idx}/{len(answer_chunks)}): {answer_part}"
                        
                        base_metadata = self._build_base_metadata(
                            document_id=document_id,
                            workspace_id=workspace_id,
                            document_type="qna",
                            source_file="qna_import",
                            **metadata_kwargs
                        )
                        
                        chunk_data = {
                            "id": self._generate_chunk_id(),
                            "text": chunk_text,
                            "metadata": {
                                **base_metadata,
                                "chunk_index": len(chunks),
                                "question": question,
                                "category": category,
                                "answer_part": part_idx,
                                "total_answer_parts": len(answer_chunks)
                            }
                        }
                        chunks.append(chunk_data)
                else:
                    # Single chunk for Q&A pair
                    base_metadata = self._build_base_metadata(
                        document_id=document_id,
                        workspace_id=workspace_id,
                        document_type="qna",
                        source_file="qna_import",
                        **metadata_kwargs
                    )
                    
                    chunk_data = {
                        "id": self._generate_chunk_id(),
                        "text": combined_text,
                        "metadata": {
                            **base_metadata,
                            "chunk_index": len(chunks),
                            "question": question,
                            "category": category,
                            "answer_part": 1,
                            "total_answer_parts": 1
                        }
                    }
                    chunks.append(chunk_data)
            
            # Update total chunks
            for chunk in chunks:
                chunk["metadata"]["total_chunks"] = len(chunks)
            
            logger.info(f"Processed {len(qna_pairs)} Q&A pairs into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing Q&A: {str(e)}")
            raise


def get_document_processor(chunk_config: Optional[ChunkConfig] = None) -> DocumentProcessor:
    """Get DocumentProcessor instance"""
    return DocumentProcessor(chunk_config)