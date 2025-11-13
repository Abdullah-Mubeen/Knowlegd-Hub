from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import uuid
import time

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

from app.utils.text_splitter import get_text_chunker, ChunkConfig
from app.utils.openai_service import get_openai_service
from app.utils.pinecone_service import get_pinecone_service

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDF documents with smart chunking"""
    
    def __init__(self):
        self.chunker = get_text_chunker(ChunkConfig(
            chunk_size=512,
            overlap_pct=0.15
        ))
        self.openai_service = get_openai_service()
        self.pinecone_service = get_pinecone_service()
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[List[Dict[str, Any]], int]:
        """Extract text from PDF page by page"""
        if PdfReader is None:
            raise ImportError("PyPDF2 required")
        
        try:
            reader = PdfReader(file_path)
            pages_data = []
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                if text.strip():
                    pages_data.append({
                        "page_number": page_num,
                        "text": text.strip()
                    })
                else:
                    logger.warning(f"Page {page_num} has no extractable text")
            
            logger.info(f"Extracted text from {len(pages_data)}/{len(reader.pages)} pages")
            return pages_data, len(reader.pages)
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise
    
    def create_smart_chunks(
        self,
        pages_data: List[Dict[str, Any]],
        document_id: str,
        business_id: str,
        source_file: str
    ) -> List[Dict[str, Any]]:
        """Create smart chunks from PDF pages"""
        all_chunks = []
        chunk_counter = 0
        
        for page_data in pages_data:
            page_num = page_data["page_number"]
            text = page_data["text"]
            
            # Smart split with paragraph preference
            text_chunks = self.chunker.smart_split(text, prefer_paragraphs=True)
            
            for chunk_text in text_chunks:
                chunk_data = {
                    "id": f"chunk_{uuid.uuid4().hex[:12]}",
                    "text": chunk_text,
                    "metadata": {
                        "document_id": document_id,
                        "business_id": business_id,
                        "document_type": "pdf",
                        "source_file": source_file,
                        "page_number": page_num,
                        "chunk_index": chunk_counter,
                        "extraction_method": "text",
                        "upload_date": datetime.utcnow().isoformat()
                    }
                }
                all_chunks.append(chunk_data)
                chunk_counter += 1
        
        # Update total chunks
        for chunk in all_chunks:
            chunk["metadata"]["total_chunks"] = len(all_chunks)
        
        logger.info(f"Created {len(all_chunks)} smart chunks from {len(pages_data)} pages")
        return all_chunks
    
    def generate_embeddings_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for all chunks"""
        try:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.openai_service.generate_embeddings_batch(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def store_chunks_in_pinecone(
        self,
        chunks: List[Dict[str, Any]],
        namespace: str
    ) -> List[str]:
        """Store chunks in Pinecone"""
        try:
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            ids = self.pinecone_service.add_documents(
                texts=texts,
                metadatas=metadatas,
                namespace=namespace
            )
            
            logger.info(f"Stored {len(ids)} chunks in Pinecone")
            return ids
            
        except Exception as e:
            logger.error(f"Error storing in Pinecone: {str(e)}")
            raise
    
    async def process_pdf(
        self,
        file_path: str,
        business_id: str,
        filename: str
    ) -> Dict[str, Any]:
        """
        Complete PDF processing pipeline:
        1. Extract text from PDF
        2. Create smart chunks
        3. Generate embeddings
        4. Store in Pinecone
        """
        start_time = time.time()
        document_id = f"pdf_{uuid.uuid4().hex[:12]}"
        
        try:
            logger.info(f"Starting PDF processing: {filename}")
            
            # Step 1: Extract text
            pages_data, total_pages = self.extract_text_from_pdf(file_path)
            
            if not pages_data:
                raise ValueError("No text could be extracted from PDF")
            
            # Step 2: Create smart chunks
            chunks = self.create_smart_chunks(
                pages_data=pages_data,
                document_id=document_id,
                business_id=business_id,
                source_file=filename
            )
            
            # Step 3: Generate embeddings
            chunks = self.generate_embeddings_batch(chunks)
            
            # Step 4: Store in Pinecone
            stored_ids = self.store_chunks_in_pinecone(
                chunks=chunks,
                namespace=business_id
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "document_id": document_id,
                "business_id": business_id,
                "filename": filename,
                "total_pages": total_pages,
                "total_chunks": len(chunks),
                "stored_ids": stored_ids,
                "processing_time": processing_time
            }
            
            logger.info(f"PDF processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise


def get_pdf_processor() -> PDFProcessor:
    """Get PDFProcessor instance"""
    return PDFProcessor()