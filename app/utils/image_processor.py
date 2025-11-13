from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import uuid
import time

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None

from app.utils.text_splitter import get_text_chunker, ChunkConfig
from app.utils.openai_service import get_openai_service
from app.utils.pinecone_service import get_pinecone_service

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process images with smart chunking and embeddings"""
    
    def __init__(self):
        self.chunker = get_text_chunker(ChunkConfig(
            chunk_size=400,
            overlap_pct=0.20
        ))
        self.openai_service = get_openai_service()
        self.pinecone_service = get_pinecone_service()
    
    def extract_text_from_image(self, file_path: str) -> Tuple[str, str, Optional[float]]:
        """Extract text from image using OCR"""
        if Image is None or pytesseract is None:
            raise ImportError("Pillow and pytesseract required")
        
        try:
            image = Image.open(file_path)
            image_size = f"{image.width}x{image.height}"
            
            # Extract text using OCR
            ocr_text = pytesseract.image_to_string(image)
            
            if not ocr_text.strip():
                logger.warning(f"No text extracted from: {file_path}")
                return "", image_size, 0.0
            
            # Simple confidence scoring
            word_count = len(ocr_text.split())
            confidence = min(0.95, word_count / 100)
            
            logger.info(f"Extracted {len(ocr_text)} chars from image")
            return ocr_text.strip(), image_size, confidence
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise
    
    def create_smart_chunks(
        self,
        text: str,
        document_id: str,
        business_id: str,
        source_file: str,
        image_size: str,
        confidence_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Create smart chunks from extracted text"""
        
        if not text.strip():
            # Metadata-only chunk
            return [{
                "id": f"chunk_{uuid.uuid4().hex[:12]}",
                "text": f"Image: {source_file}",
                "metadata": {
                    "document_id": document_id,
                    "business_id": business_id,
                    "document_type": "image",
                    "source_file": source_file,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "extraction_method": "metadata_only",
                    "image_size": image_size,
                    "upload_date": datetime.utcnow().isoformat(),
                    "requires_vision_api": True
                }
            }]
        
        # Smart split with paragraph preference
        text_chunks = self.chunker.smart_split(text, prefer_paragraphs=True)
        
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk_data = {
                "id": f"chunk_{uuid.uuid4().hex[:12]}",
                "text": chunk_text,
                "metadata": {
                    "document_id": document_id,
                    "business_id": business_id,
                    "document_type": "image",
                    "source_file": source_file,
                    "chunk_index": idx,
                    "total_chunks": len(text_chunks),
                    "extraction_method": "ocr",
                    "image_size": image_size,
                    "confidence_score": confidence_score,
                    "upload_date": datetime.utcnow().isoformat()
                }
            }
            chunks.append(chunk_data)
        
        logger.info(f"Created {len(chunks)} smart chunks")
        return chunks
    
    def generate_embeddings_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for all chunks at once"""
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
    
    async def process_image(
        self,
        file_path: str,
        business_id: str,
        filename: str,
        use_ocr: bool = True
    ) -> Dict[str, Any]:
        """
        Complete image processing pipeline:
        1. Extract text via OCR
        2. Create smart chunks
        3. Generate embeddings
        4. Store in Pinecone
        """
        start_time = time.time()
        document_id = f"img_{uuid.uuid4().hex[:12]}"
        
        try:
            logger.info(f"Starting image processing: {filename}")
            
            # Step 1: Extract text
            extracted_text, image_size, confidence = self.extract_text_from_image(file_path)
            extraction_method = "ocr" if use_ocr and extracted_text else "metadata_only"
            
            # Step 2: Create smart chunks
            chunks = self.create_smart_chunks(
                text=extracted_text,
                document_id=document_id,
                business_id=business_id,
                source_file=filename,
                image_size=image_size,
                confidence_score=confidence
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
                "image_size": image_size,
                "extraction_method": extraction_method,
                "total_chunks": len(chunks),
                "stored_ids": stored_ids,
                "processing_time": processing_time
            }
            
            logger.info(f"Image processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise


def get_image_processor() -> ImageProcessor:
    """Get ImageProcessor instance"""
    return ImageProcessor()