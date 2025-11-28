from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import uuid
import time

# NO pytesseract needed - using OpenAI Vision API only
try:
    from PIL import Image
except ImportError:
    Image = None

from app.utils.text_splitter import get_text_chunker, ChunkConfig
from app.utils.openai_service import get_openai_service
from app.utils.pinecone_service import get_pinecone_service

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process images with smart chunking and embeddings for optimal RAG - OpenAI Vision only"""
    
    def __init__(self):
        self.chunker = get_text_chunker(ChunkConfig(
            chunk_size=300,  
            overlap_pct=0.25 
        ))
        self.openai_service = get_openai_service()
        self.pinecone_service = get_pinecone_service()
    
    def _get_image_description(self, file_path: str) -> str:
        """Get a brief description of the image for context using Vision API"""
        try:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            
            # Use the existing extract_text_from_image but with description prompt
            # We'll create a helper method in openai_service for this
            import base64
            from openai import OpenAI
            
            client = OpenAI(api_key=self.openai_service.api_key)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o",  
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Provide a brief 1-2 sentence description of this image. Focus on:
- Main subject or document type
- Purpose or context
- Key visual elements

Be concise and informative."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            description = response.choices[0].message.content.strip()
            logger.info(f"Generated image description: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.warning(f"Could not get image description: {str(e)}")
            return ""
    
    def extract_text_from_image(self, file_path: str) -> Tuple[str, str, str, Optional[float]]:
        """
        Extract text and description using OpenAI Vision API only (NO pytesseract).
        Returns: (extracted_text, image_description, image_size, confidence)
        """
        try:
            # Get image dimensions
            if Image:
                image = Image.open(file_path)
                image_size = f"{image.width}x{image.height}"
            else:
                image_size = "unknown"
            
            # Read image bytes
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            
            # Extract text using your existing Vision API method
            extracted_text = self.openai_service.extract_text_from_image(image_bytes)
            
            # Get image description for additional context
            image_description = self._get_image_description(file_path)
            
            if not extracted_text or not extracted_text.strip():
                logger.warning(f"No text extracted from image: {file_path}")
                return "", image_description, image_size, 0.0
            
            # Enhanced confidence scoring based on text quality
            word_count = len(extracted_text.split())
            char_count = len(extracted_text)
            has_punctuation = any(c in extracted_text for c in '.!?,:;')
            
            # Smart confidence calculation
            if word_count > 100 and char_count > 400 and has_punctuation:
                confidence = 0.95
            elif word_count > 50 and char_count > 200:
                confidence = 0.85
            elif word_count > 20:
                confidence = 0.75
            elif word_count > 5:
                confidence = 0.60
            else:
                confidence = 0.40
            
            logger.info(f"Extracted {word_count} words ({char_count} chars) with {confidence:.2f} confidence")
            return extracted_text, image_description, image_size, confidence
        
        except Exception as e:
            logger.error(f"Vision OCR extraction failed: {str(e)}")
            raise
    
    def create_smart_chunks(
        self,
        text: str,
        image_description: str,
        document_id: str,
        workspace_id: str,
        source_file: str,
        image_size: str,
        confidence_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Create smart chunks from extracted text with image context.
        Enhanced chunking strategy for optimal RAG retrieval.
        """
        
        chunks = []
        
        # Case 1: No text extracted - create description-only chunk
        if not text.strip():
            chunk_text = f"Image: {source_file}"
            if image_description:
                chunk_text = f"{chunk_text}\n\nDescription: {image_description}"
            
            chunks.append({
                "id": f"chunk_{uuid.uuid4().hex[:12]}",
                "text": chunk_text,
                "metadata": {
                    "document_id": document_id,
                    "workspace_id": workspace_id,
                    "document_type": "image",
                    "source_file": source_file,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "extraction_method": "description_only",
                    "image_size": image_size,
                    "has_extracted_text": False,
                    "image_description": image_description or "",
                    "confidence_score": 0.0,
                    "upload_date": datetime.utcnow().isoformat(),
                    "requires_vision_api": True
                }
            })
            logger.info("Created 1 description-only chunk")
            return chunks
        
        # Case 2: Text extracted - create context-aware chunks
        # Smart split with paragraph preference for better semantic boundaries
        text_chunks = self.chunker.smart_split(text, prefer_paragraphs=True)
        
        # Add image context to first chunk only for better retrieval
        context_prefix = ""
        if image_description:
            context_prefix = f"[Image Context: {image_description}]\n\n"
        
        # Create chunks with rich metadata
        for idx, chunk_text in enumerate(text_chunks):
            # Add context to first chunk only to avoid repetition
            final_text = chunk_text
            if idx == 0 and context_prefix:
                final_text = f"{context_prefix}{chunk_text}"
            
            chunk_data = {
                "id": f"chunk_{uuid.uuid4().hex[:12]}",
                "text": final_text,
                "metadata": {
                    "document_id": document_id,
                    "workspace_id": workspace_id,
                    "document_type": "image",
                    "source_file": source_file,
                    "chunk_index": idx,
                    "total_chunks": len(text_chunks),
                    "extraction_method": "vision_ocr",
                    "image_size": image_size,
                    "confidence_score": confidence_score,
                    "has_extracted_text": True,
                    "image_description": image_description or "",
                    "upload_date": datetime.utcnow().isoformat(),
                    "text_length": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "has_context": (idx == 0 and bool(context_prefix))
                }
            }
            chunks.append(chunk_data)
        
        logger.info(f"Created {len(chunks)} smart chunks with image context")
        return chunks
    
    def generate_embeddings_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for all chunks using OpenAI's best embedding model"""
        try:
            texts = [chunk["text"] for chunk in chunks]
            
            # Use your existing batch embedding method
            # This uses text-embedding-3-large (or whatever is configured)
            embeddings = self.openai_service.generate_embeddings_batch(texts)
            
            if len(embeddings) != len(chunks):
                raise ValueError(f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}")
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            logger.info(f"Generated {len(embeddings)} embeddings for chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def store_chunks_in_pinecone(
        self,
        chunks: List[Dict[str, Any]],
        namespace: str
    ) -> List[str]:
        """Store chunks with embeddings in Pinecone"""
        try:
            # Prepare vectors for Pinecone
            vectors = []
            
            for chunk in chunks:
                vector_id = f"{chunk['metadata']['document_id']}_chunk_{chunk['metadata']['chunk_index']}"
                
                # Add text to metadata for retrieval
                metadata = chunk["metadata"].copy()
                metadata["text"] = chunk["text"]
                
                vectors.append({
                    "id": vector_id,
                    "values": chunk["embedding"],
                    "metadata": metadata
                })
            
            # Upsert to Pinecone
            self.pinecone_service.index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            
            stored_ids = [v["id"] for v in vectors]
            logger.info(f"Stored {len(stored_ids)} chunks in Pinecone namespace: {namespace}")
            return stored_ids
        
        except Exception as e:
            logger.error(f"Error storing in Pinecone: {str(e)}")
            raise
    
    async def process_image(
        self,
        file_path: str,
        workspace_id: str,
        filename: str,
        document_id: str,
        use_ocr: bool = True
    ) -> Dict[str, Any]:
        """
        Complete image processing pipeline optimized for RAG (OpenAI Vision only):
        1. Extract text via OpenAI Vision API (gpt-4o-mini)
        2. Get image description for context
        3. Create smart, context-aware chunks
        4. Generate embeddings with text-embedding-3-large
        5. Store in Pinecone with rich metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting image processing: {filename}")
            
            # Step 1: Extract text and get description (Vision API only)
            extracted_text, image_description, image_size, confidence = self.extract_text_from_image(file_path)
            
            extraction_method = "vision_ocr" if extracted_text else "description_only"
            
            logger.info(f"Extraction complete: {len(extracted_text)} chars, confidence: {confidence:.2f}")
            
            # Step 2: Create smart chunks with image context
            chunks = self.create_smart_chunks(
                text=extracted_text,
                image_description=image_description,
                document_id=document_id,
                workspace_id=workspace_id,
                source_file=filename,
                image_size=image_size,
                confidence_score=confidence
            )
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            chunks = self.generate_embeddings_batch(chunks)
            
            # Step 4: Store in Pinecone
            stored_ids = self.store_chunks_in_pinecone(
                chunks=chunks,
                namespace=workspace_id
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "document_id": document_id,
                "workspace_id": workspace_id,
                "filename": filename,
                "image_size": image_size,
                "image_description": image_description,
                "extraction_method": extraction_method,
                "extracted_text_length": len(extracted_text),
                "confidence_score": confidence,
                "total_chunks": len(chunks),
                "stored_ids": stored_ids,
                "processing_time": processing_time,
                "chunks_metadata": [c["metadata"] for c in chunks]
            }
            
            logger.info(f"✅ Image processing completed in {processing_time:.2f}s")
            return result
        
        except Exception as e:
            logger.error(f"❌ Error processing image {filename}: {str(e)}")
            raise


def get_image_processor() -> ImageProcessor:
    """Get ImageProcessor instance"""
    return ImageProcessor()