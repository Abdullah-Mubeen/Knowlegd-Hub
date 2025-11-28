from typing import List, Dict, Any
import logging
from datetime import datetime
import uuid
import time

from app.models.faq_schemas import FAQItem
from app.utils.text_splitter import get_text_chunker, ChunkConfig
from app.utils.openai_service import get_openai_service
from app.utils.pinecone_service import get_pinecone_service
from app.db import get_db

logger = logging.getLogger(__name__)


class FAQProcessor:
    """Process FAQ items with chunking and embeddings"""
    
    def __init__(self):
        self.chunker = get_text_chunker(ChunkConfig(
            chunk_size=600,
            overlap_pct=0.1
        ))
        self.openai_service = get_openai_service()
        self.pinecone_service = get_pinecone_service()
        self.db = get_db()
    
    def create_chunks_from_faq(
        self,
        faq_items: List[FAQItem],
        document_id: str,
        workspace_id: str,
        category: str
    ) -> List[Dict[str, Any]]:
        """Create chunks from FAQ items"""
        chunks = []
        
        for idx, faq in enumerate(faq_items):
            # Combine question and answer
            combined_text = f"FAQ: {faq.question}\n\n{faq.answer}"
            token_count = self.chunker.count_tokens(combined_text)
            
            # If answer is too long, split it
            if token_count > self.chunker.config.chunk_size:
                answer_chunks = self.chunker.split_by_sentences(
                    faq.answer,
                    chunk_size=self.chunker.config.chunk_size - self.chunker.count_tokens(f"FAQ: {faq.question}\n\n(part /): ")
                )
                
                for part_idx, answer_part in enumerate(answer_chunks, start=1):
                    chunk_text = f"FAQ: {faq.question}\n\n(part {part_idx}/{len(answer_chunks)}): {answer_part}"
                    
                    chunk_data = {
                        "id": f"chunk_{uuid.uuid4().hex[:12]}",
                        "text": chunk_text,
                        "metadata": {
                            "document_id": document_id,
                            "workspace_id": workspace_id,
                            "document_type": "faq",
                            "source_file": "faq_import",
                            "chunk_index": len(chunks),
                            "category": category,
                            "question": faq.question,
                            "answer_part": part_idx,
                            "total_answer_parts": len(answer_chunks),
                            "upload_date": datetime.utcnow().isoformat()
                        }
                    }
                    chunks.append(chunk_data)
            else:
                # Single chunk for FAQ item
                chunk_data = {
                    "id": f"chunk_{uuid.uuid4().hex[:12]}",
                    "text": combined_text,
                    "metadata": {
                        "document_id": document_id,
                        "workspace_id": workspace_id,
                        "document_type": "faq",
                        "source_file": "faq_import",
                        "chunk_index": len(chunks),
                        "category": category,
                        "question": faq.question,
                        "answer_part": 1,
                        "total_answer_parts": 1,
                        "upload_date": datetime.utcnow().isoformat()
                    }
                }
                chunks.append(chunk_data)
        
        # Update total chunks
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(faq_items)} FAQ items")
        return chunks
    
    def generate_embeddings_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks"""
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
    
    async def process_faq_items(
        self,
        faq_items: List[FAQItem],
        workspace_id: str,
        category: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Complete FAQ processing pipeline:
        1. Create document record in MongoDB
        2. Create chunks from FAQ items
        3. Generate embeddings
        4. Store chunks in MongoDB
        5. Store in Pinecone
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting FAQ processing: {len(faq_items)} items in {category}")
            
            # Step 1: Create document record in MongoDB
            document_record = self.db.create_document(
                document_id=document_id,
                workspace_id=workspace_id,
                document_type="faq",
                filename=f"faq_{category}.json",  # Virtual filename
                file_path=f"virtual://faq/{document_id}",  # Virtual path (no actual file)
                file_size=sum(len(item.question + item.answer) for item in faq_items),
                total_chunks=0,  # Will update after chunk creation
                file_metadata={
                    "source": "api_upload",
                    "total_items": len(faq_items),
                    "category": category
                },
                processing_metadata={
                    "extraction_method": "direct_input",
                    "format": "faq_items"
                }
            )
            
            logger.info(f"Created FAQ document record in MongoDB")
            
            # Step 2: Create chunks
            chunks = self.create_chunks_from_faq(
                faq_items=faq_items,
                document_id=document_id,
                workspace_id=workspace_id,
                category=category
            )
            
            # Step 3: Generate embeddings
            chunks = self.generate_embeddings_batch(chunks)
            
            # Step 4: Store chunks in MongoDB
            mongo_chunks = []
            for i, chunk_data in enumerate(chunks):
                mongo_chunk = {
                    "chunk_id": chunk_data["id"],
                    "document_id": document_id,
                    "workspace_id": workspace_id,
                    "text": chunk_data["text"],
                    "pinecone_id": "",  # Will be filled after Pinecone storage
                    "metadata": chunk_data["metadata"]
                }
                mongo_chunks.append(mongo_chunk)
            
            # Step 5: Store in Pinecone
            stored_ids = self.store_chunks_in_pinecone(
                chunks=chunks,
                namespace=workspace_id
            )
            
            # Update pinecone_id in MongoDB chunks
            for i, chunk in enumerate(mongo_chunks):
                chunk["pinecone_id"] = stored_ids[i]
            
            self.db.create_chunks_batch(mongo_chunks)
            
            logger.info(f"Stored {len(mongo_chunks)} chunks in MongoDB with Pinecone IDs")
            
            # Step 6: Update user statistics
            user = self.db.get_user_by_workspace(workspace_id)
            if user:
                self.db.update_user_stats(
                    user_id=user["user_id"],
                    increment_documents=1,
                    increment_chunks=len(chunks)
                )
            
            processing_time = time.time() - start_time
            
            result = {
                "document_id": document_id,
                "workspace_id": workspace_id,
                "total_chunks": len(chunks),
                "stored_ids": stored_ids,
                "processing_time": processing_time
            }
            
            logger.info(f"FAQ processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing FAQ: {str(e)}")
            raise


def get_faq_processor() -> FAQProcessor:
    """Get FAQProcessor instance"""
    return FAQProcessor()