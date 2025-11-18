from typing import List, Dict, Any, Optional
import logging
import time

from app.utils.pinecone_service import get_pinecone_service
from app.utils.openai_service import get_openai_service
from app.db import get_db

logger = logging.getLogger(__name__)


class RAGQueryService:
    """RAG query service with MongoDB logging"""
    
    def __init__(self):
        self.pinecone_service = get_pinecone_service()
        self.openai_service = get_openai_service()
        self.db = get_db()
    
    def query(
        self,
        question: str,
        user_id: str,
        workspace_id: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute RAG query: search vectors, retrieve chunks, generate answer
        
        Args:
            question: User's question
            user_id: User ID
            workspace_id: Workspace ID
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters
            
        Returns:
            Query results with answer and sources
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query from user {user_id}: {question[:100]}...")
            
            # 1. Search Pinecone for similar chunks
            search_results = self.pinecone_service.similarity_search(
                query=question,
                namespace=workspace_id,
                top_k=top_k,
                filter_dict=filters
            )
            
            if not search_results:
                logger.warning(f"No results found for query")
                return {
                    "answer": "I couldn't find any relevant information in your knowledge base to answer this question.",
                    "sources": [],
                    "total_sources": 0
                }
            
            logger.info(f"Found {len(search_results)} relevant chunks")
            
            # 2. Extract texts and metadata
            context_chunks = []
            pinecone_ids = []
            
            for text, score, metadata in search_results:
                context_chunks.append(text)
                pinecone_ids.append(metadata.get("pinecone_id"))
            
            # 3. Get full chunk data from MongoDB (for better metadata)
            mongo_chunks = self.db.get_chunks_by_pinecone_ids(pinecone_ids)
            
            # Create mapping for easy lookup
            chunk_map = {chunk["pinecone_id"]: chunk for chunk in mongo_chunks}
            
            # 4. Generate answer using RAG
            user = self.db.get_user_by_id(user_id)
            business_name = user.get("company_name") if user else None
            
            answer = self.openai_service.generate_rag_answer(
                question=question,
                context_chunks=context_chunks,
                business_name=business_name
            )
            
            logger.info(f"Generated answer (length: {len(answer)})")
            
            # 5. Prepare source information
            sources = []
            for i, (text, score, metadata) in enumerate(search_results):
                pinecone_id = metadata.get("pinecone_id")
                mongo_chunk = chunk_map.get(pinecone_id, {})
                
                source = {
                    "chunk_id": mongo_chunk.get("chunk_id", "unknown"),
                    "document_id": metadata.get("document_id"),
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "score": float(score),
                    "metadata": {
                        "document_type": metadata.get("document_type"),
                        "source_file": metadata.get("source_file"),
                        "page_number": metadata.get("page_number"),
                        "chunk_index": metadata.get("chunk_index")
                    }
                }
                sources.append(source)
            
            # 6. Log query to MongoDB
            query_metadata = {
                "response_time": round(time.time() - start_time, 2),
                "top_k": top_k,
                "num_sources": len(sources),
                "filters": filters
            }
            
            self.db.log_query(
                user_id=user_id,
                workspace_id=workspace_id,
                question=question,
                returned_chunks=[{
                    "chunk_id": s["chunk_id"],
                    "document_id": s["document_id"],
                    "score": s["score"]
                } for s in sources],
                answer=answer,
                metadata=query_metadata
            )
            
            logger.info(f"Query completed in {query_metadata['response_time']}s")
            
            return {
                "answer": answer,
                "sources": sources,
                "total_sources": len(sources),
                "response_time": query_metadata["response_time"]
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def get_query_history(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get user's query history"""
        try:
            queries = self.db.get_user_queries(user_id=user_id, limit=limit)
            
            # Format for response
            history = []
            for query in queries:
                history.append({
                    "question": query["question"],
                    "answer": query["answer"],
                    "num_sources": len(query.get("returned_chunks", [])),
                    "timestamp": query["created_at"].isoformat(),
                    "response_time": query.get("metadata", {}).get("response_time")
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting query history: {str(e)}")
            raise


# Singleton instance
_rag_service = None

def get_rag_service() -> RAGQueryService:
    """Get or create RAGQueryService singleton"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGQueryService()
    return _rag_service