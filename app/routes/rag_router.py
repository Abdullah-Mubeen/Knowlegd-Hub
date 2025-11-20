from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from app.utils.rag_query import get_rag_service
from app.db import get_db
from app.utils.rag_conv import get_conversational_rag_service, ConversationMode
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    total_sources: int
    response_time: float


class QueryHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    total: int


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None
    mode: Optional[str] = ConversationMode.CHAT.value


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    total_sources: int
    cited_sources: int
    conversation_id: str
    intent: str
    response_time: float
    rewritten_query: Optional[str] = None


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: Request,
    query_data: QueryRequest
):
    """
    Query the knowledge base using RAG
    
    Request body:
    ```json
    {
        "question": "What is your refund policy?",
        "top_k": 5,
        "filters": {
            "document_type": "pdf"
        }
    }
    ```
    
    Returns answer with source citations
    """
    try:
        # Get authenticated user
        user = request.state.user
        user_id = user["user_id"]
        workspace_id = user["workspace_id"]
        
        # Validate question
        if not query_data.question or not query_data.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Execute RAG query
        rag_service = get_rag_service()
        result = rag_service.query(
            question=query_data.question,
            user_id=user_id,
            workspace_id=workspace_id,
            top_k=query_data.top_k,
            filters=query_data.filters
        )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@router.get("/history", response_model=QueryHistoryResponse)
async def get_query_history(
    request: Request,
    limit: int = 20
):
    """
    Get user's query history
    
    Returns list of previous queries and answers
    """
    try:
        # Get authenticated user
        user = request.state.user
        user_id = user["user_id"]
        
        # Get query history
        rag_service = get_rag_service()
        history = rag_service.get_query_history(user_id=user_id, limit=limit)
        
        return QueryHistoryResponse(
            history=history,
            total=len(history)
        )
        
    except Exception as e:
        logger.error(f"Error getting query history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get query history: {str(e)}"
        )


@router.get("/workspace/stats")
async def get_workspace_stats(request: Request):
    """
    Get workspace statistics
    
    Returns document and chunk counts
    """
    try:
        # Get authenticated user
        user = request.state.user
        workspace_id = user["workspace_id"]
        
        # Get stats from MongoDB
        db = get_db()
        stats = db.get_workspace_stats(workspace_id)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting workspace stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workspace stats: {str(e)}"
        )


@router.get("/documents")
async def list_documents(
    request: Request,
    document_type: Optional[str] = None,
    limit: int = 50,
    skip: int = 0
):
    """
    List documents in workspace
    
    Query parameters:
    - document_type: Filter by type (pdf, image, qna, faq)
    - limit: Max number of results (default 50)
    - skip: Number of results to skip (pagination)
    """
    try:
        # Get authenticated user
        user = request.state.user
        workspace_id = user["workspace_id"]
        
        # Get documents from MongoDB
        db = get_db()
        documents = db.list_documents(
            workspace_id=workspace_id,
            document_type=document_type,
            limit=limit,
            skip=skip
        )
        
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "skip": skip
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    request: Request,
    document_id: str
):
    """
    Delete a document and its chunks
    
    This will:
    1. Soft delete document in MongoDB
    2. Delete chunks from MongoDB
    3. Delete vectors from Pinecone
    """
    try:
        # Get authenticated user
        user = request.state.user
        workspace_id = user["workspace_id"]
        
        db = get_db()
        
        # Verify document belongs to user's workspace
        document = db.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if document["workspace_id"] != workspace_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete chunks from MongoDB
        deleted_chunks = db.delete_chunks_by_document(document_id)
        
        # Delete vectors from Pinecone
        from app.utils.pinecone_service import get_pinecone_service
        pinecone_service = get_pinecone_service()
        pinecone_service.delete_by_metadata(
            filter_dict={"document_id": document_id},
            namespace=workspace_id
        )
        
        # Soft delete document
        db.delete_document(document_id)
        
        # Update user stats
        db.update_user_stats(
            user_id=user["user_id"],
            increment_documents=-1,
            increment_chunks=-deleted_chunks
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "deleted_chunks": deleted_chunks,
            "message": "Document deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db = get_db()
        db_health = db.health_check()
        
        return {
            "status": "healthy",
            "endpoint": "rag_query",
            "database": db_health
        }
    except Exception as e:
        return {
            "status": "degraded",
            "endpoint": "rag_query",
            "error": str(e)
        }