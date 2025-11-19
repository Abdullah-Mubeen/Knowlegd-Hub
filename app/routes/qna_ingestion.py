from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
import logging
import uuid

from app.models.qna_schemas import QnAUploadResponse, QnAPair
from app.utils.qna_processor import get_qna_processor
from app.db import get_db
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()


# Request model without workspace_id
class QnAUploadRequestAuth(BaseModel):
    qna_pairs: List[QnAPair]


@router.post("/upload", response_model=QnAUploadResponse)
async def upload_qna(
    request: Request,
    qna_data: QnAUploadRequestAuth
):
    """
    Upload and process Q&A pairs
    
    Request body example:
    ```json
    {
        "qna_pairs": [
            {
                "question": "What is your product?",
                "answer": "Our product is...",
                "category": "General"
            },
            {
                "question": "How to get support?",
                "answer": "Contact us at...",
                "category": "Support"
            }
        ]
    }
    ```
    
    **Note**: workspace_id is automatically extracted from your JWT token
    """
    try:
        # Get workspace_id from authenticated user
        user = request.state.user
        workspace_id = user["workspace_id"]
        
        # Validate Q&A pairs
        if not qna_data.qna_pairs or len(qna_data.qna_pairs) == 0:
            raise HTTPException(status_code=400, detail="At least one Q&A pair required")
        
        # Generate document ID
        document_id = f"qna_{uuid.uuid4().hex[:12]}"
        
        # Process Q&A pairs
        logger.info(f"Processing {len(qna_data.qna_pairs)} Q&A pairs for workspace: {workspace_id}")
        qna_processor = get_qna_processor()
        
        # Convert Pydantic models → dicts
        pairs_dict = [pair.dict() for pair in qna_data.qna_pairs]

        result = await qna_processor.process_qna_pairs(
            qna_pairs=pairs_dict,
            workspace_id=workspace_id,
            document_id=document_id
        )

        
        return QnAUploadResponse(
            success=True,
            document_id=result["document_id"],
            workspace_id=workspace_id,
            total_pairs=len(qna_data.qna_pairs),
            total_chunks=result["total_chunks"],
            message=f"Q&A processed successfully: {result['total_chunks']} chunks created",
            processing_time=result["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading Q&A: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process Q&A: {str(e)}"
        )


@router.get("")
async def list_qna_documents(request: Request, limit: int = 50, skip: int = 0):
    """List QnA documents for the authenticated workspace"""
    try:
        user = request.state.user
        workspace_id = user["workspace_id"]

        db = get_db()
        documents = db.list_documents(
            workspace_id=workspace_id,
            document_type="qna",
            limit=limit,
            skip=skip
        )

        return {"documents": documents, "total": len(documents), "limit": limit, "skip": skip}

    except Exception as e:
        logger.error(f"Error listing QnA documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list QnA documents: {str(e)}")


@router.delete("/{document_id}")
async def delete_qna_document(request: Request, document_id: str):
    """Delete a QnA document and its chunks/vectors"""
    try:
        user = request.state.user
        workspace_id = user["workspace_id"]

        db = get_db()
        document = db.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document["workspace_id"] != workspace_id:
            raise HTTPException(status_code=403, detail="Access denied")

        if document.get("document_type") != "qna":
            raise HTTPException(status_code=400, detail="Document is not a QnA document")

        deleted_chunks = db.delete_chunks_by_document(document_id)

        from app.utils.pinecone_service import get_pinecone_service
        pinecone_service = get_pinecone_service()
        pinecone_service.delete_by_metadata(filter_dict={"document_id": document_id}, namespace=workspace_id)

        db.delete_document(document_id)

        db.update_user_stats(user_id=user["user_id"], increment_documents=-1, increment_chunks=-deleted_chunks)

        return {"success": True, "document_id": document_id, "deleted_chunks": deleted_chunks, "message": "QnA document deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting QnA document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete QnA document: {str(e)}")