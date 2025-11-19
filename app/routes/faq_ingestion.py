from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
import logging
import uuid

from app.models.faq_schemas import FAQUploadResponse, FAQItem
from app.utils.faq_processor import get_faq_processor
from app.db import get_db
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()


# Request model without workspace_id
class FAQUploadRequestAuth(BaseModel):
    category: str
    faq_items: List[FAQItem]


@router.post("/upload", response_model=FAQUploadResponse)
async def upload_faq(
    request: Request,
    faq_data: FAQUploadRequestAuth
):
    """
    Upload and process FAQ items
    
    Request body example:
    ```json
    {
        "category": "Billing",
        "faq_items": [
            {
                "question": "What is your refund policy?",
                "answer": "We offer 30-day refunds for all products."
            },
            {
                "question": "Do you offer support?",
                "answer": "Yes, we offer 24/7 customer support via email, chat, and phone."
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
        
        # Validate category
        if not faq_data.category or not faq_data.category.strip():
            raise HTTPException(status_code=400, detail="category cannot be empty")
        
        # Validate FAQ items
        if not faq_data.faq_items or len(faq_data.faq_items) == 0:
            raise HTTPException(status_code=400, detail="At least one FAQ item required")
        
        # Generate document ID
        document_id = f"faq_{uuid.uuid4().hex[:12]}"
        
        # Process FAQ items
        logger.info(f"Processing {len(faq_data.faq_items)} FAQ items in category '{faq_data.category}' for workspace: {workspace_id}")
        faq_processor = get_faq_processor()
        result = await faq_processor.process_faq_items(
            faq_items=faq_data.faq_items,
            workspace_id=workspace_id,
            category=faq_data.category,
            document_id=document_id
        )
        
        return FAQUploadResponse(
            success=True,
            document_id=result["document_id"],
            workspace_id=workspace_id,
            category=faq_data.category,
            total_items=len(faq_data.faq_items),
            total_chunks=result["total_chunks"],
            message=f"FAQ processed successfully: {result['total_chunks']} chunks created",
            processing_time=result["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading FAQ: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process FAQ: {str(e)}"
        )


@router.get("")
async def list_faq_documents(
    request: Request,
    limit: int = 50,
    skip: int = 0
):
    """
    List FAQ documents for the authenticated workspace
    """
    try:
        user = request.state.user
        workspace_id = user["workspace_id"]

        db = get_db()
        documents = db.list_documents(
            workspace_id=workspace_id,
            document_type="faq",
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
        logger.error(f"Error listing FAQ documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list FAQ documents: {str(e)}")


@router.delete("/{document_id}")
async def delete_faq_document(request: Request, document_id: str):
    """Delete a FAQ document and its chunks/vectors"""
    try:
        user = request.state.user
        workspace_id = user["workspace_id"]

        db = get_db()

        document = db.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document["workspace_id"] != workspace_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # ensure document type matches
        if document.get("document_type") != "faq":
            raise HTTPException(status_code=400, detail="Document is not a FAQ document")

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
            "message": "FAQ document deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting FAQ document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete FAQ document: {str(e)}")