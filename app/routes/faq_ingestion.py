from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
import logging
import uuid

from app.models.faq_schemas import FAQUploadResponse, FAQItem
from app.utils.faq_processor import get_faq_processor
from app.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


class FAQUploadRequestAuth(BaseModel):
    category: str
    faq_items: List[FAQItem]


@router.post("/ingest/faq", response_model=FAQUploadResponse)
async def upload_faq(
    request: Request,
    faq_data: FAQUploadRequestAuth
):
    """
    Upload and process FAQ items
    
    **Note**: workspace_id is automatically extracted from your JWT token
    """
    try:
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


@router.delete("/ingest/faq/{preprocessed_id}")
async def delete_faq(preprocessed_id: str):
    """
    Delete a preprocessed FAQ document
    
    - **preprocessed_id**: ID of the preprocessed document
    """
    try:
        db = get_db()
        result = db.delete_document(preprocessed_id)
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting FAQ: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/ingest/faq/list")
async def list_faq(request: Request):
    """
    List all FAQ documents for the current workspace
    
    **Note**: workspace_id is automatically extracted from your JWT token
    """
    try:
        db = get_db()
        user = request.state.user
        workspace_id = user["workspace_id"]
        docs = db.list_documents(workspace_id=workspace_id, document_type="faq")
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Error listing FAQs: {e}")
        raise HTTPException(500, "Internal Server Error")