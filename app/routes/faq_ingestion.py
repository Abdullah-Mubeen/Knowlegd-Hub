from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
import logging
import uuid

from app.models.faq_schemas import FAQUploadResponse, FAQItem
from app.utils.faq_processor import get_faq_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-hub/ingest/faq", tags=["FAQ Ingestion"])


# Request model without business_id
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
            business_id=workspace_id,
            category=faq_data.category,
            document_id=document_id
        )
        
        return FAQUploadResponse(
            success=True,
            document_id=result["document_id"],
            business_id=workspace_id,
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


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "endpoint": "faq_ingestion",
        "supported_formats": ["json"],
        "methods": ["POST /upload"]
    }