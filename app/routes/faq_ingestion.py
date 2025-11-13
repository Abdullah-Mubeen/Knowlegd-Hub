from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
import logging
import uuid

from app.models.faq_schemas import FAQUploadRequest, FAQUploadResponse
from app.utils.faq_processor import get_faq_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-hub/ingest/faq", tags=["FAQ Ingestion"])


@router.post("/upload", response_model=FAQUploadResponse)
async def upload_faq(request: FAQUploadRequest):
    """
    Upload and process FAQ items
    
    Request body example:
    ```json
    {
        "business_id": "company_123",
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
    """
    try:
        # Validate business_id
        if not request.business_id or not request.business_id.strip():
            raise HTTPException(status_code=400, detail="business_id cannot be empty")
        
        # Validate category
        if not request.category or not request.category.strip():
            raise HTTPException(status_code=400, detail="category cannot be empty")
        
        # Validate FAQ items
        if not request.faq_items or len(request.faq_items) == 0:
            raise HTTPException(status_code=400, detail="At least one FAQ item required")
        
        # Generate document ID
        document_id = f"faq_{uuid.uuid4().hex[:12]}"
        
        # Process FAQ items
        logger.info(f"Processing {len(request.faq_items)} FAQ items in category '{request.category}' for business: {request.business_id}")
        faq_processor = get_faq_processor()
        result = await faq_processor.process_faq_items(
            faq_items=request.faq_items,
            business_id=request.business_id,
            category=request.category,
            document_id=document_id
        )
        
        return FAQUploadResponse(
            success=True,
            document_id=result["document_id"],
            business_id=request.business_id,
            category=request.category,
            total_items=len(request.faq_items),
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


@router.post("/upload-file", response_model=FAQUploadResponse)
async def upload_faq_file(
    file: UploadFile = File(...),
    business_id: str = None,
    category: str = None
):
    """
    Upload FAQ items from JSON file
    
    - **file**: JSON file containing FAQ items
    - **business_id**: Business identifier (optional - can be in file)
    - **category**: FAQ category (optional - can be in file)
    
    File format example:
    ```json
    {
        "business_id": "company_123",
        "category": "Billing",
        "faq_items": [
            {
                "question": "What is the refund policy?",
                "answer": "We offer 30-day refunds..."
            }
        ]
    }
    ```
    """
    try:
        import json
        
        if not file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="Only JSON files allowed")
        
        # Read file
        content = await file.read()
        
        try:
            file_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
        
        if not isinstance(file_data, dict):
            raise HTTPException(status_code=400, detail="JSON must be an object")
        
        # Get business_id and category from parameters or file
        bid = business_id or file_data.get("business_id")
        cat = category or file_data.get("category")
        
        if not bid or not bid.strip():
            raise HTTPException(status_code=400, detail="business_id is required")
        
        if not cat or not cat.strip():
            raise HTTPException(status_code=400, detail="category is required")
        
        # Parse FAQ items
        faq_data = file_data.get("faq_items", [])
        if not isinstance(faq_data, list) or len(faq_data) == 0:
            raise HTTPException(status_code=400, detail="faq_items must be non-empty array")
        
        # Validate and create request
        try:
            request = FAQUploadRequest(
                business_id=bid,
                category=cat,
                faq_items=faq_data
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid FAQ data: {str(e)}")
        
        # Generate document ID
        document_id = f"faq_{uuid.uuid4().hex[:12]}"
        
        # Process FAQ items
        logger.info(f"Processing {len(request.faq_items)} FAQ items from file in category '{cat}' for business: {bid}")
        faq_processor = get_faq_processor()
        result = await faq_processor.process_faq_items(
            faq_items=request.faq_items,
            business_id=request.business_id,
            category=request.category,
            document_id=document_id
        )
        
        return FAQUploadResponse(
            success=True,
            document_id=result["document_id"],
            business_id=request.business_id,
            category=request.category,
            total_items=len(request.faq_items),
            total_chunks=result["total_chunks"],
            message=f"FAQ file processed successfully: {result['total_chunks']} chunks created",
            processing_time=result["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading FAQ file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process FAQ file: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "endpoint": "faq_ingestion",
        "supported_formats": ["json"],
        "methods": ["POST /upload", "POST /upload-file"]
    }