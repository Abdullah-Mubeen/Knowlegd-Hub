from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
import logging
import uuid

from app.models.qna_schemas import QnAUploadRequest, QnAUploadResponse
from app.utils.qna_processor import get_qna_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-hub/ingest/qna", tags=["Q&A Ingestion"])


@router.post("/upload", response_model=QnAUploadResponse)
async def upload_qna(request: QnAUploadRequest):
    """
    Upload and process Q&A pairs
    
    Request body example:
    ```json
    {
        "business_id": "company_123",
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
    """
    try:
        # Validate business_id
        if not request.business_id or not request.business_id.strip():
            raise HTTPException(status_code=400, detail="business_id cannot be empty")
        
        # Validate Q&A pairs
        if not request.qna_pairs or len(request.qna_pairs) == 0:
            raise HTTPException(status_code=400, detail="At least one Q&A pair required")
        
        # Generate document ID
        document_id = f"qna_{uuid.uuid4().hex[:12]}"
        
        # Process Q&A pairs
        logger.info(f"Processing {len(request.qna_pairs)} Q&A pairs for business: {request.business_id}")
        qna_processor = get_qna_processor()
        result = await qna_processor.process_qna_pairs(
            qna_pairs=request.qna_pairs,
            business_id=request.business_id,
            document_id=document_id
        )
        
        return QnAUploadResponse(
            success=True,
            document_id=result["document_id"],
            business_id=request.business_id,
            total_pairs=len(request.qna_pairs),
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


@router.post("/upload-file", response_model=QnAUploadResponse)
async def upload_qna_file(
    file: UploadFile = File(...),
    business_id: str = None
):
    """
    Upload Q&A pairs from JSON file
    
    - **file**: JSON file containing Q&A pairs
    - **business_id**: Business identifier (optional - can be in file)
    
    File format example:
    ```json
    {
        "business_id": "company_123",
        "qna_pairs": [
            {
                "question": "...",
                "answer": "...",
                "category": "General"
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
        
        # Get business_id from parameter or file
        bid = business_id or file_data.get("business_id")
        if not bid or not bid.strip():
            raise HTTPException(status_code=400, detail="business_id is required")
        
        # Parse QnA pairs
        qna_data = file_data.get("qna_pairs", [])
        if not isinstance(qna_data, list) or len(qna_data) == 0:
            raise HTTPException(status_code=400, detail="qna_pairs must be non-empty array")
        
        # Validate and create request
        try:
            request = QnAUploadRequest(
                business_id=bid,
                qna_pairs=qna_data
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid Q&A data: {str(e)}")
        
        # Generate document ID
        document_id = f"qna_{uuid.uuid4().hex[:12]}"
        
        # Process Q&A pairs
        logger.info(f"Processing {len(request.qna_pairs)} Q&A pairs from file for business: {bid}")
        qna_processor = get_qna_processor()
        result = await qna_processor.process_qna_pairs(
            qna_pairs=request.qna_pairs,
            business_id=request.business_id,
            document_id=document_id
        )
        
        return QnAUploadResponse(
            success=True,
            document_id=result["document_id"],
            business_id=request.business_id,
            total_pairs=len(request.qna_pairs),
            total_chunks=result["total_chunks"],
            message=f"Q&A file processed successfully: {result['total_chunks']} chunks created",
            processing_time=result["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading Q&A file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process Q&A file: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "endpoint": "qna_ingestion",
        "supported_formats": ["json"],
        "methods": ["POST /upload", "POST /upload-file"]
    }