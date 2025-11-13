from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
import logging
import uuid

from app.models.qna_schemas import QnAUploadResponse, QnAPair
from app.utils.qna_processor import get_qna_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-hub/ingest/qna", tags=["Q&A Ingestion"])


# Request model without business_id
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
        result = await qna_processor.process_qna_pairs(
            qna_pairs=qna_data.qna_pairs,
            business_id=workspace_id,
            document_id=document_id
        )
        
        return QnAUploadResponse(
            success=True,
            document_id=result["document_id"],
            business_id=workspace_id,
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


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "endpoint": "qna_ingestion",
        "supported_formats": ["json"],
        "methods": ["POST /upload"]
    }