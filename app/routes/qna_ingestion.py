from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
import logging
import uuid

from app.models.qna_schemas import QnAUploadResponse, QnAPair
from app.utils.qna_processor import get_qna_processor
from app.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


class QnAUploadRequestAuth(BaseModel):
    qna_pairs: List[QnAPair]


@router.post("/ingest/qna", response_model=QnAUploadResponse)
async def upload_qna(
    request: Request,
    qna_data: QnAUploadRequestAuth
):
    """
    Upload and process Q&A pairs
    
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


@router.delete("/ingest/qna/{preprocessed_id}")
async def delete_qna(preprocessed_id: str):
    """
    Delete a preprocessed Q&A document
    
    - **preprocessed_id**: ID of the preprocessed document
    """
    try:
        db = get_db()
        result = db.delete_document(preprocessed_id)
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting Q&A: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/ingest/qna/list")
async def list_qna(request: Request):
    """
    List all Q&A documents for the current workspace
    
    **Note**: workspace_id is automatically extracted from your JWT token
    """
    try:
        db = get_db()
        user = request.state.user
        workspace_id = user["workspace_id"]
        docs = db.list_documents(workspace_id=workspace_id, document_type="qna")
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Error listing Q&A: {e}")
        raise HTTPException(500, "Internal Server Error")