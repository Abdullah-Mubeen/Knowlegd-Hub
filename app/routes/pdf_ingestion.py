from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import Optional
import logging
import uuid

from app.models.pdf_schemas import PDFUploadResponse
from app.utils.file_handler import get_file_handler
from app.utils.pdf_processor import get_pdf_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-hub/ingest/pdf", tags=["PDF Ingestion"])


@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload and process a PDF document
    
    - **file**: PDF file (max 60MB)
    
    Note: workspace_id is automatically extracted from JWT token
    """
    try:
        # Get workspace_id from authenticated user
        user = request.state.user
        workspace_id = user["workspace_id"]
        
        # Validate PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")
        
        file_handler = get_file_handler()
        is_valid, error_msg = file_handler.validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        await file.seek(0)
        
        # Generate document ID
        document_id = f"pdf_{uuid.uuid4().hex[:12]}"
        
        # Save file with workspace_id
        logger.info(f"Saving PDF: {file.filename} for workspace: {workspace_id}")
        file_path, file_metadata = await file_handler.save_upload_file(
            file=file,
            business_id=workspace_id,  # Use workspace_id instead of business_id
            document_id=document_id,
            validate=False
        )
        
        # Process PDF
        logger.info(f"Processing PDF: {file.filename}")
        pdf_processor = get_pdf_processor()
        result = await pdf_processor.process_pdf(
            file_path=file_path,
            business_id=workspace_id,
            filename=file.filename
        )
        
        return PDFUploadResponse(
            success=True,
            document_id=result["document_id"],
            business_id=workspace_id,
            filename=file.filename,
            total_pages=result["total_pages"],
            total_chunks=result["total_chunks"],
            file_size=file_metadata["file_size"],
            message=f"PDF processed: {result['total_chunks']} chunks created",
            processing_time=result["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "endpoint": "pdf_ingestion"}