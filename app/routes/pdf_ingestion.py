from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import Optional
import logging
import uuid

from app.models.pdf_schemas import PDFUploadResponse
from app.utils.file_handler import get_file_handler
from app.utils.pdf_processor import get_pdf_processor
from app.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest/pdf", response_model=PDFUploadResponse)
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
            workspace_id=workspace_id,  # Use workspace_id instead of workspace_id
            document_id=document_id,
            validate=False
        )
        
        # Process PDF
        logger.info(f"Processing PDF: {file.filename}")
        pdf_processor = get_pdf_processor()
        result = await pdf_processor.process_pdf(
            file_path=file_path,
            workspace_id=workspace_id,
            filename=file.filename
        )
        
        return PDFUploadResponse(
            success=True,
            document_id=result["document_id"],
            workspace_id=workspace_id,
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


@router.delete("/ingest/pdf/{preprocessed_id}")
async def delete_pdf(preprocessed_id: str):
    """
    Delete a preprocessed PDF document
    
    - **preprocessed_id**: ID of the preprocessed document
    """
    try:
        db = get_db()
        result = db.delete_document(preprocessed_id)
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/ingest/pdf/list")
async def list_pdfs(request: Request):
    """
    List all PDF documents for the current workspace
    
    Note: workspace_id is automatically extracted from JWT token
    """
    try:
        db = get_db()
        user = request.state.user
        workspace_id = user["workspace_id"]
        docs = db.list_documents(workspace_id=workspace_id, document_type="pdf")
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Error listing PDFs: {e}")
        raise HTTPException(500, "Internal Server Error")
