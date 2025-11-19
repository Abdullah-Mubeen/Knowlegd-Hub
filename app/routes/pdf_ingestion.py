from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import Optional
import logging
import uuid

from app.models.pdf_schemas import PDFUploadResponse
from app.utils.file_handler import get_file_handler
from app.utils.pdf_processor import get_pdf_processor
from app.db import get_db
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.get("")
async def list_pdf_documents(
    request: Request,
    limit: int = 50,
    skip: int = 0
):
    """List PDF documents for the workspace"""
    try:
        user = request.state.user
        workspace_id = user["workspace_id"]

        db = get_db()
        documents = db.list_documents(
            workspace_id=workspace_id,
            document_type="pdf",
            limit=limit,
            skip=skip
        )

        return {"documents": documents, "total": len(documents), "limit": limit, "skip": skip}

    except Exception as e:
        logger.error(f"Error listing PDF documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list PDFs: {str(e)}")


@router.delete("/{document_id}")
async def delete_pdf_document(request: Request, document_id: str):
    """Delete a PDF document and its chunks/vectors"""
    try:
        user = request.state.user
        workspace_id = user["workspace_id"]

        db = get_db()
        document = db.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document["workspace_id"] != workspace_id:
            raise HTTPException(status_code=403, detail="Access denied")

        if document.get("document_type") != "pdf":
            raise HTTPException(status_code=400, detail="Document is not a PDF document")

        deleted_chunks = db.delete_chunks_by_document(document_id)

        from app.utils.pinecone_service import get_pinecone_service
        pinecone_service = get_pinecone_service()
        pinecone_service.delete_by_metadata(filter_dict={"document_id": document_id}, namespace=workspace_id)

        db.delete_document(document_id)

        db.update_user_stats(user_id=user["user_id"], increment_documents=-1, increment_chunks=-deleted_chunks)

        return {"success": True, "document_id": document_id, "deleted_chunks": deleted_chunks, "message": "PDF document deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting PDF document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete PDF document: {str(e)}")
