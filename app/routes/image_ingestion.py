from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List
import logging
import uuid
from pathlib import Path

from app.models.image_schemas import ImageUploadResponse
from app.utils.file_handler import get_file_handler
from app.utils.image_processor import get_image_processor
from app.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_IMAGES_PER_UPLOAD = 10
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


@router.post("/ingest/images", response_model=ImageUploadResponse)
async def upload_images(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """
    Upload and process multiple images
    
    **Note**: workspace_id is automatically extracted from your JWT token
    """
    try:
        user = request.state.user
        workspace_id = user["workspace_id"]
        
        if len(files) > MAX_IMAGES_PER_UPLOAD:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_IMAGES_PER_UPLOAD} images per upload"
            )
        
        file_handler = get_file_handler()
        image_processor = get_image_processor()
        
        processed_images = []
        for file in files:
            if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Invalid file type")
            
            await file.seek(0)
            # Generate unique document_id for each image
            document_id = f"img_{uuid.uuid4().hex[:12]}"
            file_path, file_metadata = await file_handler.save_upload_file(
                file, workspace_id, document_id
            )
            # Await the async process_image function - pass document_id
            processed_data = await image_processor.process_image(
                file_path, workspace_id, file.filename, document_id
            )
            processed_images.append(processed_data)
        
        # Return the first processed image with aggregate data
        if not processed_images:
            raise HTTPException(status_code=400, detail="No images processed")
        
        first_image = processed_images[0]
        total_chunks = sum(img.get("total_chunks", 0) for img in processed_images)
        
        return ImageUploadResponse(
            document_id=first_image["document_id"],
            workspace_id=workspace_id,
            filename=first_image["filename"],
            image_size=first_image["image_size"],
            total_chunks=total_chunks,
            extraction_method=first_image["extraction_method"],
            extracted_text_length=first_image["extracted_text_length"],
            confidence_score=first_image.get("confidence_score"),
            stored_ids=first_image.get("stored_ids", []),
            processing_time=first_image.get("processing_time"),
            chunks_metadata=first_image.get("chunks_metadata", []),
            processed_images=processed_images
        )
    except Exception as e:
        logger.error(f"Error uploading images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.delete("/ingest/images/{preprocessed_id}")
async def delete_images(preprocessed_id: str):
    """
    Delete preprocessed image documents, their chunks, and embeddings
    
    - **preprocessed_id**: ID of the preprocessed document
    """
    try:
        db = get_db()
        
        # Verify document exists
        document = db.get_document(preprocessed_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        workspace_id = document["workspace_id"]
        
        # Delete chunks from MongoDB
        deleted_chunks = db.delete_chunks_by_document(preprocessed_id)
        
        # Delete vectors from Pinecone
        from app.utils.pinecone_service import get_pinecone_service
        pinecone_service = get_pinecone_service()
        pinecone_service.delete_by_metadata(
            filter_dict={"document_id": preprocessed_id},
            namespace=workspace_id
        )
        
        # Soft delete document
        db.delete_document(preprocessed_id)
        
        return {
            "success": True,
            "message": "Image document deleted successfully",
            "deleted_chunks": deleted_chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/ingest/images/list")
async def list_images(request: Request):
    """
    List all image documents for the current workspace
    
    **Note**: workspace_id is automatically extracted from your JWT token
    """
    try:
        db = get_db()
        user = request.state.user
        workspace_id = user["workspace_id"]
        docs = db.list_documents(workspace_id=workspace_id, document_type="image")
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Error listing images: {e}")
        raise HTTPException(500, "Internal Server Error")