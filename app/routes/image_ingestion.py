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
        document_id = f"images_{uuid.uuid4().hex[:12]}"
        
        processed_images = []
        for file in files:
            if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Invalid file type")
            
            await file.seek(0)
            file_path, file_metadata = await file_handler.save_upload_file(
                file, workspace_id, document_id
            )
            processed_data = image_processor.process_image(file_path, workspace_id, document_id)
            processed_images.append(processed_data)
        
        return ImageUploadResponse(
            document_id=document_id,
            processed_images=processed_images
        )
    except Exception as e:
        logger.error(f"Error uploading images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.delete("/ingest/images/{preprocessed_id}")
async def delete_images(preprocessed_id: str):
    """
    Delete preprocessed image documents
    
    - **preprocessed_id**: ID of the preprocessed document
    """
    try:
        db = get_db()
        result = db.delete_document(preprocessed_id)
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/ingest/images/{document_id}")
async def get_images(document_id: str):
    """
    Retrieve image documents by their ID
    
    - **document_id**: ID of the document
    """
    try:
        db = get_db()
        document = db.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        logger.error(f"Error retrieving images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")