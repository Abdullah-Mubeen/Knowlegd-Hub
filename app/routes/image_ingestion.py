from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List
import logging
import uuid
from pathlib import Path

from app.models.image_schemas import ImageUploadResponse
from app.utils.file_handler import get_file_handler
from app.utils.image_processor import get_image_processor
from app.db import get_db
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_IMAGES_PER_UPLOAD = 10
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


@router.post("/upload", response_model=dict)
async def upload_images(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """
    Upload and process multiple images with smart chunking
    
    - **files**: Image files (PNG, JPG, JPEG - max 10 files)
    
    **Note**: workspace_id is automatically extracted from your JWT token
    
    Returns list of processed images with chunks and embeddings
    """
    try:
        # Get workspace_id from authenticated user
        user = request.state.user
        workspace_id = user["workspace_id"]
        
        # Validate number of files
        if len(files) > MAX_IMAGES_PER_UPLOAD:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_IMAGES_PER_UPLOAD} images per upload"
            )
        
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        file_handler = get_file_handler()
        image_processor = get_image_processor()
        results = []
        failed = []
        
        for file in files:
            try:
                # Validate extension
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in ALLOWED_EXTENSIONS:
                    failed.append({
                        "filename": file.filename,
                        "error": f"Only {', '.join(ALLOWED_EXTENSIONS)} allowed"
                    })
                    continue
                
                # Validate file
                is_valid, error_msg = file_handler.validate_file(file)
                if not is_valid:
                    failed.append({"filename": file.filename, "error": error_msg})
                    continue
                
                await file.seek(0)
                
                # Generate document ID
                document_id = f"img_{uuid.uuid4().hex[:12]}"
                
                # Save file
                logger.info(f"Saving image: {file.filename} for workspace: {workspace_id}")
                file_path, file_metadata = await file_handler.save_upload_file(
                    file=file,
                    workspace_id=workspace_id,
                    document_id=document_id,
                    validate=False
                )
                
                # Process image (extract text, create chunks, generate embeddings)
                logger.info(f"Processing image: {file.filename}")
                result = await image_processor.process_image(
                    file_path=file_path,
                    workspace_id=workspace_id,
                    filename=file.filename,
                    use_ocr=True
                )
                
                results.append({
                    "success": True,
                    "document_id": result["document_id"],
                    "filename": file.filename,
                    "image_size": result["image_size"],
                    "total_chunks": result["total_chunks"],
                    "extraction_method": result["extraction_method"],
                    "file_size": file_metadata["file_size"],
                    "processing_time": result["processing_time"]
                })
                
                logger.info(f"Image processed: {document_id}")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                failed.append({"filename": file.filename, "error": str(e)})
        
        # Return results
        return {
            "success": len(results) > 0,
            "processed": results,
            "failed": failed,
            "total_processed": len(results),
            "total_failed": len(failed),
            "workspace_id": workspace_id,
            "message": f"Processed {len(results)}/{len(files)} images"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")



@router.get("")
async def list_image_documents(
    request: Request,
    limit: int = 50,
    skip: int = 0
):
    """List image documents for workspace"""
    try:
        user = request.state.user
        workspace_id = user["workspace_id"]

        db = get_db()
        documents = db.list_documents(
            workspace_id=workspace_id,
            document_type="image",
            limit=limit,
            skip=skip
        )

        return {"documents": documents, "total": len(documents), "limit": limit, "skip": skip}

    except Exception as e:
        logger.error(f"Error listing image documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list images: {str(e)}")


@router.delete("/{document_id}")
async def delete_image_document(request: Request, document_id: str):
    """Delete image document and its associated chunks/vectors"""
    try:
        user = request.state.user
        workspace_id = user["workspace_id"]

        db = get_db()
        document = db.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document["workspace_id"] != workspace_id:
            raise HTTPException(status_code=403, detail="Access denied")

        if document.get("document_type") != "image":
            raise HTTPException(status_code=400, detail="Document is not an image document")

        deleted_chunks = db.delete_chunks_by_document(document_id)

        from app.utils.pinecone_service import get_pinecone_service
        pinecone_service = get_pinecone_service()
        pinecone_service.delete_by_metadata(filter_dict={"document_id": document_id}, namespace=workspace_id)

        db.delete_document(document_id)

        db.update_user_stats(user_id=user["user_id"], increment_documents=-1, increment_chunks=-deleted_chunks)

        return {"success": True, "document_id": document_id, "deleted_chunks": deleted_chunks, "message": "Image document deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image document: {str(e)}")
