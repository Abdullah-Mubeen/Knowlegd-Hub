from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ImageUploadRequest(BaseModel):
    """Request schema for image upload"""
    workspace_id: str = Field(..., description="Business identifier", min_length=1)
    description: Optional[str] = Field(None, description="Image description")
    tags: Optional[List[str]] = Field(default_factory=list, description="Image tags")
    use_ocr: bool = Field(True, description="Whether to use OCR for text extraction")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ImageChunkMetadata(BaseModel):
    """Metadata for a single image chunk"""
    document_id: str
    workspace_id: str
    document_type: str = "image"
    source_file: str
    chunk_index: int
    total_chunks: int
    extraction_method: str
    image_size: str
    upload_date: datetime
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None


class ImageUploadResponse(BaseModel):
    """Response schema for image upload"""
    success: bool
    document_id: str
    workspace_id: str
    filename: str
    image_size: str
    total_chunks: int
    extraction_method: str
    file_size: int
    message: str
    processing_time: Optional[float] = None


class ImageProcessingError(BaseModel):
    """Error response for image processing"""
    success: bool = False
    error: str
    detail: Optional[str] = None