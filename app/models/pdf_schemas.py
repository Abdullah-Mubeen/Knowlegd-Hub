from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class PDFUploadRequest(BaseModel):
    """Request schema for PDF upload"""
    workspace_id: str = Field(..., description="Business identifier", min_length=1)
    section_title: Optional[str] = Field(None, description="Document section/category")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class PDFChunkMetadata(BaseModel):
    """Metadata for a single PDF chunk"""
    document_id: str
    workspace_id: str
    document_type: str = "pdf"
    source_file: str
    page_number: int
    chunk_index: int
    total_chunks: int
    section_title: Optional[str] = None
    extraction_method: str
    upload_date: datetime
    tags: List[str] = Field(default_factory=list)


class PDFUploadResponse(BaseModel):
    """Response schema for PDF upload"""
    success: bool
    document_id: str
    workspace_id: str
    filename: str
    total_pages: int
    total_chunks: int
    file_size: int
    message: str
    processing_time: Optional[float] = None


class PDFProcessingError(BaseModel):
    """Error response for PDF processing"""
    success: bool = False
    error: str
    detail: Optional[str] = None