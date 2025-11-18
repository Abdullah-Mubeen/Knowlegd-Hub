from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    QNA = "qna"
    IMAGE = "image"
    FAQ = "faq"
    ARTICLE = "article"
    API = "api"


class BaseMetadata(BaseModel):
    """Base metadata for all chunks"""
    document_id: str
    workspace_id: str
    document_type: DocumentType
    source_file: str
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    chunk_index: int
    total_chunks: int


class ChunkData(BaseModel):
    """Individual chunk with text and metadata"""
    id: str
    text: str
    metadata: Dict[str, Any]


class DocumentUploadRequest(BaseModel):
    """Request for document upload"""
    workspace_id: str = Field(..., description="Business identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    document_id: str
    workspace_id: str
    document_type: DocumentType
    total_chunks: int
    status: str = "success"
    message: Optional[str] = None


class QnAPair(BaseModel):
    """Single Q&A pair"""
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    category: Optional[str] = "General"


class QnAUploadRequest(BaseModel):
    """Request for Q&A upload"""
    workspace_id: str
    qna_pairs: List[QnAPair] = Field(..., min_items=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class FAQItem(BaseModel):
    """FAQ item"""
    question: str
    answer: str


class FAQUploadRequest(BaseModel):
    """Request for FAQ upload"""
    workspace_id: str
    category: str
    faqs: List[FAQItem] = Field(..., min_items=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ArticleUploadRequest(BaseModel):
    """Request for article upload"""
    workspace_id: str
    title: str
    content: str = Field(..., min_length=1)
    author: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Request for querying knowledge base"""
    workspace_id: str
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None


class SourceReference(BaseModel):
    """Source reference for answer"""
    text: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response for query"""
    answer: str
    sources: List[SourceReference]
    workspace_id: str
    confidence: Optional[float] = None


class DocumentListResponse(BaseModel):
    """Response for document listing"""
    documents: List[Dict[str, Any]]
    total: int
    workspace_id: str


class DeleteDocumentResponse(BaseModel):
    """Response for document deletion"""
    document_id: str
    workspace_id: str
    status: str = "deleted"
    message: Optional[str] = None