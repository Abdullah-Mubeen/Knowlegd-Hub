
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


# ============ Q&A SCHEMAS ============

class QnAPair(BaseModel):
    """Single Q&A pair with validation"""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question text"
    )
    answer: str = Field(
        ...,
        min_length=5,
        max_length=10000,
        description="Answer text"
    )
    category: str = Field(
        default="General",
        min_length=1,
        max_length=100,
        description="Q&A category for organization"
    )
    
    @validator('question', 'answer', 'category')
    def strip_whitespace(cls, v):
        """Strip whitespace from all string fields"""
        if isinstance(v, str):
            return v.strip()
        return v
    
    @validator('question', 'answer')
    def validate_not_empty_after_strip(cls, v):
        """Validate that strings are not empty after stripping"""
        if not v or not v.strip():
            raise ValueError("Cannot be empty")
        return v
    
    class Config:
        example = {
            "question": "What is your product?",
            "answer": "Our product is an innovative solution...",
            "category": "General"
        }


class QnAUploadRequest(BaseModel):
    """Request for Q&A upload with validation"""
    business_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique business identifier"
    )
    qna_pairs: List[QnAPair] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of Q&A pairs (max 1000 per request)"
    )
    
    @validator('business_id')
    def strip_business_id(cls, v):
        """Strip whitespace from business_id"""
        if isinstance(v, str):
            return v.strip()
        return v
    
    class Config:
        example = {
            "business_id": "company_123",
            "qna_pairs": [
                {
                    "question": "What is your product?",
                    "answer": "Our product is...",
                    "category": "General"
                },
                {
                    "question": "How to contact support?",
                    "answer": "You can reach us at...",
                    "category": "Support"
                }
            ]
        }


class QnAUploadResponse(BaseModel):
    """Response for Q&A upload"""
    success: bool = Field(description="Operation success status")
    document_id: str = Field(description="Unique document identifier")
    business_id: str = Field(description="Business identifier")
    total_pairs: int = Field(description="Total Q&A pairs processed")
    total_chunks: int = Field(description="Total chunks created from Q&A pairs")
    message: str = Field(description="Operation message")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        example = {
            "success": True,
            "document_id": "qna_abc123def456",
            "business_id": "company_123",
            "total_pairs": 2,
            "total_chunks": 5,
            "message": "Q&A processed successfully: 5 chunks created",
            "processing_time": 2.34
        }


class QnAChunk(BaseModel):
    """Q&A chunk with metadata"""
    id: str
    text: str
    document_id: str
    business_id: str
    question: str
    category: str
    chunk_index: int
    total_chunks: int
    answer_part: Optional[int] = None
    total_answer_parts: Optional[int] = None
    upload_date: datetime
