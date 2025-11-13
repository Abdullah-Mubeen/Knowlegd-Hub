from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class FAQItem(BaseModel):
    """Single FAQ item with validation"""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="FAQ question"
    )
    answer: str = Field(
        ...,
        min_length=5,
        max_length=10000,
        description="FAQ answer"
    )
    
    @validator('question', 'answer')
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
            "question": "What is your refund policy?",
            "answer": "We offer 30-day refunds for all products..."
        }


class FAQUploadRequest(BaseModel):
    """Request for FAQ upload with validation"""
    business_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique business identifier"
    )
    category: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="FAQ category (e.g., Billing, Technical, General)"
    )
    faq_items: List[FAQItem] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of FAQ items (max 1000 per request)"
    )
    
    @validator('business_id', 'category')
    def strip_fields(cls, v):
        """Strip whitespace from business_id and category"""
        if isinstance(v, str):
            return v.strip()
        return v
    
    class Config:
        example = {
            "business_id": "company_123",
            "category": "Billing",
            "faq_items": [
                {
                    "question": "What is your refund policy?",
                    "answer": "We offer 30-day refunds..."
                },
                {
                    "question": "Do you offer support?",
                    "answer": "Yes, 24/7 support available..."
                }
            ]
        }


class FAQUploadResponse(BaseModel):
    """Response for FAQ upload"""
    success: bool = Field(description="Operation success status")
    document_id: str = Field(description="Unique document identifier")
    business_id: str = Field(description="Business identifier")
    category: str = Field(description="FAQ category")
    total_items: int = Field(description="Total FAQ items processed")
    total_chunks: int = Field(description="Total chunks created from FAQ items")
    message: str = Field(description="Operation message")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        example = {
            "success": True,
            "document_id": "faq_xyz789uvw012",
            "business_id": "company_123",
            "category": "Billing",
            "total_items": 2,
            "total_chunks": 5,
            "message": "FAQ processed successfully: 5 chunks created",
            "processing_time": 1.87
        }


class FAQChunk(BaseModel):
    """FAQ chunk with metadata"""
    id: str
    text: str
    document_id: str
    business_id: str
    category: str
    question: str
    chunk_index: int
    total_chunks: int
    answer_part: Optional[int] = None
    total_answer_parts: Optional[int] = None
    upload_date: datetime