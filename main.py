from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.routes import faq_ingestion, pdf_ingestion, image_ingestion, qna_ingestion
from app.config import get_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Knowledge Hub API for document ingestion and querying",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pdf_ingestion.router)
app.include_router(image_ingestion.router)
app.include_router(qna_ingestion.router)
app.include_router(faq_ingestion.router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Knowledge Hub API",
        "version": "1.0.0",
        "endpoints": {
            "root": "/",
            "pdf_ingestion": "/pdf/upload",
            "image_ingestion": "/image/upload",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }
