from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
import logging

from app.routes import faq_ingestion, pdf_ingestion, image_ingestion, qna_ingestion, auth_routes, rag_router
from app.middleware.auth import AuthMiddleware
from app.config import get_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Knowledge Hub API for document ingestion and querying with JWT Authentication",
    version="1.0.0"
)

# CORS middleware (must be added before AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Authentication Middleware
app.add_middleware(AuthMiddleware)

# Security scheme for Swagger UI
security = HTTPBearer()

# Include routers  
app.include_router(auth_routes.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(pdf_ingestion.router, prefix="/api/knowledge-hub/ingest/pdf", tags=["Knowledge Hub"])
app.include_router(image_ingestion.router, prefix="/api/knowledge-hub/ingest/image", tags=["Knowledge Hub"])
app.include_router(qna_ingestion.router, prefix="/api/knowledge-hub/ingest/qna", tags=["Knowledge Hub"])
app.include_router(faq_ingestion.router, prefix="/api/knowledge-hub/ingest/faq", tags=["Knowledge Hub"])
app.include_router(rag_router.router, prefix="/api/rag", tags=["RAG Chat System"])


def custom_openapi():
    """Custom OpenAPI schema with JWT Bearer authentication"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.APP_NAME,
        version="1.0.0",
        description="Knowledge Hub API with JWT Authentication",
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your JWT token (without 'Bearer' prefix)"
        }
    }
    
    # Apply security globally to all endpoints except auth
    for path, path_item in openapi_schema["paths"].items():
        # Skip auth endpoints
        if "/api/auth/" in path or path == "/" or "/health" in path:
            continue
        
        for operation in path_item.values():
            if isinstance(operation, dict) and "summary" in operation:
                operation["security"] = [{"bearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Knowledge Hub API",
        "version": "1.0.0",
        "authentication": "JWT Required (except /api/auth endpoints)",
        "endpoints": {
            "authentication": {
                "register": "/api/auth/register",
                "login": "/api/auth/login"
            },
            "ingestion": {
                "pdf": "/api/knowledge-hub/ingest/pdf/upload",
                "image": "/api/knowledge-hub/ingest/image/upload",
                "qna": "/api/knowledge-hub/ingest/qna/upload",
                "faq": "/api/knowledge-hub/ingest/faq/upload"
            },
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "instructions": {
            "step_1": "Register at /api/auth/register to get JWT token",
            "step_2": "Click 'Authorize' button in Swagger UI (top-right)",
            "step_3": "Paste your JWT token (without 'Bearer' prefix)",
            "step_4": "Access protected endpoints with authentication"
        }
    }