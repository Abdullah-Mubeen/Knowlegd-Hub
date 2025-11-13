from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, HTTPException
from jose import jwt, JWTError
from starlette.responses import JSONResponse
import os
import re

# Shared secret key used to verify JWTs
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"

# Paths that don't require authentication
ALLOWED_PATHS = {
    "/",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/favicon.ico",
    "/api/auth/register",
    "/api/auth/login",
}

# Path patterns that don't require authentication
ALLOWED_PATH_PATTERNS = [
    r"^/api/knowledge-hub/ingest/.+/health$",  # Health check endpoints
]


class AuthMiddleware(BaseHTTPMiddleware):
    """JWT Authentication Middleware"""
    
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        # Bypass authentication for allowed paths
        if path in ALLOWED_PATHS:
            return await call_next(request)
        
        # Bypass authentication for pattern-matched paths
        for pattern in ALLOWED_PATH_PATTERNS:
            if re.match(pattern, path):
                return await call_next(request)
        
        # Extract token from cookies or Authorization header
        token = (
            request.cookies.get("auth_token")
            or request.headers.get("Authorization")
        )
        
        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing JWT token"}
            )
        
        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]
        
        try:
            # Decode and verify JWT token
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            workspace_id = payload.get("workspace_id")
            user_id = payload.get("sub")
            email = payload.get("email")
            
            if not workspace_id or not user_id:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Missing workspace_id or user_id in token"}
                )
            
            # Attach user info to request state
            request.state.user = {
                "user_id": user_id,
                "workspace_id": workspace_id,
                "email": email,
            }
            
        except JWTError as e:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid JWT token", "error": str(e)}
            )
        
        return await call_next(request)