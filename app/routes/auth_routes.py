from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
from jose import jwt
import bcrypt
import uuid
import logging
import os

from app.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    company_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    workspace_id: str
    user_id: str
    email: str


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """
    Register a new user and generate workspace
    
    - **email**: User email (unique)
    - **password**: User password (min 8 characters)
    - **name**: Optional user name
    - **company_name**: Optional company/business name
    
    Returns JWT token with workspace_id for authentication
    """
    try:
        db = get_db()
        
        # Validate password strength
        if len(user_data.password) < 8:
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 8 characters long"
            )
        
        # Check if user already exists
        existing_user = db.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        # Generate unique IDs
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        workspace_id = f"ws_{uuid.uuid4().hex[:12]}"
        
        # Hash password
        hashed_password = hash_password(user_data.password)
        
        # Store user in MongoDB
        db.create_user(
            user_id=user_id,
            email=user_data.email,
            hashed_password=hashed_password,
            workspace_id=workspace_id,
            name=user_data.name,
            company_name=user_data.company_name
        )
        
        # Create JWT token
        token_data = {
            "sub": user_id,
            "email": user_data.email,
            "workspace_id": workspace_id
        }
        access_token = create_access_token(token_data)
        
        logger.info(f"User registered: {user_data.email} with workspace: {workspace_id}")
        
        return TokenResponse(
            access_token=access_token,
            workspace_id=workspace_id,
            user_id=user_id,
            email=user_data.email
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """
    Login with email and password
    
    - **email**: User email
    - **password**: User password
    
    Returns JWT token for authentication
    """
    try:
        db = get_db()
        
        # Check if user exists
        user = db.get_user_by_email(credentials.email)
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Verify password
        if not verify_password(credentials.password, user["password"]):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Create JWT token
        token_data = {
            "sub": user["user_id"],
            "email": user["email"],
            "workspace_id": user["workspace_id"]
        }
        access_token = create_access_token(token_data)
        
        logger.info(f"User logged in: {credentials.email}")
        
        return TokenResponse(
            access_token=access_token,
            workspace_id=user["workspace_id"],
            user_id=user["user_id"],
            email=user["email"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Login failed: {str(e)}"
        )


@router.get("/me")
async def get_current_user():
    """
    Get current authenticated user info
    
    Note: This endpoint requires authentication
    """
    return {
        "message": "This endpoint requires authentication",
        "info": "Use /register or /login to get a JWT token"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db = get_db()
        db_health = db.health_check()
        
        return {
            "status": "healthy",
            "endpoint": "authentication",
            "database": db_health
        }
    except Exception as e:
        return {
            "status": "degraded",
            "endpoint": "authentication",
            "error": str(e)
        }