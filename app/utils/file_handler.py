import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, BinaryIO, Tuple
import logging
from datetime import datetime
import hashlib
import mimetypes
from fastapi import UploadFile, HTTPException

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FileHandler:
    """Handle file uploads, validation, and storage"""
    
    # Allowed file extensions and MIME types
    ALLOWED_EXTENSIONS = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".json": "application/json",
        ".csv": "text/csv"
    }
    
    MAX_FILE_SIZE = 60 * 1024 * 1024  # 60MB 
    
    def __init__(self, storage_dir: str = "storage"):
        """
        Initialize file handler
        
        Args:
            storage_dir: Base directory for file storage
        """
        self.storage_dir = Path(storage_dir)
        self.uploads_dir = self.storage_dir / "uploads"
        self.temp_dir = self.storage_dir / "temp"
        
        # Create directories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure storage directories exist"""
        directories = [
            self.storage_dir,
            self.uploads_dir,
            self.temp_dir,
            self.uploads_dir / "pdf",
            self.uploads_dir / "images",
            self.uploads_dir / "documents"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension in lowercase"""
        return Path(filename).suffix.lower()
    
    def _get_file_hash(self, file_content: bytes) -> str:
        """Generate SHA256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()[:16]
    
    def _get_storage_path(self, workspace_id: str, file_type: str) -> Path:
        """
        Get storage path for business and file type
        
        Args:
            workspace_id: Business identifier
            file_type: Type of file (pdf, images, documents)
            
        Returns:
            Path to storage directory
        """
        storage_path = self.uploads_dir / file_type / workspace_id
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path
    
    def validate_file(
        self,
        file: UploadFile,
        max_size: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file
        
        Args:
            file: Uploaded file
            max_size: Maximum file size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        max_size = max_size or self.MAX_FILE_SIZE
        
        # Check filename
        if not file.filename:
            return False, "Filename is required"
        
        # Check extension
        extension = self._get_file_extension(file.filename)
        if extension not in self.ALLOWED_EXTENSIONS:
            allowed = ", ".join(self.ALLOWED_EXTENSIONS.keys())
            return False, f"File type not allowed. Allowed types: {allowed}"
        
        # Check content type
        expected_mime = self.ALLOWED_EXTENSIONS[extension]
        if file.content_type and not file.content_type.startswith(expected_mime.split('/')[0]):
            return False, f"Invalid content type: {file.content_type}"
        
        # Check file size (if file object has size attribute)
        if hasattr(file, 'size') and file.size:
            if file.size > max_size:
                return False, f"File too large. Maximum size: {max_size / 1024 / 1024:.1f}MB"
        
        return True, None
    
    async def save_upload_file(
        self,
        file: UploadFile,
        workspace_id: str,
        document_id: str,
        validate: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Save uploaded file to storage
        
        Args:
            file: Uploaded file
            workspace_id: Business identifier
            document_id: Document identifier
            validate: Whether to validate file
            
        Returns:
            Tuple of (file_path, file_metadata)
        """
        try:
            # Validate file
            if validate:
                is_valid, error_msg = self.validate_file(file)
                if not is_valid:
                    raise HTTPException(status_code=400, detail=error_msg)
            
            # Read file content
            content = await file.read()
            file_size = len(content)
            
            # Check size
            if file_size > self.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {file_size / 1024 / 1024:.1f}MB. Max: {self.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
                )
            
            # Determine file type for storage
            extension = self._get_file_extension(file.filename)
            if extension == ".pdf":
                file_type = "pdf"
            elif extension in [".png", ".jpg", ".jpeg"]:
                file_type = "images"
            else:
                file_type = "documents"
            
            # Get storage path
            storage_path = self._get_storage_path(workspace_id, file_type)
            
            # Generate unique filename
            file_hash = self._get_file_hash(content)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{document_id}_{timestamp}_{file_hash}{extension}"
            
            file_path = storage_path / safe_filename
            
            # Write file
            with open(file_path, "wb") as f:
                f.write(content)
            
            logger.info(f"Saved file: {file_path} ({file_size} bytes)")
            
            # Build metadata
            metadata = {
                "original_filename": file.filename,
                "stored_filename": safe_filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_type": file_type,
                "extension": extension,
                "content_type": file.content_type,
                "file_hash": file_hash,
                "upload_timestamp": datetime.utcnow().isoformat()
            }
            
            return str(file_path), metadata
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    async def save_temp_file(
        self,
        file: UploadFile,
        prefix: str = "temp"
    ) -> str:
        """
        Save file to temporary directory
        
        Args:
            file: Uploaded file
            prefix: Filename prefix
            
        Returns:
            Path to temporary file
        """
        try:
            extension = self._get_file_extension(file.filename)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"{prefix}_{timestamp}{extension}"
            temp_path = self.temp_dir / temp_filename
            
            content = await file.read()
            
            with open(temp_path, "wb") as f:
                f.write(content)
            
            logger.debug(f"Saved temp file: {temp_path}")
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Error saving temp file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save temp file: {str(e)}")
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete file from storage
        
        Args:
            file_path: Path to file
            
        Returns:
            True if deleted successfully
        """
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File not found: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False
    
    def delete_business_files(self, workspace_id: str) -> int:
        """
        Delete all files for a business
        
        Args:
            workspace_id: Business identifier
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        
        try:
            for file_type in ["pdf", "images", "documents"]:
                business_dir = self.uploads_dir / file_type / workspace_id
                
                if business_dir.exists():
                    # Delete all files in directory
                    for file_path in business_dir.iterdir():
                        if file_path.is_file():
                            file_path.unlink()
                            deleted_count += 1
                    
                    # Remove empty directory
                    business_dir.rmdir()
                    logger.info(f"Deleted directory: {business_dir}")
            
            logger.info(f"Deleted {deleted_count} files for business: {workspace_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting business files: {str(e)}")
            return deleted_count
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files
        
        Args:
            max_age_hours: Maximum age of temp files in hours
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        current_time = datetime.utcnow().timestamp()
        max_age_seconds = max_age_hours * 3600
        
        try:
            for file_path in self.temp_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old temp file: {file_path}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} temp files")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning temp files: {str(e)}")
            return deleted_count
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get file information
        
        Args:
            file_path: Path to file
            
        Returns:
            File information dict or None
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return None
            
            stat = path.stat()
            
            return {
                "filename": path.name,
                "file_path": str(path),
                "file_size": stat.st_size,
                "extension": path.suffix,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return None
    
    def list_business_files(
        self,
        workspace_id: str,
        file_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all files for a business
        
        Args:
            workspace_id: Business identifier
            file_type: Optional filter by file type (pdf, images, documents)
            
        Returns:
            List of file information dicts
        """
        files = []
        
        file_types = [file_type] if file_type else ["pdf", "images", "documents"]
        
        try:
            for ftype in file_types:
                business_dir = self.uploads_dir / ftype / workspace_id
                
                if business_dir.exists():
                    for file_path in business_dir.iterdir():
                        if file_path.is_file():
                            file_info = self.get_file_info(str(file_path))
                            if file_info:
                                file_info["file_type"] = ftype
                                files.append(file_info)
            
            logger.info(f"Found {len(files)} files for business: {workspace_id}")
            return files
            
        except Exception as e:
            logger.error(f"Error listing business files: {str(e)}")
            return []


# Singleton instance
_file_handler = None

def get_file_handler() -> FileHandler:
    """Get or create FileHandler singleton"""
    global _file_handler
    if _file_handler is None:
        _file_handler = FileHandler()
    return _file_handler