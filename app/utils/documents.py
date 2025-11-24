# app/utils/document_utils.py
import uuid
import os
from fastapi import UploadFile, HTTPException
from typing import Optional

ALLOWED_EXT = {"pdf", "jpg", "jpeg", "png", "txt"}

def generate_document_id() -> str:
    return f"doc_{uuid.uuid4().hex[:12]}"

def get_file_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower()

def detect_document_type(filename: str) -> str:
    ext = get_file_extension(filename)

    if ext == "pdf":
        return "pdf"
    if ext in {"jpg", "jpeg", "png"}:
        return "image"
    if ext == "txt":
        return "text"
    
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

async def save_upload_file(directory: str, file: UploadFile) -> str:
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return file_path
