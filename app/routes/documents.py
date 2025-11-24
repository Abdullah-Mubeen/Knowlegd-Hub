from fastapi import APIRouter, UploadFile, Form, File, HTTPException, Request
from typing import Optional, List
import json
import logging

from app.db import get_db
from app.utils.documents import (
    save_upload_file,
    detect_document_type,
    generate_document_id
)
from app.utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["Documents"])

processor = DocumentProcessor()


# -------------------------------
# UPLOAD DOCUMENT
# -------------------------------
@router.post("")
async def upload_document(
    workspace_id: str = Form(...),
    user_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    text_input: Optional[str] = Form(None),
    qna_json: Optional[str] = Form(None),
):
    db = get_db()

    # Detect document type
    if file:
        document_type = detect_document_type(file.filename)
    elif qna_json:
        document_type = "qna"
    elif text_input:
        document_type = "text"
    else:
        raise HTTPException(400, "Provide either file, text_input, or qna_json")

    document_id = generate_document_id()
    file_path = None
    file_size = 0

    # Process file upload
    if file:
        file_path = await save_upload_file("storage/uploads", file)
        file_size = len((await file.read()) or b"")  # reset read buffer

        if document_type == "pdf":
            chunks = processor.process_pdf(file_path, document_id, workspace_id)
        elif document_type == "image":
            chunks = processor.process_image(file_path, document_id, workspace_id)
        elif document_type == "text":
            raw_text = (await file.read()).decode("utf-8")
            chunks = processor.process_text(raw_text, document_id, workspace_id)

    # Process TEXT input
    elif text_input:
        chunks = processor.process_text(text_input, document_id, workspace_id)

    # Process Q&A JSON input
    elif qna_json:
        try:
            qna_pairs = json.loads(qna_json)
            if not isinstance(qna_pairs, list):
                raise ValueError
        except:
            raise HTTPException(400, "qna_json must be valid JSON list of objects")

        chunks = processor.process_qna(qna_pairs, document_id, workspace_id)

    # Insert DOCUMENT record
    doc_record = db.create_document(
        document_id=document_id,
        workspace_id=workspace_id,
        document_type=document_type,
        filename=file.filename if file else "inline_input",
        file_path=file_path or "inline_text",
        file_size=file_size,
        total_chunks=len(chunks),
    )

    # Save CHUNKS in batch
    chunk_docs = []
    for chunk in chunks:
        chunk_docs.append({
            "chunk_id": chunk["id"],
            "document_id": document_id,
            "workspace_id": workspace_id,
            "text": chunk["text"],
            "pinecone_id": "",  # (later inserted by vector DB)
            "metadata": chunk["metadata"]
        })

    db.create_chunks_batch(chunk_docs)

    # Update user stats
    db.update_user_stats(
        user_id,
        increment_documents=1,
        increment_chunks=len(chunk_docs)
    )

    return {
        "status": "success",
        "document_id": document_id,
        "chunks_created": len(chunk_docs),
    }


# -------------------------------
# LIST DOCUMENTS
# -------------------------------
@router.get("")
async def list_documents(request: Request, limit: int = 100, skip: int = 0):
    """
    List all documents for the current user's workspace.
    Requires AuthMiddleware to attach workspace_id in request.state.user
    """
    db = get_db()
    user_info = getattr(request.state, "user", None)
    if not user_info:
        raise HTTPException(401, "Missing user info in request")

    workspace_id = user_info["workspace_id"]
    documents = db.list_documents(workspace_id=workspace_id, limit=limit, skip=skip)
    return {"status": "success", "count": len(documents), "documents": documents}


# -------------------------------
# DELETE DOCUMENT
# -------------------------------
@router.delete("/{document_id}")
async def delete_document(document_id: str, request: Request):
    """
    Soft delete a document by its ID.
    """
    db = get_db()
    user_info = getattr(request.state, "user", None)
    if not user_info:
        raise HTTPException(401, "Missing user info in request")

    # Optional: verify workspace ownership
    document = db.get_document(document_id)
    if not document or document["workspace_id"] != user_info["workspace_id"]:
        raise HTTPException(404, "Document not found or unauthorized")

    success = db.delete_document(document_id)
    return {"status": "success" if success else "failed"}
