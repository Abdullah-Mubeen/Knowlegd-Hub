from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import json

from app.utils.rag_conv import (
    get_conversational_rag_service,
    ConversationMode
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for continuity")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of sources to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    mode: Optional[str] = Field("chat", description="Mode: chat, qa, research")
    stream: Optional[bool] = Field(False, description="Enable streaming response")


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    total_sources: int
    cited_sources: int
    conversation_id: str
    intent: str
    response_time: float
    rewritten_query: Optional[str] = None


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[Dict[str, Any]]
    total_exchanges: int


@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: Request,
    chat_data: ChatRequest
):
    """
    🚀 MAIN CONVERSATIONAL ENDPOINT - Natural chat with your knowledge base
    
    This endpoint provides:
    ✅ Natural conversation with memory
    ✅ Automatic intent detection (greeting, question, clarification, etc.)
    ✅ Context-aware query rewriting for follow-ups
    ✅ Source citations in answers
    ✅ Conversation continuity with conversation_id
    
    Example Request:
    ```json
    {
        "message": "What is your refund policy?",
        "conversation_id": null,
        "top_k": 5,
        "mode": "chat"
    }
    ```
    
    Example Follow-up:
    ```json
    {
        "message": "What about international orders?",
        "conversation_id": "conv_abc123",
        "top_k": 5
    }
    ```
    
    Response includes:
    - Natural conversational answer with [1], [2] citations
    - Source documents that were cited
    - Conversation ID for follow-ups
    - Detected intent (greeting, question, clarification, etc.)
    - Query rewriting info (if applicable)
    """
    try:
        # Get authenticated user
        user = request.state.user
        user_id = user["user_id"]
        workspace_id = user["workspace_id"]
        
        # Validate message
        if not chat_data.message or not chat_data.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Parse mode
        mode_map = {
            "chat": ConversationMode.CHAT,
            "qa": ConversationMode.QA,
            "research": ConversationMode.RESEARCH
        }
        mode = mode_map.get(chat_data.mode, ConversationMode.CHAT)
        
        # Get conversational RAG service
        rag_service = get_conversational_rag_service()
        
        # Process message
        result = await rag_service.chat(
            message=chat_data.message,
            user_id=user_id,
            workspace_id=workspace_id,
            conversation_id=chat_data.conversation_id,
            top_k=chat_data.top_k,
            filters=chat_data.filters,
            mode=mode
        )
        
        return ChatResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat message: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message: {str(e)}"
        )


@router.post("/stream")
async def stream_message(
    request: Request,
    chat_data: ChatRequest
):
    """
    🌊 STREAMING ENDPOINT - Real-time chat responses
    
    Returns a stream of Server-Sent Events (SSE) for real-time responses.
    Perfect for creating a ChatGPT-like typing effect in your UI.
    
    Stream format:
    ```
    {"type": "intent", "content": "question"}
    {"type": "token", "content": "The "}
    {"type": "token", "content": "refund "}
    {"type": "token", "content": "policy... "}
    {"type": "done", "metadata": {...}}
    ```
    
    Frontend example (JavaScript):
    ```javascript
    const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: "Hello!" })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const data = JSON.parse(chunk);
        
        if (data.type === 'token') {
            displayText += data.content;
        }
    }
    ```
    """
    try:
        user = request.state.user
        user_id = user["user_id"]
        workspace_id = user["workspace_id"]
        
        if not chat_data.message or not chat_data.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        rag_service = get_conversational_rag_service()
        
        async def generate():
            try:
                async for chunk in rag_service.stream_chat(
                    message=chat_data.message,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    conversation_id=chat_data.conversation_id,
                    top_k=chat_data.top_k,
                    filters=chat_data.filters
                ):
                    yield chunk
            except Exception as e:
                logger.error(f"Error in stream: {str(e)}")
                yield json.dumps({"type": "error", "content": str(e)}) + "\n"
        
        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stream endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream message: {str(e)}"
        )


@router.get("/conversation/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation(
    request: Request,
    conversation_id: str
):
    """
    Get conversation history
    
    Returns all messages in a conversation for context or review.
    """
    try:
        user = request.state.user
        
        rag_service = get_conversational_rag_service()
        history = rag_service.get_conversation_history(conversation_id)
        
        # Calculate exchanges (pairs of user + assistant)
        exchanges = len([m for m in history if m["role"] == "user"])
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=history,
            total_exchanges=exchanges
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation: {str(e)}"
        )


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(
    request: Request,
    conversation_id: str
):
    """
    Clear conversation history
    
    Removes all messages from the conversation memory.
    Useful for "New Chat" functionality.
    """
    try:
        user = request.state.user
        
        rag_service = get_conversational_rag_service()
        rag_service.clear_conversation(conversation_id)
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "message": "Conversation cleared"
        }
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation: {str(e)}"
        )


@router.post("/quick-answer", response_model=ChatResponse)
async def quick_answer(
    request: Request,
    message: str = Query(..., description="Quick question")  # Use Query here
):
    """
    Quick answer endpoint (no conversation memory)
    
    For simple one-off questions where conversation context isn't needed.
    Faster than the main chat endpoint since it skips memory management.
    
    Example:
    ```
    POST /api/chat/quick-answer?message=What are your hours?
    ```
    """
    try:
        user = request.state.user
        user_id = user["user_id"]
        workspace_id = user["workspace_id"]
        
        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        rag_service = get_conversational_rag_service()
        
        # Use QA mode (no memory)
        result = await rag_service.chat(
            message=message,
            user_id=user_id,
            workspace_id=workspace_id,
            conversation_id=None,
            top_k=5,
            filters=None,
            mode=ConversationMode.QA
        )
        
        return ChatResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quick answer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get quick answer: {str(e)}"
        )
    
@router.get("/health")
async def health_check():
    """Health check for conversational RAG service"""
    try:
        rag_service = get_conversational_rag_service()
        
        return {
            "status": "healthy",
            "endpoint": "conversational_rag",
            "features": [
                "conversation_memory",
                "intent_detection",
                "context_aware_rewriting",
                "streaming_support",
                "multi_mode_support",
                "source_citations"
            ],
            "modes": ["chat", "qa", "research"],
            "active_conversations": len(rag_service.memory.conversations)
        }
    except Exception as e:
        return {
            "status": "degraded",
            "endpoint": "conversational_rag",
            "error": str(e)
        }