from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import time
import json
from datetime import datetime, timedelta
from enum import Enum

from app.utils.pinecone_service import get_pinecone_service
from app.utils.openai_service import get_openai_service
from app.db import get_db

logger = logging.getLogger(__name__)


class ConversationMode(Enum):
    """Conversation modes for different use cases"""
    CHAT = "chat"  # Natural conversation with context
    QA = "qa"  # Single question-answer, no memory
    RESEARCH = "research"  # Deep research with multiple queries


class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10, max_tokens: int = 2000):
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.conversations: Dict[str, List[Dict]] = {}
    
    def add_exchange(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict] = None
    ):
        """Add a Q&A exchange to conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        })
        
        self.conversations[conversation_id].append({
            "role": "assistant",
            "content": assistant_message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Trim to max_history
        if len(self.conversations[conversation_id]) > self.max_history * 2:
            self.conversations[conversation_id] = self.conversations[conversation_id][-(self.max_history * 2):]
    
    def get_history(
        self,
        conversation_id: str,
        last_n: Optional[int] = None
    ) -> List[Dict]:
        """Get conversation history"""
        if conversation_id not in self.conversations:
            return []
        
        history = self.conversations[conversation_id]
        if last_n:
            return history[-(last_n * 2):]
        return history
    
    def get_context_summary(self, conversation_id: str) -> str:
        """Get a summary of the conversation context"""
        history = self.get_history(conversation_id, last_n=3)
        if not history:
            return ""
        
        summary_parts = []
        for msg in history:
            if msg["role"] == "user":
                summary_parts.append(f"User: {msg['content'][:100]}")
            else:
                summary_parts.append(f"Assistant: {msg['content'][:100]}")
        
        return "\n".join(summary_parts)
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]


class ConversationalRAGService:
    """
    Advanced conversational RAG service with:
    - Conversation memory and context tracking
    - Query intent detection
    - Dynamic retrieval strategy
    - Streaming responses
    - Source citations
    """
    
    def __init__(self):
        self.pinecone_service = get_pinecone_service()
        self.openai_service = get_openai_service()
        self.db = get_db()
        self.memory = ConversationMemory()
    
    def _detect_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Detect user intent and conversation type
        
        Returns:
            {
                "is_followup": bool,
                "requires_retrieval": bool,
                "is_clarification": bool,
                "is_greeting": bool,
                "intent_type": str
            }
        """
        message_lower = message.lower().strip()
        
        # Check for greetings
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "greetings"]
        is_greeting = any(g in message_lower for g in greetings) and len(message.split()) <= 3
        
        # Check for clarifications
        clarifications = ["what do you mean", "can you explain", "tell me more", "elaborate"]
        is_clarification = any(c in message_lower for c in clarifications)
        
        # Check for follow-ups
        followup_indicators = ["what about", "how about", "and", "also", "this", "that", "it"]
        is_followup = (
            len(conversation_history) > 0 and
            any(ind in message_lower for ind in followup_indicators) and
            len(message.split()) < 15
        )
        
        # Check if retrieval is needed
        requires_retrieval = not (is_greeting or (is_clarification and len(conversation_history) > 0))
        
        # Determine intent type
        if is_greeting:
            intent_type = "greeting"
        elif is_clarification:
            intent_type = "clarification"
        elif is_followup:
            intent_type = "followup"
        elif any(word in message_lower for word in ["compare", "difference", "vs", "versus"]):
            intent_type = "comparison"
        elif any(word in message_lower for word in ["how to", "steps", "process", "guide"]):
            intent_type = "procedural"
        else:
            intent_type = "factual"
        
        return {
            "is_followup": is_followup,
            "requires_retrieval": requires_retrieval,
            "is_clarification": is_clarification,
            "is_greeting": is_greeting,
            "intent_type": intent_type
        }
    
    def _rewrite_with_context(
        self,
        message: str,
        conversation_history: List[Dict]
    ) -> str:
        """
        Rewrite message to be standalone using conversation context
        """
        if not conversation_history:
            return message
        
        try:
            # Get last 3 exchanges for context
            recent_history = conversation_history[-6:]  # 3 Q&A pairs
            history_text = "\n".join([
                f"{msg['role'].title()}: {msg['content']}"
                for msg in recent_history
            ])
            
            prompt = f"""Given this conversation history, rewrite the user's message to be a standalone, complete question.

Conversation History:
{history_text}

User's Message: {message}

Rewrite this as a complete, standalone question that includes all necessary context. Return ONLY the rewritten question:"""
            
            rewritten = self.openai_service.generate_answer(prompt)
            rewritten = rewritten.strip().strip('"')
            
            logger.info(f"Rewrote: '{message}' → '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.warning(f"Context rewriting failed: {e}")
            return message
    
    def _retrieve_relevant_chunks(
        self,
        query: str,
        workspace_id: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        intent_info: Optional[Dict] = None
    ) -> List[tuple]:
        """
        Retrieve relevant chunks with adaptive strategy
        """
        # Adjust retrieval based on intent
        if intent_info and intent_info.get("intent_type") == "comparison":
            top_k = min(top_k * 2, 10)  # Get more results for comparisons
        
        # Search Pinecone
        results = self.pinecone_service.similarity_search(
            query=query,
            namespace=workspace_id,
            top_k=top_k,
            filter_dict=filters
        )
        
        return results
    
    def _generate_conversational_answer(
        self,
        message: str,
        retrieved_chunks: List[tuple],
        conversation_history: List[Dict],
        business_name: Optional[str],
        intent_info: Dict
    ) -> tuple[str, List[int]]:
        """
        Generate conversational answer with citations
        
        Returns: (answer, cited_chunk_indices)
        """
        # Format context with citations
        context_parts = []
        for i, (text, score, metadata) in enumerate(retrieved_chunks, 1):
            source = metadata.get('source_file', 'Unknown')
            page = f" (Page {metadata['page_number']})" if metadata.get('page_number') else ""
            context_parts.append(f"[{i}] {text}\n(Source: {source}{page})")
        
        context = "\n\n".join(context_parts)
        
        # Format conversation history
        history_text = ""
        if conversation_history:
            recent = conversation_history[-4:]  # Last 2 exchanges
            history_text = "\n".join([
                f"{msg['role'].title()}: {msg['content']}"
                for msg in recent
            ])
        
        # Build system message based on intent
        business_context = f" for {business_name}" if business_name else ""
        
        if intent_info.get("is_greeting"):
            system_message = f"""
        You are a warm, friendly, and genuinely helpful assistant{business_context}.
        When the user greets you, respond with enthusiasm and positive energy.
        Make them feel welcome, appreciated, and excited to interact with you.
        Keep it natural—like a real human who loves helping others."""
            
        elif intent_info.get("is_clarification"):
            system_message = f"""
        You are an empathetic, patient, and supportive assistant{business_context}.
        The user is asking for clarification, so slow down, simplify, and explain things clearly.
        Break complex ideas into easy, digestible parts.
        Encourage the user, validate their confusion, and make them feel completely understood."""
            
        else:
            system_message = f"""
        You are a warm, knowledgeable, and highly adaptive assistant{business_context}.
        Your goal is to have natural, human-like conversations while giving accurate, context-aware answers.

        Core Guidelines:
        - Sound human: natural, conversational, and approachable
        - Adapt your tone to the user's industry and business context
        - Show genuine interest and emotional intelligence
        - Maintain context throughout the conversation
        - Ground your answers in the data provided to you
        - Use citations like [1], [2] only when needed and do so naturally
        - If you lack information, be honest and guide the user helpfully
        - Keep step-by-step explanations simple and easy to follow
        - Ask clarifying questions when something is unclear
        - Stay positive, encouraging, and solution-oriented

        Your mission is to feel like a real expert support agent built specifically for their business—smart, friendly, and truly helpful."""

        
        # Build prompt
        prompt_parts = []
        
        if history_text:
            prompt_parts.append(f"Conversation History:\n{history_text}\n")
        
        if context:
            prompt_parts.append(f"Relevant Information:\n{context}\n")
        
        prompt_parts.append(f"User: {message}\n")
        prompt_parts.append("Assistant: Share your response in a natural, conversational way. Be helpful, warm, and genuine. Include citations where relevant:")
        
        prompt = "\n".join(prompt_parts)
        
        # Generate answer
        answer = self.openai_service.generate_answer(prompt, system_message)
        
        # Extract cited indices
        import re
        cited_indices = [int(x) - 1 for x in re.findall(r'\[(\d+)\]', answer)]
        
        return answer, cited_indices
    
    def _handle_greeting(self, business_name: Optional[str]) -> str:
        """Generate greeting response"""
        business_context = f" from {business_name}" if business_name else ""
        greetings = [
            f"Hey there! 👋 Welcome{business_context}. I'm here to help you find exactly what you're looking for. Whether you have questions about our services, policies, or anything else—just ask away!",
            f"Hi! 😊 Great to see you{business_context}. I'm your go-to assistant for all your questions. Feel free to ask about anything, and I'll do my best to help!",
            f"Welcome{business_context}! 🙌 I'm thrilled to help you out. Got questions? I've got answers! What can I assist you with today?"
        ]
        import random
        return random.choice(greetings)
    
    def _handle_clarification(
        self,
        message: str,
        conversation_history: List[Dict],
        business_name: Optional[str]
    ) -> str:
        """Handle clarification requests"""
        if not conversation_history:
            return "Of course! I'd love to help clarify things. Let me know what you'd like to understand better, and I'll break it down for you. 😊"
        
        # Get last assistant message
        last_assistant = None
        for msg in reversed(conversation_history):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break
        
        if not last_assistant:
            return "You know what, I totally understand if something wasn't clear. Let me explain it in a different way that makes more sense! What part would you like me to dive deeper into?"
        
        prompt = f"""The user is asking you to clarify something from your previous response. They want a clearer, more understandable explanation.

Your previous response was:
{last_assistant}

Their clarification request:
{message}

Explain this more clearly and in an easy-to-understand way. Use examples if it helps. Make it conversational and friendly:"""
        
        system_message = f"""You are a patient, friendly assistant who genuinely cares about helping people understand. 
- Explain complex things simply
- Use helpful analogies and examples
- Be warm and encouraging
- Show empathy if something was confusing
- Break down information into easy steps"""
        
        return self.openai_service.generate_answer(prompt, system_message)
    
    async def chat(
        self,
        message: str,
        user_id: str,
        workspace_id: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        mode: ConversationMode = ConversationMode.CHAT
    ) -> Dict[str, Any]:
        """
        Main conversational endpoint
        
        Args:
            message: User's message
            user_id: User ID
            workspace_id: Workspace ID
            conversation_id: Optional conversation ID for continuity
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters
            mode: Conversation mode
            
        Returns:
            Response with answer, sources, and conversation metadata
        """
        start_time = time.time()
        
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                import uuid
                conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
            
            logger.info(f"Processing chat message: {message[:50]}... (conv: {conversation_id})")
            
            # Get conversation history
            conversation_history = self.memory.get_history(conversation_id)
            
            # Detect intent
            intent_info = self._detect_intent(message, conversation_history)
            logger.info(f"Detected intent: {intent_info['intent_type']}")
            
            # Get user info
            user = self.db.get_user_by_id(user_id)
            business_name = user.get("company_name") if user else None
            
            # Handle greetings without retrieval
            if intent_info["is_greeting"]:
                answer = self._handle_greeting(business_name)
                response_time = round(time.time() - start_time, 2)
                
                # Add to memory
                self.memory.add_exchange(conversation_id, message, answer)
                
                return {
                    "answer": answer,
                    "sources": [],
                    "total_sources": 0,
                    "cited_sources": 0,
                    "conversation_id": conversation_id,
                    "intent": intent_info["intent_type"],
                    "response_time": response_time,
                    "requires_retrieval": False
                }
            
            # Handle clarifications
            if intent_info["is_clarification"] and not intent_info["requires_retrieval"]:
                answer = self._handle_clarification(message, conversation_history, business_name)
                response_time = round(time.time() - start_time, 2)
                
                # Add to memory
                self.memory.add_exchange(conversation_id, message, answer)
                
                return {
                    "answer": answer,
                    "sources": [],
                    "total_sources": 0,
                    "cited_sources": 0,
                    "conversation_id": conversation_id,
                    "intent": intent_info["intent_type"],
                    "response_time": response_time,
                    "requires_retrieval": False
                }
            
            # Rewrite query with context if needed
            search_query = message
            if intent_info["is_followup"] or conversation_history:
                search_query = self._rewrite_with_context(message, conversation_history)
            
            # Retrieve relevant chunks
            retrieved_chunks = self._retrieve_relevant_chunks(
                query=search_query,
                workspace_id=workspace_id,
                top_k=top_k,
                filters=filters,
                intent_info=intent_info
            )
            
            if not retrieved_chunks:
                answer = "I couldn't find relevant information in the knowledge base to answer your question. Could you rephrase or ask something else?"
                response_time = round(time.time() - start_time, 2)
                
                # Add to memory
                self.memory.add_exchange(conversation_id, message, answer, {
                    "no_results": True
                })
                
                # Log query
                self.db.log_query(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    question=message,
                    returned_chunks=[],
                    answer=answer,
                    metadata={
                        "conversation_id": conversation_id,
                        "intent": intent_info["intent_type"],
                        "no_results": True,
                        "response_time": response_time
                    }
                )
                
                return {
                    "answer": answer,
                    "sources": [],
                    "total_sources": 0,
                    "cited_sources": 0,
                    "conversation_id": conversation_id,
                    "intent": intent_info["intent_type"],
                    "response_time": response_time
                }
            
            # Generate conversational answer
            answer, cited_indices = self._generate_conversational_answer(
                message=message,
                retrieved_chunks=retrieved_chunks,
                conversation_history=conversation_history,
                business_name=business_name,
                intent_info=intent_info
            )
            
            # Get chunk details from MongoDB
            pinecone_ids = [meta.get("pinecone_id") for _, _, meta in retrieved_chunks]
            mongo_chunks = self.db.get_chunks_by_pinecone_ids(pinecone_ids)
            chunk_map = {chunk["pinecone_id"]: chunk for chunk in mongo_chunks}
            
            # Build sources
            sources = []
            for i, (text, score, metadata) in enumerate(retrieved_chunks):
                pinecone_id = metadata.get("pinecone_id")
                mongo_chunk = chunk_map.get(pinecone_id, {})
                
                source = {
                    "citation_number": i + 1,
                    "cited": i in cited_indices,
                    "chunk_id": mongo_chunk.get("chunk_id", "unknown"),
                    "document_id": metadata.get("document_id"),
                    "text": text[:300] + "..." if len(text) > 300 else text,
                    "score": float(score),
                    "metadata": {
                        "document_type": metadata.get("document_type"),
                        "source_file": metadata.get("source_file"),
                        "page_number": metadata.get("page_number"),
                        "chunk_index": metadata.get("chunk_index")
                    }
                }
                sources.append(source)
            
            response_time = round(time.time() - start_time, 2)
            
            # Add to memory
            self.memory.add_exchange(
                conversation_id,
                message,
                answer,
                metadata={
                    "sources_used": len(sources),
                    "cited_sources": len(cited_indices),
                    "intent": intent_info["intent_type"]
                }
            )
            
            # Log query
            self.db.log_query(
                user_id=user_id,
                workspace_id=workspace_id,
                question=message,
                returned_chunks=[{
                    "chunk_id": s["chunk_id"],
                    "document_id": s["document_id"],
                    "score": s["score"],
                    "cited": s["cited"]
                } for s in sources],
                answer=answer,
                metadata={
                    "conversation_id": conversation_id,
                    "intent": intent_info["intent_type"],
                    "rewritten_query": search_query if search_query != message else None,
                    "response_time": response_time,
                    "mode": mode.value
                }
            )
            
            logger.info(f"Chat completed in {response_time}s")
            
            return {
                "answer": answer,
                "sources": sources,
                "total_sources": len(sources),
                "cited_sources": len(cited_indices),
                "conversation_id": conversation_id,
                "intent": intent_info["intent_type"],
                "response_time": response_time,
                "rewritten_query": search_query if search_query != message else None
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}", exc_info=True)
            raise
    
    async def stream_chat(
        self,
        message: str,
        user_id: str,
        workspace_id: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream conversational responses (for real-time chat UX)
        
        Yields JSON chunks: {"type": "token", "content": "..."} or {"type": "done", "metadata": {...}}
        """
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                import uuid
                conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
            
            # Get conversation history
            conversation_history = self.memory.get_history(conversation_id)
            
            # Detect intent
            intent_info = self._detect_intent(message, conversation_history)
            
            # Send intent info
            yield json.dumps({
                "type": "intent",
                "content": intent_info["intent_type"]
            }) + "\n"
            
            # Handle non-retrieval intents
            if not intent_info["requires_retrieval"]:
                user = self.db.get_user_by_id(user_id)
                business_name = user.get("company_name") if user else None
                
                if intent_info["is_greeting"]:
                    answer = self._handle_greeting(business_name)
                else:
                    answer = self._handle_clarification(message, conversation_history, business_name)
                
                # Stream the answer word by word
                for word in answer.split():
                    yield json.dumps({
                        "type": "token",
                        "content": word + " "
                    }) + "\n"
                
                self.memory.add_exchange(conversation_id, message, answer)
                
                yield json.dumps({
                    "type": "done",
                    "metadata": {
                        "conversation_id": conversation_id,
                        "sources": []
                    }
                }) + "\n"
                return
            
            # Rewrite query with context
            search_query = message
            if intent_info["is_followup"] or conversation_history:
                search_query = self._rewrite_with_context(message, conversation_history)
            
            # Retrieve chunks
            retrieved_chunks = self._retrieve_relevant_chunks(
                query=search_query,
                workspace_id=workspace_id,
                top_k=top_k,
                filters=filters,
                intent_info=intent_info
            )
            
            if not retrieved_chunks:
                answer = "I couldn't find relevant information to answer your question."
                for word in answer.split():
                    yield json.dumps({
                        "type": "token",
                        "content": word + " "
                    }) + "\n"
                
                yield json.dumps({
                    "type": "done",
                    "metadata": {"conversation_id": conversation_id, "sources": []}
                }) + "\n"
                return
            
            # For streaming, we need to generate the full answer first
            # (True streaming with LLM would require OpenAI streaming API)
            user = self.db.get_user_by_id(user_id)
            business_name = user.get("company_name") if user else None
            
            answer, cited_indices = self._generate_conversational_answer(
                message=message,
                retrieved_chunks=retrieved_chunks,
                conversation_history=conversation_history,
                business_name=business_name,
                intent_info=intent_info
            )
            
            # Stream answer word by word
            for word in answer.split():
                yield json.dumps({
                    "type": "token",
                    "content": word + " "
                }) + "\n"
            
            # Build sources
            sources = []
            for i, (text, score, metadata) in enumerate(retrieved_chunks):
                if i in cited_indices:
                    sources.append({
                        "citation_number": i + 1,
                        "text": text[:200],
                        "source_file": metadata.get("source_file"),
                        "score": float(score)
                    })
            
            # Add to memory
            self.memory.add_exchange(conversation_id, message, answer)
            
            # Send completion
            yield json.dumps({
                "type": "done",
                "metadata": {
                    "conversation_id": conversation_id,
                    "sources": sources,
                    "intent": intent_info["intent_type"]
                }
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Error in stream_chat: {str(e)}")
            yield json.dumps({
                "type": "error",
                "content": str(e)
            }) + "\n"
    
    def get_conversation_history(
        self,
        conversation_id: str
    ) -> List[Dict]:
        """Get full conversation history"""
        return self.memory.get_history(conversation_id)
    
    def clear_conversation(
        self,
        conversation_id: str
    ):
        """Clear conversation history"""
        self.memory.clear_conversation(conversation_id)

    async def summarize_conversation(self, conversation_id: str) -> str:
        """
        Generate a full conversation summary using the LLM.
        Includes: problem, context, answers, resolutions.
        """
        history = self.memory.get_history(conversation_id)
        if not history:
            return "No conversation history found."

        # Format all messages
        formatted = []
        for msg in history:
            who = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{who}: {msg['content']}")

        history_text = "\n".join(formatted)

        prompt = f"""
    Summarize the following conversation between a user and an AI assistant.

    Your summary MUST include:
    - What the user wanted
    - The main problems discussed
    - Key answers the AI gave
    - Any conclusions or resolutions
    - Make the summary clear, meaningful, and structured.

    Conversation:
    {history_text}

    Write the final summary below:
    """

        system_message = "You are a highly skilled summarization assistant."

        summary = self.openai_service.generate_answer(
            prompt=prompt,
            system_message=system_message
        )

        return summary.strip()

# Singleton instance
_conversational_rag_service = None

def get_conversational_rag_service() -> ConversationalRAGService:
    """Get or create ConversationalRAGService singleton"""
    global _conversational_rag_service
    if _conversational_rag_service is None:
        _conversational_rag_service = ConversationalRAGService()
    return _conversational_rag_service