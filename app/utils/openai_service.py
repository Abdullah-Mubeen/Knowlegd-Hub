from typing import Dict, Any
from aiohttp_retry import Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List, Optional, AsyncGenerator
import logging
import base64
import hashlib
from enum import Enum
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class QueryComplexity(Enum):
    """Query complexity levels for model selection"""
    SIMPLE = -1      # Use cheap model
    MODERATE = 0     # Use default model  
    COMPLEX = 1      # Use premium model


class OpenAIService:
    """
    Enhanced OpenAI service with:
    - Smart model routing (gpt-4o-mini for simple, gpt-4 for complex)
    - Vision API for images (gpt-4o for best quality)
    - Embedding caching & deduplication
    - Streaming support & batch processing
    """
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY.get_secret_value()
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=self.api_key,
            dimensions=settings.PINECONE_DIMENSION
        )
        
        # Simple query model (cheap: gpt-4o-mini)
        self.simple_model = ChatOpenAI(
            model=settings.SIMPLE_QUERY_MODEL,
            temperature=0.3,  # Lower temp for factual answers
            max_tokens=500,   # Simple queries need fewer tokens
            openai_api_key=self.api_key
        )
        
        # Complex query model (powerful: gpt-4)
        self.complex_model = ChatOpenAI(
            model=settings.COMPLEX_QUERY_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            openai_api_key=self.api_key
        )
        
        # Default chat model (for backward compatibility)
        self.chat_model = ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            openai_api_key=self.api_key
        )
        
        # Streaming chat model
        self.streaming_chat_model = ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            openai_api_key=self.api_key,
            streaming=True
        )
        
        # Embedding cache for deduplication
        self._embedding_cache: Dict[str, List[float]] = {}

    def _detect_query_complexity(self, text: str) -> QueryComplexity:
        """
        Detect if query is simple or complex to route to appropriate model
        
        Scoring:
        - SIMPLE: < threshold (30 words), no complex keywords
        - MODERATE: medium length or some complexity
        - COMPLEX: long text, complex keywords (analyze, compare, explain, etc.)
        """
        if not text:
            return QueryComplexity.SIMPLE
        
        text_lower = text.lower().strip()
        word_count = len(text_lower.split())
        
        # Parse complex keywords from config
        complex_keywords = [kw.strip() for kw in settings.COMPLEX_KEYWORDS.split(",")]
        
        # Score the query
        score = 0
        
        # Word count score
        if word_count < settings.SIMPLE_QUERY_THRESHOLD:
            score -= 1  # Likely simple
        elif word_count > 60:
            score += 1  # Likely complex
        
        # Check for complex keywords
        for keyword in complex_keywords:
            if keyword in text_lower:
                score += 1
                break  # One complex keyword is enough
        
        # Check for question complexity indicators
        question_patterns = ["how would you", "can you explain", "tell me why", "what would you recommend"]
        for pattern in question_patterns:
            if pattern in text_lower:
                score += 1
                break
        
        # Determine complexity level
        if score <= -1:
            return QueryComplexity.SIMPLE
        elif score >= 1:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.MODERATE
    
    def _get_model_for_query(self, query: str) -> ChatOpenAI:
        """
        Select appropriate model based on query complexity
        """
        if not settings.ENABLE_SMART_ROUTING:
            return self.chat_model
        
        complexity = self._detect_query_complexity(query)
        
        if complexity == QueryComplexity.SIMPLE:
            logger.debug(f"📝 Simple query detected → using {settings.SIMPLE_QUERY_MODEL}")
            return self.simple_model
        elif complexity == QueryComplexity.COMPLEX:
            logger.debug(f"🧠 Complex query detected → using {settings.COMPLEX_QUERY_MODEL}")
            return self.complex_model
        else:
            logger.debug(f"⚖️ Moderate query detected → using default model")
            return self.chat_model
    
    def _get_text_hash(self, text: str) -> str:
        """Get SHA256 hash of text for embedding deduplication"""
        return hashlib.sha256(text.strip().encode()).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if available"""
        text_hash = self._get_text_hash(text)
        return self._embedding_cache.get(text_hash)
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for future use"""
        text_hash = self._get_text_hash(text)
        self._embedding_cache[text_hash] = embedding
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector
        """
        try:
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            embedding = self.embeddings.embed_query(text.strip())
            logger.info(f"Generated embedding (dim={len(embedding)}) for text length={len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings with deduplication cache to avoid duplicate API calls
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (avoid rate limits)
            
        Returns:
            List of embedding vectors in same order as input
        """
        try:
            if not texts:
                return []
            
            # Filter and clean texts
            valid_texts = [t.strip() for t in texts if t and t.strip()]
            
            if not valid_texts:
                raise ValueError("No valid texts to embed")
            
            # Check cache for all texts first (deduplication)
            embeddings_map = {}  # hash -> embedding
            texts_to_embed = []  # texts not in cache
            
            for text in valid_texts:
                cached = self._get_cached_embedding(text)
                if cached:
                    text_hash = self._get_text_hash(text)
                    embeddings_map[text_hash] = cached
                    logger.debug(f"✓ Cache hit for embedding")
                else:
                    texts_to_embed.append(text)
            
            # Only call API for texts not in cache
            if texts_to_embed:
                logger.info(f"Cache optimization: {len(texts_to_embed)}/{len(valid_texts)} texts need API call")
                
                for i in range(0, len(texts_to_embed), batch_size):
                    batch = texts_to_embed[i:i + batch_size]
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    
                    for text, embedding in zip(batch, batch_embeddings):
                        text_hash = self._get_text_hash(text)
                        embeddings_map[text_hash] = embedding
                        self._cache_embedding(text, embedding)
                    
                    logger.info(f"Generated batch {i//batch_size + 1}: {len(batch)} embeddings")
            
            # Reconstruct embeddings in original order
            result = []
            for text in valid_texts:
                text_hash = self._get_text_hash(text)
                result.append(embeddings_map[text_hash])
            
            logger.info(f"Generated {len(result)} total embeddings (with cache optimization)")
            return result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def extract_text_from_image(self, image_bytes: bytes) -> str:
            """
            Extract text from image using OpenAI Vision API (GPT-4o for best quality)
            Optimized: reduced tokens from 4096 to 2500, smart detail mode
            
            Args:
                image_bytes: Raw image bytes
                
            Returns:
                Extracted text from image
            """
            try:
                from openai import OpenAI
                
                client = OpenAI(api_key=self.api_key)
                
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                # Keep gpt-4o for best quality, but optimize tokens
                response = client.chat.completions.create(
                    model="gpt-4o",  # Best quality for images
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Extract ALL visible text from this image with high accuracy.

        Instructions:
        - Extract every word, number, and text element you can see
        - Preserve the original structure and formatting as much as possible
        - Include headers, titles, body text, captions, labels
        - Include table contents and structured data if present
        - Maintain paragraph breaks and logical text flow
        - If there's no readable text, return an empty response

        Return only the extracted text, nothing else."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "auto"  # Smart detail: 'auto' balances quality vs cost
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2500,  # Reduced from 4096 (saves ~40% tokens)
                    temperature=0.0  # Deterministic (better for OCR)
                )
                
                extracted_text = response.choices[0].message.content
                
                if extracted_text:
                    extracted_text = extracted_text.strip()
                    logger.info(f"✅ Extracted {len(extracted_text)} chars from image [gpt-4o, optimized tokens]")
                else:
                    logger.warning("⚠️ No text extracted from image")
                    extracted_text = ""
                
                return extracted_text
                
            except Exception as e:
                logger.error(f"❌ Error extracting text from image: {str(e)}")
                raise


    # OPTIONAL
    def describe_image(self, image_bytes: bytes) -> str:
            """
            Get a brief description of the image for context
            
            Args:
                image_bytes: Raw image bytes
                
            Returns:
                Image description (1-2 sentences)
            """
            try:
                from openai import OpenAI
                
                client = OpenAI(api_key=self.api_key)
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Provide a brief 1-2 sentence description of this image.

        Focus on:
        - Main subject or document type
        - Purpose or context
        - Key visual elements

        Be concise and informative for search context."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low"  # Low detail is sufficient for description
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=100,  # Reduced from 150 (1-2 sentences is enough)
                    temperature=0.0  # Deterministic
                )
                
                description = response.choices[0].message.content.strip()
                logger.info(f"Generated image description [optimized]")
                return description
                
            except Exception as e:
                logger.warning(f"Could not get image description: {str(e)}")
                return ""
    
    def generate_answer(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate answer using smart model routing based on complexity
        
        Args:
            prompt: User prompt/question
            system_message: Optional system context
            
        Returns:
            Generated answer
        """
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            # Smart model selection based on query complexity
            model = self._get_model_for_query(prompt)
            response = model.invoke(messages)
            answer = response.content
            
            logger.info(f"Generated answer (length={len(answer)})")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    async def generate_answer_stream(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate answer with streaming (for real-time responses)
        
        Args:
            prompt: User prompt/question
            system_message: Optional system context
            
        Yields:
            Token chunks as they're generated
        """
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            async for chunk in self.streaming_chat_model.astream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error in streaming answer: {str(e)}")
            raise
    
    def generate_rag_answer(
        self,
        question: str,
        context_chunks: List[str],
        business_name: Optional[str] = None,
        include_citations: bool = True,
        conversation_mode: bool = False
    ) -> str:
        """
        Generate RAG answer with retrieved context
        
        Args:
            question: User's question
            context_chunks: Retrieved relevant chunks
            business_name: Optional business identifier
            include_citations: Whether to include [1], [2] citation markers
            conversation_mode: Whether to use conversational tone
            
        Returns:
            Contextual answer
        """
        try:
            # Build context with or without citations
            if include_citations:
                context = "\n\n".join([
                    f"[{i+1}] {chunk}"
                    for i, chunk in enumerate(context_chunks)
                ])
            else:
                context = "\n\n".join([
                    f"Document {i+1}:\n{chunk}"
                    for i, chunk in enumerate(context_chunks)
                ])
            
            business_context = f" for {business_name}" if business_name else ""
            
            # Build system message based on mode
            if conversation_mode:
                system_message = f"""You are a friendly, conversational AI assistant{business_context}.
Your role is to have natural conversations while providing accurate information.

Guidelines:
- Use a warm, conversational tone
- Answer based on the provided context
- Be concise but complete
- If information isn't in context, say so naturally
- Stay helpful and professional"""
            else:
                system_message = f"""You are an intelligent business assistant{business_context}.
Your role is to answer questions accurately based on the provided business documents and information.

Guidelines:
- Answer directly and concisely based on the context
- Use specific details and numbers from the context
- If information is not in the context, clearly state: "I don't have this information in the knowledge base."
- Be professional and helpful
- If context is ambiguous, acknowledge the ambiguity
- Don't make assumptions beyond the provided information"""
            
            # Add citation instructions
            if include_citations:
                system_message += """

IMPORTANT - Citation Format:
- Cite sources using [1], [2], etc. at the end of sentences
- Place citations after the relevant statement: "Information here [1]."
- Use multiple citations if combining information: "Combined info [1][2]."
- Every factual claim should have a citation"""
            
            prompt = f"""Based on the following information, answer the question.

Context:
{context}

Question: {question}

Provide a clear, accurate answer:"""
            
            return self.generate_answer(prompt, system_message)
            
        except Exception as e:
            logger.error(f"Error generating RAG answer: {str(e)}")
            raise
    
    def summarize_text(
        self,
        text: str,
        max_words: int = 150,
        style: str = "concise"
    ) -> str:
        """
        Summarize text
        
        Args:
            text: Text to summarize
            max_words: Maximum words in summary
            style: "concise", "detailed", or "bullet"
            
        Returns:
            Summary text
        """
        try:
            style_instructions = {
                "concise": "Create a brief, concise summary.",
                "detailed": "Create a comprehensive summary covering key points.",
                "bullet": "Create a bullet-point summary of key points."
            }
            
            instruction = style_instructions.get(style, style_instructions["concise"])
            
            prompt = f"""{instruction} Maximum {max_words} words.

Text:
{text}

Summary:"""
            
            system_message = "You are a precise summarization assistant. Create clear, informative summaries."
            
            return self.generate_answer(prompt, system_message)
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            raise
    
    def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify the intent of user input
        
        Returns:
            {
                "primary_intent": str,
                "confidence": float,
                "entities": List[str]
            }
        """
        try:
            prompt = f"""Classify the intent of this user message. Return ONLY valid JSON.

Message: {text}

Return format:
{{
    "primary_intent": "greeting|question|command|clarification|feedback",
    "confidence": 0.0-1.0,
    "entities": ["entity1", "entity2"],
    "is_followup": true|false
}}

Classification:"""
            
            response = self.chat_model.invoke([{"role": "user", "content": prompt}])
            
            import json
            classification = json.loads(response.content.strip().strip('`').replace('json\n', ''))
            
            logger.info(f"Classified intent: {classification['primary_intent']}")
            return classification
            
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")
            return {
                "primary_intent": "question",
                "confidence": 0.5,
                "entities": [],
                "is_followup": False
            }
    
    def rewrite_query(
        self,
        query: str,
        conversation_context: Optional[str] = None
    ) -> str:
        """
        Rewrite query to be more search-friendly
        
        Args:
            query: Original query
            conversation_context: Optional conversation history
            
        Returns:
            Rewritten query
        """
        try:
            if conversation_context:
                prompt = f"""Given this conversation context, rewrite the user's query to be a complete, standalone search query.

Context:
{conversation_context}

User Query: {query}

Rewritten Query (return only the query, no explanation):"""
            else:
                prompt = f"""Rewrite this query to be more specific and search-friendly. Expand abbreviations and add context.

Original: {query}

Rewritten (return only the query):"""
            
            rewritten = self.generate_answer(prompt)
            return rewritten.strip().strip('"')
            
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return query


# Singleton instance
_openai_service = None

def get_openai_service() -> OpenAIService:
    """Get or create OpenAIService singleton"""
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service