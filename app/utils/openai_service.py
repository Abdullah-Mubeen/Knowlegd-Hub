from typing import Dict, Any
from aiohttp_retry import Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List, Optional, AsyncGenerator
import logging
import base64
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIService:
    """
    Enhanced OpenAI service with:
    - Vision API for image text extraction
    - Streaming support
    - Better error handling
    - Batch processing with rate limiting
    - Improved prompts
    """
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY.get_secret_value()
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=self.api_key,
            dimensions=settings.PINECONE_DIMENSION
        )
        
        # Initialize chat model
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
        
        logger.info(f"Initialized OpenAI with model: {settings.OPENAI_CHAT_MODEL}")
    
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
        Generate embeddings for multiple texts efficiently with batching
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (avoid rate limits)
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            # Filter and clean texts
            valid_texts = [t.strip() for t in texts if t and t.strip()]
            
            if not valid_texts:
                raise ValueError("No valid texts to embed")
            
            # Process in batches to avoid rate limits
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated batch {i//batch_size + 1}: {len(batch)} embeddings")
            
            logger.info(f"Generated {len(all_embeddings)} total embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def extract_text_from_image(self, image_bytes: bytes) -> str:
            """
            Extract text from image using OpenAI Vision API (GPT-4o for best accuracy)
            
            Args:
                image_bytes: Raw image bytes
                
            Returns:
                Extracted text from image
            """
            try:
                from openai import OpenAI
                
                client = OpenAI(api_key=self.api_key)
                
                # Convert to base64
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                # Call Vision API with GPT-4o (better than gpt-4o-mini for OCR)
                response = client.chat.completions.create(
                    model="gpt-4o",  # ✅ UPGRADED from gpt-4o-mini for better accuracy
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
                                        "detail": "high"  # High detail for best OCR quality
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4096,  # ✅ Increased from 1500 for longer documents
                    temperature=0.1   # ✅ Low temperature for accuracy
                )
                
                extracted_text = response.choices[0].message.content
                
                if extracted_text:
                    extracted_text = extracted_text.strip()
                    logger.info(f"✅ Extracted {len(extracted_text)} characters from image using Vision API")
                else:
                    logger.warning("⚠️ No text extracted from image")
                    extracted_text = ""
                
                return extracted_text
                
            except Exception as e:
                logger.error(f"❌ Error extracting text from image: {str(e)}")
                raise


        # OPTIONAL: Add this new method for getting image descriptions
    
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
                    max_tokens=150,
                    temperature=0.3
                )
                
                description = response.choices[0].message.content.strip()
                logger.info(f"Generated image description: {description[:100]}...")
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
        Generate answer using chat model
        
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
            
            response = self.chat_model.invoke(messages)
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