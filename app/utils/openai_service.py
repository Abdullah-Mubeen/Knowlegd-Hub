from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List, Optional
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIService:
    """Enhanced OpenAI service using LangChain"""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY.get_secret_value()
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=self.api_key,
            dimensions=settings.PINECONE_DIMENSION  # text-embedding-3-large supports 1024
        )
        
        # Initialize chat model
        self.chat_model = ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            openai_api_key=self.api_key
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
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            
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
            
            embeddings = self.embeddings.embed_documents(valid_texts)
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
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
            from langchain.schema import SystemMessage, HumanMessage
            
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
    
    def generate_rag_answer(
        self,
        question: str,
        context_chunks: List[str],
        business_name: Optional[str] = None
    ) -> str:
        """
        Generate RAG answer with retrieved context
        
        Args:
            question: User's question
            context_chunks: Retrieved relevant chunks
            business_name: Optional business identifier
            
        Returns:
            Contextual answer
        """
        try:
            # Build context
            context = "\n\n".join([
                f"Document {i+1}:\n{chunk}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            business_context = f" for {business_name}" if business_name else ""
            
            system_message = f"""You are an intelligent business assistant{business_context}.
Your role is to answer questions accurately based on the provided business documents and information.

Guidelines:
- Answer directly and concisely based on the context
- If information is not in the context, clearly state that
- Cite specific details when possible
- Be professional and helpful
- If context is ambiguous, ask for clarification"""
            
            prompt = f"""Based on the following business information, answer the question.

Context:
{context}

Question: {question}

Provide a clear, accurate answer:"""
            
            return self.generate_answer(prompt, system_message)
            
        except Exception as e:
            logger.error(f"Error generating RAG answer: {str(e)}")
            raise


# Singleton instance
_openai_service = None

def get_openai_service() -> OpenAIService:
    """Get or create OpenAIService singleton"""
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service