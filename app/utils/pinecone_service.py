from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from app.config import get_settings
from app.utils.openai_service import get_openai_service

logger = logging.getLogger(__name__)
settings = get_settings()


class PineconeService:
    """Enhanced Pinecone service with LangChain integration"""

    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY.get_secret_value()
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.index = None
        self.vector_store = None
        
        self._ensure_index_exists()
        self._initialize_vector_store()
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist, with proper initialization wait"""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                
                self.pc.create_index(
                    name=self.index_name,
                    dimension=settings.PINECONE_DIMENSION,
                    metric=settings.PINECONE_METRIC,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.PINECONE_ENVIRONMENT
                    )
                )
                
                # Wait for index to be ready
                logger.info("Waiting for index to initialize...")
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                
                logger.info(f"Index {self.index_name} created and ready")
            else:
                logger.info(f"Index {self.index_name} already exists")
            
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Error ensuring index exists: {str(e)}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize LangChain vector store"""
        try:
            openai_service = get_openai_service()
            
            self.vector_store = PineconeVectorStore(
                index=self.index,
                embedding=openai_service.embeddings,
                text_key="text",
                namespace=""  # Can be customized per business
            )

            logger.info("LangChain vector store initialized")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        namespace: str = "",
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to vector store using LangChain
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dicts for each chunk
            namespace: Optional namespace (use business_id)
            batch_size: Batch size for uploads
            
        Returns:
            List of document IDs
        """
        try:
            if not texts:
                raise ValueError("No texts provided")
            
            if len(texts) != len(metadatas):
                raise ValueError("Texts and metadatas must have same length")
            
            # Update vector store namespace if needed
            if namespace:
                openai_service = get_openai_service()
                self.vector_store = PineconeVectorStore(
                    index=self.index,
                    embedding=openai_service.embeddings,
                    text_key="text",
                    namespace=namespace
                )
            
            # Add documents in batches
            all_ids = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                ids = self.vector_store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
                all_ids.extend(ids)
                
                logger.info(f"Added batch {i//batch_size + 1}: {len(ids)} documents")
            
            logger.info(f"Total documents added: {len(all_ids)}")
            return all_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        namespace: str = "",
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            namespace: Optional namespace to search in
            top_k: Number of results
            filter_dict: Metadata filters
            
        Returns:
            List of (text, score, metadata) tuples
        """
        try:
            # Update namespace if needed
            if namespace:
                openai_service = get_openai_service()
                self.vector_store = PineconeVectorStore(
                    index=self.index,
                    embedding=openai_service.embeddings,
                    text_key="text",
                    namespace=namespace
                )
            
            # Search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = [
                (doc.page_content, score, doc.metadata)
                for doc, score in results
            ]
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def delete_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Delete vectors by metadata filter
        
        Args:
            filter_dict: Metadata filter (e.g., {"document_id": "doc123"})
            namespace: Namespace to delete from
            
        Returns:
            Delete response
        """
        try:
            response = self.index.delete(
                filter=filter_dict,
                namespace=namespace
            )
            logger.info(f"Deleted vectors with filter: {filter_dict}")
            return response
            
        except Exception as e:
            logger.error(f"Error deleting by metadata: {str(e)}")
            raise
    
    def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """
        Delete all vectors in a namespace (e.g., when business removes all data)
        
        Args:
            namespace: Namespace to delete
            
        Returns:
            Delete response
        """
        try:
            response = self.index.delete(
                delete_all=True,
                namespace=namespace
            )
            logger.info(f"Deleted namespace: {namespace}")
            return response
            
        except Exception as e:
            logger.error(f"Error deleting namespace: {str(e)}")
            raise
    
    def get_stats(self, namespace: str = "") -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            
            if namespace and 'namespaces' in stats:
                namespace_stats = stats['namespaces'].get(namespace, {})
                return {
                    "total_vectors": namespace_stats.get('vector_count', 0),
                    "namespace": namespace
                }
            
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "namespaces": list(stats.get('namespaces', {}).keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            raise


# Singleton instance
_pinecone_service = None

def get_pinecone_service() -> PineconeService:
    """Get or create PineconeService singleton"""
    global _pinecone_service
    if _pinecone_service is None:
        _pinecone_service = PineconeService()
    return _pinecone_service