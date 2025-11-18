from pymongo import MongoClient, ASCENDING, DESCENDING, IndexModel
from pymongo.collection import Collection
from pymongo.database import Database
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MongoDBService:
    """MongoDB service for RAG system with Users, Documents, Chunks, and Query logs"""
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.users: Optional[Collection] = None
        self.documents: Optional[Collection] = None
        self.chunks: Optional[Collection] = None
        self.queries: Optional[Collection] = None
        
        self._connect()
        self._create_indexes()
    
    def _connect(self):
        """Connect to MongoDB and initialize collections"""
        try:
            mongodb_uri = settings.MONGODB_URI
            if not mongodb_uri:
                raise ValueError("MONGODB_URI not set in environment variables")
            
            self.client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            
            # Test connection
            self.client.server_info()
            
            # Get database (extract from URI or use default)
            self.db = self.client.get_database()
            
            # Initialize collections
            self.users = self.db["users"]
            self.documents = self.db["documents"]
            self.chunks = self.db["chunks"]
            self.queries = self.db["queries"]
            
            logger.info("Successfully connected to MongoDB")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def _create_indexes(self):
        """Create indexes for all collections"""
        try:
            # Users indexes
            self.users.create_indexes([
                IndexModel([("email", ASCENDING)], unique=True, name="email_unique"),
                IndexModel([("user_id", ASCENDING)], unique=True, name="user_id_unique"),
                IndexModel([("workspace_id", ASCENDING)], name="workspace_id_idx"),
                IndexModel([("created_at", DESCENDING)], name="created_at_idx")
            ])
            
            # Documents indexes
            self.documents.create_indexes([
                IndexModel([("document_id", ASCENDING)], unique=True, name="document_id_unique"),
                IndexModel([("workspace_id", ASCENDING)], name="workspace_id_idx"),
                IndexModel([("document_type", ASCENDING)], name="document_type_idx"),
                IndexModel([("workspace_id", ASCENDING), ("document_type", ASCENDING)], name="workspace_type_idx"),
                IndexModel([("created_at", DESCENDING)], name="created_at_idx"),
                IndexModel([("status", ASCENDING)], name="status_idx")
            ])
            
            # Chunks indexes
            self.chunks.create_indexes([
                IndexModel([("chunk_id", ASCENDING)], unique=True, name="chunk_id_unique"),
                IndexModel([("document_id", ASCENDING)], name="document_id_idx"),
                IndexModel([("workspace_id", ASCENDING)], name="workspace_id_idx"),
                IndexModel([("pinecone_id", ASCENDING)], name="pinecone_id_idx"),
                IndexModel([("document_id", ASCENDING), ("chunk_index", ASCENDING)], name="doc_chunk_idx"),
                IndexModel([("created_at", DESCENDING)], name="created_at_idx")
            ])
            
            # Queries indexes
            self.queries.create_indexes([
                IndexModel([("user_id", ASCENDING)], name="user_id_idx"),
                IndexModel([("workspace_id", ASCENDING)], name="workspace_id_idx"),
                IndexModel([("created_at", DESCENDING)], name="created_at_idx"),
                IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)], name="user_created_idx")
            ])
            
            logger.info("Successfully created MongoDB indexes")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            raise

    # ==================== USER OPERATIONS ====================
    
    def create_user(
        self,
        user_id: str,
        email: str,
        hashed_password: str,
        workspace_id: str,
        name: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new user
        
        Args:
            user_id: Unique user identifier
            email: User email
            hashed_password: Hashed password
            workspace_id: Unique workspace identifier
            name: Optional user name
            company_name: Optional company name
            
        Returns:
            Created user document
        """
        try:
            user_doc = {
                "user_id": user_id,
                "email": email,
                "password": hashed_password,
                "workspace_id": workspace_id,
                "name": name,
                "company_name": company_name,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "is_active": True,
                "total_documents": 0,
                "total_chunks": 0,
                "total_queries": 0
            }
            
            result = self.users.insert_one(user_doc)
            user_doc["_id"] = str(result.inserted_id)
            
            logger.info(f"Created user: {email} with workspace: {workspace_id}")
            return user_doc
            
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            user = self.users.find_one({"email": email})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            raise
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user_id"""
        try:
            user = self.users.find_one({"user_id": user_id})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"Error getting user by id: {str(e)}")
            raise
    
    def get_user_by_workspace(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """Get user by workspace_id"""
        try:
            user = self.users.find_one({"workspace_id": workspace_id})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"Error getting user by workspace: {str(e)}")
            raise
    
    def update_user_stats(
        self,
        user_id: str,
        increment_documents: int = 0,
        increment_chunks: int = 0,
        increment_queries: int = 0
    ) -> bool:
        """Update user statistics"""
        try:
            update_fields = {"updated_at": datetime.utcnow()}
            
            if increment_documents != 0:
                update_fields["total_documents"] = increment_documents
            if increment_chunks != 0:
                update_fields["total_chunks"] = increment_chunks
            if increment_queries != 0:
                update_fields["total_queries"] = increment_queries
            
            result = self.users.update_one(
                {"user_id": user_id},
                {
                    "$inc": {k: v for k, v in update_fields.items() if k.startswith("total_")},
                    "$set": {"updated_at": update_fields["updated_at"]}
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating user stats: {str(e)}")
            raise
    
    # ==================== DOCUMENT OPERATIONS ====================
    
    def create_document(
        self,
        document_id: str,
        workspace_id: str,
        document_type: str,
        filename: str,
        file_path: str,
        file_size: int,
        total_chunks: int,
        file_metadata: Optional[Dict[str, Any]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new document record
        
        Args:
            document_id: Unique document identifier
            workspace_id: Workspace/business identifier
            document_type: Type of document (pdf, image, qna, faq, etc.)
            filename: Original filename
            file_path: Storage path
            file_size: File size in bytes
            total_chunks: Number of chunks created
            file_metadata: Additional file metadata
            processing_metadata: Processing details
            
        Returns:
            Created document
        """
        try:
            doc = {
                "document_id": document_id,
                "workspace_id": workspace_id,
                "document_type": document_type,
                "filename": filename,
                "file_path": file_path,
                "file_size": file_size,
                "total_chunks": total_chunks,
                "file_metadata": file_metadata or {},
                "processing_metadata": processing_metadata or {},
                "status": "active",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "deleted_at": None
            }
            
            result = self.documents.insert_one(doc)
            doc["_id"] = str(result.inserted_id)
            
            logger.info(f"Created document: {document_id} for workspace: {workspace_id}")
            return doc
            
        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            raise
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        try:
            doc = self.documents.find_one({"document_id": document_id})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception as e:
            logger.error(f"Error getting document: {str(e)}")
            raise
    
    def list_documents(
        self,
        workspace_id: str,
        document_type: Optional[str] = None,
        status: str = "active",
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """List documents for a workspace"""
        try:
            query = {"workspace_id": workspace_id, "status": status}
            if document_type:
                query["document_type"] = document_type
            
            docs = list(
                self.documents
                .find(query)
                .sort("created_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            
            for doc in docs:
                doc["_id"] = str(doc["_id"])
            
            return docs
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise
    
    def update_document_status(
        self,
        document_id: str,
        status: str
    ) -> bool:
        """Update document status (active, deleted, archived)"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if status == "deleted":
                update_data["deleted_at"] = datetime.utcnow()
            
            result = self.documents.update_one(
                {"document_id": document_id},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating document status: {str(e)}")
            raise
    
    def delete_document(self, document_id: str) -> bool:
        """Soft delete a document"""
        return self.update_document_status(document_id, "deleted")
    
    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get statistics for a workspace"""
        try:
            total_docs = self.documents.count_documents({
                "workspace_id": workspace_id,
                "status": "active"
            })
            
            total_chunks = self.chunks.count_documents({
                "workspace_id": workspace_id
            })
            
            doc_types = list(self.documents.aggregate([
                {"$match": {"workspace_id": workspace_id, "status": "active"}},
                {"$group": {"_id": "$document_type", "count": {"$sum": 1}}}
            ]))
            
            return {
                "workspace_id": workspace_id,
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "documents_by_type": {item["_id"]: item["count"] for item in doc_types}
            }
            
        except Exception as e:
            logger.error(f"Error getting workspace stats: {str(e)}")
            raise
    
    # ==================== CHUNK OPERATIONS ====================
    
    def create_chunk(
        self,
        chunk_id: str,
        document_id: str,
        workspace_id: str,
        text: str,
        pinecone_id: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a chunk record
        
        Args:
            chunk_id: Unique chunk identifier
            document_id: Parent document ID
            workspace_id: Workspace identifier
            text: Chunk text content
            pinecone_id: Pinecone vector ID
            metadata: Chunk metadata (page_number, chunk_index, etc.)
            
        Returns:
            Created chunk document
        """
        try:
            chunk_doc = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "workspace_id": workspace_id,
                "text": text,
                "pinecone_id": pinecone_id,
                "metadata": metadata,
                "chunk_index": metadata.get("chunk_index", 0),
                "created_at": datetime.utcnow()
            }
            
            result = self.chunks.insert_one(chunk_doc)
            chunk_doc["_id"] = str(result.inserted_id)
            
            return chunk_doc
            
        except Exception as e:
            logger.error(f"Error creating chunk: {str(e)}")
            raise
    
    def create_chunks_batch(
        self,
        chunks_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Batch insert chunks
        
        Args:
            chunks_data: List of chunk documents
            
        Returns:
            List of inserted chunk IDs
        """
        try:
            # Add created_at to all chunks
            for chunk in chunks_data:
                chunk["created_at"] = datetime.utcnow()
                chunk["chunk_index"] = chunk.get("metadata", {}).get("chunk_index", 0)
            
            result = self.chunks.insert_many(chunks_data)
            
            logger.info(f"Batch inserted {len(result.inserted_ids)} chunks")
            return [str(id) for id in result.inserted_ids]
            
        except Exception as e:
            logger.error(f"Error batch creating chunks: {str(e)}")
            raise
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID"""
        try:
            chunk = self.chunks.find_one({"chunk_id": chunk_id})
            if chunk:
                chunk["_id"] = str(chunk["_id"])
            return chunk
        except Exception as e:
            logger.error(f"Error getting chunk: {str(e)}")
            raise
    
    def get_chunks_by_document(
        self,
        document_id: str,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        try:
            chunks = list(
                self.chunks
                .find({"document_id": document_id})
                .sort("chunk_index", ASCENDING)
                .limit(limit)
            )
            
            for chunk in chunks:
                chunk["_id"] = str(chunk["_id"])
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks by document: {str(e)}")
            raise
    
    def get_chunks_by_pinecone_ids(
        self,
        pinecone_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get chunks by Pinecone IDs (for query results)"""
        try:
            chunks = list(
                self.chunks.find({"pinecone_id": {"$in": pinecone_ids}})
            )
            
            for chunk in chunks:
                chunk["_id"] = str(chunk["_id"])
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks by pinecone IDs: {str(e)}")
            raise
    
    def delete_chunks_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document"""
        try:
            result = self.chunks.delete_many({"document_id": document_id})
            logger.info(f"Deleted {result.deleted_count} chunks for document: {document_id}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting chunks: {str(e)}")
            raise
    
    def delete_chunks_by_workspace(self, workspace_id: str) -> int:
        """Delete all chunks for a workspace"""
        try:
            result = self.chunks.delete_many({"workspace_id": workspace_id})
            logger.info(f"Deleted {result.deleted_count} chunks for workspace: {workspace_id}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting workspace chunks: {str(e)}")
            raise
    
    # ==================== QUERY LOG OPERATIONS ====================
    
    def log_query(
        self,
        user_id: str,
        workspace_id: str,
        question: str,
        returned_chunks: List[Dict[str, Any]],
        answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a RAG query
        
        Args:
            user_id: User who made the query
            workspace_id: Workspace context
            question: User's question
            returned_chunks: List of chunks returned from vector search
            answer: Generated answer
            metadata: Additional metadata (response_time, model, etc.)
            
        Returns:
            Created query log
        """
        try:
            query_log = {
                "user_id": user_id,
                "workspace_id": workspace_id,
                "question": question,
                "returned_chunks": returned_chunks,
                "answer": answer,
                "metadata": metadata or {},
                "created_at": datetime.utcnow()
            }
            
            result = self.queries.insert_one(query_log)
            query_log["_id"] = str(result.inserted_id)
            
            # Update user query count
            self.update_user_stats(user_id, increment_queries=1)
            
            logger.info(f"Logged query for user: {user_id}")
            return query_log
            
        except Exception as e:
            logger.error(f"Error logging query: {str(e)}")
            raise
    
    def get_user_queries(
        self,
        user_id: str,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Get query history for a user"""
        try:
            queries = list(
                self.queries
                .find({"user_id": user_id})
                .sort("created_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            
            for query in queries:
                query["_id"] = str(query["_id"])
            
            return queries
            
        except Exception as e:
            logger.error(f"Error getting user queries: {str(e)}")
            raise
    
    def get_workspace_queries(
        self,
        workspace_id: str,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Get query history for a workspace"""
        try:
            queries = list(
                self.queries
                .find({"workspace_id": workspace_id})
                .sort("created_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            
            for query in queries:
                query["_id"] = str(query["_id"])
            
            return queries
            
        except Exception as e:
            logger.error(f"Error getting workspace queries: {str(e)}")
            raise
    
    # ==================== UTILITY METHODS ====================
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def health_check(self) -> Dict[str, Any]:
        """Check MongoDB health"""
        try:
            self.client.server_info()
            
            return {
                "status": "healthy",
                "database": self.db.name,
                "collections": {
                    "users": self.users.count_documents({}),
                    "documents": self.documents.count_documents({}),
                    "chunks": self.chunks.count_documents({}),
                    "queries": self.queries.count_documents({})
                }
            }
        except Exception as e:
            logger.error(f"MongoDB health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Singleton instance
_db_service = None

def get_db() -> MongoDBService:
    """Get or create MongoDB service singleton"""
    global _db_service
    if _db_service is None:
        _db_service = MongoDBService()
    return _db_service


def close_db():
    """Close MongoDB connection"""
    global _db_service
    if _db_service is not None:
        _db_service.close()
        _db_service = None