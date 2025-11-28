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
        """
        Update user statistics with proper increments
        
        Args:
            user_id: User identifier
            increment_documents: Amount to increment total_documents by
            increment_chunks: Amount to increment total_chunks by
            increment_queries: Amount to increment total_queries by
            
        Returns:
            True if update was successful
        """
        try:
            inc_fields = {}
            
            if increment_documents != 0:
                inc_fields["total_documents"] = increment_documents
            if increment_chunks != 0:
                inc_fields["total_chunks"] = increment_chunks
            if increment_queries != 0:
                inc_fields["total_queries"] = increment_queries
            
            # Build update query
            update_query = {"$set": {"updated_at": datetime.utcnow()}}
            if inc_fields:
                update_query["$inc"] = inc_fields
            
            result = self.users.update_one(
                {"user_id": user_id},
                update_query
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating user stats: {str(e)}")
            raise
    
    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user statistics"""
        try:
            user = self.users.find_one(
                {"user_id": user_id},
                {
                    "total_documents": 1,
                    "total_chunks": 1,
                    "total_queries": 1,
                    "created_at": 1,
                    "updated_at": 1
                }
            )
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"Error getting user stats: {str(e)}")
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
    
    def count_chunks_by_document(self, document_id: str) -> int:
        """Count chunks for a specific document"""
        try:
            count = self.chunks.count_documents({"document_id": document_id})
            return count
        except Exception as e:
            logger.error(f"Error counting chunks by document: {str(e)}")
            raise
    
    def count_documents_by_type(self, workspace_id: str, document_type: str) -> int:
        """Count documents of a specific type in a workspace"""
        try:
            count = self.documents.count_documents({
                "workspace_id": workspace_id,
                "document_type": document_type,
                "status": "active"
            })
            return count
        except Exception as e:
            logger.error(f"Error counting documents by type: {str(e)}")
            raise
    
    def list_documents_by_type(
        self,
        workspace_id: str,
        document_type: str,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """List documents of a specific type"""
        try:
            docs = list(
                self.documents
                .find({
                    "workspace_id": workspace_id,
                    "document_type": document_type,
                    "status": "active"
                })
                .sort("created_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            
            for doc in docs:
                doc["_id"] = str(doc["_id"])
            
            return docs
            
        except Exception as e:
            logger.error(f"Error listing documents by type: {str(e)}")
            raise
    
    def search_documents(
        self,
        workspace_id: str,
        search_term: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search documents by filename or metadata"""
        try:
            docs = list(
                self.documents
                .find({
                    "workspace_id": workspace_id,
                    "status": "active",
                    "$or": [
                        {"filename": {"$regex": search_term, "$options": "i"}},
                        {"document_type": search_term}
                    ]
                })
                .sort("created_at", DESCENDING)
                .limit(limit)
            )
            
            for doc in docs:
                doc["_id"] = str(doc["_id"])
            
            return docs
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
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
    
    def get_chunks_by_workspace(
        self,
        workspace_id: str,
        limit: int = 1000,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a workspace"""
        try:
            chunks = list(
                self.chunks
                .find({"workspace_id": workspace_id})
                .sort("created_at", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            
            for chunk in chunks:
                chunk["_id"] = str(chunk["_id"])
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks by workspace: {str(e)}")
            raise
    
    def update_chunk_metadata(
        self,
        chunk_id: str,
        metadata_update: Dict[str, Any]
    ) -> bool:
        """Update chunk metadata"""
        try:
            result = self.chunks.update_one(
                {"chunk_id": chunk_id},
                {
                    "$set": {
                        "metadata": metadata_update,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating chunk metadata: {str(e)}")
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
    
    def get_query_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get query statistics for a workspace"""
        try:
            total_queries = self.queries.count_documents({
                "workspace_id": workspace_id
            })
            
            # Get top questions
            pipeline = [
                {"$match": {"workspace_id": workspace_id}},
                {"$group": {"_id": "$question", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
            
            top_questions = list(self.queries.aggregate(pipeline))
            
            return {
                "workspace_id": workspace_id,
                "total_queries": total_queries,
                "top_questions": top_questions
            }
            
        except Exception as e:
            logger.error(f"Error getting query stats: {str(e)}")
            raise
    
    def delete_workspace_data(self, workspace_id: str) -> Dict[str, int]:
        """
        Completely delete all data for a workspace
        
        Args:
            workspace_id: Workspace identifier
            
        Returns:
            Dict with counts of deleted items
        """
        try:
            # Delete chunks
            deleted_chunks = self.chunks.delete_many({
                "workspace_id": workspace_id
            }).deleted_count
            
            # Delete documents
            deleted_docs = self.documents.delete_many({
                "workspace_id": workspace_id
            }).deleted_count
            
            # Delete queries
            deleted_queries = self.queries.delete_many({
                "workspace_id": workspace_id
            }).deleted_count
            
            logger.info(f"Deleted workspace {workspace_id}: {deleted_docs} docs, {deleted_chunks} chunks, {deleted_queries} queries")
            
            return {
                "documents": deleted_docs,
                "chunks": deleted_chunks,
                "queries": deleted_queries
            }
            
        except Exception as e:
            logger.error(f"Error deleting workspace data: {str(e)}")
            raise
    
    # ==================== UTILITY METHODS ====================
    
    def get_collection_sizes(self) -> Dict[str, int]:
        """Get approximate sizes of all collections"""
        try:
            return {
                "users": self.users.count_documents({}),
                "documents": self.documents.count_documents({}),
                "chunks": self.chunks.count_documents({}),
                "queries": self.queries.count_documents({})
            }
        except Exception as e:
            logger.error(f"Error getting collection sizes: {str(e)}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            total_users = self.users.count_documents({})
            total_docs = self.documents.count_documents({})
            total_chunks = self.chunks.count_documents({})
            total_queries = self.queries.count_documents({})
            
            # Get document type breakdown
            doc_types = list(self.documents.aggregate([
                {"$match": {"status": "active"}},
                {"$group": {"_id": "$document_type", "count": {"$sum": 1}}}
            ]))
            
            return {
                "total_users": total_users,
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_queries": total_queries,
                "document_types": {item["_id"]: item["count"] for item in doc_types},
                "avg_chunks_per_document": round(total_chunks / max(total_docs, 1), 2)
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            raise
    
    def bulk_update_document_metadata(
        self,
        document_ids: List[str],
        metadata_update: Dict[str, Any]
    ) -> int:
        """Bulk update metadata for multiple documents"""
        try:
            result = self.documents.update_many(
                {"document_id": {"$in": document_ids}},
                {
                    "$set": {
                        "file_metadata": metadata_update,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            logger.info(f"Updated metadata for {result.modified_count} documents")
            return result.modified_count
            
        except Exception as e:
            logger.error(f"Error bulk updating document metadata: {str(e)}")
            raise
    
    def find_orphaned_chunks(self, workspace_id: str) -> List[Dict[str, Any]]:
        """
        Find chunks whose parent documents don't exist (orphaned chunks)
        
        Args:
            workspace_id: Workspace identifier
            
        Returns:
            List of orphaned chunks
        """
        try:
            # Get all document IDs in workspace
            docs = self.documents.find(
                {"workspace_id": workspace_id},
                {"document_id": 1}
            )
            doc_ids = {doc["document_id"] for doc in docs}
            
            # Find chunks not in document set
            orphaned = list(
                self.chunks.find({
                    "workspace_id": workspace_id,
                    "document_id": {"$nin": list(doc_ids)}
                })
            )
            
            for chunk in orphaned:
                chunk["_id"] = str(chunk["_id"])
            
            logger.warning(f"Found {len(orphaned)} orphaned chunks in workspace {workspace_id}")
            return orphaned
            
        except Exception as e:
            logger.error(f"Error finding orphaned chunks: {str(e)}")
            raise
    
    def delete_orphaned_chunks(self, workspace_id: str) -> int:
        """Delete all orphaned chunks in a workspace"""
        try:
            orphaned = self.find_orphaned_chunks(workspace_id)
            chunk_ids = [c["chunk_id"] for c in orphaned]
            
            if not chunk_ids:
                return 0
            
            result = self.chunks.delete_many({
                "chunk_id": {"$in": chunk_ids}
            })
            
            logger.info(f"Deleted {result.deleted_count} orphaned chunks")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting orphaned chunks: {str(e)}")
            raise
    
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