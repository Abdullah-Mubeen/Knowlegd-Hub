import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from app.utils.openai_service import get_openai_service
from app.utils.pinecone_service import get_pinecone_service

def test_services():
    print("🧪 Testing OpenAI Service...")
    openai_svc = get_openai_service()
    
    # Test embedding
    embedding = openai_svc.generate_embedding("Test business document")
    print(f"✅ Embedding dimension: {len(embedding)}")
    
    print("\n🧪 Testing Pinecone Service...")
    pinecone_svc = get_pinecone_service()
    
    # Test add documents
    test_texts = [
        "Our company offers cloud services",
        "We provide 24/7 customer support"
    ]
    test_metadata = [
        {"business_id": "test_biz", "doc_type": "service"},
        {"business_id": "test_biz", "doc_type": "support"}
    ]
    
    ids = pinecone_svc.add_documents(
        texts=test_texts,
        metadatas=test_metadata,
        namespace="test_business"
    )
    print(f"✅ Added {len(ids)} documents")
    
    # Test search
    results = pinecone_svc.similarity_search(
        query="customer support",
        namespace="test_business",
        top_k=2
    )
    print(f"✅ Search returned {len(results)} results")
    for text, score, metadata in results:
        print(f"  - Score: {score:.3f} | {text[:50]}...")
    
    # Get stats
    stats = pinecone_svc.get_stats(namespace="test_business")
    print(f"✅ Index stats: {stats}")
    
    print("\n✅ All services working with LangChain!")

if __name__ == "__main__":
    test_services()