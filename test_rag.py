import chromadb
from chromadb.utils import embedding_functions

try:
    print("Testing ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_data")
    collection = client.get_or_create_collection(name="tutor_docs")
    
    collection.add(
        documents=["This is a test document about artificial intelligence."],
        metadatas=[{"source": "test"}],
        ids=["id1"]
    )
    print("Added document!")
    print(f"Collection count: {collection.count()}")
    
    results = collection.query(
        query_texts=["What is artificial intelligence?"],
        n_results=1
    )
    print("Query results:", results)
except Exception as e:
    print(f"Error: {e}")
