import os
import uuid
import chromadb
from PyPDF2 import PdfReader

# Initialize ChromaDB persistently
CHROMA_DATA_PATH = os.path.join(os.getcwd(), "chroma_data")
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection_name = "tutor_docs"

def chunk_text(text, chunk_size=400, overlap=50):
    """Simple character-based chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_and_add_pdf(pdf_file_path):
    """
    Reads a PDF, chunks its text, and stores it in ChromaDB.
    Returns the number of chunks processed.
    """
    try:
        reader = PdfReader(pdf_file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        if not text.strip():
            return 0
            
        chunks = chunk_text(text)
        
        # Get or create collection
        # Uses default all-MiniLM-L6-v2 embedding function behind the scenes
        collection = client.get_or_create_collection(name=collection_name)
        
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": "pdf_upload"} for _ in chunks]
        
        if chunks:
            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
        return len(chunks)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return 0

def query_knowledge_base(query_text, n_results=3):
    """
    Queries ChromaDB for the most relevant chunks given a text query.
    Returns a concatenated string of the relevant texts.
    """
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        # Collection might not exist yet
        return ""
    
    # If the collection is empty, querying will fail, so let's handle that securely.
    if collection.count() == 0:
        return ""
        
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, collection.count())
        )
        
        if not results or not results['documents'] or not results['documents'][0]:
            return ""
        
        return "\n\n".join(results['documents'][0])
    except Exception as e:
        print(f"Error querying knowledge base: {e}")
        return ""

def clear_knowledge_base():
    """Utility to clear the chromadb collection if needed."""
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
