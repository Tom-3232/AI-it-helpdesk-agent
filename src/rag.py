import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except Exception:
    pass

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

_vectorstore = None
_embeddings = None

def get_embeddings():
    """Lazy load HuggingFace Embeddings model."""
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings

def get_or_create_vectorstore():
    """Initializes or loads the ChromaDB vector database from knowledge_base/ documents."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if persistent vectorstore exists and has documents
    try:
        vectorstore = Chroma(
            collection_name="it_helpdesk_kb",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        # Check if collection is populated
        if vectorstore._collection.count() > 0:
            _vectorstore = vectorstore
            return _vectorstore
    except Exception as e:
        print(f"[RAG Info] Initializing new Chroma collection: {e}")

    # Build vectorstore from knowledge_base directory
    if not KB_DIR.exists():
        KB_DIR.mkdir(parents=True, exist_ok=True)

    loader = DirectoryLoader(
        str(KB_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()

    if not docs:
        print("[RAG Warning] No documents found in knowledge_base/")
        _vectorstore = Chroma(
            collection_name="it_helpdesk_kb",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        return _vectorstore

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="it_helpdesk_kb",
        persist_directory=str(CHROMA_DIR)
    )
    _vectorstore = vectorstore
    return _vectorstore

def query_rag(query: str, top_k: int = 3) -> dict:
    """
    Queries ChromaDB vectorstore and retrieves top_k relevant doc chunks.
    Returns dict: {'chunks': list, 'context': str}
    """
    try:
        vs = get_or_create_vectorstore()
        results = vs.similarity_search(query, k=top_k)
        chunks = [doc.page_content for doc in results]
        context = "\n---\n".join(chunks)
        return {"chunks": chunks, "context": context}
    except Exception as e:
        print(f"[RAG Error] Query failed: {e}")
        return {"chunks": [], "context": f"Failed to retrieve context from knowledge base: {e}"}

def get_indexed_chunk_count() -> int:
    """Returns the total number of chunks currently indexed in ChromaDB."""
    try:
        vs = get_or_create_vectorstore()
        return vs._collection.count()
    except Exception:
        return 0
