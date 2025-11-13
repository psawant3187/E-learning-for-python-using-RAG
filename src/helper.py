import os
import json
import redis
from typing import List
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# 🔧 Global Paths & Settings
# ---------------------------
DATA_DIR = os.path.join(os.getcwd(), "chat_assistant", "data", "tutorials")
INDEX_PATH = os.path.join(os.getcwd(), "chat_assistant", "data", "faiss_index")
CACHE_FILE = os.path.join(os.getcwd(), "chat_assistant", "data", "cache.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

# ---------------------------
# ⚡ Redis Setup
# ---------------------------
def init_redis():
    """Initialize Redis for local caching."""
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        return r
    except redis.exceptions.ConnectionError:
        return None

redis_client = init_redis()

# ---------------------------
# 🧠 Tutorial Ingestion
# ---------------------------
def add_tutorial_to_index(file_path: str):
    """Load tutorial file and add it to FAISS index."""
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    embedding = OllamaEmbeddings(model="mistral")

    if os.path.exists(INDEX_PATH):
        db = FAISS.load_local(
            INDEX_PATH, 
            embedding, 
            allow_dangerous_deserialization=True
        )
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embedding)

    db.save_local(INDEX_PATH)

# ---------------------------
# ⚙️ Custom Hybrid Retriever
# ---------------------------
class HybridRetriever:
    """Combine FAISS (semantic) + BM25 (keyword) retrievers with weighted scores."""
    def __init__(self, faiss_retriever, bm25_retriever, alpha=0.6):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha  # weight toward FAISS results

    def _safe_retrieve(self, retriever, query: str):
        """Safely retrieve documents using either invoke or get_relevant_documents."""
        try:
            # Try LangChain 0.3.* invoke method first
            if hasattr(retriever, 'invoke'):
                return retriever.invoke(query)
            elif hasattr(retriever, 'get_relevant_documents'):
                return retriever.get_relevant_documents(query)
            else:
                # Direct call as fallback
                return retriever(query)
        except Exception:
            return []

    def get_relevant_documents(self, query: str):
        """Retrieve documents from both retrievers and merge."""
        faiss_docs = self._safe_retrieve(self.faiss_retriever, query)
        bm25_docs = self._safe_retrieve(self.bm25_retriever, query)

        # Merge + deduplicate by content
        all_docs = {d.page_content: d for d in faiss_docs}
        for d in bm25_docs:
            if d.page_content not in all_docs:
                all_docs[d.page_content] = d

        combined = list(all_docs.values())
        return combined[:5]  # top few docs

    def invoke(self, input_data):
        """LangChain 0.3.* compatibility - invoke method."""
        # Handle both string queries and dict inputs
        if isinstance(input_data, dict):
            query = input_data.get("question", input_data.get("query", ""))
        else:
            query = str(input_data)
        return self.get_relevant_documents(query)
    
    def __call__(self, query: str):
        """Direct call compatibility."""
        return self.get_relevant_documents(query)

# ---------------------------
# 🔍 Build Hybrid RAG
# ---------------------------
def get_or_build_rag():
    """Return hybrid FAISS + BM25 retriever safely."""
    embedding = OllamaEmbeddings(model="mistral")

    # ✅ If FAISS index exists, just load it
    if os.path.exists(INDEX_PATH):
        faiss_store = FAISS.load_local(
            INDEX_PATH, 
            embedding, 
            allow_dangerous_deserialization=True
        )
    else:
        # ✅ Build new index from tutorials
        all_docs = []
        tutorial_files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]

        # ⚠️ Handle empty dataset
        if not tutorial_files:
            # Create a sample file to avoid crash
            sample_text = """# Python Basics
This is a sample tutorial file.
Python is an interpreted, object-oriented programming language.

# Python Loops
Loops allow you to repeat code. Python has for loops and while loops.

For loop example:
for i in range(5):
    print(i)

While loop example:
count = 0
while count < 5:
    print(count)
    count += 1"""
            sample_path = os.path.join(DATA_DIR, "sample_tutorial.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(sample_text)
            tutorial_files.append("sample_tutorial.txt")

        # ✅ Load documents
        for f in tutorial_files:
            loader = TextLoader(os.path.join(DATA_DIR, f), encoding="utf-8")
            all_docs.extend(loader.load())

        if not all_docs:
            raise ValueError("❌ No documents found to build the RAG index.")

        faiss_store = FAISS.from_documents(all_docs, embedding)
        faiss_store.save_local(INDEX_PATH)

    # ✅ Build BM25 retriever
    docs_for_bm25 = faiss_store.similarity_search("python", k=10)
    if not docs_for_bm25:
        raise ValueError("❌ BM25 retriever could not find any base documents.")
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 3
    
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 3})

    # ✅ Combine retrievers
    hybrid_retriever = HybridRetriever(faiss_retriever, bm25_retriever, alpha=0.6)
    return hybrid_retriever


# ---------------------------
# 🧠 Cache utilities
# ---------------------------
def cache_response(question, response):
    """Cache response to Redis (if available) and JSON file."""
    if redis_client:
        try:
            redis_client.setex(question, 3600, response)  # 1 hour TTL
        except Exception:
            pass
    
    try:
        cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        cache[question] = {"response": response}
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def get_cached_response(question):
    """Retrieve cached response from Redis or JSON."""
    if redis_client:
        try:
            cached = redis_client.get(question)
            if cached:
                return cached
        except Exception:
            pass
    
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if question in cache:
                return cache[question]["response"]
    except Exception:
        pass
    
    return None

# ---------------------------
# 🧩 RAG Query (Direct Method)
# ---------------------------
def format_docs(docs):
    """Format documents for context."""
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs[:3])

def query_lesson_fast(question, retriever):
    """Query with direct LLM invocation - optimized for speed."""
    # Check cache first
    cached = get_cached_response(question)
    if cached:
        return f"[Cached]\n{cached}"

    # Get relevant documents
    try:
        docs = retriever.get_relevant_documents(question)
        context = format_docs(docs)
    except Exception:
        context = "Unable to retrieve context."

    # Initialize LLM with optimized settings
    llm = Ollama(
        model="gemma:2b",
        temperature=0.7,
        num_predict=512,  # Limit response length for speed
    )

    # Create prompt
    prompt_text = f"""You are a Python e-learning assistant.
Use the following context to answer the user's question clearly and concisely.

Context:
{context}

Question:
{question}

Provide a clear explanation with code examples when relevant."""

    # Direct invocation
    try:
        answer = llm.invoke(prompt_text)
        # Handle different response types
        if hasattr(answer, 'content'):
            answer = answer.content
        elif not isinstance(answer, str):
            answer = str(answer)
    except Exception:
        answer = "I'm having trouble generating a response. Please try rephrasing your question."
    
    # Cache the response
    cache_response(question, answer)
    return answer