"""
RAG Vectorstore - Compatible with LangChain 0.2+
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BRAND_DOCS_DIR = Path(__file__).parent / "brand_docs"
VECTORSTORE_PATH = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore() -> FAISS:
    docs = []
    for txt_file in BRAND_DOCS_DIR.glob("*.txt"):
        loader = TextLoader(str(txt_file))
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTORSTORE_PATH))
    print(f"[RAG] Vectorstore built — {len(chunks)} chunks saved.")
    return vectorstore


def load_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if VECTORSTORE_PATH.exists():
        print("[RAG] Loading existing vectorstore...")
        return FAISS.load_local(
            str(VECTORSTORE_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    print("[RAG] No index found — building now...")
    return build_vectorstore()


def get_retriever(k: int = 3):
    vs = load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})
