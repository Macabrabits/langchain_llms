# pip install langchain-chroma langchain-huggingface langchain-text-splitters langchain-community sentence-transformers chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Fixed
from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import torch
import dotenv
import os
dotenv.load_dotenv()

# Check GPU availability
print(f"🎮 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🎮 CUDA version: {torch.version.cuda}")

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
MARKDOWNS_DIR = os.path.join(DATA_DIR, "markdowns")

# Carregar markdowns
loader = DirectoryLoader(MARKDOWNS_DIR, glob="**/*.md")
docs = loader.load()

print(f"📄 Loaded {len(docs)} documents")

# Dividir texto em pedaços
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

print(f"✂️ Split into {len(chunks)} chunks")
# USE GPU! 🚀
print("🔄 Loading embedding model on GPU...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda'},  # USE GPU!
    encode_kwargs={'batch_size': 32}  # Larger batch for GPU
)

print(f"🔄 Creating vector database on GPU...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB_DIR,
    collection_name="example_collection"
)

print(f"✅ Base vetorial criada em {CHROMA_DB_DIR}")
print(f"📊 Total de {db._collection.count()} documentos no banco")


retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Test retrieval
query = "how to scale nodes to zero in karpenter?"
retrieved_docs = retriever.invoke(query)

print(f"🔍 Retrieved {len(retrieved_docs)} documents for: {query}\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"📄 Result {i}:")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print("-" * 80)
