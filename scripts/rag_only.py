from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import dotenv
import os

dotenv.load_dotenv()

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# Setup
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="qwen2.5:0.5b", temperature=0)

# RAG prompt
template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Query
response = rag_chain.invoke("How does karpanter deal with spot nodes?")
print(response.content)
