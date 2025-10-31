# from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
import dotenv

dotenv.load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # Where to save data locally, remove if not necessary
)



@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


llm = ChatOllama(
    # model="llama3:8b", #incompatible with tool use
    model="qwen2.5:0.5b",
    temperature=0,
    # other params...
)




tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(
    llm,
    tools,
    system_prompt=prompt
    )

messages = [
    {"role": "system", "content": "You are a helpful assistant. Give straight forward answers."},
    {"role": "user", "content": "How does karpanter deal with spot nodes?"},
]

ai_msg = agent.invoke({"messages": messages})
print(ai_msg['messages'][-1].content)