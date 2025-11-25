# from langchain.agents import create_agent
# from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrockConverse
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
import dotenv
import os

dotenv.load_dotenv()

llm = ChatBedrockConverse(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    max_tokens=4096,
    # additional_model_request_fields={
    #     "thinking": {"type": "enabled", "budget_tokens": 1024},
    # },
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR,
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



docs = vector_store.similarity_search("test", k=1)
print("✅ Vector store working:", len(docs), "docs found")



result = retrieve_context.invoke("spot nodes")
# print(result)
print("✅ Tool working")


tools = [retrieve_context]
# If desired, specify custom instructions
# prompt = (
#     "You have access to a tool that retrieves context from a blog post. "
#     "Use the tool to help answer user queries."
# )
agent = create_agent(
    llm,
    tools,
    # system_prompt=prompt
    )

messages = [
    {"role": "system", "content": "You are a helpful assistant. Give straight forward, short answers."},
    {"role": "system", "content": "Use the tools available, do not use internal knowledge.."},
    # {"role": "user", "content": "Como o Spot IO lida com nodes spot?"},
    {"role": "user", "content": "Como o karpenter lida com nodes spot?"},
]


# ai_msg = llm.invoke(messages)
# print(ai_msg.content)

ai_msg = agent.invoke({"messages": messages})
print(ai_msg['messages'][-1].content)
