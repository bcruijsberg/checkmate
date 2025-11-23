import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from tavily import TavilyClient
import json
from utils import format_docs
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv(".env", override=True)

#  Load the LLM
llm_tuned = ChatOllama(model="Socratic-8B.Q4_K_M:latest", temperature=0.1, base_url="http://localhost:11434")
llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.1)

# Load Tavily
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))

# ───────────────────────────────────────────────────────────────────────
# LOAD FAISS DATABASE WITH VERIFIED CLAIMS
# ───────────────────────────────────────────────────────────────────────

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load existing FAISS index
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Set up retriever
retriever = vectorstore.as_retriever()

# ───────────────────────────────────────────────────────────────────────
# TOOLS
# ───────────────────────────────────────────────────────────────────────
@tool
def tavily_search(query: str) -> str:
    """Search the web with Tavily and return JSON-like results."""
    if not query.strip():
        return "No query provided."
    resp = tavily_client.search(query=query, max_results=5, search_depth="advanced")
    return json.dumps(resp, indent=2)


@tool
def retriever_tool(query: str) -> str:
    """Search the FACTors dataset and return context + allowed URLs."""
    docs = retriever.invoke(query)
    context_block = format_docs(docs)
    urls = [d.metadata.get("url", "") for d in docs if d.metadata.get("url")]
    allowed = dict(enumerate(urls))
    return (
        "CONTEXT:\n" + context_block + "\n\n"
        "ALLOWED_URLS:\n" + json.dumps(allowed, indent=2)
    )

tools = [retriever_tool, tavily_search]
llm_tools = llm.bind_tools(tools)
tools_dict = {t.name: t for t in tools}
