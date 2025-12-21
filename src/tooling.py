import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from tavily import TavilyClient
import json
from utils import format_docs
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from state_scope import SearchResult, TavilySearchOutput

load_dotenv(".env", override=True)

#  Load the LLM
llm_tuned = ChatOllama(model="mistral7b-q4km:latest", temperature=0.5, base_url="http://localhost:11434")
llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.1)
#qwen3:1.7b was also tested, but did not provide explanation in the retrieve information node
#llm = ChatOllama(model="qwen3:4b", temperature=0.3, base_url="http://localhost:11434")
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
def tavily_search(query: str, max_results: int = 10) -> TavilySearchOutput:
    """Search the web with Tavily and return JSON-like results."""
    if not query.strip():
        return TavilySearchOutput(query=query, results=[], error="No query provided.")

    resp = tavily_client.search(query=query, max_results=max_results, search_depth="advanced")
    results = [
        SearchResult(
            title=r.get("title"),
            url=r.get("url"),
        )
        for r in resp.get("results", [])
    ]
    return TavilySearchOutput(query=query, results=results)


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
