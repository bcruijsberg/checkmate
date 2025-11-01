import sys
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
#from langchain_ollama import ChatOllama
from tavily import TavilyClient
import streamlit as st

# Load alle the API keys
load_dotenv(dotenv_path=".env", override=True)

# Initialize Tavily client 
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))

#os.environ["LANGSMITH_TRACING"] = "true"
#os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
#os.environ["LANGSMITH_PROJECT"]="pr-left-technician-100"

#low temperature for more factual answers,
#llm = ChatOllama(model="qwen3:4b", temperature=0.1, base_url="http://localhost:11434")
llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.1)

# location for src files
sys.path.append(os.path.abspath("./src"))

# ───────────────────────────────────────────────────────────────────────
# LOAD FAISS DATABASE WITH VERIFIED CLAIMS
# ───────────────────────────────────────────────────────────────────────

from langchain.vectorstores import FAISS
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
from langchain.tools import tool
from utils import format_docs

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns information from the FACTors dataset.
    It returns two blocks as one string:
    1) CONTEXT: numbered snippets without URLs
    2) ALLOWED_URLS: a JSON dictionary of index -> url that the model must cite by index only
    """
    print(f"\n--retriever called, query: {query}--\n")
    docs = retriever.invoke(query)
    context_block = format_docs(docs)

    # Build a list of allowed URLs, and index them.
    urls = [d.metadata.get("url", "") for d in docs if d.metadata.get("url")]
    allowed = dict(enumerate(urls))
    print(f"Allowed URLs: {allowed}")

    # Return a single string so your existing tool plumbing still works.
    return (
        "CONTEXT (read-only; do NOT copy or invent URLs):\n"
        f"{context_block}\n\n"
        "ALLOWED_URLS (index -> url):\n"
        f"{json.dumps(allowed, indent=2)}\n\n"
        "INSTRUCTIONS: When citing, use indices from ALLOWED_URLS only (e.g., [0], [2]). "
        "Do not output raw URLs unless they come from ALLOWED_URLS."
    )


@tool
def tavily_search(query: str) -> str:
    """
    General-purpose web search using Tavily.
    Use this when the user gives a source (URL, outlet, author, platform) and subject.
    Returns a JSON-like text block with titles, urls and snippets.
    """
    print(f"\n--tavily_search called, query: {query}--\n")
    if not query or not query.strip():
        return "No query provided."

    resp = tavily_client.search(
        query=query,
        max_results=5,
        search_depth="advanced",
        include_raw_content=False,
    )

    # resp looks like: {"results": [...], "query": "...", ...}
    # To match your current pattern, we return a single string.
    results = resp.get("results", [])
    return json.dumps(
        {
            "SOURCE": "tavily",
            "query": query,
            "results": results,
            "INSTRUCTIONS": (
                "Cite using the URLs in `results`. Do NOT invent URLs. "
                "Prefer the most relevant/high-authority result."
            ),
        },
        indent=2,
    )


tools = [retriever_tool, tavily_search]
llm_tools = llm.bind_tools(tools)

# Dict for lookup by name
tools_dict = {our_tool.name: our_tool for our_tool in tools}
print(f"\n--tools registered: {list(tools_dict.keys())}--\n")

# ───────────────────────────────────────────────────────────────────────
# CLAIM GRAPH
# ───────────────────────────────────────────────────────────────────────
from claim_nodes import checkable_fact,checkable_confirmation,retrieve_information,clarify_information,produce_summary,get_confirmation
from langgraph.graph import StateGraph, START, END
from state_scope import AgentStateClaim

claim.add_node("checkable_fact", checkable_fact)
claim.add_node("checkable_confirmation", checkable_confirmation)
claim.add_node("retrieve_information", retrieve_information)
claim.add_node("clarify_information", clarify_information)
claim.add_node("produce_summary", produce_summary)
claim.add_node("get_confirmation", get_confirmation)

# Entry point
claim.add_edge(START, "checkable_fact")
claim.add_edge("checkable_fact", "checkable_confirmation")
claim.add_edge("retrieve_information", "clarify_information")
claim.add_edge("produce_summary", "get_confirmation")
claim.add_edge("get_confirmation", END)

claim_flow = claim.compile()

# ───────────────────────────────────────────────────────────────────────
# SOURCE GRAPH
# ───────────────────────────────────────────────────────────────────────
from source_nodes import claim_matching,match_or_continue,get_source,get_primary_source,locate_primary_source,select_primary_source,research_claim
from langgraph.graph import StateGraph, START, END
from state_scope import AgentStateSource

source = StateGraph(AgentStateSource)

source.add_node("claim_matching", claim_matching)
source.add_node("match_or_continue", match_or_continue)
source.add_node("get_source", get_source)
source.add_node("get_primary_source", get_primary_source)
source.add_node("locate_primary_source", locate_primary_source)
source.add_node("select_primary_source", select_primary_source)
source.add_node("research_claim", research_claim)

# Entry point
source.add_edge(START, "claim_matching")
source.add_edge("claim_matching", "match_or_continue")
source.add_edge("match_or_continue", "get_source")
source.add_edge("get_source", "get_primary_source")
source.add_edge("locate_primary_source", "select_primary_source")
source.add_edge("research_claim", END)

source_flow = source.compile()

# ───────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="CheckMate", page_icon="✅")

st.title("🕵️ CheckMate – Claim checker")

# init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "claim_result" not in st.session_state:
    st.session_state.claim_result = None
if "source_result" not in st.session_state:
    st.session_state.source_result = None

# show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_input = st.chat_input("What claim do you want to investigate?" if not st.session_state.claim_result else "Add info or clarify:")

if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 1) first run claim flow if not done yet
    if st.session_state.claim_result is None:
        state_claim: AgentStateClaim = {
            "messages": [HumanMessage(content=user_input)],
            "claim": user_input,
            "checkable": None,
            "subject": None,
            "quantitative": None,
            "precision": None,
            "based_on": None,
            "confirmed": False,
            "question": None,
            "alerts": [],
            "summary": None,
        }
        claim_out = claim_flow.invoke(state_claim)
        st.session_state.claim_result = claim_out
        # show assistant
        with st.chat_message("assistant"):
            st.write(claim_out.get("summary", "I analyzed the claim."))
        st.session_state.messages.append({"role": "assistant", "content": claim_out.get("summary", "")})
    else:
        # 2) once claim is done, run source flow
        prev = st.session_state.claim_result
        state_source: AgentStateSource = {
            "messages": prev.get("messages", []) + [HumanMessage(content=user_input)],
            "claim": prev.get("claim", user_input),
            "checkable": True,
            "subject": prev.get("subject", ""),
            "quantitative": prev.get("quantitative", ""),
            "precision": prev.get("precision", ""),
            "based_on": prev.get("based_on", ""),
            "confirmed": False,
            "search_queries": [],
            "tavily_context": None,
            "research_focus": None,
            "research_results": [],
            "alerts": prev.get("alerts", []),
            "summary": prev.get("summary", ""),
            "claim_url": None,
            "claim_source": None,
            "primary_source": False,
            "match": False,
        }
        source_out = source_flow.invoke(state_source)
        st.session_state.source_result = source_out
        with st.chat_message("assistant"):
            st.write(source_out.get("summary", "I looked into the source of the claim."))
        st.session_state.messages.append({"role": "assistant", "content": source_out.get("summary", "")})