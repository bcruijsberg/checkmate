import sys
import os
import streamlit as st

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGSMITH_PROJECT"] = "pr-left-technician-100"

# location for src files
sys.path.append(os.path.abspath("./src"))

# ───────────────────────────────────────────────────────────────────────
# CLAIM GRAPH
# ───────────────────────────────────────────────────────────────────────
from claim_nodes import (
    router,
    checkable_fact,
    checkable_confirmation,
    retrieve_information,
    clarify_information,
    produce_summary,
    get_confirmation,
    get_rag_queries,
    confirm_rag_queries,
    route_rag_confirm,
    rag_retrieve_worker,
    reduce_rag_results,
    structure_claim_matching,
    match_or_continue,
    get_source,
    get_location_source,
    get_source_queries,
    get_search_queries,
    confirm_search_queries,
    reset_search_state,
    route_after_confirm,
    find_sources_worker,
    reduce_sources,
    select_primary_source,
    iterate_search,
    critical_question,
)
import asyncio
from langgraph.graph import StateGraph, START, END
from state_scope import AgentStateClaim
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="CheckMate", page_icon="✅")

st.markdown("""
<style>
/* Use percentage-based width */
[data-testid="stSidebar"] {
    width: 30% !important;           /* Sidebar takes 30% of screen width */
    min-width: 280px !important;     /* (optional) Prevent it from becoming too small */
}

/* Ensure inside container scales too */
[data-testid="stSidebar"] > div:first-child {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────
# GRAPH SETUP
# ───────────────────────────────────────────────────────────────────────

claim = StateGraph(AgentStateClaim)

#getting all info about the claim nodes
claim.add_node("checkable_fact", checkable_fact)
claim.add_node("checkable_confirmation", checkable_confirmation)
claim.add_node("retrieve_information", retrieve_information)
claim.add_node("clarify_information", clarify_information)
claim.add_node("produce_summary", produce_summary)
claim.add_node("get_confirmation", get_confirmation)
claim.add_node("critical_question", critical_question)

#Claim matching nodes
claim.add_node("get_rag_queries", get_rag_queries)
claim.add_node("confirm_rag_queries", confirm_rag_queries)
claim.add_node("rag_retrieve_worker", rag_retrieve_worker)
claim.add_node("reduce_rag_results", reduce_rag_results)
claim.add_node("structure_claim_matching", structure_claim_matching)
claim.add_node("match_or_continue", match_or_continue)

# Source finding nodes and search query nodes
claim.add_node("get_source", get_source)
claim.add_node("get_location_source", get_location_source)
claim.add_node("get_source_queries", get_source_queries)
claim.add_node("confirm_search_queries", confirm_search_queries)
claim.add_node("reset_search_state", reset_search_state)
claim.add_node("find_sources_worker", find_sources_worker)
claim.add_node("reduce_sources", reduce_sources)
claim.add_node("select_primary_source", select_primary_source)
claim.add_node("get_search_queries", get_search_queries)
claim.add_node("iterate_search",iterate_search)
claim.add_node("router", router)

# Entry point
claim.add_edge(START, "router")
claim.add_edge("checkable_fact", "critical_question")
claim.add_edge("retrieve_information", "clarify_information")
claim.add_edge("produce_summary", "critical_question")

# Connecting claim matching nodes
claim.add_conditional_edges("confirm_rag_queries", route_rag_confirm)
claim.add_edge("rag_retrieve_worker", "reduce_rag_results")
claim.add_edge("reduce_rag_results", "structure_claim_matching")

# Connecting source finding and search query nodes
claim.add_edge("get_source_queries", "critical_question")
claim.add_edge("get_search_queries", "critical_question")
claim.add_edge("confirm_search_queries", "reset_search_state")
claim.add_conditional_edges("reset_search_state", route_after_confirm)
claim.add_edge("find_sources_worker", "reduce_sources")

claim_flow = claim.compile()


def flush_new_ai_messages():
    """
    Take new AI messages from claim_state["messages"]) and render them as assistant messages
    """
    final_messages = st.session_state.claim_state.get("messages", [])
    start_idx = st.session_state.graph_cursor

    new_texts = []
    for m in final_messages[start_idx:]:
        if isinstance(m, AIMessage):
            st.session_state.messages.append(
                {"role": "assistant", "content": m.content}
            )
            new_texts.append(m.content)

    # Update cursor so we don't re-render old messages
    st.session_state.graph_cursor = len(final_messages)

    # Render just the new ones
    for text in new_texts:
        with st.chat_message("assistant"):
            st.write(text)


# ───────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────────────────────────────

st.title("🕵️ CheckMate – Claim checker")

claim_question = "What claim do you want to investigate?"

# Initialize UI history for MAIN chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": claim_question}]

# LangGraph message cursor to avoid duplicating AI turns
if "graph_cursor" not in st.session_state:
    st.session_state.graph_cursor = 0

# Claim state (graph state)
if "claim_state" not in st.session_state:
    st.session_state.claim_state = {
        "messages": [],
        "messages_critical": [],
        "claim": None,
        "checkable": None,
        "additional_context": None,
        "subject": None,
        "quantitative": None,
        "precision": None,
        "based_on": None,
        "queries_confirmed": False,
        "question": None,
        "alerts": [],
        "summary": None,
        "awaiting_user": False,
        "explanation": None,
        "tool_trace":None,
        "rag_trace": [],
        "next_node": None,
        "search_queries": [],
        "tavily_context": [],
        "current_query": None,
        "research_focus": None,
        "claim_url": None,
        "claim_source": None,
        "primary_source": False,
        "source_description": None,
        "match": False,
        "critical_question": None,
        "reasoning_summary": None
    }
    st.session_state.graph_cursor = 0

# Optional done flag
if "claim_done" not in st.session_state:
    st.session_state.claim_done = False

# ───────────────────────────────────────────────────────────────────────
# MAIN FACT-CHECK CHAT (center)
# ───────────────────────────────────────────────────────────────────────

# Render full MAIN chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


# ───────────────────────────────────────────────────────────────────────
# GLOBAL MAIN CHAT INPUT (single chat_input, pinned at bottom)
# ───────────────────────────────────────────────────────────────────────

main_prompt = st.chat_input(
    "Type your response",
    key="main_chat_input",
)

# ───────────────────────────────────────────────────────────────────────
# HANDLE INPUTS
# ───────────────────────────────────────────────────────────────────────


def handle_graph_output(claim_out):
    """Shared post-processing after claim_flow.invoke."""
    st.session_state.claim_state = claim_out
    awaiting = claim_out.get("awaiting_user", False)
    if (not awaiting) and (
        claim_out.get("research_results") is not None
        or claim_out.get("primary_source")
    ):
        st.session_state.claim_done = True


# MAIN CHAT INPUT (has priority)
if main_prompt:

    # Immediately show the user message
    with st.chat_message("user"):
        st.write(main_prompt)

    # Save to UI history
    st.session_state.messages.append({"role": "user", "content": main_prompt})

    # Save into LangGraph state
    if st.session_state.claim_state["claim"] is None:
        st.session_state.claim_state["claim"] = main_prompt
        st.session_state.claim_state["messages"] = [
            HumanMessage(content=main_prompt)
        ]
    else:
        st.session_state.claim_state["messages"].append(
            HumanMessage(content=main_prompt)
        )

    # Run the graph node flow
    with st.spinner("🔎 Searching for sources and analyzing results..."):
        claim_out = asyncio.run(claim_flow.ainvoke(st.session_state.claim_state))

    handle_graph_output(claim_out)

    # Render new AI messages immediately
    flush_new_ai_messages()

with st.sidebar:
    st.subheader("Critical thinking chat")
    st.caption(
        "Socratic helper — keeps you doing the thinking. "
        "It will nudge with open questions instead of giving answers."
    )

    # Only show AI messages from messages_critical
    for msg in st.session_state.claim_state.get("messages_critical", []):
        if isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
