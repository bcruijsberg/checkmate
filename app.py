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
    checkable_fact,
    checkable_confirmation,
    identify_url,
    retrieve_information,
    clarify_information,
    produce_summary,
    get_confirmation,
    get_rag_queries,
    confirm_rag_queries,
    route_rag_confirm,
    rag_retrieve_worker,
    reduce_claim_matching,
    match_or_continue,
    primary_source,
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
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

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

# Save the compiled graph in session state to avoid recompiling on every interaction
if "compiled_graph" not in st.session_state:
    workflow = StateGraph(AgentStateClaim)

    #getting all info about the claim nodes
    workflow.add_node("checkable_fact", checkable_fact)
    workflow.add_node("checkable_confirmation", checkable_confirmation)
    workflow.add_node("identify_url", identify_url)
    workflow.add_node("retrieve_information", retrieve_information)
    workflow.add_node("clarify_information", clarify_information)
    workflow.add_node("produce_summary", produce_summary)
    workflow.add_node("get_confirmation", get_confirmation)
    workflow.add_node("critical_question", critical_question)

    #Claim matching nodes
    workflow.add_node("get_rag_queries", get_rag_queries)
    workflow.add_node("confirm_rag_queries", confirm_rag_queries)
    workflow.add_node("rag_retrieve_worker", rag_retrieve_worker)
    workflow.add_node("reduce_claim_matching", reduce_claim_matching)
    workflow.add_node("match_or_continue", match_or_continue)

    # Source finding nodes and search query nodes
    workflow.add_node("primary_source", primary_source)
    workflow.add_node("get_source_queries", get_source_queries)
    workflow.add_node("confirm_search_queries", confirm_search_queries)
    workflow.add_node("reset_search_state", reset_search_state)
    workflow.add_node("find_sources_worker", find_sources_worker)
    workflow.add_node("reduce_sources", reduce_sources)
    workflow.add_node("select_primary_source", select_primary_source)
    workflow.add_node("get_search_queries", get_search_queries)
    workflow.add_node("iterate_search",iterate_search)

    # Entry point
    workflow.add_edge(START, "checkable_fact")
    workflow.add_edge("checkable_fact", "critical_question")
    workflow.add_edge("checkable_fact", "checkable_confirmation")
    workflow.add_edge("identify_url", "retrieve_information")
    workflow.add_edge("retrieve_information", "clarify_information")
    workflow.add_edge("produce_summary", "get_confirmation")

    # Connecting claim matching nodes
    workflow.add_edge("get_rag_queries", "confirm_rag_queries")
    workflow.add_edge("get_rag_queries", "critical_question")
    workflow.add_conditional_edges("confirm_rag_queries", route_rag_confirm)
    workflow.add_edge("rag_retrieve_worker", "reduce_claim_matching")

    # Connecting source finding and search query nodes
    workflow.add_edge("get_source_queries", "critical_question")
    workflow.add_edge("get_source_queries", "confirm_search_queries")
    workflow.add_edge("get_search_queries", "critical_question")
    workflow.add_edge("confirm_search_queries", "reset_search_state")
    workflow.add_conditional_edges("reset_search_state", route_after_confirm)
    workflow.add_edge("find_sources_worker", "reduce_sources")
    workflow.add_edge("select_primary_source", "get_search_queries")
    workflow.add_edge("get_search_queries", "confirm_search_queries")

    # Ensure the critical_question branch terminates
    workflow.add_edge("critical_question", END)

    # compile the graph
    memory = MemorySaver()
    st.session_state.compiled_graph = workflow.compile(checkpointer=memory)

# Use the persistent version
claim_flow = st.session_state.compiled_graph

# ───────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────

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

def handle_graph_output(claim_out):
    """Update local session state with the new graph state."""
    st.session_state.claim_state = claim_out
    
    # Check if teh graph was pause by an interrupt
    snapshot = claim_flow.get_state(st.session_state.graph_config)
    if not snapshot.next:
        st.session_state.claim_done = True


# ───────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────────────────────────────

st.title("🕵️ CheckMate – Claim checker")

claim_question = "What claim do you want to investigate?"

# The key to the memory, so it remembers to what state it should return
if "graph_config" not in st.session_state:
    st.session_state.graph_config = {"configurable": {"thread_id": "streamlit_session"}}

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
        "queries_confirmed": False,
        "alerts": [],
        "awaiting_user": False,
        "rag_trace": [],
        "search_queries": [],
        "tavily_context": [],
        "primary_source": False,
        "match": False
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
# HANDLE INPUTS and OUTPUTS
# ───────────────────────────────────────────────────────────────────────

# MAIN CHAT INPUT (has priority)
if main_prompt:
    with st.chat_message("user"):
        st.write(main_prompt)
    st.session_state.messages.append({"role": "user", "content": main_prompt})

    # Get the latest state from the persistent memory
    snapshot = claim_flow.get_state(st.session_state.graph_config)
    
    with st.spinner("🔎 Analyzing..."):
        # Check if the graph is waiting at an interrupt
        if snapshot.next:
            # RESUME
            claim_out = asyncio.run(claim_flow.ainvoke(
                Command(resume=main_prompt), 
                config=st.session_state.graph_config
            ))
        else:
            # START FRESH
            # Only send the state the first time
            initial_state = {
                "messages": [HumanMessage(content=main_prompt)],
                "claim": main_prompt
            }
            claim_out = asyncio.run(claim_flow.ainvoke(
                initial_state, 
                config=st.session_state.graph_config
            ))

    # 3. Process output
    handle_graph_output(claim_out)
    flush_new_ai_messages()

    # Check if the graph is currently paused and display the question it's waiting on
    snapshot = claim_flow.get_state(st.session_state.graph_config)
    if snapshot.tasks:
        for task in snapshot.tasks:
            if task.interrupts:
                # This grabs the question text from your interrupt()
                interrupt_msg = task.interrupts[0].value 
                
                # Check UI history to avoid double-rendering if the user refreshes
                if not st.session_state.messages or st.session_state.messages[-1]["content"] != interrupt_msg:
                    st.session_state.messages.append({"role": "assistant", "content": interrupt_msg})
                    with st.chat_message("assistant"):
                        st.write(interrupt_msg)

# ───────────────────────────────────────────────────────────────────────
# CRITICAL THINKING SIDEBAR CHAT
# ───────────────────────────────────────────────────────────────────────
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
