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
    claim_matching,
    match_or_continue,
    get_source,
    get_primary_source,
    locate_primary_source,
    select_primary_source,
    research_claim,
    critical_question,
)
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

claim.add_node("checkable_fact", checkable_fact)
claim.add_node("checkable_confirmation", checkable_confirmation)
claim.add_node("retrieve_information", retrieve_information)
claim.add_node("clarify_information", clarify_information)
claim.add_node("produce_summary", produce_summary)
claim.add_node("get_confirmation", get_confirmation)
claim.add_node("critical_question", critical_question)
claim.add_node("claim_matching", claim_matching)
claim.add_node("match_or_continue", match_or_continue)
claim.add_node("get_source", get_source)
claim.add_node("get_primary_source", get_primary_source)
claim.add_node("locate_primary_source", locate_primary_source)
claim.add_node("select_primary_source", select_primary_source)
claim.add_node("research_claim", research_claim)
claim.add_node("router", router)

# Entry point
claim.add_edge(START, "router")
claim.add_edge("checkable_fact", "checkable_confirmation")
claim.add_edge("retrieve_information", "clarify_information")
claim.add_edge("produce_summary", "get_confirmation")
claim.add_edge("critical_question", "claim_matching")
claim.add_edge("claim_matching", "match_or_continue")
claim.add_edge("locate_primary_source", "select_primary_source")
claim.add_edge("research_claim", END)

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
        "subject": None,
        "quantitative": None,
        "precision": None,
        "based_on": None,
        "confirmed": False,
        "question": None,
        "alerts": [],
        "summary": None,
        "awaiting_user": False,
        "explanation": None,
        "next_node": None,
        "search_queries": [],
        "tavily_context": None,
        "research_focus": None,
        "research_results": [],
        "claim_url": None,
        "claim_source": None,
        "primary_source": False,
        "match": False,
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
    claim_out = claim_flow.invoke(st.session_state.claim_state)
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
