import sys
import os
import streamlit as st

#os.environ["LANGSMITH_TRACING"] = "true"
#os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
#os.environ["LANGSMITH_PROJECT"]="pr-left-technician-100"

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
    critical_response
)
from langgraph.graph import StateGraph, START, END
from state_scope import AgentStateClaim
from langchain_core.messages import HumanMessage,AIMessage
import streamlit as st

claim = StateGraph(AgentStateClaim)

claim.add_node("checkable_fact", checkable_fact)
claim.add_node("checkable_confirmation", checkable_confirmation)
claim.add_node("retrieve_information", retrieve_information)
claim.add_node("clarify_information", clarify_information)
claim.add_node("produce_summary", produce_summary)
claim.add_node("get_confirmation", get_confirmation)
claim.add_node("critical_question", critical_question)
claim.add_node("critical_response", critical_response)
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
#claim.add_edge("get_confirmation", "critical_question")
claim.add_edge("claim_matching", "match_or_continue")
claim.add_edge("locate_primary_source", "select_primary_source")
claim.add_edge("research_claim", END)

claim_flow = claim.compile()

def flush_new_ai_messages():
    """Render any new AI messages produced by the graph since last UI cursor."""

    final_messages = st.session_state.claim_state.get("messages", [])
    start_idx = st.session_state.graph_cursor

    for m in final_messages[start_idx:]:
        if isinstance(m, AIMessage):
            st.session_state.messages.append({"role": "assistant", "content": m.content})
            with st.chat_message("assistant"):
                st.write(m.content)

    # Update cursor so we don't re-render old messages
    st.session_state.graph_cursor = len(final_messages)

# Modal chat window for critical thinking
@st.dialog("Critical Thinking Chat", width="large")
def critical_chat_modal():
    st.caption("Socratic helper — keeps you doing the thinking. "
               "It will nudge with open questions instead of giving answers.")

    # Seed UI history only if empty
    if not st.session_state.get("messages_critical"):
        st.session_state.messages_critical = [
            {"role": "assistant", "content": st.session_state.claim_state.get("question", "")}
        ]

    # Render history
    for m in st.session_state.messages_critical:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # Chat input LAST so it stays at the bottom
    user_msg = st.chat_input("Type your reply...")
    if user_msg:
        # Append to UI history
        st.session_state.messages_critical.append({"role": "user", "content": user_msg})

        # Append the latest user reply
        st.session_state.claim_state["messages_critical"].append(HumanMessage(content=user_msg))

        out = claim_flow.invoke(st.session_state.claim_state)
        st.session_state.claim_state = out

        st.rerun()  # re-render so the input returns to the bottom

# ───────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CheckMate", page_icon="✅")
st.title("🕵️ CheckMate – Claim checker")

claim_question = "What claim do you want to investigate?"

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": claim_question}]

# LangGraph message cursor to avoid duplicating AI turns
if "graph_cursor" not in st.session_state:
    st.session_state.graph_cursor = 0

# Claim state (graph state)
if "claim_state" not in st.session_state:
    st.session_state.claim_state = {
        "messages": [],
        "messages_critical":[],
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
        "critical_mode":False
    }
    st.session_state.graph_cursor = 0

# Optional done flag
if "claim_done" not in st.session_state:
    st.session_state.claim_done = False

# As long as we're in critical mode, only show the modal
if st.session_state.claim_state.get("critical_mode", False):
    critical_chat_modal()
    st.stop()

if st.session_state.get("skip_input_once"):
    flush_new_ai_messages()
    st.session_state["skip_input_once"] = False
    st.stop()
# ───────────────────────────────────────────────────────────────────────
# Fact Check mode
# ───────────────────────────────────────────────────────────────────────

# Render full chat history every run
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Get user input
prompt = st.chat_input("")

if not prompt:
    # No new input; just show what we rendered above and exit this run
    st.stop()

# Show and store the new user message
with st.chat_message("user"):
    st.write(prompt)
st.session_state.messages.append({"role": "user", "content": prompt})

# At the first run, the user message is the claim
if st.session_state.claim_state["claim"] is None:
    st.session_state.claim_state["claim"] = prompt
    st.session_state.claim_state["messages"] = [HumanMessage(content=prompt)]
else:
    # append the user message
    st.session_state.claim_state["messages"].append(HumanMessage(content=prompt))

# Run graph
claim_out = claim_flow.invoke(st.session_state.claim_state)
st.session_state.claim_state = claim_out

# Append any new AI messages to history and render them now
flush_new_ai_messages()

# 👇 NEW: if we just switched into critical mode, open the modal *now*
if claim_out.get("critical_mode", False):
    critical_chat_modal()
    st.stop()

# ── Bookkeeping flags (optional) ───────────────────────────────────────
awaiting = claim_out.get("awaiting_user", False)
if (not awaiting) and (
    claim_out.get("research_results") is not None or claim_out.get("primary_source")
):
    st.session_state.claim_done = True
