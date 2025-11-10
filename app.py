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

# ───────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

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
        "chat_mode":"fact-check"
    }
    st.session_state.graph_cursor = 0

# Optional done flag
if "claim_done" not in st.session_state:
    st.session_state.claim_done = False

print(st.session_state.claim_state["chat_mode"])
# check if we are in fact-check mode or critical mode 
if st.session_state.claim_state["chat_mode"]=="fact-check":

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
        st.session_state.claim_state["claim"]=prompt
        st.session_state.claim_state["messages"]=[HumanMessage(content=prompt)]
    else:
        #append the user message
        st.session_state.claim_state["messages"].append(HumanMessage(content=prompt))
        
    # Run graph
    claim_out = claim_flow.invoke(st.session_state.claim_state)
    st.session_state.claim_state = claim_out

    # Append any new AI messages to history and render them now
    final_messages = claim_out.get("messages", [])
    start_idx = st.session_state.graph_cursor
    for m in final_messages[start_idx:]:
        if isinstance(m, AIMessage):
            st.session_state.messages.append({"role": "assistant", "content": m.content})
            with st.chat_message("assistant"):
                st.write(m.content)

    # Advance the cursor to the end of the graph message list
    st.session_state.graph_cursor = len(final_messages)

    # ── Bookkeeping flags (optional) ───────────────────────────────────────
    awaiting = claim_out.get("awaiting_user", False)
    if (not awaiting) and (
        claim_out.get("research_results") is not None or claim_out.get("primary_source")
    ):
        st.session_state.claim_done = True
else:
    # ───────────────────────────────────────────────────────────────────────
    # Critical mode
    # ───────────────────────────────────────────────────────────────────────
    @st.dialog("Critical Thinking Chat", width="large")
    def critical_chat_modal():
        st.caption(
            "Socratic helper — keeps you doing the thinking. "
            "It will nudge with open questions instead of giving answers."
        )

        # chat history for the modal
        if "messages_critical" not in st.session_state:
            st.session_state.messages_critical = [
                {"role": "assistant", "content": "What's the core claim you’re examining?"}
            ]

        # render history
        for m in st.session_state.messages_critical:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        # chat input inside the modal
        user_msg = st.chat_input("Type your reply...")
        if user_msg:
            st.session_state.messages_critical.append({"role": "user", "content": user_msg})

            # 👉 replace this with your model call
            socratic_nudge = (
                "What assumption are you making here, and how could you test it "
                "without relying on a single source?"
            )

            with st.chat_message("assistant"):
                st.write(socratic_nudge)
            st.session_state.messages_critical.append({"role": "assistant", "content": socratic_nudge})

    # ---- Auto-open the modal on first load ----
        critical_chat_modal()  # pops up immediately
