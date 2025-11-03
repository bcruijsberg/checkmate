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
from claim_nodes import checkable_fact,checkable_confirmation,retrieve_information,clarify_information,produce_summary,get_confirmation,await_user
from langgraph.graph import StateGraph, START, END
from state_scope import AgentStateClaim, AgentStateSource
from source_nodes import claim_matching,match_or_continue,get_source,get_primary_source,locate_primary_source,select_primary_source,research_claim

claim = StateGraph(AgentStateClaim)

claim.add_node("checkable_fact", checkable_fact)
claim.add_node("checkable_confirmation", checkable_confirmation)
claim.add_node("await_user", await_user)
claim.add_node("retrieve_information", retrieve_information)
claim.add_node("clarify_information", clarify_information)
claim.add_node("produce_summary", produce_summary)
claim.add_node("get_confirmation", get_confirmation)


# Entry point
claim.add_edge(START, "checkable_fact")
claim.add_edge("checkable_fact", "checkable_confirmation")
claim.add_edge("retrieve_information", "clarify_information")
claim.add_edge("produce_summary", "get_confirmation")
claim.add_edge("await_user", END)

claim_flow = claim.compile()

# the second graph
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

# helper to show only the messages that were added in this run
def show_new_ai_messages(prev_len, final_messages):
    new_msgs = final_messages[prev_len:]
    for m in new_msgs:
        if isinstance(m, AIMessage):
            st.session_state.messages.append({"role": "assistant", "content": m.content})
            with st.chat_message("assistant"):
                st.write(m.content)

# ───────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

st.set_page_config(page_title="CheckMate", page_icon="✅")
st.title("🕵️ CheckMate – Claim checker")

# First question
claim_question="What claim do you want to investigate?"

# initialize session messages, ask the first question and add it to messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": claim_question})
    with st.chat_message("assistant"):
        st.write(claim_question)

# Initialize booleans to determine the FLOW
if "claim_done" not in st.session_state:
    st.session_state.claim_done = False
if "source_done" not in st.session_state:
    st.session_state.source_done = False

# Get user input
prompt = st.chat_input("")

if not prompt:
    # stop execution until user provides input
    st.stop()

# When we reach here, user has submitted input
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.write(prompt)

# ───────────────────────────────────────────────────────────────────
# PHASE 1: CLAIM FLOW
# ───────────────────────────────────────────────────────────────────
if not st.session_state.claim_done:

    # initialize session state, if it does not exist
    if "claim_state" not in st.session_state:
        st.session_state.claim_state = {
                "messages": [AIMessage(content=claim_question),HumanMessage(content=prompt)],
                "claim": prompt,
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
            }
    else:
        # add new user message to existing claim state
        st.session_state.claim_state["messages"].append(HumanMessage(content=prompt))

    # remember how many messages we had before this run, to determine new messages
    prev_len = len(st.session_state.claim_state["messages"])

    # run claim graph for THIS turn
    claim_out = claim_flow.invoke(st.session_state.claim_state)
    st.session_state.claim_state = claim_out

    # show only messages produced in THIS turn
    final_messages = claim_out.get("messages", [])
    show_new_ai_messages(prev_len, final_messages)

    # did the graph explicitly say “I’m waiting for the user”?
    awaiting = claim_out.get("awaiting_user", False)

    # if it's NOT waiting, and we now have a summary → claim phase is done
    if (not awaiting) and claim_out.get("summary"):
        st.session_state.claim_done = True

# ───────────────────────────────────────────────────────────────────
# PHASE 2: SOURCE FLOW
# ───────────────────────────────────────────────────────────────────
elif not st.session_state.source_done:

    #initialize the source_state
    if "source_state" not in st.session_state:
        prev = st.session_state.claim_state
        st.session_state.source_state = {
            "messages": prev.get("messages", []) + [HumanMessage(content=prompt)],
            "claim": prev.get("claim", ""),
            "checkable": True,
            "subject": prev.get("subject", ""),
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
            "explanation": None,
            "awaiting_user": False,

        }
    else:
        st.session_state.source_state["messages"].append(HumanMessage(content=prompt))

    prev_len = len(st.session_state.source_state["messages"])

    source_out = source_flow.invoke(st.session_state.source_state)
    st.session_state.source_state = source_out

    final_messages = source_out.get("messages", [])
    show_new_ai_messages(prev_len, final_messages)

    awaiting = source_out.get("awaiting_user", False)

    # mark source done only if not waiting AND we have results / source
    if (not awaiting) and (
        source_out.get("research_results") is not None
        or source_out.get("primary_source")
    ):
        st.session_state.source_done = True
