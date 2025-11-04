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
from claim_nodes import router,checkable_fact,checkable_confirmation,retrieve_information,clarify_information,produce_summary,get_confirmation
from langgraph.graph import StateGraph, START, END
from state_scope import AgentStateClaim
from claim_nodes import claim_matching,match_or_continue,get_source,get_primary_source,locate_primary_source,select_primary_source,research_claim

claim = StateGraph(AgentStateClaim)

claim.add_node("checkable_fact", checkable_fact)
claim.add_node("checkable_confirmation", checkable_confirmation)
claim.add_node("retrieve_information", retrieve_information)
claim.add_node("clarify_information", clarify_information)
claim.add_node("produce_summary", produce_summary)
claim.add_node("get_confirmation", get_confirmation)
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
claim.add_edge("claim_matching", "match_or_continue")
#claim.add_edge("match_or_continue", "get_source")
#claim.add_edge("get_source", "get_primary_source")
claim.add_edge("locate_primary_source", "select_primary_source")
claim.add_edge("research_claim", END)

claim_flow = claim.compile()



if "render_cursor_claim" not in st.session_state:
    st.session_state.render_cursor_claim = 0

def show_new_ai_messages(final_messages):
    start = st.session_state.render_cursor_claim
    for m in final_messages[start:]:
        if isinstance(m, AIMessage):
            with st.chat_message("assistant"):
                st.write(m.content)
            st.session_state.messages.append({"role": "assistant", "content": m.content})
    st.session_state.render_cursor_claim = len(final_messages)

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

# Get user input
prompt = st.chat_input("")

if not prompt:
    # stop execution until user provides input
    st.stop()

# When we reach here, user has submitted input
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.write(prompt)

# initialize session state, if it does not exist
if "claim_state" not in st.session_state:
    st.session_state.claim_state = {
        "messages": [HumanMessage(content=prompt)],
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
else:
    # add new user message to existing claim state
    st.session_state.claim_state["messages"].append(HumanMessage(content=prompt))

# run claim graph 
claim_out = claim_flow.invoke(st.session_state.claim_state)

#output state    
st.session_state.claim_state = claim_out

# show only messages produced in THIS turn
final_messages = claim_out.get("messages", [])
show_new_ai_messages(final_messages)

# did the graph explicitly say “I’m waiting for the user”?
awaiting = claim_out.get("awaiting_user", False)

# mark claim done only if not waiting AND we have results 
if (not awaiting) and (
    claim_out.get("research_results") is not None
    or claim_out.get("primary_source")
):
    st.session_state.claim_done = True

