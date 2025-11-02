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

# init session state, ask the first question and add it to messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    claim_question="What claim do you want to investigate?"
    st.session_state.messages.append({"role": "assistent", "content": claim_question})
    with st.chat_message("assistent"):
        st.write(claim_question)

# Get user input
prompt = st.chat_input("")
if prompt:
    # show user message in UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# running graph state
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
            }
    st.session_state.claim_done = False
else:
    # add new user message to running claim state
    st.session_state.claim_state["messages"].append(HumanMessage(content=prompt))



# if "source_state" not in st.session_state:
#     st.session_state.source_state = None
# if "source_done" not in st.session_state:
#     st.session_state.source_done = False

# show chat history
# Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # prompt depends on where we are
# prompt_text = "What claim do you want to investigate?"
# if st.session_state.claim_done and not st.session_state.source_done:
#     prompt_text = "Add a source/URL or say where the claim appeared:"
# elif st.session_state.claim_done and st.session_state.source_done:
#     prompt_text = "You can continue or start a new claim."

# user_input = st.chat_input(prompt_text)

# ───────────────────────────────────────────────────────────────────
# PHASE 1: CLAIM FLOW
# ───────────────────────────────────────────────────────────────────
if not st.session_state.claim_done:
    
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
    if st.session_state.source_state is None:
        prev = st.session_state.claim_state
        st.session_state.source_state = {
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
    else:
        st.session_state.source_state["messages"].append(HumanMessage(content=user_input))

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
