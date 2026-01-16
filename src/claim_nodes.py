
""" All the nodes """

from prompts import (
    checkable_check_prompt,
    confirmation_prompt,
    retrieve_info_prompt,
    get_summary_prompt,
    rag_queries_prompt,
    match_check_prompt,
    structure_claim_prompt,
    source_prompt,
    source_queries_prompt,
    confirm_queries_prompt,
    select_primary_source_prompt,
    search_queries_prompt,
    eval_search_prompt,
    iterate_search_prompt,
    get_socratic_question,
)
from state_scope import (
    AgentStateClaim, 
    CheckableOutput, 
    ConfirmationOutput,
    MoreInfoOutput, 
    SummaryOutput, 
    ConfirmationMatch,
    ClaimMatchingOutput,
    SourceOutput,
    GetSearchQueries,
    SearchSynthesis,
)
from langgraph.types import Overwrite, interrupt
from langchain_core.messages import HumanMessage,AIMessage, get_buffer_string
from typing import Dict, Any, Literal
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send
from utils import get_new_user_reply,_domain
from tooling import llm, llm_tuned, tools_dict, tavily_client

# Maximum number of messages to send to the prompt
MAX_HISTORY_MESSAGES = 6

# ───────────────────────────────────────────────────────────────────────
# CRITICAL QUESTION NODE
# ───────────────────────────────────────────────────────────────────────

async def critical_question(state: AgentStateClaim):

    """ Ask a socratic question to make the user think about the consequences of a fact checking a claim """

    # retrieve alerts and format to string for the prompt
    alerts=state.get("alerts", [])
    alerts_str= "\n".join(f"- {a}" for a in alerts)

    # retrieve conversation history fact-check messages and critical messages
    conversation_history_critical = list(state.get("messages_critical", []))

    # Add the last messages into a string for the prompt
    messages_critical_str = get_buffer_string(conversation_history_critical[-MAX_HISTORY_MESSAGES:] )

    # Create a prompt
    prompt  =  get_socratic_question.format(
        alerts=alerts_str,
        claim=state.get("claim"),
        summary=state.get("summary"),
        messages_critical=messages_critical_str 
    )
    try:
        #invoke the LLM and store the output
        result = await llm_tuned.ainvoke([HumanMessage(content=prompt)])
        content = result.content
    except Exception as e:
        # Graceful fallback question
        content = (
            "I’m having trouble formulating a good question right now. "
            "Let’s pause and think: what part of this claim would be most important to verify first?"
        )

        # Optional: log for debugging
        print("critical_question error:", repr(e))

    ai_chat_msg = AIMessage(content=content)
  
    return {
        "critical_question": content,
        "messages_critical": [ai_chat_msg],
    }

# ───────────────────────────────────────────────────────────────────────
# CHECKABLE_FACT NODE
# ───────────────────────────────────────────────────────────────────────

async def checkable_fact(state: AgentStateClaim):

    """ Check if a claim is potentially checkable. """

    #Retrieve conversation history
    conversation_history = list(state.get("messages", []))

    # Add the last messages into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:] 
    messages_str = get_buffer_string(recent_messages)

    # Use structured output
    structured_llm = llm.with_structured_output(CheckableOutput, method="json_mode")

    # Create a prompt
    prompt = checkable_check_prompt.format(
        claim=state.get("claim", ""),
        additional_context=state.get("additional_context", ""),
        messages=messages_str,
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

        # checkable is a boolean in State
        is_checkable = result.checkable == "POTENTIALLY CHECKABLE"

        # human-readable assistant message for the chat
        explanation_text = (
            f"### Checkability analysis\n"
            f"- Checkable: `{result.checkable}`\n"
            f"- Reason: {result.explanation}\n"
        )

        ai_chat_msg = AIMessage(content=explanation_text)

        # Goto next node and update State
        return {
            "question": result.question,
            "checkable": is_checkable,
            "explanation": result.explanation,
            "messages": [ai_chat_msg],
        }
    except Exception as e:
        # fallback: ask a clarifying question
        print("checkable_fact error:", repr(e))

        fallback_question = (
            "I hit a technical issue while analyzing checkability. "
            "Quick question: what exactly should be verified (a number, a date, a quote, or an event)?"
        )

        ai_chat_msg = AIMessage(
            content=(
                "### Checkability analysis\n"
                "- Checkable: `UNKNOWN`\n"
                "- Reason: I couldn’t run the analysis due to a temporary error.\n\n"
                f"{fallback_question}"
            )
        )

        return {
            "question": fallback_question,
            "checkable": False,          # safest default
            "explanation": "LLM error during checkability analysis.",
            "messages": [ai_chat_msg],
        }

# ───────────────────────────────────────────────────────────────────────
# CHECKABLE_CONFIRMATION NODE
# ───────────────────────────────────────────────────────────────────────

async def checkable_confirmation(state: AgentStateClaim) -> Command[Literal["identify_url","__end__","checkable_fact"]]:
    
    """ Get confirmation from user on the gathered information. """

    # Get an answer from the user
    user_answer = interrupt(state.get("question", "Is the information correct?"))

    # Use structured output
    structured_llm = llm.with_structured_output(ConfirmationOutput, method="json_mode")

    # Create a prompt
    prompt = confirmation_prompt.format(
        user_answer=user_answer,
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        confirmed = bool(result.confirmed)
    
    except Exception as e:
        # Fallback: assume NOT confirmed and loop back
        print("checkable_confirmation error:", repr(e))
        confirmed = False

        # Message to the user
        ai_chat_msg = AIMessage(
            content="I had a technical hiccup reading that confirmation. Let’s try again."
        )

        # Goto next node and update State
        return Command(
            goto="checkable_fact",
            update={
                "messages": [ai_chat_msg],
                "additional_context": user_answer,
            },
        )

    # human-readable assistant message for the chat
    if confirmed:
        confirm_text = "We'll continue with this claim."
    else:
        confirm_text = "Okay let's revise the claim or stop here."

    ai_chat_msg = AIMessage(content=confirm_text)

    # Goto next node and update State
    if confirmed:
        if state.get("checkable"):
            return Command(
                    goto="identify_url", 
                    update={
                        "messages": [ai_chat_msg],
                        "additional_context": None,
                    }
            )   
        else: 
            # user confirmed but claim is not checkable → end
            end_msg = AIMessage(content="This claim appears to be uncheckable, so we'll stop the process here.")
            return Command(
                    goto="__end__", 
                    update={
                        "messages": [ai_chat_msg, end_msg]
                    }
            )   
    else:
        return Command(
                goto="checkable_fact", 
                update={
                    "messages": [ai_chat_msg],
                    "additional_context": user_answer,
                }
        )

# ───────────────────────────────────────────────────────────────────────
# GET URL NODE
# ───────────────────────────────────────────────────────────────────────
import re

URL_RE = re.compile(r"(https?://[^\s)>\]]+|www\.[^\s)>\]]+)", re.IGNORECASE)

def identify_url(state: AgentStateClaim):
    """Ask once, extract everything, and route the research path."""
    
    # Get an answer from the user
    user_answer = interrupt("Do you have a URL where you found this claim? ")

    m = URL_RE.search(user_answer or "")
    if not m:
        return {"claim_url": None}

    url = m.group(0)
    if url.lower().startswith("www."):
        url = "https://" + url

    return {
        "claim_url": url
    }

# ───────────────────────────────────────────────────────────────────────
# RETRIEVE_INFORMATION NODE
# ───────────────────────────────────────────────────────────────────────
async def retrieve_information(state: AgentStateClaim):
    
    """ Analyse the content of the source URL and update the claim summary and source verification. """

    #retrieve the claim URL and scrape the content
    claim_url = state.get("claim_url", "")
    page_content = "No valid URL provided to extract content."
    
    if claim_url and claim_url.startswith("http"):
        try:
            extract_response = tavily_client.extract(urls=[claim_url])
            page_content = extract_response["results"][0].get("raw_content", "")
        except Exception as e:
            print("retrieve_information: tavily extract failed:", repr(e))
            page_content = "Failed to extract content from the provided URL."
    
    #Retrieve conversation history
    conversation_history = list(state.get("messages", []))

    # Add the last messages into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # Use structured output
    structured_llm = llm.with_structured_output(MoreInfoOutput, method="json_mode")

    # Create a prompt
    prompt = retrieve_info_prompt.format(
        page_content=page_content,
        claim=state.get("claim", ""),
        additional_context=state.get("additional_context", ""),
        messages=messages_str,
    )

    try:
        # Invoke the LLM and store the output
        result = await structured_llm.ainvoke(prompt)

    except Exception as e:
        # fallback: minimal, safe state update
        print("retrieve_information: LLM failed:", repr(e))

        ai_chat_msg = AIMessage(
            content=(
                "I had trouble analyzing the source content due to a technical issue. "
                "We can continue, but some details may be missing."
            )
        )

        return {
            "alerts": ["Source analysis failed"],
            "messages": [ai_chat_msg],
            "question": "Would you like to try once more or continue anyway?",
        }

    # human-readable assistant message for the chat
    details_text = (
        "**Here’s what I extracted from your claim:**\n"
        f"- Claim source: {result.claim_source or 'not clearly specified'}\n"
        f"- Source description: {result.details_claim.source_description or 'not clearly specified'}\n"
        f"- Subject: {result.details_claim.subject or 'not clearly specified'}\n"
        f"- Data type: {result.details_claim.data_type}\n"
        f"- Precision: {result.details_claim.precision}\n"
        f"- Based on: {result.details_claim.based_on}\n"
        f"- Geography: {result.details_claim.geography or 'not clearly specified'}\n"
        f"- Time Period: {result.details_claim.time_period or 'not clearly specified'}\n"
    )

    if result.alerts:
        details_text += "\n**Missing / to verify:**\n" + "\n".join(f"- {a}" for a in result.alerts)

    ai_chat_msg = AIMessage(content=details_text)

    # Goto next node and update State
    return {
            "claim_source": result.claim_source,
            "primary_source": result.primary_source,
            "alerts": result.alerts or [],
            "messages": [ai_chat_msg],
            "question": result.question,
            "details_claim": result.details_claim,
    }

# ───────────────────────────────────────────────────────────────────────
# CLARIFY_INFORMATION NODE
# ───────────────────────────────────────────────────────────────────────

async def clarify_information(state: AgentStateClaim) -> Command[Literal["produce_summary", "retrieve_information"]]:

    """ Get confirmation from user on the gathered information. """

    # Get an answer from the user
    user_answer = interrupt(state.get("question", "Is the information correct?"))

    # Use structured output
    structured_llm = llm.with_structured_output(ConfirmationOutput, method="json_mode")

    # Create a prompt
    prompt  =  confirmation_prompt.format(
        user_answer=user_answer,
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        confirmed = bool(result.confirmed)

    except Exception as e:
        # fallback: treat as NOT confirmed
        print("clarify_information error:", repr(e))
        confirmed = False

        ai_chat_msg = AIMessage(
            content="I had a technical hiccup processing that. Let’s add a bit more detail and try again."
        )

        return Command(
            goto="retrieve_information",
            update={
                "messages": [ai_chat_msg],
                "additional_context": user_answer,
            },
        )       

    # human-readable assistant message for the chat
    if confirmed:
        confirm_text = "Thanks, I’ll use this information to draft the summary."
    else:
        confirm_text = "Let’s collect a bit more information."

    ai_chat_msg = AIMessage(content=confirm_text)

    # Goto next node and update State
    if confirmed:
        return Command(
                goto="produce_summary", 
                update={
                    "messages": [ai_chat_msg],
                    "additional_context": None,
                    "confirmed": False,
                }
        )       
    else:
        return Command(
                goto="retrieve_information", 
                update={
                    "messages": [ai_chat_msg],
                    "additional_context": user_answer,
                }
        )

# ───────────────────────────────────────────────────────────────────────
# PRODUCE SUMMARY NODE
# ───────────────────────────────────────────────────────────────────────

async def produce_summary(state: AgentStateClaim):

    """ Get a summary on the gathered information. """

    #retrieve the claim URL and scrape the content
    claim_url = state.get("claim_url", "")
    page_content = "No valid URL provided to extract content."

    if claim_url and claim_url.startswith("http"):
        try:
            extract_response = tavily_client.extract(urls=[claim_url])
            page_content = extract_response["results"][0].get("raw_content", "")
        except Exception as e:
            print("produce_summary: tavily extract failed:", repr(e))
            page_content = "Failed to extract content from the provided URL."

    # retrieve all details
    details = state.get("details_claim")

    # retrieve alerts and format to string for the prompt
    alerts=state.get("alerts", [])
    alerts_str= "\n".join(f"- {a}" for a in alerts)

    # retrieve conversation history
    conversation_history = list(state.get("messages", []))

    # Add the last messages into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # Use structured output
    structured_llm = llm.with_structured_output(SummaryOutput, method="json_mode")

    # Create a prompt
    prompt = get_summary_prompt.format(
        claim=state.get("claim", ""),
        claim_source=state.get("claim_source", ""),
        source_description=details.source_description if details else "",
        subject=details.subject if details else "",
        quantitative=details.data_type if details else "",
        precision=details.precision if details else "",
        based_on=details.based_on if details else "",
        geography=details.geography if details else "",
        time_period=details.time_period if details else "",
        additional_context=state.get("additional_context", ""),
        page_content=page_content,
        alerts=alerts_str,
        messages=messages_str,
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
    
    except Exception as e:
        # fallback: keep flow alive with a minimal summary
        print("produce_summary: LLM failed:", repr(e))

        fallback_text = (
            "**Summary**\n\n"
            "I hit a technical issue while drafting the summary. "
            "To keep going, tell me: what is the *main point* of the claim in one sentence, "
        )

        return {
            "summary": "",
            "question": "What’s the main point of the claim, and what detail should we verify first?",
            "messages": [AIMessage(content=fallback_text)],
            "alerts": (alerts or []) + ["Summary generation failed"],
        }
    
    # human-readable assistant message for the chat
    chat_lines = [
        f"{result.summary}\n\n"
    ]

    if result.alerts:
        chat_lines.append("\n\n **Potential Issues Identified:**")
        for alert in result.alerts:
            chat_lines.append(f"- {alert}")

    # Print the lines for chat
    chat_text = "\n".join(chat_lines)

    # Goto next node and update State
    return {
        "summary": result.summary,
        "question": result.question,
        "messages": [AIMessage(content=chat_text)],
        "alerts": result.alerts or [],
     }      

# ───────────────────────────────────────────────────────────────────────
# GET_CONFIRMATION NODE
# ───────────────────────────────────────────────────────────────────────
   
async def get_confirmation(state: AgentStateClaim) -> Command[Literal["produce_summary", "get_rag_queries"]]:

    """ Get confirmation from user on the gathered information."""

    # Get an answer from the user
    user_answer = interrupt(state.get("question", "Is the information correct?"))

    # Use structured output
    structured_llm = llm.with_structured_output(ConfirmationOutput, method="json_mode")

    # Create a prompt
    prompt  =  confirmation_prompt.format(
        user_answer=user_answer,
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        confirmed = bool(result.confirmed)
    
    except Exception as e:
        # fallback: treat as NOT confirmed so user can revise summary
        print("get_confirmation error:", repr(e))
        confirmed = False

        ai_chat_msg = AIMessage(
            content="I had a technical hiccup reading that. Let’s revisit the summary and adjust it if needed."
        )

        return Command(
            goto="produce_summary",
            update={
                "messages": [ai_chat_msg],
                "additional_context": user_answer,
            },
        )
    
    # human-readable assistant message for the chat
    if confirmed:
        confirm_text = "Let's check if this claim has already been researched"
    else:
        confirm_text = "Let's revisit the summary and adjust it if needed."

    ai_chat_msg = AIMessage(content=confirm_text)

    # Goto next node and update State
    if confirmed:
        return Command(
                goto="get_rag_queries", 
                update={
                    "messages": [ai_chat_msg],
                    "additional_context": None,
                    "confirmed": False,
                }
        )       
    else:
        return Command(
                goto="produce_summary", 
                update={
                    "messages": [ai_chat_msg],
                    "additional_context": user_answer,
                }
        )

# ───────────────────────────────────────────────────────────────────────
# GENERATE QUERIES FOR CLAIM MATCHING NODE
# ───────────────────────────────────────────────────────────────────────
async def get_rag_queries(state: AgentStateClaim):

    """ Generate queries to locate the primary source of the claim. """

    # retrieve conversation history
    conversation_history = list(state.get("messages", []))

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # Use structured output
    structured_llm = llm.with_structured_output(GetSearchQueries, method="json_mode")

    # Create a prompt
    prompt  = rag_queries_prompt.format(
        summary=state.get("summary", ""),
        messages=messages_str,
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
    except Exception as e:
        # fallback: try once more
        print("get_rag_queries failed:", repr(e))

        ai_chat_msg = AIMessage(
            content=(
                "I ran into a technical issue while generating search queries. "
                "Please try again in a moment."
            )
        )

        return {
            "search_queries": [],
            "messages": [ai_chat_msg],
            "question": state.get("question","Would you like to try again?"),
        }
    
    # create a human-readable assistant message for the chat
    queries_text = "\n".join(
        f"- **{q.query}**\n\n  *{q.reasoning}*" 
        for q in result.queries if q.query
    )
    
    chat_text = (
        "I will perform a search with these queries:\n\n"
        f"{queries_text}"
    )

    ai_chat_msg = AIMessage(content=chat_text)

    # Extract onlythe query strings for storage in State
    query_strings = [q.query for q in result.queries if q.query]

    # Goto next node and update State  
    return {
        "search_queries": query_strings,
        "messages":  [ai_chat_msg],
        "question": "would you like to add or change something? Or do you want to continue?"
    }  

# ───────────────────────────────────────────────────────────────────────
# CONFIRM CLAIM MATCHING QUERIES NODE
# ───────────────────────────────────────────────────────────────────────
async def confirm_rag_queries(state: AgentStateClaim) -> dict:
    
    """Update state based on user confirmation. Routing is handled by a router."""
    
    # Get an answer from the user
    user_answer = interrupt(state.get("question", "Is the information correct?"))

    # retrieve search_queries and format to string for the prompt
    search_queries = state.get("search_queries", [])
    search_queries_str = "\n".join(f"- {q}" for q in search_queries)

    # Use structured output
    structured_llm = llm.with_structured_output(GetSearchQueries, method="json_mode")

    # Create a prompt
    prompt  =  confirm_queries_prompt.format(
        search_queries=search_queries_str,
        user_answer=user_answer,
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
    
    except Exception as e:
        # fallback: treat as NOT confirmed
        print("confirm_rag_queries error:", repr(e))

        return {
            "queries_confirmed": False,
            "search_queries": search_queries, 
            "question": state.get(
                "question",
                "Is the information correct?"
            ),
        }
    
    # Format display for the 'else' block (with reasoning)
    updated_display = "\n".join(
        f"- **{q.query}**\n\n  *{q.reasoning}*" 
        for q in result.queries if q.query
    )

    # Extract ONLY the strings for the update
    query_strings = [q.query for q in result.queries if q.query]

    # Create the base update
    update = {
        "queries_confirmed": result.confirmed,
        "search_queries": query_strings,
    }

    # Add messages only if confirmed, els reask the question
    if result.confirmed:
        confirm_text = "From the retrieved information, these existing claims might be relevant to what you're investigating.\n"
        update["messages"] = [AIMessage(content=confirm_text)]
    else:
        update["question"] = (
            "Is there anything else you would like to change about the search queries?\n"
            f"{updated_display}"
        )

    return update
# ───────────────────────────────────────────────────────────────────────
# ROUTER CLAIM MATCHING NODE
# ───────────────────────────────────────────────────────────────────────

def route_rag_confirm(state: AgentStateClaim):
    """ Route based on user confirmation of RAG queries. """

    # If not confirmed or search queries are empty, go back to "get_rag_queries"
    if not state.get("queries_confirmed", False):
        return "confirm_rag_queries" 
    
    search_queries = state.get("search_queries", [])

    if not search_queries:
        return "confirm_rag_queries"

    # If confirmed, proceed to retrieval
    return [
        Send("rag_retrieve_worker", {"current_query": q})
        for q in search_queries if q
    ]

# ───────────────────────────────────────────────────────────────────────
# WORKER CLAIM MATCHING NODE
# ───────────────────────────────────────────────────────────────────────

async def rag_retrieve_worker(state: AgentStateClaim) -> Dict[str, Any]:
    """ Perform retrieval for a given query. """

    # Get the current query from state
    q = state["current_query"]

    # Pass the subject as a safety net
    details = state.get("details_claim")
    subject=details.subject if details else ""

    # Call the retriever tool
    retriever_tool = tools_dict["retriever_tool"] 
    
    try:
        #invoke the LLM and store the output
        out = await retriever_tool.ainvoke({"query": q, "subject": subject})  
    
    except Exception as first_error:
        # fallback: try once more
        print("rag_retrieve_worker first attempt failed:", repr(first_error))

        try:
            #invoke the LLM and store the output
            out = await retriever_tool.ainvoke({"query": q, "subject": subject})

        except Exception as second_error:
            # fallback: just continue
            print("rag_retrieve_worker second attempt failed:", repr(second_error))

            return {
                "rag_trace": [{
                    "tool_name": "retriever_tool",
                    "args": {"query": q, "subject": subject},
                    "error": str(second_error),
                }],
                "messages": [
                    AIMessage(
                        content=(
                            "I ran into a technical issue while retrieving information for this query. "
                            "I tried again but it still failed, so I’ll skip it and continue."
                        )
                    )
                ],
            }
    
    # Return the RAG trace entry
    return {
        "rag_trace": [{
            "tool_name": "retriever_tool",
            "args": {"query": q, "subject": subject},
            "output": out,
        }]
    }

# ───────────────────────────────────────────────────────────────────────
# REDUCER CLAIM MATCHING NODE
# ───────────────────────────────────────────────────────────────────────

async def reduce_claim_matching(state: AgentStateClaim) -> Command[Literal["match_or_continue", "primary_source"]]:
    """Combines RAG traces and turns them into structured output."""

    # Gather all RAG traces
    raw_traces = state.get("rag_trace", [])

    # Format traces for the prompt
    formatted_trace = ""
    for entry in raw_traces:
        formatted_trace += f"\nQuery: {entry['args'].get('query')}\nOutput: {entry['output']}\n{'-'*20}"

    # Use structured output
    structured_llm = llm.with_structured_output(ClaimMatchingOutput,method="json_mode")

    # Create a prompt
    prompt = structure_claim_prompt.format(
        summary=state.get("summary", ""),
        rag_trace=formatted_trace, 
    )

    try:
        # Invoke the LLM and store the output
        result = await structured_llm.ainvoke(prompt)
    except Exception as e:
        # fallback: continue if failed
        print("reduce_claim_matching failed:", repr(e))

        ai_chat_msg = AIMessage(
            content=(
                "I ran into a technical issue while matching this claim to existing research. "
                "We’ll skip matching and move on to finding the primary source."
            )
        )

        return Command(
            goto="primary_source",
            update={
                "messages": [ai_chat_msg],
                "queries_confirmed": False,
            },
        )
    
    # human-readable assistant message for the chat
    explanation_lines = ["### Claim Matching Analysis\n", f"{result.explanation}\n"]
    
    if result.top_claims:
        explanation_lines.append("**Matched Claims:**")
        for i, c in enumerate(result.top_claims, start=1):
            url_part = f" ([Source]({c.allowed_url}))" if c.allowed_url else ""
            explanation_lines.append(f"{i}. {c.short_summary}{url_part}")
            explanation_lines.append(f"   *Rationale:* {c.alignment_rationale}")
    else:
        explanation_lines.append("_No strong matching claims were found for this specific assertion._")

    ai_chat_msg = AIMessage(content="\n".join(explanation_lines))

    # Determine next node based on whether matches were found
    goto_node = "match_or_continue" if result.top_claims else "primary_source"
    
    # Goto next node and update State
    return Command(
        goto=goto_node,
        update={
            "messages": [ai_chat_msg],
            "claim_matching_result": result,
            "queries_confirmed": False 
        }
    )
# ───────────────────────────────────────────────────────────────────────
# MATCHED OR CONTUE RESEARCH NODE
# ───────────────────────────────────────────────────────────────────────

async def match_or_continue(state: AgentStateClaim) -> Command[Literal["primary_source", "__end__"]]:

    """ Decide whether to continue researching or end the process if a matching claim was found."""

    # Get an answer from the user
    question = "Do any of these match your claim? Or do you want to continue researching as suggested?"
    user_answer = interrupt(question)

    # retrieve conversation history
    conversation_history = list(state.get("messages", []))

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # Use structured output
    structured_llm = llm.with_structured_output(ConfirmationMatch, method="json_mode")

    # Create a prompt
    prompt =  match_check_prompt.format(
        messages=messages_str,
        user_answer=user_answer,
    )

    try:
        # Invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
    
    except Exception as e:
        # fallback: ask to answer once more
        print("match_or_continue failed:", repr(e))

        # Goto next node and update State
        return Command(
            goto="match_or_continue",
            update={
                "messages": [
                    AIMessage(
                        content="I ran into a technical issue interpreting that. Could you answer once more?"
                    )
                ],
                "question": question,
            },
        )
    
    # human-readable assistant message for the chat
    if result.match:
        ai_chat_msg = AIMessage(
            content=(
                "This claim appears to match an already researched claim. "
                "We can stop the process here."
            )
        )
    else:
        ai_chat_msg = AIMessage(
            content=(
                "No exact match found. Let's continue researching."
            )
        )
    
    # Goto next node and update State
    if result.match:
        return Command(
                goto="__end__", 
                update={
                    "match": result.match,
                    "messages": [ai_chat_msg], 
                }
        )       
    else:
        return Command(
                goto="primary_source", 
                update={
                    "messages": [ai_chat_msg],  
                }
        )

# ───────────────────────────────────────────────────────────────────────
# RETRIEVE SOURCE, AND CHECK IF PRIMARY SOURCE KNOWN
# ───────────────────────────────────────────────────────────────────────
async def primary_source(state: AgentStateClaim) -> Command[Literal["get_search_queries", "get_source_queries"]]:
    """Ask once, extract everything, and route the research path."""
    
    # Prepare the question to ask the user
    claim_source = state.get("claim_source")
    primary_source= state.get("primary_source")
    if primary_source:
        ask_msg = f"Is it correct that **{claim_source}** is the primary source? If not please provide the correct primary source."
    elif claim_source != "":
        ask_msg = f"is **{claim_source}** the primary source? If not please provide the correct primary source."
    else:
        ask_msg = "From whom did you retrieve this claim? And was this the primary source? "

    # Get an answer from the user
    user_answer = interrupt(ask_msg)

    # Use structured output
    structured_llm = llm.with_structured_output(SourceOutput, method="json_mode")
    
    # Create a prompt
    prompt= source_prompt.format(
        user_answer=user_answer,
        claim_source=claim_source,
        primary_source=primary_source
    )

    try:
        # Invoke the LLM and store the output
        result = await structured_llm.ainvoke(prompt)

    except Exception as e:
        # fallback: ask to answer once more
        print("primary_source failed:", repr(e))

        # Don’t route. Re-ask.
        return Command(
            goto="primary_source",
            update={
                "messages": [
                    AIMessage(
                        content="I ran into a technical issue interpreting that. Could you answer once more?"
                    )
                ],
                "question": ask_msg,
            },
        )

    # Determine Routing and Messaging
    if result.primary_source:
        chat_text = f"Identified primary source: **{result.claim_source}**. Proceeding to verify the claim."
        goto_node = "get_search_queries"
    else:
        chat_text = f"Source noted as **{result.claim_source}**. I will first try to locate the original primary source."
        goto_node = "get_source_queries"

    return Command(
        goto=goto_node,
        update={
            "claim_source": result.claim_source,
            "primary_source": result.primary_source,
            "messages": [AIMessage(content=chat_text)],
            "search_queries": [] 
        }
    )

# ───────────────────────────────────────────────────────────────────────
# GENERATE QUERIES TO LOCATE PRIMARY SOURCE NODE
# ───────────────────────────────────────────────────────────────────────
async def get_source_queries(state: AgentStateClaim):

    """ Generate queries to locate the primary source of the claim. """

    # retrieve conversation history
    conversation_history = list(state.get("messages", []))

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # Use structured output
    structured_llm = llm.with_structured_output(GetSearchQueries, method="json_mode")

    # Create a prompt
    prompt  = source_queries_prompt.format(
        messages=messages_str,
        summary = state.get("summary", ""),
        claim = state.get("claim", ""),
        claim_source=state.get("claim_source", ""),
        claim_description = state.get("claim_description", "")
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

    except Exception as e:
        # fallback: try again
        print("get_source_queries failed:", repr(e))

        ai_chat_msg = AIMessage(
            content=(
                "I ran into a technical issue while generating source-search queries. "
                "Please try again."
            )
        )

        return {
            "search_queries": [],
            "messages": [ai_chat_msg],
            "research_focus": "select_primary_source",
            "question": "Please try again.",
        }
    
    # Format text for the user: include Reasoning for clarity
    queries_display = "\n".join(
        f"- **{q.query}**\n\n  *{q.reasoning}*" 
        for q in result.queries if q.query
    )
    
    chat_text = f"I will perform a search to find the primary source using these queries:\n\n{queries_display}"

    # Extract ONLY strings for State storage
    query_strings = [q.query for q in result.queries if q.query]

    # Goto next node and update State  
    return {
        "search_queries": query_strings,
        "messages":  [AIMessage(content=chat_text)],
        "research_focus": "select_primary_source",
        "question": "would you like to add or change something? Or do you want to continue?"
    }

# ───────────────────────────────────────────────────────────────────────
# CONFIRM SEARCH QUERIES NODE
# ───────────────────────────────────────────────────────────────────────
async def confirm_search_queries(state: AgentStateClaim) -> dict:
    
    """Update state based on user confirmation. Routing is handled by a router."""

    # Get an answer from the user
    question = state.get("question", "Is the information correct?")
    user_answer = interrupt(question)
    
    # retrieve search_queries and format to string for the prompt
    search_queries = state.get("search_queries", [])
    search_queries_str = "\n".join(f"- {q}" for q in search_queries)

    # Use structured output
    structured_llm = llm.with_structured_output(GetSearchQueries, method="json_mode")

    # Create a prompt
    prompt  =  confirm_queries_prompt.format(
        search_queries=search_queries_str,
        user_answer=user_answer,
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

    except Exception as e:
        # fallback: treat as NOT confirmed
        print("confirm_search_queries failed:", repr(e))

        # 🔁 Do not confirm, do not change anything, re-ask
        return {
            "queries_confirmed": False,
            "search_queries": search_queries,  # unchanged
            "messages": [
                AIMessage(content="I ran into a technical issue interpreting that. Could you answer once more?")
            ],
            "question": question,
        }    

    # Format display for the user (re-generating reasoning in the process)
    updated_display = "\n".join(
        f"- **{q.query}**\n\n  *{q.reasoning}*" 
        for q in result.queries if q.query
    )

    # Strip reasoning for State storage
    query_strings = [q.query for q in result.queries if q.query]
    
    # Create the base update
    update = {
        "queries_confirmed": result.confirmed,
        "search_queries": query_strings,
    }

    # Add messages only if confirmed, els reask the question
    if result.confirmed:
        confirm_text = "We will continue searching.\n"
        update["messages"] = [AIMessage(content=confirm_text)]
    else:
        update["question"] = (
            "Is there anything else you would like to change about the search queries? Or do you want to continue?\n"
            f"{updated_display}"
        )

    return update
# ───────────────────────────────────────────────────────────────────────
# RESET THE SEARCH STATE NODE
# ───────────────────────────────────────────────────────────────────────

def reset_search_state(state: AgentStateClaim):
    return {
        "tavily_context": Overwrite([]),
        }

# ───────────────────────────────────────────────────────────────────────
# FIND SOURCES ORCHESTRATOR NODE
# ───────────────────────────────────────────────────────────────────────

def route_after_confirm(state: AgentStateClaim):

    """ Route based on user confirmation of search queries. """

    # If not confirmed or search queries are empty, go back to "confirm_search_queries"
    if not state.get("queries_confirmed", False):
        return "confirm_search_queries"
    
    search_queries = state.get("search_queries", [])
    if not search_queries:
        return "confirm_search_queries"

    # If confirmed, proceed to retrieval
    return [
        Send("find_sources_worker", {"current_query": q})
        for q in search_queries if q
    ]

# ───────────────────────────────────────────────────────────────────────
# FIND SOURCES WORKER NODE
# ───────────────────────────────────────────────────────────────────────

async def find_sources_worker(state: AgentStateClaim) -> Dict[str, Any]:

    """Worker: run Tavily for one query, return one compact result block."""

    # Run tavily search for the current query, get top 18 results
    q = state["current_query"]
    tavily_tool = tools_dict.get("tavily_search")

    try:
        # Call the Tavily tool
        tool_output = await tavily_tool.ainvoke({"query": q, "max_results": 10})
        out_dict = tool_output.model_dump() if hasattr(tool_output, "model_dump") else dict(tool_output)

        # Compact the results
        compact_results = []
        for r in (out_dict.get("results") or []):
            r_data = r.model_dump() if hasattr(r, "model_dump") else dict(r)
            compact_results.append({
                "title": r_data.get("title"),
                "url": r_data.get("url"),
                "snippet": r_data.get("content", "")[:400]
            })

        # Return raw data for later evaluation
        return {
            "tavily_context": [{
                "query": q,
                "results": compact_results[:9]
            }]
        }

    except Exception as e:
        # fallback: just continue
        print("find_sources_worker failed:", repr(e))

        # No guessing: return empty results + record the error
        return {
            "tavily_context": [{
                "query": q,
                "results": [],
                "error": str(e),
            }],
            "messages": [
                AIMessage(
                    content="I ran into a technical issue while searching for sources. We’ll skip this query and continue."
                )
            ],
        }

# ───────────────────────────────────────────────────────────────────────
# FIND SOURCES REDUCER NODE
# ───────────────────────────────────────────────────────────────────────

async def reduce_sources(state: AgentStateClaim) -> Command[Literal["select_primary_source", "iterate_search"]]:
    """Merge worker outputs, perform global evidence evaluation, and build message."""
    
    tavily_results = state.get("tavily_context", [])
    used_urls, used_domains = set(), set()
    final_blocks = []
    evaluation_text = "" # To feed into the LLM

    # Deduplication & Compaction
    for block in tavily_results:
        query = block.get("query")
        results = block.get("results", [])
        if not results: continue

        # Extract relevant fields and deduplicate
        compact = {"query": query, "results": []}
        for r in results:
            url = (r.get("url") or "").strip()
            dom = _domain(url)
            if not url or url in used_urls or dom in used_domains:
                continue

            # Add to compacted results
            compact["results"].append({
                "title": r.get("title"), 
                "url": url, 
                "snippet": r.get("snippet", "") # Ensure snippet is passed for evaluation
            })
            used_urls.add(url)
            used_domains.add(dom)
            
            # Feed this into the evaluation text
            evaluation_text += f"\nQuery: {query}\nSource: {r.get('title')}\nSnippet: {r.get('snippet', '')}\n"

            if len(compact["results"]) >= 3: break
        final_blocks.append(compact)

    # Evaluate results and build message if in Search mode (not in primary source selection)
    lines = []
    research_focus = state.get("research_focus")
    if research_focus=="iterate_search":
        # retrieve conversation history
        conversation_history = list(state.get("messages", []))

        # Add the last message into a string for the prompt
        recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
        messages_str = get_buffer_string(recent_messages)

        # retrieve alerts and format to string for the prompt
        alerts=state.get("alerts", [])
        alerts_str= "\n".join(f"- {a}" for a in alerts)

        # Use structured output 
        structured_llm = llm.with_structured_output(SearchSynthesis, method="json_mode")
        
        # Create a prompt
        prompt  = eval_search_prompt.format(
            messages=messages_str,
            alerts=alerts_str,
            summary = state.get("summary", ""),
            claim = state.get("claim", ""),
            claim_source=state.get("claim_source", ""),
            claim_description = state.get("claim_description", ""),
            evaluation_text = evaluation_text
        )

        try:
            #invoke the LLM and store the output
            result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

            # human-readable assistant message
            lines = ["### Search Synthesis", f"{result.overall_summary}",""]

            if result.missing_info:
                lines.append("**Missing Information:**")
                for gap in result.missing_info:
                    lines.append(f"- {gap}")
                lines.append("")

            lines.append("---")

        except Exception as e:
            # fallback: just continue
            print("reduce_sources synthesis failed:", repr(e))

            lines = [
                "### Search Synthesis",
                "I ran into a technical issue while synthesizing the search results. "
                "Here are the sources I found so far:",
                "",
                "---",
            ]

    # Build final message with detailed sources
    lines.append("### Detailed Sources")
    
    startnr = 1
    for block in final_blocks:
        lines.append(f"\n**Query:** {block['query']}")
        lines.append("")
        for r in block['results']:
            lines.append(f"{startnr}. **[{r['title']}]({r['url']})**")
            startnr += 1

    new_msgs = [AIMessage(content="\n".join(lines))]

    # Routing to the correct next node
    state_next_node = "select_primary_source" if research_focus == "select_primary_source" else "iterate_search"
    
    return Command(
        goto=state_next_node,
        update={
            "queries_confirmed": False,
            "tavily_context": final_blocks,
            "messages": new_msgs,
            "search_queries": [],
        },
    )
# ───────────────────────────────────────────────────────────────────────
# SELECT PRIMARY SOURCE
# ───────────────────────────────────────────────────────────────────────

async def select_primary_source(state: AgentStateClaim):

    """ pick the best / most likely primary source. """

    # Get an answer from the user
    question = "Does any of these sources correspond to the primary source of the claim?"
    user_answer = interrupt(question)

    # Get the context and conversation history
    conversation_history = list(state.get("messages", []))

    # Use structured output 
    structured_llm = llm.with_structured_output(SourceOutput, method="json_mode")

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # Create a prompt
    prompt = select_primary_source_prompt.format(
        claim_source=state.get("claim_source", ""),
        user_answer=user_answer,
        messages=messages_str,
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

    except Exception as e:
        # fallback: askm once more
        print("select_primary_source failed:", repr(e))

        # no decision, no state changes
        return {
            "messages": [
                AIMessage(
                    content="I ran into a technical issue interpreting that. Could you answer once more?"
                )
            ],
            "question": question,
        }

    # retrieve existing alerts
    alerts = list(state.get("alerts", []))

    # human-readable assistant message
    chat_text = ""
    if result.primary_source:
        chat_text=(
                f"Great, you have identified **{result.claim_source}** as the primary source of the claim. "
                "I'll proceed with the research."
            )
    else:   
        # Add an alert if the primary source is not found
        alerts.append("primary source not found")   
        chat_text=(
            "I couldn’t identify a clear primary source from these results. "
            "I'll continue with the research anyway, but note that the original source is still missing."
        )

    ai_chat_msg = AIMessage(content=chat_text)

    # Goto next node and update State
    return {
        "primary_source": result.primary_source,
        "claim_source": result.claim_source,
        "messages": [ai_chat_msg],
        "alerts": alerts,
    }

# ───────────────────────────────────────────────────────────────────────
# GENERATE QUERIES TO FALSIFY OR VERIFY CLAIM NODE
# ───────────────────────────────────────────────────────────────────────
async def get_search_queries(state: AgentStateClaim):

    """ Generate queries to locate the primary source of the claim. """

    # retrieve conversation history
    conversation_history = list(state.get("messages", []))

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # retrieve alerts and format to string for the prompt
    alerts=state.get("alerts", [])
    alerts_str= "\n".join(f"- {a}" for a in alerts)

    # Use structured output
    structured_llm = llm.with_structured_output(GetSearchQueries, method="json_mode")

    # Create a prompt
    prompt  = search_queries_prompt.format(
        messages=messages_str,
        alerts=alerts_str,
        summary = state.get("summary", ""),
        claim = state.get("claim", ""),
        claim_source=state.get("claim_source", ""),
        #claim_description = state.get("claim_description", "")
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

    except Exception as e:
        # fallback: Continue
        print("get_search_queries failed:", repr(e))

        return {
            "search_queries": [],  
            "messages": [
                AIMessage(
                    content=(
                        "I ran into a technical issue while generating search queries. "
                        "Please try again."
                    )
                )
            ],
            "research_focus": "iterate_search",
            "question": "Please try again.",
        }

    #Format user-facing text (Query + Reasoning)
    queries_display = "\n".join(
        f"- **{q.query}**\n\n  *{q.reasoning}*" 
        for q in result.queries if q.query
    )
    
    chat_text = (
        "Based on our discussion and the missing details I've flagged, "
        "I will perform a search with these queries:\n\n"
        f"{queries_display}"
    )

    #Extract only query strings for State storage
    query_strings = [q.query for q in result.queries if q.query]

    ai_chat_msg = AIMessage(content=chat_text)

    # Goto next node and update State  
    return {
        "search_queries": query_strings,
        "messages":  [ai_chat_msg],
        "research_focus": "iterate_search",
        "question": "would you like to add or change something?"
    }
# ───────────────────────────────────────────────────────────────────────
# ASK TO ITERATE SEARCH NODE
# ───────────────────────────────────────────────────────────────────────

async def iterate_search(state: AgentStateClaim) -> Command[Literal["get_search_queries","__end__"]]:

    """ pick the best / most likely primary source. """

    # Get an answer from the user
    question = "Do you want to search once more?"
    user_answer = interrupt(question)

    # Get the context and conversation history
    conversation_history = list(state.get("messages", []))

    # Use structured output 
    structured_llm = llm.with_structured_output(ConfirmationOutput, method="json_mode")

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # Create a prompt
    prompt = iterate_search_prompt.format(
                user_answer=user_answer,
                messages=messages_str,
    )

    try:
        #invoke the LLM and store the output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

    except Exception as e:
        # fallback: treat as NOT confirmed
        print("iterate_search failed:", repr(e))

        # 🔁 no decision; re-ask
        return Command(
            goto="iterate_search",
            update={
                "messages": [
                    AIMessage(
                        content="I ran into a technical issue interpreting that. Could you answer once more?"
                    )
                ],
                "question": question,
            },
        )
    
    # retrieve existing alerts
    alerts = list(state.get("alerts", []))

    # human-readable assistant message
    if result.confirmed:
        ai_chat_msg = AIMessage(content="Let's search once more.")
    else:   
        ai_chat_msg = AIMessage(content="Good luck with your research, we have completed the search process.")

    # Goto next node and update State
    if result.confirmed:
        return Command(
            goto="get_search_queries",
            update={
                "messages": [ai_chat_msg],
                "next_node": None,
            },
        )
    else:
        return Command(
            goto="__end__",
            update={
                "messages": [ai_chat_msg],
            }
        )
            