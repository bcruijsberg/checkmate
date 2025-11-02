
""" Nodes for claim matching, investigating the source and finally performing an search """

from prompts import retrieve_claims_prompt, match_check_prompt, identify_source_prompt, primary_source_prompt, select_primary_source_prompt, research_prompt
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage,get_buffer_string
from state_scope import AgentStateSource, ConfirmationMatch, GetSource, PrimarySourcePlan, PrimarySourceSelection, ResearchPlan
from langgraph.types import Command
from typing_extensions import Literal, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from utils import get_new_user_reply
from tooling import llm, llm_tools, tools_dict

# Maximum number of messages to send to the prompt
MAX_HISTORY_MESSAGES = 6

# ───────────────────────────────────────────────────────────────────────
# CLAIM MATCHING NODE
# ───────────────────────────────────────────────────────────────────────

def claim_matching(stateSource: AgentStateSource) -> Command[Literal["match_or_continue"]]:

    """ Call the retriever tool iteratively to find if similar claims have already been researched. """

    # retrieve conversation history
    conversation_history = list(stateSource.get("messages", []))

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    #Create a prompt
    prompt = retrieve_claims_prompt.format(
        summary=stateSource.get("summary", ""),
        subject=stateSource.get("subject", ""),
        messages=messages_str,
    )

    # Start with a single HumanMessage
    human = HumanMessage(content=prompt)

    # First model call: only the prompt
    result = llm_tools.invoke([human])

    # Keep a log in state 
    conversation_history.append(result)   

    # Iterate tool calls
    while getattr(result, "tool_calls", None):

        # empty tool messages list to contain tool outputs
        tool_msgs: List[ToolMessage] = []

        # loop over each tool call
        for t in result.tool_calls:
            name = t["name"]
            args = t.get("args") or {}

            # invoke the tool
            out = tools_dict[name].invoke(args)

            # append tool output as ToolMessage
            tool_msgs.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=name,
                    content=str(out),
                )
            )

        # Next model call, and decide if more tool calls are needed
        result = llm_tools.invoke([human, result, *tool_msgs])

        # Log messages to conversation history
        conversation_history.append(result)

    # After we finish tool-calling, ask the user to decide
    followup_msg = AIMessage(
        content=(
            "I found some possibly related or previously researched claims.\n"
            "Do any of these match your claim? Or do you want to continue researching?"
        )
    )
    
    # build updated history
    new_messages = conversation_history + [followup_msg]

    # Goto next node and update State
    return Command( 
            goto="match_or_continue",
            update={
                "messages": new_messages
            }
    )  

# ───────────────────────────────────────────────────────────────────────
# MATCHED OR CONTUE RESEARCH NODE
# ───────────────────────────────────────────────────────────────────────

def match_or_continue(stateSource: AgentStateSource) -> Command[Literal["get_source", "__end__"]]:

    """ Decide whether to continue researching or end the process if a matching claim was found."""

    # retrieve conversation history
    conversation_history = list(stateSource.get("messages", []))
    user_answer = get_last_user_message(conversation_history)

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

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())

   # human-readable assistant message for the chat
    if result.match:
        chat_msg = AIMessage(
            content=(
                "This claim appears to match an already researched claim. "
                "We can stop the process here."
            )
        )
    else:
        chat_msg = AIMessage(
            content=(
                "No exact match found. Let's continue researching.\n"
                "Do you have a URL for the source of the claim? If so, please share it.\n"
                "If not, tell me who made the claim and in what medium (article, video, social media, etc.)."
            )
        )
    
    # build updated history
    new_messages = conversation_history + [ai_msg, chat_msg]
    
    # Goto next node and update State
    if result.match:
        print("Since this claim has already been researched, we will end the process here.")
        return Command(
                goto=END, 
                update={
                    "match": result.match,
                    "explanation": result.explanation,
                    "messages": new_messages,  
                }
        )       
    else:
        return Command(
                goto="get_source", 
                update={
                    "explanation": result.explanation,
                    "messages": new_messages,  
                }
        )

# ───────────────────────────────────────────────────────────────────────
# RETRIEVE SOURCE
# ───────────────────────────────────────────────────────────────────────

def get_source(stateSource: AgentStateSource) -> Command[Literal["get_primary_source"]]:

    """ Ask the user for the  source of the claim if no match was found."""

    # retrieve conversation history
    conversation_history = list(stateSource.get("messages", []))
    user_answer = get_last_user_message(conversation_history)

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # Use structured output
    structured_llm = llm.with_structured_output(GetSource, method="json_mode")

    # Create a prompt
    prompt  =  identify_source_prompt.format(
        messages=messages_str,
        user_answer=user_answer,
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())

    # human-readable assistant message for the chat
    if result.claim_source:
        followup_text = (
            f"Thanks, I captured the source as:\n\n**{result.claim_source}**\n\n"
            "Do you know whether this is the **original / primary** source of the claim? "
            "If not, tell me the original source or share its URL."
        )
    else:
        followup_text = (
            "I couldn’t identify a concrete source from that.\n"
            "Can you tell me where the claim was published (URL, outlet, platform) "
            "and, if possible, who made it?"
        )

    ai_chat_msg = AIMessage(content=followup_text)

    # build updated history
    new_messages = conversation_history + [ai_msg, ai_chat_msg]

    # Goto next node and update State
    return Command(
            goto=END, 
            update={
                "claim_source": result.claim_source,
                "claim_url": result.claim_url,
                "messages":new_messages,
            }
    ) 

# ───────────────────────────────────────────────────────────────────────
# GET MORE INFO
# ───────────────────────────────────────────────────────────────────────

def get_primary_source(stateSource: AgentStateSource) -> Command[Literal["research_claim","locate_primary_source"]]:

    """ Ask the user for the original primary source of the claim, and more background on the source """

    # retrieve conversation history
    conversation_history = list(stateSource.get("messages", []))
    user_answer = get_last_user_message(conversation_history)

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # Use structured output
    structured_llm = llm.with_structured_output(PrimarySourcePlan, method="json_mode")

    # Create a prompt
    prompt  = primary_source_prompt.format(
        messages=messages_str,
        user_answer=user_answer,
        summary = stateSource.get("summary", ""),
        subject = stateSource.get("subject", ""),
        claim_source=stateSource.get("claim_source", ""),
        claim_url = stateSource.get("claim_url", "")
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())
    conversation_history.append(ai_msg)

    # human-readable assistant message for the chat
    if result.primary_source:
        chat_text = (
            f"We now have a primary source:\n\n**{result.claim_source}**\n\n"
            "I'll use this to continue researching the claim."
        )
    else:
        # we didn't get a clear primary source, so we will search for it in the next node
        if result.search_queries:
            queries_text = "\n".join(f"- {q}" for q in result.search_queries if q)
            chat_text = (
                "I still don’t see a clear original / primary source.\n"
                "I'll run a web search to try to locate it, using these queries:\n"
                f"{queries_text}"
            )
        else:
            chat_text = (
                "I still don’t see a clear original / primary source.\n"
                "I'll run a web search to try to locate it."
            )

    ai_chat_msg = AIMessage(content=chat_text)

    # build updated history
    new_messages = conversation_history + [ai_msg, ai_chat_msg]

    # Goto next node and update State
    if result.primary_source:
        return Command(
            goto="research_claim", 
            update={
                "primary_source": result.primary_source,
                "claim_source": result.claim_source,
                "search_queries": result.search_queries,
                "messages": new_messages,
            }
        )    
    else:
        return Command(
            goto="locate_primary_source",
            update={
                "primary_source":result.primary_source,
                "claim_source": result.claim_source,
                "search_queries": result.search_queries,
                "messages": new_messages,
            }
        )  

# ───────────────────────────────────────────────────────────────────────
# LOCATE PRIMARY SOURCE
# ───────────────────────────────────────────────────────────────────────

def locate_primary_source(stateSource: AgentStateSource) -> Command[Literal["select_primary_source"]]:

    """ Run Tavily for each prepared query and store all results. """

    # Get the context from stateSource
    conversation_history = list(stateSource.get("messages", []))
    search_queries = stateSource.get("search_queries", []) or []

    # A list to store the search results
    tavily_results = []

    # Loop over the search_queries
    for q in search_queries:
        if not q:
            continue

        tavily_tool = tools_dict.get("tavily_search")
        if tavily_tool is None:
            continue

        print(f"\n-- locate_primary_source: running tavily_search for: {q} --")
        tool_output = tavily_tool.invoke({"query": q})

        tavily_results.append({
            "query": q,
            "result": tool_output,
        })

        #show the output
        print(tool_output)

        conversation_history.append(
            ToolMessage(
                name="tavily_search",
                content=str(tool_output),
                tool_call_id=f"tavily-{hash(q)}",
            )
        )

    # add a chat message telling the user what's next
    if tavily_results:
        followup_msg = AIMessage(
            content=(
                "I searched for possible original / official sources using the queries we prepared.\n"
                "Please tell me which of the results looks like the **original / primary** source, "
                "or say 'none' if none of them is correct."
            )
        )
    else:
        followup_msg = AIMessage(
            content=(
                "I couldn’t run a useful search because there were no queries. "
                "Tell me more about the original source (URL, outlet, author, or organization)."
            )
        )
    
    ai_chat_msg = AIMessage(content=followup_text)

    # build updated history
    new_messages = conversation_history + [ai_chat_msg]

    # Goto next node and update State
    return Command(
        goto="select_primary_source",
        update={
            "tavily_context": tavily_results,
            "messages": new_messages,
        },
    )


# ───────────────────────────────────────────────────────────────────────
# SELECT PRIMARY SOURCE
# ───────────────────────────────────────────────────────────────────────

def select_primary_source(stateSource: AgentStateSource) -> Command[Literal["research_claim"]]:

    """ pick the best / most likely primary source. """

    # Use structured output 
    structured_llm = llm.with_structured_output(PrimarySourceSelection, method="json_mode")

    # Get the context and conversation history
    conversation_history = list(stateSource.get("messages", []))
    tavily_context = stateSource.get("tavily_context", [])
    user_answer = get_last_user_message(conversation_history)

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # turn tavily_context into something the LLM can read
    tavily_pretty = json.dumps(tavily_context, indent=2)

    # Create a prompt
    prompt = select_primary_source_prompt.format(
        summary=stateSource.get("summary", ""),
        subject=stateSource.get("subject", ""),
        claim_source=stateSource.get("claim_source", ""),
        claim_url=stateSource.get("claim_url", ""),
        tavily_context=tavily_pretty,
        user_answer=user_answer,
        messages=messages_str,
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())

   # Add a warning if the primary source is not found
    alerts = list(stateSource.get("alerts", []))
    if not result.primary_source:
        alerts.append("primary source not found")

    # human-readable assistant message
    if result.primary_source:
        ai_chat_msg = AIMessage(
            content=(
                f"✅ I'll treat this as the primary source:\n\n**{result.claim_source or result.claim_url}**\n\n"
                "I will now research the claim further."
            )
        )
    else:
        ai_chat_msg = AIMessage(
            content=(
                "I couldn’t identify a clear primary source from these results. "
                "I'll continue with the research anyway, but note that the original source is still missing."
            )
        )

    # build updated history
    new_messages = conversation_history + [ai_msg, ai_chat_msg]

    # Goto next node and update State
    return Command(
        goto="research_claim",
        update={
            "primary_source": result.primary_source,
            "claim_source": result.claim_source or stateSource.get("claim_source", ""),
            "claim_url": result.claim_url or stateSource.get("claim_url", ""),
            "messages": new_messages,
            "alerts": alerts
        },
    )

# ───────────────────────────────────────────────────────────────────────
# RESEARCH CLAIM
# ───────────────────────────────────────────────────────────────────────

def research_claim(stateSource: AgentStateSource) -> Command[Literal["__end__"]]:

    """ create several research queries, and run tavily_search for each query """

    # Use structured output 
    structured_llm = llm.with_structured_output(ResearchPlan, method="json_mode")

    # retrieve conversation history and alerts
    conversation_history = list(stateSource.get("messages", []))
    alerts = list(stateSource.get("alerts", []))

    # Add the last message into a string for the prompt
    recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
    messages_str = get_buffer_string(recent_messages)

    # Create a prompt
    prompt = research_prompt.format(
        summary=stateSource.get("summary", ""),
        subject=stateSource.get("subject", ""),
        claim_source=stateSource.get("claim_source", ""),
        claim_url=stateSource.get("claim_url", ""),
        alerts=alerts,
        messages=messages_str,
    )

    # 
    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())
    conversation_history.append(ai_msg)

    # 2) run Tavily for each proposed query
    research_results = []
    for q in result.research_queries:
        if not q:
            continue

        tavily_tool = tools_dict.get("tavily_search")
        if tavily_tool is None:
            continue

        print(f"\n-- research_claim: running tavily_search for: {q} --")
        tool_output = tavily_tool.invoke({"query": q})

        research_results.append({
            "query": q,
            "result": tool_output,
        })

        # add to history
        conversation_history.append(
            ToolMessage(
                name="tavily_search",
                content=str(tool_output),
                tool_call_id=f"tavily-research-{hash(q)}",
            )
        )

    # human-readable assistant message
    research_summary_lines = []
    if result.research_queries:
        research_summary_lines.append("I searched for additional evidence using these queries:")
        research_summary_lines.extend(f"- {q}" for q in result.research_queries if q)
    if alerts:
        research_summary_lines.append(
            "Note: some details were marked as missing earlier, so the search also tried to address those."
        )

    if not research_summary_lines:
        research_summary_lines.append("I tried to collect additional evidence for this claim.")

    ai_chat_msg = AIMessage(content="\n".join(research_summary_lines))

    # build updated history
    new_messages = conversation_history + [ai_chat_msg]

    # Goto next node and update State
    return Command(
        goto=END,
        update={
            "research_queries": result.research_queries,
            "research_focus": result.research_focus,
            "research_results": research_results,
            "messages": new_messages,
        },
    )
