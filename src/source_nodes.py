
from prompts import retrieve_claims_prompt, match_check_prompt, identify_source_prompt, primary_source_prompt, select_primary_source_prompt, research_prompt
from langchain_core.messages import HumanMessage, ToolMessage,AIMessage
from state_scope import AgentStateSource, ConfirmationMatch, GetSource, PrimarySourcePlan, PrimarySourceSelection, ResearchPlan
from langgraph.types import Command
from typing_extensions import Literal, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from utils import get_last_user_message

# ───────────────────────────────────────────────────────────────────────
# CLAIM MATCHING NODE
# ───────────────────────────────────────────────────────────────────────

def claim_matching(stateSource: AgentStateSource) -> Command[Literal["match_or_continue"]]:

    """ Call the retriever tool iteratively to find if similar claims have already been researched. """

    # retrieve conversation history
    conversation_history = list(stateSource.get("messages", []))

    # Get the summary of the previous steps
    summary = stateSource.get("summary", "")
    subject = stateSource.get("subject", "")

    # System prompt for checkability
    prompt = retrieve_claims_prompt.format(
        summary=summary,
        subject=subject,
        messages=conversation_history
    )

    # Start with a single HumanMessage
    human = HumanMessage(content=prompt)

    # First model call: only the prompt
    result = llm_tools.invoke([human])

    # Keep a log in state (messages must be BaseMessage objects)
    conversation_history.append(result)   

    # Iterate tool calls
    while getattr(result, "tool_calls", None):

        # empty tool messages list to contain tool outputs
        tool_msgs: List[ToolMessage] = []

        # loop over echt tool call
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

        print(result.content)
        print("Do one of the claims match? Or do you want to continue researching?")

        # Log messages to conversation history
        conversation_history.append(result)


    return Command( 
            update={
                "goto":"match_or_continue",
                "messages": conversation_history
            }
    )  

# ───────────────────────────────────────────────────────────────────────
# MATCHED OR CONTUE RESEARCH NODE
# ───────────────────────────────────────────────────────────────────────

def match_or_continue(stateSource: AgentStateSource) -> Command[Literal["get_source", "__end__"]]:

    """ Decide whether to continue researching or end the process if a matching claim was found."""

    # retrieve conversation history
    conversation_history = list(stateSource.get("messages", []))

    # Get user's reply
    user_answer = input("> ")
    conversation_history.append(HumanMessage(content=user_answer))


    # Use structured output to get a boolean and all details as output
    structured_llm = llm.with_structured_output(ConfirmationMatch, method="json_mode")

    # System prompt for checkability
    prompt =  match_check_prompt.format(
        messages=conversation_history,
        user_answer=user_answer,
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())
    conversation_history.append(ai_msg)


    # print output
    print(f"confirmed: {result.match}")
    print(f"explanation: {result.explanation}")

    if result.match:
        print("Since this claim has already been researched, we will end the process here.")
        return Command(
                goto=END, 
                update={
                    "match": result.match,
                    "explanation": result.explanation,
                    "messages": conversation_history,  
                }
        )       
    else:
        print("Since no matching claim was found, we'll continue researching.")
        print("Do you have a URL for the source of the claim? If so, please share it.")
        print("If not, could you tell me who the author/ source is and what medium the claim appeared in (e.g., article, video, social media post)?")
        return Command(
                goto="get_source", 
                update={
                    "explanation": result.explanation,
                    "messages": conversation_history,  
                }
        )

# ───────────────────────────────────────────────────────────────────────
# RETRIEVE SOURCE
# ───────────────────────────────────────────────────────────────────────

def get_source(stateSource: AgentStateSource) -> Command[Literal["get_primary_source"]]:

    """ Ask the user for the  source of the claim if no match was found."""

    # retrieve conversation history
    conversation_history = list(stateSource.get("messages", []))

    # Get user's reply
    user_answer = input("> ")
    conversation_history.append(HumanMessage(content=user_answer))


    # Use structured output to get a boolean and all details as output
    structured_llm = llm.with_structured_output(GetSource, method="json_mode")

    # System prompt for checkability
    prompt  =  identify_source_prompt.format(
        messages=conversation_history,
        user_answer=user_answer,
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())
    conversation_history.append(ai_msg)


    # Ask more details on the source
    if {result.claim_source}!= "":
        print(f"Is the author/ source {result.claim_source} an expert, a lobiest, an NGO, government, opposition, civil service, think tank, a company or a citizen?")
    else:
        print("Do you know if the author is an expert, a lobiest, an NGO, government, opposition, civil service, think tank, a company or a citizen?")

    # Ask for the primary source
    print("Do you know if this is the orignal primary source? If not do you know the original source? If so tell me or add a url")

    return Command(
            goto=END, 
            update={
                "claim_source": result.claim_source,
                "claim_url": result.claim_url,
                "messages": conversation_history,
            }
    ) 

# ───────────────────────────────────────────────────────────────────────
# GET MORE INFO
# ───────────────────────────────────────────────────────────────────────

def get_primary_source(stateSource: AgentStateSource) -> Command[Literal["research_claim","locate_primary_source"]]:

    """ Ask the user for the original primary source of the claim, and more background on the source """

    # retrieve conversation history
    conversation_history = list(stateSource.get("messages", []))

    # Get user's reply
    user_answer = input("> ")
    conversation_history.append(HumanMessage(content=user_answer))


    # Use structured output to get a boolean and all details as output
    structured_llm = llm.with_structured_output(PrimarySourcePlan, method="json_mode")

    # System prompt for checkability
    prompt  = primary_source_prompt.format(
        messages=conversation_history,
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

    # Print the output
    print(f"primary source: {result.primary_source}")
    print(f"claim source: {result.claim_source}")
    print(f"search queries: {result.search_queries}")

    # print output
    if result.primary_source:
        return Command(
            goto="research_claim", 
            update={
                "primary_source": result.primary_source,
                "claim_source": result.claim_source,
                "search_queries": result.search_queries,
                "messages": conversation_history,
            }
        )    
    else:
        return Command(
            goto="locate_primary_source",
            update={
                "primary_source":result.primary_source,
                "claim_source": result.claim_source,
                "search_queries": result.search_queries,
                "messages": conversation_history,
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

    # write into state
    # Do you think one of these resources is the primary source?
    return Command(
        goto="select_primary_source",
        update={
            "tavily_context": tavily_results,
            "messages": conversation_history,
        },
    )


# ───────────────────────────────────────────────────────────────────────
# SELECT PRIMARY SOURCE
# ───────────────────────────────────────────────────────────────────────

def select_primary_source(stateSource: AgentStateSource) -> Command[Literal["research_claim"]]:

    """ pick the best / most likely primary source. """

    # Use structured output 
    structured_llm = llm.with_structured_output(PrimarySourceSelection, method="json_mode")

    conversation_history = list(stateSource.get("messages", []))
    tavily_context = stateSource.get("tavily_context", [])
    claim_source = stateSource.get("claim_source", "")
    claim_url = stateSource.get("claim_url", "")
    summary = stateSource.get("summary", "")
    subject = stateSource.get("subject", "")

    # Get user's reply
    user_answer = input("> ")
    conversation_history.append(HumanMessage(content=user_answer))

    # turn tavily_context into something the LLM can read
    tavily_pretty = json.dumps(tavily_context, indent=2)

    prompt = select_primary_source_prompt.format(
        summary=summary,
        subject=subject,
        claim_source=claim_source,
        claim_url=claim_url,
        tavily_context=tavily_pretty,
        user_answer=user_answer,
        messages=conversation_history
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())
    conversation_history.append(ai_msg)

    # Print the output
    print(f"primary source: {result.primary_source}")
    print(f"claim source: {result.claim_source}")
    print(f"claim url: {result.claim_url}")

    # Add a warning if the primary source is not found
    alerts = list(stateSource.get("alerts", []))
    if not result.primary_source:
        alerts.append("primary source not found")


    return Command(
        goto="research_claim",
        update={
            "primary_source": result.primary_source,
            "claim_source": result.claim_source or stateSource.get("claim_source", ""),
            "claim_url": result.claim_url or stateSource.get("claim_url", ""),
            "messages": conversation_history,
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

    # Get all context
    conversation_history = list(stateSource.get("messages", []))

    primary_source = stateSource.get("primary_source", False)
    claim_source = stateSource.get("claim_source", "")
    claim_url = stateSource.get("claim_url", "")
    summary = stateSource.get("summary", "")
    subject = stateSource.get("subject", "")
    alerts = list(stateSource.get("alerts", []))

    # Create research queries
    prompt = research_prompt.format(
        summary=summary,
        subject=subject,
        claim_source=claim_source,
        claim_url=claim_url,
        alerts=alerts,
        messages=conversation_history
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

       # Print the output
    print(f"research queries: {result.research_queries}")
    print(f"research focus: {result.research_focus}")
    print(research_results)

    # 3) write everything back to state
    return Command(
        goto=END,
        update={
            "research_queries": result.research_queries,
            "research_focus": result.research_focus,
            "research_results": research_results,
            "messages": conversation_history,
        },
    )
