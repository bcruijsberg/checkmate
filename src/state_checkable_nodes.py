
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage,AIMessage
from prompts import clarify_with_user_instructions

# ──────────────────────────────
# NODE FUNCTION TO CHECK IF THE CLAIM is CHECKABLE
# ──────────────────────────────
def check_fact_checkability(state: AgentState) -> AgentState:
    """
    Check if a claim is checkable.
    """
    # Use structured output to get a boolean and explanation as output
    structured_llm = llm.with_structured_output(Checkability)

    # System prompt for checkability
    checkability_prompt = f"""
    Task:
    Decide if a single claim is CHECKABLE (factual, verifiable) or UNCHECKABLE (opinion, prediction, or too vague).

    Definitions:
    CHECKABLE: The claim asserts something that can be verified with evidence (e.g., data, records, studies), even without exact numbers or precise details.
    UNCHECKABLE: The claim is an opinion, value judgment, future prediction, or too vague to investigate.

    Rules:
    -Evidence link: If the claim refers to a measurable property or documentable fact, mark CHECKABLE. Lack of an exact figure does not disqualify it.
    -Comparisons: “More/less,” “higher/lower,” etc. are CHECKABLE if the attribute is measurable.
    -Clarity: If subject + attribute are specific enough that a journalist could look them up, it’s CHECKABLE. If too vague, UNCHECKABLE.
    -Future claims: Predictions are always UNCHECKABLE.
    -Value judgments: Normative or taste claims (“better,” “should,” “harmful”) are UNCHECKABLE, unless they define a measurable outcome.

    Examples:
    “Omicron spreads faster than other Covid-19 strains.” → CHECKABLE
    “Chocolate ice cream is better than vanilla.” → UNCHECKABLE
    “Unemployment was higher in 2023 than 2022.” → CHECKABLE
    “The stock market will crash next month.” → UNCHECKABLE
    “City X has more residents than City Y.” → CHECKABLE
    “This policy is harmful.” → UNCHECKABLE unless linked to specific metrics.

    {format_checkability}
    """
    msgs = [
        SystemMessage(content=checkability_prompt),
        HumanMessage(content=f'Claim: "{state["claim"]}"'),
    ]

    result: Checkability = structured_llm.invoke(msgs)

    # print output
    print("\n=== 1. CHECKABLE? ===")
    print(f"{result.checkable}")
    print(f"{result.explanation}")

    # Add the model's response as an AIMessage for traceability
    ai_msg = AIMessage(content=result.model_dump_json())

    return {
        **state,
        "checkable": result.checkable,
        "messages": list(state.get("messages", [])) + [ai_msg],
    }

# ─────────────────────────────
# CONDITIONAL FUNCTION: IF CHECKABLE -> CONTINUE
# ──────────────────────────────
def branch_on_checkable(state: AgentState) -> bool:
        return state.get("checkable")

# ─────────────────────────────
# NODE FUNCTION TO GATHER MORE EVIDENCE
# ──────────────────────────────
def research(state: AgentState) -> AgentState:
    """
    Call the tools iteratively (retriever tool) to gather evidence about the claim. 
    Add this evidence to the conversation history. 
    """
    claim = state.get("claim", "")

    # System prompt for research
    research_prompt = f"""
    INSTRUCTIONS
    - Call `retriever_tool` with short, focused queries directly about the {claim} (population, exposure, outcome, timeframe),
    or if the last HumanMessag is a "Follow up question" focus on this question.
    - Reformulate and call again ONLY if needed. Do not ask unrelated trivia.
    - Use retrieved CONTEXT and ALLOWED_URLS to decide if you need more queries.
    - Do NOT produce a final verdict or JSON in this turn.
    - When you have enough, stop calling tools and give a brief, ordinary reply (e.g., "Proceed to finalize").
    """
    msgs = [
        SystemMessage(content=research_prompt),
        HumanMessage(content="If you need evidence, call `retriever_tool` with a focused query, relevant to the claim.")
    ]
    messages = list(state.get("messages", [])) + msgs

    # Iteratively call tools until done, use llm_tools to enable tool calling
    while True:
        ai = llm_tools.invoke(messages) 
        print(f"AI Response: {ai.content}\n")
        messages.append(ai)

        # check if there are tool calls
        tc = getattr(ai, "tool_calls", None)
        if not tc:
            break

        # run tools and append ToolMessages
        tool_msgs = []
        for t in tc:
            out = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            tool_msgs.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(out)))
        messages += tool_msgs

    return {**state, "messages": messages}

# ─────────────────────────────
# NODE FUNCTION TO STRUCTURE FINAL OUTPUT
# ──────────────────────────────
def finalize(state: AgentState) -> AgentState:
    """
    Finalize the fact-check by producing a verdict, explanation, and citations based on all retrieved evidence. 
    Use structured output to ensure the response is in the correct format.
    """
    claim = state.get("claim", "")

    # System prompt for finalization
    finalize_prompt = f"""
    You are finalizing a fact-check for the claim: "{claim}".

    You must rely ONLY on evidence already retrieved (CONTEXT and ALLOWED_URLS from prior tool calls in the conversation). 
    Do NOT call any tools now.

    VERDICTS
    - TRUE / FALSE/ MOSTLY TRUE / MOSTLY FALSE / UNCHECKABLE

    RULES
    - Match metric, population, geography, and timeframe.
    - Prefer primary/official sources; explain conflicts briefly.
    - Cite passages ≤40 words and reference sources by ALLOWED_URLS index only (e.g., "0", "2").
    - Include the urls from ALLOWED_URLS in your citations.

    {format_fact_check}
    """

    #use structured output to get a verdict eplanation and citations from ALLOWED_URLS
    structured_llm = llm.with_structured_output(FactCheckResult)

    msgs = list(state.get("messages", [])) + [
        SystemMessage(content=finalize_prompt),
        HumanMessage(content="Return ONLY the JSON for the final verdict based on retrieved evidence.")
    ]
    result: FactCheckResult = structured_llm.invoke(msgs)

    # print output
    print("\n=== 2. ANSWER ===")
    print(f"{result.verdict}")
    print(f"{result.explanation}")
    for citation in result.citations:
        print(f"[{citation['index']}] {citation['title']}")
        print(f"URL: {citation['url']}")
        print(f"Passage: {citation['passage']}\n")


    ai_msg = AIMessage(content=result.model_dump_json())
    return {
        **state,
        "verdict": result.verdict,
        "explanation": result.explanation,
        "citations": result.citations,
        "messages": list(state.get("messages", [])) + [ai_msg],
    }

# ─────────────────────────────
# CONDITIONAL FUNCTION: IF FOLLOW UP -> REPEAT FLOW
# ──────────────────────────────
def follow_up(state: AgentState) -> AgentState:
    """
    Check if the user has a follow-up question.
    """
    user_input = input("\nDo you have a follow-up question? (yes (y) /no (n)): ")

    if user_input == 'yes' or user_input == 'y':
        # if the user has a follow-up question, get the question and add it to the messages
        user_followup=input("\nWhat is your question? : ")
        return {
            **state,
            "follow_up_Q": user_followup,
            "messages": list(state.get("messages", [])) + [HumanMessage(content="follow up question: "+user_followup)],
        }       
    else:
        print("No follow-up question. Ending the fact-checking process.")
        return {**state,
                "follow_up_Q": None,
        }

    # ─────────────────────────────
# CONDITIONAL FUNCTION: IF FOLLOW-UP -> CONTINUE
# ──────────────────────────────
def branch_on_follow_up(state: AgentState) -> bool:
    if state.get("follow_up_Q")==None:
        return False
    else:
        return True
