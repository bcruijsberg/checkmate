
"""Nodes for checking if a claim is potentially checkable."""

from langchain_core.messages import HumanMessage,AIMessage,get_buffer_string
from prompts import checkable_check_prompt,confirmation_checkable_prompt, get_information_prompt, confirmation_clarification_prompt, get_summary_prompt, confirmation_check_prompt
from state_scope import AgentStateClaim, SubjectResult, MoreInfoResult, SummaryResult, ConfirmationResult,ConfirmationFinalResult
from typing_extensions import Literal, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from utils import get_last_user_message

# ───────────────────────────────────────────────────────────────────────
# CHECKABLE_FACT NODE
# ───────────────────────────────────────────────────────────────────────

def checkable_fact(state: AgentStateClaim) -> Command[Literal["checkable_confirmation"]]:

    """ Check if a claim is potentially checkable. """

    # Use structured output to get a boolean and explanation as output
    structured_llm = llm.with_structured_output(SubjectResult, method="json_mode")

    # System prompt for checkability
    prompt  =  checkable_check_prompt.format(
        claim=state.get("claim", ""),
        messages=get_buffer_string(state.get("messages", [])),
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())

    # checkable is a boolean in State
    is_checkable = result.checkable == "POTENTIALLY CHECKABLE"

    # print output
    print("\n=== 1. CHECKABLE? ===")
    print(f"{result.checkable}")
    print(f"{result.explanation}")
    print(f"{result.question}")


    return Command(
        goto="checkable_confirmation", 
        update={
            "question": result.question,
            "checkable": is_checkable,
            "messages": list(state.get("messages", [])) + [ai_msg],
        }
    )

# ───────────────────────────────────────────────────────────────────────
# CHECKABLE_CONFIRMATION NODE
# ───────────────────────────────────────────────────────────────────────

def checkable_confirmation(state: AgentStateClaim) -> Command[Literal["retrieve_information","__end__","checkable_fact"]]:

    """ Get confirmation from user on the gathered information. """

    #Retrieve conversation history
    conversation_history = list(state.get("messages", []))

    # Get user's reply
    user_answer = input("> ")
    conversation_history.append(HumanMessage(content=user_answer))


    # Use structured output to get a boolean and ethe messages as output
    structured_llm = llm.with_structured_output(ConfirmationResult, method="json_mode")

    # System prompt for checkability
    prompt  =  confirmation_checkable_prompt.format(
        claim=state.get("claim", ""),
        checkable=state.get("checkable", ""),
        explanation=state.get("explanation", ""),
        user_answer=user_answer,
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())


    # print output
    print(f"confirmed: {result.confirmed}")


    if result.confirmed:
        if state.get("checkable"):
            return Command(
                    goto="retrieve_information", 
                    update={
                        "confirmed": result.confirmed,
                        "messages": conversation_history + [ai_msg],
                    }
            )   
        else: 
            print("Since this claim is uncheckable, we will end the process here.")
            return Command(
                    goto=END, 
                    update={
                        "confirmed": result.confirmed,
                        "messages": conversation_history + [ai_msg],
                    }
            )   
    else:
        return Command(
                goto="checkable_fact", 
                update={
                    "messages": conversation_history + [ai_msg],
                }
        )

# ───────────────────────────────────────────────────────────────────────
# RETRIEVE_INFORMATION NODE
# ───────────────────────────────────────────────────────────────────────

def retrieve_information(state: AgentStateClaim) -> Command[Literal["clarify_information"]]:

    """ Gather more information about a potentially checkable claim. """

    # Use structured output to get aal the details as output
    structured_llm = llm.with_structured_output(MoreInfoResult, method="json_mode")


    # System prompt for checkability
    prompt  =  get_information_prompt.format(
        claim=state.get("claim", ""),
        messages=get_buffer_string(state.get("messages", [])),
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())


    # print output
    print("\n=== 2. Evidence retrieved? ===")
    print(f"subject: {result.subject}")
    print(f"quantitative: {result.quantitative}")
    print(f"precision: {result.precision}")
    print(f"based on: {result.based_on}")
    print(f"question: {result.question}")
    print(f"alerts: {result.alerts}")

    return Command(
        goto="clarify_information", 
        update={
            "subject": result.subject,
            "quantitative": result.quantitative,
            "precision": result.precision,
            "based_on": result.based_on,
            "question": result.question,
            "alerts": result.alerts or [],
        }
    )   

# ───────────────────────────────────────────────────────────────────────
# CLARIFY_INFORMATION NODE
# ───────────────────────────────────────────────────────────────────────

def clarify_information(state: AgentStateClaim) -> Command[Literal["produce_summary", "retrieve_information"]]:

    """ Get confirmation from user on the gathered information. """
    #retrieve alerts and format to string for the prompt
    alerts=state.get("alerts", [])
    alerts_str= "\n".join(f"- {a}" for a in alerts)

    #retreive conversation history
    conversation_history = list(state.get("messages", []))

    # Get user's reply
    user_answer = input("> ")
    conversation_history.append(HumanMessage(content=user_answer))


    # Use structured output to get a boolean and explanation as output
    structured_llm = llm.with_structured_output(ConfirmationResult, method="json_mode")

    # System prompt for checkability
    prompt  =  confirmation_clarification_prompt.format(
        subject=state.get("subject", ""),
        quantitative=state.get("quantitative", ""),
        precision=state.get("precision", ""),
        based_on=state.get("based_on", ""),
        claim=state.get("claim", ""),
        question=state.get("question", ""),
        alerts=alerts_str,
        messages=conversation_history,
        user_answer=user_answer,
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())


    # print output
    print(f"confirmed: {result.confirmed}")


    if result.confirmed:
        return Command(
                goto="produce_summary", 
                update={
                    "confirmed": result.confirmed,
                    "messages": conversation_history + [ai_msg],
                }
        )       
    else:
        return Command(
                goto="retrieve_information", 
                update={
                    "messages": conversation_history + [ai_msg],
                }
        )

# ───────────────────────────────────────────────────────────────────────
# PRODUCE SUMMARY NODE
# ───────────────────────────────────────────────────────────────────────

def produce_summary(state: AgentStateClaim) -> Command[Literal["get_confirmation"]]:

    """ Get a summary on the gathered information. """

   #retrieve alerts and format to string for the prompt
    alerts=state.get("alerts", [])
    alerts_str= "\n".join(f"- {a}" for a in alerts)

    # retrieve conversation history
    conversation_history = list(state.get("messages", []))


    # Use structured output to get all details the messages as output
    structured_llm = llm.with_structured_output(SummaryResult, method="json_mode")

    # System prompt for checkability
    prompt  =  get_summary_prompt.format(
        claim=state.get("claim", ""),
        subject=state.get("subject", ""),
        quantitative=state.get("quantitative", ""),
        precision=state.get("precision", ""),
        based_on=state.get("based_on", ""),
        alerts=alerts_str,
        messages=conversation_history ,
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())


    # print output
    print("\n=== 3. SUMMARY ===")
    print(f"To verify, this is a summary of our discussion: {result.summary}")
    print(f"{result.question}")

    return Command( 
            update={
                "summary": result.summary,
                "question": result.question,
                "messages": list(state.get("messages")) + [ai_msg],
                "subject": result.subject,
                "quantitative": result.quantitative,
                "precision": result.precision,
                "based_on": result.based_on,
                "alerts": result.alerts or []
            }
    )       

# ───────────────────────────────────────────────────────────────────────
# GET_CONFIRMATION NODE
# ───────────────────────────────────────────────────────────────────────
#    
def get_confirmation(state: AgentStateClaim) -> Command[Literal["produce_summary", "__end__"]]:

    """ Get confirmation from user on the gathered information."""

    # retrieve conversation history
    conversation_history = list(state.get("messages", []))

    # Get user's reply
    user_answer = input("> ")
    conversation_history.append(HumanMessage(content=user_answer))


    # Use structured output to get a boolean and all details as output
    structured_llm = llm.with_structured_output(ConfirmationFinalResult, method="json_mode")

    # System prompt for checkability
    prompt  =  confirmation_check_prompt.format(
        summary=state.get("summary", ""),
        user_answer=user_answer,
    )

    #invoke the LLM and store the output
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=result.model_dump_json())
    conversation_history.append(ai_msg)


    # print output
    print(f"confirmed: {result.confirmed}")

    if result.confirmed:
        return Command(
                goto=END, 
                update={
                    "confirmed": result.confirmed,
                    "subject": result.subject,
                    "quantitative": result.quantitative,
                    "precision": result.precision,
                    "based_on": result.based_on,
                    "question": result.question,
                    "alerts": result.alerts or [],
                    "messages": conversation_history,

                }
        )       
    else:
        return Command(
                goto="produce_summary", 
                update={
                    "messages":  conversation_history,
                    "subject": result.subject,
                    "quantitative": result.quantitative,
                    "precision": result.precision,
                    "based_on": result.based_on,
                    "question": result.question,
                    "alerts": result.alerts or [],
                }
        )
