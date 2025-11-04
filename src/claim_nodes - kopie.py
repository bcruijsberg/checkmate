
""" Nodes for investigating the claim itself (workflow 1) """

from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,get_buffer_string
from prompts import checkable_check_prompt,confirmation_checkable_prompt, get_information_prompt, confirmation_clarification_prompt, get_summary_prompt, confirmation_check_prompt
from state_scope import AgentStateClaim, SubjectResult, MoreInfoResult, SummaryResult, ConfirmationResult,ConfirmationFinalResult
from typing_extensions import Literal, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from utils import get_new_user_reply
from tooling import llm

# Maximum number of messages to send to the prompt
MAX_HISTORY_MESSAGES = 6
# ───────────────────────────────────────────────────────────────────────
# CHECKABLE_FACT NODE
# ───────────────────────────────────────────────────────────────────────

def checkable_fact(state: AgentStateClaim) -> Command[Literal["checkable_confirmation"]]:

    """ Check if a claim is potentially checkable. """

    print(f"this is the state: {state.get("next_node")}")
    print("check")

    if state.get("next_node") is None:
    
        #Retrieve conversation history
        conversation_history = list(state.get("messages", []))

        # Add the last message into a string for the prompt
        recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
        messages_str = get_buffer_string(recent_messages)

        # Use structured output
        structured_llm = llm.with_structured_output(SubjectResult, method="json_mode")

        # Create a prompt
        prompt = checkable_check_prompt.format(
            claim=state.get("claim", ""),
            messages=messages_str,
        )

        #invoke the LLM and store the output
        result = structured_llm.invoke([HumanMessage(content=prompt)])

        # checkable is a boolean in State
        is_checkable = result.checkable == "POTENTIALLY CHECKABLE"

        # human-readable assistant message for the chat
        explanation_text = (
            f"**Checkability analysis**\n"
            f"- Checkable: `{result.checkable}`\n"
            f"- Reason: {result.explanation}\n"
        )
        if result.question:
            explanation_text += f"- Follow-up for you: {result.question}\n"

        ai_chat_msg = AIMessage(content=explanation_text)

        # Goto next node and update State
        return Command(
            goto="checkable_confirmation", 
            update={
                "question": result.question,
                "checkable": is_checkable,
                "explanation": result.explanation,
                "messages": [ai_chat_msg],
                "awaiting_user": True,
                "next_node": None,
            }
        )

# ───────────────────────────────────────────────────────────────────────
# CHECKABLE_CONFIRMATION NODE
# ───────────────────────────────────────────────────────────────────────

def checkable_confirmation(state: AgentStateClaim) -> Command[Literal["retrieve_information","__end__","checkable_fact"]]:
    
    """ Get confirmation from user on the gathered information. """
    print("confirm check")
    
    if state.get("next_node")=="checkable_confirmation" or state.get("next_node") is None:

        if state.get("awaiting_user"):

            ask_msg = AIMessage(content="Does this look like a claim you want to fact-check? You can reply with 'yes' or 'no'.")
            return Command(
                goto="__end__", 
                update={
                    "messages": [ask_msg],
                    "next_node": "checkable_confirmation",
                    "awaiting_user": False,
                },
            )
        else:
            # Retrieve conversation history
            conversation_history = list(state.get("messages", []))

            # Get user reply, if the last message was a user message
            user_answer = get_new_user_reply(conversation_history)

            # Use structured output
            structured_llm = llm.with_structured_output(ConfirmationResult, method="json_mode")

            # Create a prompt
            prompt = confirmation_checkable_prompt.format(
                claim=state.get("claim", ""),
                checkable=state.get("checkable", ""),
                explanation=state.get("explanation", ""),
                user_answer=user_answer,
            )

            #invoke the LLM and store the output
            result = structured_llm.invoke([HumanMessage(content=prompt)])
            
            # human-readable assistant message for the chat
            if result.confirmed:
                confirm_text = "We'll continue with this claim."
            else:
                confirm_text = "Okay let's revise the claim or stop here."

            ai_chat_msg = AIMessage(content=confirm_text)

            # Goto next node and update State
            if result.confirmed:
                if state.get("checkable"):
                    return Command(
                            goto="retrieve_information", 
                            update={
                                "confirmed": result.confirmed,
                                "messages": [ai_chat_msg],
                                "awaiting_user": False,
                                "next_node": None
                            }
                    )   
                else: 
                    # user confirmed but claim is not checkable → end
                    end_msg = AIMessage(content="This claim appears to be uncheckable, so we'll stop the process here.")
                    return Command(
                            goto=END, 
                            update={
                                "confirmed": result.confirmed,
                                "messages": [ai_chat_msg] + [end_msg],
                                "awaiting_user": False,
                                "next_node": None
                            }
                    )   
            else:
                return Command(
                        goto="checkable_fact", 
                        update={
                            "messages": [ai_chat_msg],
                            "awaiting_user": False,
                            "next_node": None
                        }
                )
    else:
        return Command(
                        goto="retrieve_information", 
                )

# ───────────────────────────────────────────────────────────────────────
# RETRIEVE_INFORMATION NODE
# ───────────────────────────────────────────────────────────────────────

def retrieve_information(state: AgentStateClaim) -> Command[Literal["clarify_information"]]:

    """ Gather more information about a potentially checkable claim. """
    print("info")
    if state.get("next_node") is None:

        #Retrieve conversation history
        conversation_history = list(state.get("messages", []))

        # Add the last message into a string for the prompt
        recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
        messages_str = get_buffer_string(recent_messages)

        # Use structured output
        structured_llm = llm.with_structured_output(MoreInfoResult, method="json_mode")

        # Create a prompt
        prompt  =  get_information_prompt.format(
            claim=state.get("claim", ""),
            messages=messages_str,
        )

        #invoke the LLM and store the output
        result = structured_llm.invoke([HumanMessage(content=prompt)])

        # human-readable assistant message for the chat
        details_text = (
            "**Here’s what I extracted from your claim:**\n"
            f"- Subject: {result.subject or 'not clearly specified'}\n"
            f"- Quantitative: {result.quantitative or 'no'}\n"
            f"- Precision: {result.precision or 'not specified'}\n"
            f"- Based on: {result.based_on or 'not specified'}\n"
        )
        if result.question:
            details_text += f"\n**Follow-up for you:** {result.question}\n"
        if result.alerts:
            details_text += "\n**Missing / to verify:**\n" + "\n".join(f"- {a}" for a in result.alerts)

        ai_chat_msg = AIMessage(content=details_text)

        # Goto next node and update State
        return Command(
            goto="clarify_information", 
            update={
                "subject": result.subject,
                "quantitative": result.quantitative,
                "precision": result.precision,
                "based_on": result.based_on,
                "question": result.question,
                "alerts": result.alerts or [],
                "messages": [ai_chat_msg],
                "awaiting_user": True,
                "next_node": None,
            }
        )   

# ───────────────────────────────────────────────────────────────────────
# CLARIFY_INFORMATION NODE
# ───────────────────────────────────────────────────────────────────────

def clarify_information(state: AgentStateClaim) -> Command[Literal["produce_summary", "retrieve_information"]]:

    """ Get confirmation from user on the gathered information. """
    print("clarify ")
    if state.get("next_node")=="clarify_information" or state.get("next_node") is None:

        if state.get("awaiting_user"):

            ask_msg = AIMessage(content="If you have more details, could your provide them, and mention if you want to continue.")
            return Command(
                goto="__end__", 
                update={
                    "messages": [ask_msg],
                    "next_node": "clarify_information",
                    "awaiting_user": False,
                },
            )
        else:

            #retrieve alerts and format to string for the prompt
            alerts=state.get("alerts", [])
            alerts_str= "\n".join(f"- {a}" for a in alerts)

            # retrieve conversation history
            conversation_history = list(state.get("messages", []))
            user_answer = get_new_user_reply(conversation_history)

            # Add the last message into a string for the prompt
            recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
            messages_str = get_buffer_string(recent_messages)

            # Use structured output
            structured_llm = llm.with_structured_output(ConfirmationResult, method="json_mode")

            # Create a prompt
            prompt  =  confirmation_clarification_prompt.format(
                subject=state.get("subject", ""),
                quantitative=state.get("quantitative", ""),
                precision=state.get("precision", ""),
                based_on=state.get("based_on", ""),
                claim=state.get("claim", ""),
                question=state.get("question", ""),
                alerts=alerts_str,
                messages=messages_str,
                user_answer=user_answer,
            )

            #invoke the LLM and store the output
            result = structured_llm.invoke([HumanMessage(content=prompt)])

            # human-readable assistant message for the chat
            if result.confirmed:
                confirm_text = "Thanks, I’ll use this information to draft the summary."
            else:
                confirm_text = "Let’s collect a bit more information."

            ai_chat_msg = AIMessage(content=confirm_text)

            # Goto next node and update State
            if result.confirmed:
                return Command(
                        goto="produce_summary", 
                        update={
                            "confirmed": result.confirmed,
                            "messages": [ai_chat_msg],
                            "next_node": None,
                        }
                )       
            else:
                return Command(
                        goto="retrieve_information", 
                        update={
                            "messages": [ai_chat_msg],
                            "next_node": None,
                        }
                )

# ───────────────────────────────────────────────────────────────────────
# PRODUCE SUMMARY NODE
# ───────────────────────────────────────────────────────────────────────

def produce_summary(state: AgentStateClaim) -> Command[Literal["get_confirmation"]]:

    """ Get a summary on the gathered information. """
    print("summary  ")
    if state.get("next_node") is None:

        # retrieve alerts and format to string for the prompt
        alerts=state.get("alerts", [])
        alerts_str= "\n".join(f"- {a}" for a in alerts)

        # retrieve conversation history
        conversation_history = list(state.get("messages", []))

        # Add the last message into a string for the prompt
        recent_messages = conversation_history[-MAX_HISTORY_MESSAGES:]  # tune this number
        messages_str = get_buffer_string(recent_messages)

        # Use structured output
        structured_llm = llm.with_structured_output(SummaryResult, method="json_mode")

        # Create a prompt
        prompt  =  get_summary_prompt.format(
            claim=state.get("claim", ""),
            subject=state.get("subject", ""),
            quantitative=state.get("quantitative", ""),
            precision=state.get("precision", ""),
            based_on=state.get("based_on", ""),
            alerts=alerts_str,
            messages=messages_str,
        )

        #invoke the LLM and store the output
        result = structured_llm.invoke([HumanMessage(content=prompt)])

        # human-readable assistant message for the chat
        summary_text = (
            f"**Summary of our findings so far:**\n\n{result.summary}\n\n"
        )
        if result.question:
            summary_text += f"**Next step / question:** {result.question}\n"

        ai_chat_msg = AIMessage(content=summary_text)

        # Goto next node and update State
        return Command( 
                goto="get_confirmation",
                update={
                    "summary": result.summary,
                    "question": result.question,
                    "messages": [ai_chat_msg],
                    "subject": result.subject,
                    "quantitative": result.quantitative,
                    "precision": result.precision,
                    "based_on": result.based_on,
                    "alerts": result.alerts or [],
                    "awaiting_user": True,
                }
        )       

# ───────────────────────────────────────────────────────────────────────
# GET_CONFIRMATION NODE
# ───────────────────────────────────────────────────────────────────────
   
def get_confirmation(state: AgentStateClaim) -> Command[Literal["produce_summary", "__end__"]]:

    """ Get confirmation from user on the gathered information."""

    if state.get("next_node")=="get_confirmation" or state.get("next_node") is None:

        if state.get("awaiting_user"):

            ask_msg = AIMessage(content="If you have more details, could your provide them, and mention if you want to continue.")
            return Command(
                goto="__end__", 
                update={
                    "messages": [ask_msg],
                    "next_node": "get_confirmation",
                    "awaiting_user": False,
                },
            )
        else:
            # retrieve conversation history
            conversation_history = list(state.get("messages", []))
            user_answer = get_new_user_reply(conversation_history)

            # Use structured output
            structured_llm = llm.with_structured_output(ConfirmationFinalResult, method="json_mode")

            # Create a prompt
            prompt  =  confirmation_check_prompt.format(
                summary=state.get("summary", ""),
                user_answer=user_answer,
            )

            #invoke the LLM and store the output
            result = structured_llm.invoke([HumanMessage(content=prompt)])

            # human-readable assistant message for the chat
            if result.confirmed:
                confirm_text = "Your confirmation has been recorded. We'll close this check here."
            else:
                confirm_text = "Let's revisit the summary and adjust it if needed."

            ai_chat_msg = AIMessage(content=confirm_text)

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
                            "messages": [ai_chat_msg],
                            "next_node": None,
                        }
                )       
            else:
                return Command(
                        goto="produce_summary", 
                        update={
                            "messages": [ai_chat_msg],
                            "subject": result.subject,
                            "quantitative": result.quantitative,
                            "precision": result.precision,
                            "based_on": result.based_on,
                            "question": result.question,
                            "alerts": result.alerts or [],
                            "next_node": None,
                        }
                )
    
