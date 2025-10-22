
"""State Definitions and Pydantic Schemas for CheckMate.

This defines the state objects and structured schemas used for
the CheckMate scoping workflow, states and output schemas.
"""

import operator
from typing_extensions import Optional, Annotated, Sequence, List, Literal,TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

# Create an object to hold the state of the agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    claim: str 
    checkable: Optional[bool]
    subject: Optional[str]
    quantitative: Optional[bool]
    precision: Optional[str]
    based_on: Optional[str]
    confirmed: bool
    question: Optional[str]
    alerts: List[str]
    
#output models for structured output
class SubjectResult(BaseModel):
    subject: str = Field("", description="Main subject of the claim, if identifiable")
    checkable: Literal["POTENTIALLY CHECKABLE", "UNCHECKABLE"]
    explanation: str
    question: str = Field("", description="Question to user for clarification if needed")
    alerts: List[str] = Field([], description="Any alerts or warnings about the claim")


class MoreInfoResult(BaseModel):
    quantitative: bool
    precision: str = Field("", description="How precise is it?")
    based_on: str = Field("", description="how was the data collected or derived?")
    explanation: str
    question: str = Field("", description="Question to user for clarification if needed")
    alerts: List[str] = Field([], description="Any alerts or warnings about the claim")
