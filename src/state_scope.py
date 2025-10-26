
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

# Input state for the full agent
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
    alerts: List[str] = Field(default_factory=list)
    summary: Optional[str]
    claim_author: Optional[str]
    claim_source: Optional[str]
    primary_source: Optional[str]
    
#output models for structured output
class SubjectResult(BaseModel):
    checkable: Literal["POTENTIALLY CHECKABLE", "UNCHECKABLE"]
    explanation: str
    question: str = Field("", description="Question to user for confirmation")

class MoreInfoResult(BaseModel):
    subject: str
    quantitative: bool
    precision: str = Field("", description="How precise is it?")
    based_on: str = Field("", description="how was the data collected or derived?")
    explanation: str
    question: str = Field("", description="Question to user for clarification if needed")
    alerts: List[str] = Field([], description="Any alerts or warnings about the claim")

class SummaryResult(BaseModel):
    summary: str
    question: str = Field("", description="Question to user for verification")

class ConfirmationResult(BaseModel):
    confirmed: bool = Field(False, description="Whether the user confirmed the claim as checkable")