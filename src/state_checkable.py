"""State Definitions and Pydantic Schemas for testing if a claim is checkable.

This defines the state objects and structured schemas used for Checking on checkability and check worthiness
of claims.
"""

from typing import TypedDict, Annotated, Sequence, Literal, List
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from operator import add as add_messages

# Define the structure of a citation
class Citation(TypedDict):
    index: str
    url: str
    title: str
    passage: str

# Create an object to hold the state of the agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    claim: str
    checkable: bool
    verdict: Literal["true", "false", "mixed", "insufficient"]
    explanation: str
    citations: List[Citation]
    follow_up_Q: str

#output models for structured output
class Checkability(BaseModel):
    checkable: bool = Field(..., description="True if the claim is checkable, else false.")
    explanation: str = Field(..., min_length=1, description="Short justification for the decision.")

parser_checkability = PydanticOutputParser(pydantic_object=Checkability)
format_checkability = parser_checkability.get_format_instructions()
