
""" State Definitions and Pydantic Schemas for testing if a claim is checkable. """

from typing_extensions import Annotated, Literal, List, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from operator import add as add_messages
from langgraph.graph.message import MessagesState


class AgentStateClaim(MessagesState):
    messages: Annotated[List[BaseMessage], add_messages]
    messages_critical: Annotated[List[BaseMessage], add_messages]
    claim: str 
    checkable: Optional[bool]
    additional_context: Optional[str]
    subject: Optional[str]
    quantitative: Optional[str]
    precision: Optional[str]
    based_on: Optional[str]
    confirmed: bool
    question: Optional[str]
    alerts: List[str] = Field(default_factory=list)
    summary: Optional[str]
    awaiting_user: bool
    explanation: Optional[str]
    next_node: Optional[str]
    search_queries:  List[str] = Field(default_factory=list)
    tavily_context: Optional[str]
    research_focus: Optional[str]
    research_results: List[str] = Field(default_factory=list)
    claim_url: Optional[str]
    claim_source: Optional[str]
    primary_source: Optional[bool]
    match: Optional[bool]
    critical_question: Optional[str]
    reasoning_summary: Optional[str]

#output models for structured output
class SubjectResult(BaseModel):
    checkable: Literal["POTENTIALLY CHECKABLE", "UNCHECKABLE"]
    explanation: str = Field("", description="Explanation for the classification")
    question: str = Field("", description="Question to user for confirmation")

class MoreInfoResult(BaseModel):
    subject: str = Field("", description="The subject of the claim")
    quantitative: str = Field("", description="Is the claim quantitative?")
    precision: str = Field("", description="How precise is it?")
    based_on: str = Field("", description="how was the data collected or derived?")
    question: str = Field("", description="Question to user for clarification if needed")
    alerts: List[str] = Field([], description="Any alerts or warnings about the claim")

class SummaryResult(BaseModel):
    summary: str = Field("", description="A concise summary of the claim")
    question: str = Field("", description="Question to user for verification")
    subject: str = Field("", description="The subject of the claim")
    quantitative: str = Field("", description="Is the claim quantitative?")
    precision: str = Field("", description="How precise is it?")
    based_on: str = Field("", description="how was the data collected or derived?")
    question: str = Field("", description="Question to user for clarification if needed")
    alerts: List[str] = Field([], description="Any alerts or warnings about the claim")

class ConfirmationResult(BaseModel):
    confirmed: bool = Field(False, description="Whether the user confirmed the claim as checkable")

class ConfirmationFinalResult(BaseModel):
    confirmed: bool = Field(False, description="Whether the user confirmed the claim as checkable")
    subject: str = Field("", description="The subject of the claim")
    quantitative: str = Field("", description="Is the claim quantitative?")
    precision: str = Field("", description="How precise is it?")
    based_on: str = Field("", description="how was the data collected or derived?")
    question: str = Field("", description="Question to user for clarification if needed")
    alerts: List[str] = Field([], description="Any alerts or warnings about the claim")

class ConfirmationMatch(BaseModel):
    match: bool = Field(False, description="Whether the user confirmed their was a matching claim")
    explanation: str = Field("", description="Explanation of the matching")

class QueryItem(BaseModel):
    query: str
    reasoning: str

class TopClaim(BaseModel):
    short_summary: str
    allowed_url: Optional[str]
    alignment_rationale: str

class ClaimMatchingOutput(BaseModel):
    queries: List[QueryItem]          # ← now flat, not grouped
    top_claims: List[TopClaim]
    follow_up_question: str

class GetSource(BaseModel):
    claim_source: str = Field("", description="What is the source of this claim?")
    claim_url: str = Field("", description = "What is the url of this claim?")

class PrimarySourcePlan(BaseModel):
    claim_source: str = Field("", description="Best current source for the claim (URL, site, platform, etc.)")
    primary_source: bool = Field(False, description="True if the user already provided the original/official source")
    search_queries: List[str] = Field(default_factory=list, description="Ordered list of queries to run in Tavily to find the primary source")

class PrimarySourceSelection(BaseModel):
    primary_source: bool = Field(..., description="True if a credible/original source was found among the Tavily results.")
    claim_source: str = Field("", description="The best/most likely primary source (URL or title).")
    claim_url: str = Field("", description="The URL of the primary source if available, otherwise ''.")
    alerts: List[str] = Field([], description="Any alerts or warnings about the claim")

class ResearchPlan(BaseModel):
    research_queries: List[str] = Field([], description="Ordered list of search queries to run to gather evidence for the claim.")
    research_focus: str = Field("", description="What the research should focus on (e.g. fact checks, official statements, datasets).")

class CriticalQuestion(BaseModel):
    critical_question: str = Field("", description="A critical socratic question to stimulate critical thinking")
    reasoning_summary: str = Field("", description="A summary of the reasoning done by the llm")