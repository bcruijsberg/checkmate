
""" State Definitions and Pydantic Schemas for testing if a claim is checkable. """

import operator
from typing import Annotated, Literal, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from operator import add as add_messages
from langgraph.graph.message import MessagesState

#initial state for claim checking
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
    queries_confirmed: bool
    question: Optional[str]
    alerts: List[str] = Field(default_factory=list)
    summary: Optional[str]
    explanation: Optional[str]
    tool_trace: Optional[str]
    rag_trace: Annotated[List[Dict[str, Any]], operator.add]
    claim_matching_result: Optional[str]
    search_queries:  List[str] = Field(default_factory=list)
    tavily_context: Annotated[List[Dict[str, Any]], operator.add]
    current_query: str
    research_focus: Optional[str]
    claim_url: Optional[str]
    claim_source: Optional[str]
    primary_source: Optional[bool]
    match: Optional[bool]
    critical_question: Optional[str]
    reasoning_summary: Optional[str]

#output models for structured output

# Structured output models for the first nodes, gathering all needed info about the claim
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
    claim_source: str = Field("", description="What is the source of this claim?")

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
    claim_source: str = Field("", description="What is the source of this claim?")

class ConfirmationMatch(BaseModel):
    match: bool = Field(False, description="Whether the user confirmed their was a matching claim")

# Structured output models for the claim matching node
class QueryItem(BaseModel):
    query: str = Field("", description="The search query to run")
    reasoning: str = Field("", description="The reasoning behind this query")

class TopClaim(BaseModel):
    short_summary: str= Field("", description="A short summary of the claim")
    allowed_url: Optional[str] = Field(None, description="An allowed URL supporting or refuting the claim")
    alignment_rationale: str = Field("", description="Rationale for how this claim aligns or conflicts with the user's claim")

class ClaimMatchingOutput(BaseModel): 
    queries: List[QueryItem] = Field(..., description="List of search queries with reasoning")
    top_claims: List[TopClaim] = Field(..., description="Top matching claims from the database")

# Structured output models for primary source identification nodes
class GetSource(BaseModel):
    claim_source: str = Field("", description="What is the source of this claim?")
    primary_source: bool = Field(False, description="True if the user provided the original/official source")

class GetSourceLocation(BaseModel):
    claim_url: str = Field("", description="The URL of the primary source if available, otherwise ''.")
    source_description: str = Field("", description="Description of the source if no URL is available.")

# Structured output model for search nodes (primary source and final research)
class GetSearchQueries(BaseModel):
    search_queries: List[str] = Field(default_factory=list, description="Ordered list of queries to run in Tavily to find the primary source")
    confirmed: bool = Field(False, description="Whether the user confirmed the search queries")

class SearchResult(BaseModel):
    title: Optional[str] = Field("", description="Title of the search result")
    url: Optional[str] = Field("", description="URL of the search result")
    snippet: Optional[str] = Field("", description="Snippet or summary of the search result")
    score: Optional[float] = Field(None, description="Relevance score if available")

class TavilySearchOutput(BaseModel):
    query: str
    results: List[SearchResult] = Field(default_factory=list)

class PrimarySourceSelection(BaseModel):
    primary_source: bool = Field(..., description="True if a credible/original source was found among the Tavily results.")
    claim_source: str = Field("", description="The best/most likely primary source (URL or title).")
    claim_url: str = Field("", description="The URL of the primary source if available, otherwise ''.")