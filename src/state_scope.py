
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
    geography: Optional[str]
    time_period: Optional[str]
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
    claim_source: str = Field(description="The entity who made the claim.")
    primary_source: bool = Field(description="True if this is the original/foundational source.")
    source_description: str = Field(description="Contextual description of where it was found.")
    geography: str = Field("", description="Geographic scope of the claim")
    time_period: str = Field("", description="Time frame relevant to the claim")

class SummaryResult(BaseModel):
    summary: str = Field("", description="A concise summary of the claim")
    question: str = Field("", description="Question to user for verification")
    subject: str = Field("", description="The subject of the claim")
    alerts: List[str] = Field([], description="Any alerts or warnings about the claim")

class ConfirmationResult(BaseModel):
    confirmed: bool = Field(False, description="Whether the user confirmed the claim as checkable")

# Structured output models for the claim matching node
class ConfirmationMatch(BaseModel):
    match: bool = Field(False, description="Whether the user confirmed their was a matching claim")

class QueryItem(BaseModel):
    query: str = Field("", description="The search query to run")
    reasoning: str = Field("", description="The reasoning behind this query")

class GetSearchQueries(BaseModel):
    queries: List[QueryItem] = Field(..., description="List of search queries with reasoning")
    confirmed: bool = Field(False, description="Whether the user confirmed the search queries")

class TopClaim(BaseModel):
    short_summary: str = Field(default="", description="Summary of the found claim")
    allowed_url: Optional[str] = Field(None, description="The URL from the trace")
    alignment_rationale: str = Field(default="", description="Comparison with user claim")

class ClaimMatchingOutput(BaseModel): 
    top_claims: List[TopClaim] = Field(default_factory=list, description="List of matches")
    explanation: str = Field(default="", description="Summary of findings")

# Structured output models for primary source identification nodes
class SourceExtraction(BaseModel):
    claim_source: str = Field(description="The entity who made the claim.")
    primary_source: bool = Field(description="True if this is the original/foundational source.")

class SourceExtraction2(BaseModel):
    claim_source: str = Field(description="The entity who made the claim.")
    primary_source: bool = Field(description="True if this is the original/foundational source.")
    claim_url: str = Field(description="Direct URL if provided, else empty string.")
    source_description: str = Field(description="Contextual description of where it was found.")
    question: str = Field("", description="Question to user for verification")
    subject: str = Field("", description="The subject of the claim")
    quantitative: str = Field("", description="Is the claim quantitative?")
    precision: str = Field("", description="How precise is it?")
    based_on: str = Field("", description="how was the data collected or derived?")

# Structured output model for search nodes (primary source and final research)
class SearchResult(BaseModel):
    title: str = Field("", description="Title of the search result")
    url: str = Field("", description="URL of the search result")
    content: str = Field("", description="Content snippet of the search result")

class TavilySearchOutput(BaseModel):
    query: str = Field(..., description="The original search query used")
    results: List[SearchResult] = Field(default_factory=list)
    answer: Optional[str] = Field(None, description="An AI-generated answer if available")
    error: Optional[str] = Field(None, description="Error message if the search failed")

class SearchSynthesis(BaseModel):
    overall_summary: str = Field(description="A brief synthesis of what the combined search results tell us.")
    missing_info: List[str] = Field(description="Specific facets of the claim still unsupported by the results.")
    coverage_score: int = Field(description="Score from 1-10 on how well the search results cover the subject.")

class PrimarySourceSelection(BaseModel):
    primary_source: bool = Field(..., description="True if a credible/original source was found among the Tavily results.")
    claim_source: str = Field("", description="The best/most likely primary source (URL or title).")