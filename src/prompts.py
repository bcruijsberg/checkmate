# Generate a socratic question
get_socratic_question = """
### Role
Pedagogical Facilitator and Socratic Coach. 

### Objective
Your goal is to be a "thought partner." Instead of pointing out errors, you ask questions that lead the student to discover gaps in the claim's logic or evidence themselves.

### Inputs
- {claim}
- {summary}

- Gaps in claim: {alerts}

- Critical questions so far (if any):
<History>
{messages_critical}
</History>

### Now generate the question.
"""

# Prompt to determine if a claim is checkable
checkable_check_prompt = """
### Role
Neutral Fact-Checking Analyst. Focus on objective evaluation and guiding the user's reasoning through reflective inquiry rather than providing definitive answers.

### Context
<History>{messages}</History>

Claim to Evaluate: {claim}

### Additional context the user provided
"{additional_context}"

### Task
Classify the claim and determine if it can be fact-checked.

### Classification Logic
- **UNCHECKABLE**: 
  - *Opinion*: Beliefs, judgments, or values (e.g., "Policy X is bad").
  - *Prediction*: Future events or uncertain outcomes.
- **POTENTIALLY CHECKABLE**: 
  - *Fact*: Assertions about the past or present that can be verified with evidence.

### Instructions
1. Analyze the claim, prioritizing any "Additional Context" provided.
2. Categorize the claim and draft a brief justification.
3. Formulate a polite **question** to confirm your assessment with the user, and ALWAYS ADD "Or do you want to continue to the next step?".

### Output (JSON)
{{
  "checkable": "POTENTIALLY CHECKABLE | UNCHECKABLE",
  "explanation": "Brief justification for the category chosen.",
  "question": "Polite confirmation question asking if the user agrees, and wants to continue"
}}
"""

# Prompt to retrieve information from the claim's source
retrieve_info_prompt = """
### Role
Neutral Fact-Checking Analyst. Focus on objective evaluation and guiding the user's reasoning through reflective inquiry rather than providing definitive answers.

### Context
<History>{messages}</History>
- claim origin: "{page_content}"
- Claim: {claim}

### Additional context the user provided
"{additional_context}"

### Task 1: Source & Intent Extraction
1. **claim_source**: Identify the person or organization who originated the claim.
2. **primary_source**: Set to true ONLY if the evidence confirms this is the original/foundational origin.
3. **source_description**: Describe the medium (e.g., "Official PDF", "Social Media Post").

### Task 2: Factual Dimension Analysis
Analyze the claim's logic based on all available evidence:
1. **Subject**: Identify the core entity or event.
2. **Quantitative/Qualitative**: Explain if it is measurable data or a description.
3. **Precision**: Categorize as Precise, Vague, or Absolute (100%), and provide specific numbers, or names from the evidence.
4. **Based On**: Identify the likely methodology (e.g., Official stats, Survey, research). Provide a brief explanation.
5. **Geography**: Identify the geographic scope of the claim.
6. **Time Period**: Identify the time frame relevant to the claim.

### Task 3: Guidance & Risk
1. **Alerts**: Flag missing Geography, Time Period, unclear subject, qualitative claim, vague quantitative claim, geography missing, time period missing, methodological details absent. Do not flag if the info is present.
2. **The Question**: Formulate a polite open **question** asking for additional information, and ALWAYS ADD "Or do you want to continue to the next step?".
3. **details** : Include specific details (dates, numbers, names) from the evidence, to support your analysis from:
{page_content}

### Output Format
Respond in the following structured JSON format:
{{
  "claim_source": A specific Person or Organisation or "unknown",
  "primary_source": boolean,
  "source_description": description of the medium,
  "subject": "subject text" or "unclear",
  "quantitative": "quantitative" or "qualitative", and a short explanation,
  "precision": "precise" or "vague" or "absolute (100%)" or "", specific numbers, dates, and names from the evidence,
  "based_on": "methodology" or "unclear", and a short explanation,
  "question": "Polite open question asking for addional information, or if the user wants to continue",
  "alerts": ["each alert as a short string; [] if none"],
  "geography": "geographic scope" or "unclear",
  "time_period": "time frame relevant to the claim" or "unclear"
}}
"""

# Prompt to confirm the extracted information with the user
confirmation_prompt = """
### Role
Linguistic Analyst specializing in intent detection.

### Context
- User's Response: "{user_answer}"

### Task
Determine if the User's Response provides a "Green Light" to proceed.

### Decision Rules
**Set "confirmed": true IF:**
- User explicitly agrees (e.g., "Yes," "Correct," "Exactly").
- User provides a neutral command somwhere in the answer to proceed (e.g., "Continue," "Next" "Proceed" "Move on").
- User admits they have no more information (e.g., "I don't know," "That's all I have," "No more details").

**Set "confirmed": false IF:**
- User provides **new additional context or corrections** (even if they agree with the rest).
- User expresses uncertainty or asks a new question.

### Important Rules
- Maintain a neutral, analytical tone.

### Output (JSON)
{{
  "confirmed": boolean
}}
"""

# Prompt to produce a summary of the claim and its characteristics so far
get_summary_prompt = """
### Role
Neutral Fact-Checking Analyst. Focus on objective evaluation and guiding the user's reasoning through reflective inquiry.

### Context
<History>{messages}</History>

- Extracted Info: {{Subject: {subject}, Type: {quantitative}, Precision: {precision}, Basis: {based_on}, Time Period: {time_period}, Geography: {geography}}}
- Extracted source info: {{claim_source: {claim_source}, source_description: "{source_description}"}}
- Current Claim: {claim}
- Alerts: "{alerts}"
- claim origin: "{page_content}"

### Additional context the user provided
"{additional_context}"

### Task
Synthesize the current understanding of the claim into a concise report for the user to review before research begins.

### Instructions
1. **Summarize**: Create a brief overview of the claim, with as much specific detail as possible.
2. **Subject Refinement**: Refine the "subject" field to be as specific as possible based on all available information.
3. **Alerts**: List all relevant alerts based on the current extracted info.
4. **Question**: Formulate one polite, open-ended question to ensure the user is satisfied with this framing, and ALWAYS ADD "Or do you want to continue to the next step?".
5. **details** : Include specific details (dates, numbers, names) from the evidence, to support your analysis from:
{page_content}

### Output (JSON)
{{
  "summary": "Concise summary of the claim, its characteristics, and discussion so far."
  "subject": refine the "subject text", keep it short
  "question": "Polite open question asking for addional information, or if the user wants to continue"
  "alerts": ["each alert as a short string; [] if none"
}}
"""

# Prompt to generate RAG search queries
rag_queries_prompt = """
### Role
Neutral Fact-Checking Analyst. Focus on objective evaluation and aiding retrieval through precise, diverse search strategies.

### Objective
Generate 3 distinct search queries to locate semantically similar previously fact-checked claims in the FACTors database.

### Context
<History>{messages}</History>

- Current Summary: "{summary}"
- Subject: {subject}

### Query Generation Logic
To maximize retrieval recall, generate three queries with different structural focuses:
1. **The Proposition Query**: Focus on the core assertion and its actors (e.g., "Mayor Smith corruption allegations").
2. **The Data/Outcome Query**: Focus on specific statistics, quantities, or measurable outcomes (e.g., "15% increase in city crime rates 2024").
3. **The Contextual/Paraphrased Query**: Use synonyms and alternative phrasing to capture different ways the same claim might have been recorded (e.g., "Metropolis municipal budget deficit claims").

### Constraints
- No commentary or conversational filler.
- Queries must be concise (6–10 words).
- Output exactly 3 queries in the specified JSON.

### Output (JSON)
{{
  "queries": [
    {{
      "query": "The concrete search string used",
      "reasoning": "1-2 sentences on why this query was chosen."
    }}
  ],
  "confirmed": false
}}
"""

# Prompt to confirm or modify the generated RAG search queries
confirm_queries_prompt = """
### Role
Linguistic Analyst specializing in intent detection.

### Context
<Search Queries>
{search_queries}
</Search Queries>

- User's Response: "{user_answer}"

### Task
Determine if the user's latest response indicates approval to proceed with the current search queries or a request for modification.

### Decision Rules
**Set "confirmed": true IF:**
- User explicitly agrees (e.g., "Fine" "continue" "Correct," "Ok").
- User provides a neutral command somwhere in the answer to proceed (e.g., "Continue,"  "Next" "Proceed" "Move on").
- User admits they have no more information (e.g., "I don't know," "That's all I have," "No more details").

**Set "confirmed": false IF:**
- User provides **new additional context or corrections** (even if they agree with the rest).
- User expresses uncertainty or asks a new question.

### Instructions
1. **If confirmed is true**: Return the "{search_queries}" exactly as provided.
2. **If confirmed is false**: Modify or replace the queries based on the user's specific feedback in "{user_answer}". 
3. Maintain exactly 3 queries in the final list.

### Output (JSON)
{{
    "queries": [
    {{
      "query": "The concrete search string used",
      "reasoning": "1-2 sentences on why this query was chosen."
    }}
  ],
  "confirmed": boolean
}}
"""

# Prompt to structure the claim matching process
structure_claim_prompt = """
### Role
Neutral Fact-Checking Analyst. Focus on objective synthesis of evidence and logical evaluation of claim alignment.

### Context
- Claim Summary: {summary}
- Subject: {subject}
- Retrieval Trace: {rag_trace}

### Objective
Organize the retrieval results into a structured evaluation. Identify the search queries used and map them to potentially relevant existing claims found in the trace.

### Task Guidelines
1. **Query Analysis**: Identify how the search queries targeted specific facets (subject, statistics, or timeframe) of the claim.
2. **Claim Matching**: Extract up to 5 relevant claims from the `{rag_trace}`.
3. **Alignment Rationale**: For each match, explicitly compare facets (entities, geography, quantities) to the user's claim.
4. **Strict Evidence**: Use ONLY the information provided in the `{rag_trace}`. Do not invent claims.

### Important Rules
- **Tone**: Maintain a neutral, analytical, and pedagogical tone.

### Output (JSON)
Respond only with a JSON object in this format:
{{
  "top_claims": [
    {{
      "short_summary": "Description of the claim",
      "allowed_url": "URL or null",
      "alignment_rationale": "Comparison logic"
    }}
  ],
  "explanation": "Summary of search results and why matches were or were not found."
}}
"""

# Check if a matching claim has been found based on the user's answer
match_check_prompt = """
### Role
Linguistic Analyst specializing in intent detection.

### Context
<History>{messages}</History>

- User's Latest Response: "{user_answer}"

### Task
Determine if the user's response indicates they have found a satisfactory match among the previously presented claims or if they wish to continue the search.

### Decision Rules
- **Set "match": true** IF the user explicitly identifies a match (e.g., "Found it," "That's the one," "Yes, this claim matches").
- **Set "match": false** IF the user indicates no match was found, expresses dissatisfaction with results, or gives a command to keep looking (e.g., "None of these," "Keep searching," "No," "Continue").

### Guidelines
- **Ambiguity**: If the response is unclear, lean toward `false` to ensure the investigation is thorough rather than stopping prematurely.
- **Evidence**: Base your decision strictly on the user's intent expressed in `{user_answer}` relative to the `{messages}` history.

### Output (JSON)
{{
  "match": boolean,
  "explanation": "Brief, analytical justification for the decision."
}}
"""

# Prompt to identify and locate the primary source of the claim
source_prompt = """
### Role
Linguistic Analyst specializing in intent detection and source verification.

### Context
- User's Response: "{user_answer}"
- Previously Known Source: "{claim_source}"
- Current Primary Source Status: "{primary_source}"

### Task
Extract the source identity and the location of the claim from the user's response.

### Extraction Rules
1. **claim_source**: Update the person or organization if the user provide additional or corrected information. If not known, mention "unknown".
2. **primary_source**: Update the status if the user provide additional or corrected information.

### Output (JSON)
{{
  "claim_source": A specific Person or Organisation or "unknown",
  "primary_source": boolean,
}}
"""

# Generate 3 queries to find the primary source of the claim
source_queries_prompt = """
### Role
Neutral Fact-Checking Analyst specializing in forensic source tracing.

### Objective
Generate 3 distinct search queries designed to locate the primary or original source of the claim (e.g., the foundational statement, official document, or first instance of publication).

### Context
<History>{messages}</History>

- Original Claim: {claim}
- Known Source: "{claim_source}"
- Source Description: "{claim_description}"
- Current Summary: "{summary}"

### Query Generation Strategy
1. **The Attribution Query**: Combine the core claim with the alleged author/organization to find the direct quote or official press release.
2. **The Forensic Query**: Use specific identifiers (dates, unique keywords, or event names) to bypass secondary news reports and find the original platform (e.g., a specific social media thread or archive).
3. **The Alternative Framing Query**: Use synonyms or broader categories for the event/statistic to find official database entries or earlier versions of the claim.

### Constraints
- Do NOT invent sources or URLs.
- Each query must be unique and prioritize primary documentation over news summaries.
- Output exactly 3 queries with their respective logical reasoning.

### Output (JSON)
{{
    "queries": [
    {{
      "query": "The concrete search string used",
      "reasoning": "1-2 sentences on why this query was chosen."
    }}
  ],
  "confirmed": boolean
}}
"""

# Synthesize the search results to evaluate claim coverage
eval_search_prompt = """
### Role
Neutral Fact-Checking Analyst. Your goal is to evaluate search evidence with objective rigor and identify logical gaps in the current evidence base.

### Objective
Synthesize the provided search results to determine how much of the original claim is supported and what specific facets remain unverified.

### Context
<History>{messages}</History>

- Original Claim: {claim}
- Known Source: "{claim_source}"
- Source Description: "{claim_description}"
- Current Summary: "{summary}"

### Search Evidence (Traces)
{evaluation_text}

### Analytical Tasks
1. **Facet Matching**: Cross-reference the "who, what, where, and when" of the claim against the snippets.
2. **Evidence Synthesis**: Summarize the consensus or contradictions found in the search results.
3. **Gap Analysis**: Explicitly list facets of the claim that are NOT covered (e.g., missing official data, lack of primary sources, or timeframe mismatches).

### Strict Output Requirements (JSON)
- **Format**: Respond ONLY with a valid JSON object following the `SearchSynthesis` schema.
- **Stability Rule**: DO NOT use apostrophes ('), quotation marks ("), or curly braces ({{}}) inside text fields. Use plain, neutral descriptions only.
- **Constraint**: If no evidence is found, set `coverage_score` to 1 and list all facets as missing.

{{
  "overall_summary": "Neutral synthesis of found evidence. Avoid all special characters.",
  "missing_info": ["Specific unverified detail 1", "Specific unverified detail 2"],
  "coverage_score": (1-10 rating of evidence strength)
}}
"""

# Select the primary source
select_primary_source_prompt = """
### Role
Linguistic Analyst specializing in intent detection.

### Context
<History>{messages}</History>

- User's Response: "{user_answer}"
- claim Source: "{claim_source}"

### Task
Analyze the user's response to determine if they have identified a primary source from the search results or if they wish to keep looking.

### Decision Rules
0. **Explicit Rejection (Highest Priority)**  
   If the user explicitly says **no**, **none**, **not any**, **neither**, **I don’t see it**, **none of these**, or similar negative language:
   - Set `primary_source` to **false**
   - Do NOT select any source
   - Do NOT invent or infer a source
   - Keep `claim_source` unchanged unless the user provides a new one explicitly

1. **Identify Selection**  
   Detect if the user picks a specific result (e.g., "the first one," "number 3," "the BBC link," "that official report").
   - If a selection is made:
     - Update `claim_source` and `claim_url` with that specific information
     - Set `primary_source` to **true**

2. **Handle Non-Selection / New Info**  
   If the user provides new information but does NOT confirm it as the primary source, or if they ask to continue searching:
   - Set `primary_source` to **false**
   - Update `claim_source` only if new source information is provided

3. **Defaults (No Clear Signal)**  
   If no choice is made and no new data is provided:
   - Retain all values from the current state
   
### Guidelines
- **Precision**: Only set `primary_source: true` if the user expresses confidence that the source is the original or official origin.
- **Integrity**: Do not invent URLs. Use empty strings "" if a value is not explicitly identified.
- Maintain a neutral and analytical tone.

### Output (JSON)
{{
  "primary_source": boolean,
  "claim_source": A specific Person or Organisation or "unknown",
}}
"""

# Generate 3 search queries to find information to falsify or verify the claim
search_queries_prompt = """
### Role
Neutral Fact-Checking Analyst specializing in verification strategy and forensic research.

### Objective
Generate 3 distinct, high-leverage research queries designed to either verify or falsify the core factual assertions of the claim.

### Context
<History>{messages}</History>

- Gaps in claim: {alerts}
- Claim Summary: "{summary}"
- Claim: {claim}
- Source: "{claim_source}"

### Research Strategy
1. **The Statistical/Factual Check**: Target the specific numbers, dates, or events using authoritative databases (e.g., government reports, NGO data).
2. **The Context/Nuance Check**: Search for the broader event or statement to see if the claim is taken out of context or missing critical qualifiers.
3. **The Counter-Evidence Check**: Phrase queries neutrally or slightly toward the opposite of the claim to find potential rebuttals or corrections.

### Guidelines
- **Address Alerts**: Prioritize queries that fill the gaps identified in the "{alerts}" section.
- **Avoid Echo Chambers**: Do not use the claim's exact wording; use objective, investigative language.
- **Diversity of Angle**: Each query must target a different "facet" (e.g., one for the person involved, one for the statistic, one for the location).

### Constraints
- Do NOT invent sources or URLs.
- Output exactly 3 queries with their respective logical reasoning.

### Output (JSON)
{{
    "queries": [
    {{
      "query": "The concrete search string used",
      "reasoning": "1-2 sentences on why this query was chosen."
    }}
  ],
  "confirmed": boolean
}}
"""

# Ask the user if they want to search one more time
iterate_search_prompt = """
### Role
Linguistic Analyst specializing in intent detection and conversational flow control.

### Context
<History>{messages}</History>

- User's Latest Input: "{user_answer}"

### Objective
Determine if the user's intent is to continue the investigation with more searching or to conclude the current research phase and proceed to the next step.

### Intent Classification Rules
1. **Set "confirmed": true** if the user indicates a desire to keep looking. 
   - Keywords/Intent: "Yes," "Keep going," "Try another query," "Search again," "Maybe look for [topic]," "Not enough info yet."
2. **Set "confirmed": false** if the user indicates they are finished, satisfied, or want to move on.
   - Keywords/Intent: "No," "Stop," "Proceed," "Show results," "That's fine," "I'm done," "Next step."
3. **Handle Ambiguity**:
   - If the user provides a *new search query* or *additional facts* without saying yes/no, assume **"confirmed": true**.
   - If the user is non-committal or asks "What else is there?", default to **"confirmed": true** to ensure research thoroughness.

### Guidelines
- Focus strictly on the *direction* of the workflow.
- Do not evaluate the truth of the conversation; only evaluate the user's "go/stop" signal.
- Base your decision on the semantic meaning of "{user_answer}" within the context of "{messages}".

### Output (JSON)
{{
  "confirmed": boolean
}}
"""