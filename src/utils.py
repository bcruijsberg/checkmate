from newspaper import Article
from typing_extensions import List, Optional
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage
from urllib.parse import urlparse

# ────────────────────────────────────────────────────────────
# HELPER FUNCTION TO FETCH FULL ARTICLE CONTENT
# ────────────────────────────────────────────────────────────
def fetch_full_article(url) -> tuple[str, bool]:
    
    """ Fetch and return the full text of an article from a given URL. """

    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text, True
    except Exception as e:
        return f"[Failed to fetch article content from {url}: {e}]", False

#set a limit to the number of characters from the full article
max_chars = 1200

# ───────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION TO COMBINE SUMMARY AND FULL ARTICLE CONTENT, AND SEPERATE URLS
# ───────────────────────────────────────────────────────────────────────────────
def format_docs(docs) -> str:
    """Format the found documents using only the local vector store summary."""
    formatted = []
    
    for i, d in enumerate(docs):
        # Extract metadata from the document
        title = d.metadata.get("title", "No Title")
        summary = (d.page_content or "").strip()
        verdict = d.metadata.get("verdict", "N/A")
        url = d.metadata.get("url", "")

        # Create a clean, concise block for the LLM
        # We include the URL here so the LLM can map it to 'allowed_url'
        combined = (
            f"[{i}] SUMMARY:\n"
            f"Title: {title}\n"
            f"Content: {summary}\n"
            f"Verdict: {verdict}\n"
            f"Source URL: {url}"
        )
        formatted.append(combined)
        
    return "\n\n---\n\n".join(formatted)

# ────────────────────────────────────────────────────────────
# HELPER FUNCTION TO TAKE INPUT FROM USER
# ────────────────────────────────────────────────────────────

""" Get newest user reply """

def get_new_user_reply(messages: List[BaseMessage]) -> Optional[str]:
    last_h = -1
    last_a = -1
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_h = i
        elif isinstance(m, AIMessage):
            last_a = i
    # user spoke after assistant → real reply
    if last_h > last_a:
        return messages[last_h].content
    return None

# ────────────────────────────────────────────────────────────
# HELPER FUNCTION TO RETRIEVE DOMAIN FROM URL
# ────────────────────────────────────────────────────────────
from urllib.parse import urlparse
def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""