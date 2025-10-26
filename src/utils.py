from newspaper import Article

# ──────────────────────────────
# HELPER FUNCTION TO FETCH FULL ARTICLE CONTENT
# ──────────────────────────────
def fetch_full_article(url) -> tuple[str, bool]:
    """
    Fetch and return the full text of an article from a given URL.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text, True
    except Exception as e:
        return f"[Failed to fetch article content from {url}: {e}]", False

#set a limit to the number of characters from the full article
max_chars = 1200

# ──────────────────────────────
# HELPER FUNCTION TO COMBINE SUMMARY AND FULL ARTICLE CONTENT, AND SEPERATE URLS
# ──────────────────────────────
def format_docs(docs) -> str:
    """
    Format the found documents to format
    """
    formatted = []
    for i, d in enumerate(docs, start=0):
        summary = (d.page_content or "").strip()
        url = d.metadata.get("url", "")
        full_content, page_exsists = fetch_full_article(url)
        
        #rerturn content only if the page exsists 
        if page_exsists:
            if len(full_content) > max_chars:
                full_content = full_content[:max_chars] + "…"
            combined = f"""
            [{i}] SUMMARY:
            {summary}

            [{i}] FULL ARTICLE CONTENT:
            {full_content}
            """
            formatted.append(combined)
        else:
            formatted.append(f"this url: {url} does not exist, don't use it in your answer")
    return "\n\n---\n\n".join(formatted)
