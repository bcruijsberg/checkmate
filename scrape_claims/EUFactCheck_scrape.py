#!/usr/bin/env python3
"""
EUFactCheck scraper (2019–2025)
Collects post URL, rating, cleaned title, and year.

Output: data/eufactcheck_posts_2019_2025.csv
Columns: url, title, rating, year
"""

import csv
import os
import re
import sys
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from urllib import robotparser

# -------- Settings --------
BASE = "https://eufactcheck.eu/"
YEARS = list(range(2019, 2026))
OUTFILE = os.path.join("data", "eufactcheck_posts_2019_2025.csv")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EUFactCheckScraper/1.7; +https://example.org/bot)"
}

RATINGS = ["True", "Mostly True", "False", "Mostly False", "Uncheckable"]

# -------- Helpers --------
def allowed_by_robots(url: str) -> bool:
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(BASE, "robots.txt"))
    try:
        rp.read()
    except Exception:
        return True
    return rp.can_fetch(HEADERS["User-Agent"], url)

def get_soup(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException:
        return None

def is_archive_url(href: str) -> bool:
    """Skip pure archive links like /YYYY/, /YYYY/MM/, /YYYY/MM/DD/"""
    try:
        path = urlparse(href).path.strip("/")
    except Exception:
        return True
    parts = [p for p in path.split("/") if p]
    if not parts:
        return True
    if len(parts) in {1, 2, 3} and all(p.isdigit() for p in parts):
        return True
    return False

def is_post_url(href: str) -> bool:
    return bool(href and href.startswith(BASE) and not is_archive_url(href))

def detect_rating(title: str) -> str:
    """Detect if title starts with a known rating keyword; return empty if none."""
    t = title.strip()
    for rating in RATINGS:
        if re.match(rf"^{rating}\b[:\-–]?", t, flags=re.IGNORECASE):
            return rating.upper()
    return ""  # empty cell = NA

def clean_title(title: str) -> str:
    """
    Remove any prefix like 'True:' or 'Blog:' from the start of the title.
    Keeps the part after the first colon.
    """
    title = title.strip()
    cleaned = re.sub(r"^[^:]+:\s*", "", title)
    return cleaned.strip()

def extract_title_anchor(article: BeautifulSoup):
    """Within an archive card, grab the anchor that holds the real post title."""
    for sel in ["h1 a", "h2 a", "h3 a", ".entry-title a"]:
        a = article.select_one(sel)
        if a and a.get_text(strip=True) and a.get("href"):
            href = a["href"]
            if is_post_url(href):
                return a
    return None

def iter_archive_pages(year: int):
    page = 1
    while True:
        yield urljoin(BASE, f"{year}/" if page == 1 else f"{year}/page/{page}/")
        page += 1

# -------- Core scraping --------
def scrape_year(year: int, seen: set[str]):
    print(f"=== Year {year} ===", file=sys.stderr)
    for archive_url in iter_archive_pages(year):
        if not allowed_by_robots(archive_url):
            print(f"[robots] skip {archive_url}", file=sys.stderr)
            break

        soup = get_soup(archive_url)
        if not soup:
            print(f"[end] no page {archive_url}", file=sys.stderr)
            break

        articles = soup.find_all("article")
        if not articles:
            print(f"[end] no articles on {archive_url}", file=sys.stderr)
            break

        count = 0
        for art in articles:
            a = extract_title_anchor(art)
            if not a:
                continue

            href = a["href"].strip()
            title = a.get_text(strip=True)
            if not (href and title) or href in seen or not is_post_url(href):
                continue

            rating = detect_rating(title)
            title_cleaned = clean_title(title)

            seen.add(href)
            count += 1

            yield (href, title_cleaned, rating, year)

        print(f"[archive] {archive_url} -> {count} posts", file=sys.stderr)
        time.sleep(0.6)
        if count == 0:
            break

def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    seen = set()
    rows = 0
    with open(OUTFILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "title", "rating", "year"])
        for year in YEARS:
            for rec in scrape_year(year, seen):
                writer.writerow(rec)
                rows += 1
    print(f"Done. Wrote {rows} rows to {OUTFILE}")

if __name__ == "__main__":
    main()
