"""URL reader — stub awaiting implementation.

Intended implementation: use requests to fetch the URL, then strip HTML with
BeautifulSoup (or html2text) to produce clean plain text. Handle redirects,
timeouts, and non-HTML content types gracefully.
"""

from __future__ import annotations


class URLReader:
    """Fetches a URL and returns its plain-text content."""

    def supports(self, source: str) -> bool:
        return source.lower().startswith(("http://", "https://"))

    def read(self, source: str) -> str:
        raise NotImplementedError(
            "URLReader is not yet implemented. "
            "Use requests.get(source, timeout=30) to fetch the page, "
            "then parse the HTML with BeautifulSoup and extract visible text."
        )
