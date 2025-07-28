class LangSearchAPIHandler:
    def __init__(self):
        self.api_key = os.getenv("LANGSEARCH_API_KEY")
        self.search_url = os.getenv("LANGSEARCH_BASE_URL","https://api.langsearch.com/v1/web-search")
        logger.info("LangSearchAPIHandler initialized.")

    def load_keywords_from_file(self, filepath: str = "relevant_keywords.txt") -> list[str]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                keywords = [kw.strip() for kw in content.split(",") if kw.strip()]
            return keywords
        except FileNotFoundError:
            logger.error(f"Keyword file not found: {filepath}")
            return []

    # Define the web_search function
    def web_search(self, query: str, required_keywords: list[str] = []) -> tuple[str, str, list[str]] | None:
        try:
            # Define get_starting_sentences function
            def get_starting_sentences(text: str, max_sentences: int = 3) -> str:
                """Fetch the first few sentences of the text."""
                sentences = re.split(r'(?<=[.!?]) +', text)
                return " ".join(sentences[:max_sentences])

            def clean_text(text: str) -> str:
                text = re.sub(r"\[.*?\]", "", text)
                text = text.replace("\n", " ")
                text = re.sub(r"\s+", " ", text)
                text = text.strip()
                return text

            def fetch_logo_from_page(url: str) -> str:
                try:
                    headers = {
                         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    # Send a request to the page
                    response = requests.get(url)
                    response.raise_for_status()

                    # Parse the page content
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Try to find the favicon from the <link rel="icon"> tag
                    favicon_url = None
                    favicon_tag = soup.find("link", rel="icon")
                    if favicon_tag:
                        favicon_url = favicon_tag.get("href")

                    # If favicon not found, try to find the logo in meta tags like og:image
                    if not favicon_url:
                        meta_logo = soup.find("meta", property="og:image")
                        if meta_logo:
                            favicon_url = meta_logo.get("content")

                    if not favicon_url:
                        meta_logo = soup.find("meta", name="logo:image")
                        if meta_logo:
                            favicon_url = meta_logo.get("content")

                    # If favicon or logo URL is not absolute, make it absolute by joining with base URL
                    if favicon_url and not favicon_url.startswith("http"):
                        favicon_url = urllib.parse.urljoin(url, favicon_url)

                    # If no logo found, return a default icon or logo
                    if not favicon_url:
                        favicon_url = "No Logo Available"

                    return favicon_url

                except Exception as e:
                    logger.error(f"Error fetching logo from {url}: {e}")
                    return "No Logo Available"

            # Set up the API request headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Define the payload for the API request
            payload = {
                "query": f"site:c-zentrix.com {query}",
                "freshness": "onLimit",
                "summary": True,
                "count": 3,
            }

            # Log the search query
            logger.info(f"Searching LangSearch with query: {payload['query']}")
            response = requests.post(self.search_url, json=payload, headers=headers)
            response.raise_for_status()
            results = response.json()

            suggestions = []
            sources = {}

            # Extract summaries and logo URLs from the results
            data = results.get("data", {})
            web_pages = data.get("webPages", {})
            search_results = web_pages.get("value", [])

            for item in search_results:
                summary = item.get("summary", "")
                page_url = item.get("url", "")
                logo_url = item.get("logo", None)

                if not logo_url:
                    # Fetch logo if not present in the API result
                    logo_url = fetch_logo_from_page(page_url)

                if summary:  # If there is a summary, add it to suggestions
                    cleaned_summary = clean_text(summary)

                    preview_summary = get_starting_sentences(cleaned_summary, max_sentences=2)

                    # Avoid duplicates
                    if preview_summary not in suggestions:
                        suggestions.append(preview_summary)
                        sources[page_url] = logo_url

            if not suggestions:
                suggestions = ["No summaries available."]

            # Filter summaries based on required keywords
            matched_summaries = []
            for summary in suggestions:
                if any(keyword.lower() in summary.lower() for keyword in required_keywords):
                    matched_summaries.append(summary)

            if matched_summaries:
                preview = matched_summaries[0]
                return sources, preview, suggestions
            else:
                logger.info("No summaries match the required keywords.")
                return sources, suggestions[0], suggestions

        except Exception as e:
            logger.error(f"LangSearch API error: {e}")
            return None, None, []

def normalize(text):
        """Lowercase and remove punctuation for better matching"""
        return re.sub(r'[^\w\s]', '', text.lower().strip())

def strip_html_tags(text):
    """Remove all HTML tags from the text"""
    return re.sub(r'<[^>]+>', '', text)