import asyncio
import logging
import sys
import json
import os
import time
from typing import Any
import aiohttp
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pathlib import Path

from datetime import datetime
from mcp.server.fastmcp import FastMCP

try:
    from ddgs_fork import DDGS
    from ddgs_fork.exceptions import RatelimitException, TimeoutException
    DDGS_IMPORT_ERROR = None
except Exception as e:
    DDGS = None
    DDGS_IMPORT_ERROR = str(e)

    class RatelimitException(Exception):
        pass

    class TimeoutException(Exception):
        pass

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./mcp_server.log'),
        logging.StreamHandler(sys.stderr)  # Send logs to stderr to avoid interfering with stdio
    ]
)

logger = logging.getLogger("websearch-scraper")


def _load_config() -> dict:
    """Load search configuration from app_config.json (single-config)."""
    default_search_config = {
        "searxng": {
            "search_url": "http://127.0.0.1:5002/search",
            "health_url": "http://127.0.0.1:5002",
            "health_check_timeout": 2.0,
        },
        "backend_cache_ttl_seconds": 20,
    }

    app_config_path = Path(
        os.environ.get(
            "APP_CONFIG_PATH",
            Path(__file__).resolve().parent.parent / "app_config.json",
        )
    )
    try:
        if app_config_path.exists():
            with open(app_config_path, "r") as f:
                app_config = json.load(f)
                search_config = app_config.get("search")
                if isinstance(search_config, dict):
                    logger.info(f"✅ Search configuration loaded from {app_config_path}")
                    return search_config
                logger.warning(f"⚠️ Missing 'search' key in {app_config_path}; using default values")
    except Exception as e:
        logger.warning(f"⚠️ Error loading {app_config_path}: {e}")

    logger.info("ℹ️ Using default search configuration")
    return default_search_config


# Load configuration
_config = _load_config()

SEARXNG_SEARCH_URL = os.environ.get("SEARXNG_SEARCH_URL", _config["searxng"]["search_url"])
SEARXNG_HEALTH_URL = os.environ.get("SEARXNG_HEALTH_URL", _config["searxng"]["health_url"])
SEARXNG_HEALTH_TIMEOUT = float(os.environ.get("SEARXNG_HEALTH_TIMEOUT", _config["searxng"]["health_check_timeout"]))
BACKEND_CACHE_TTL_SECONDS = int(os.environ.get("SEARCH_BACKEND_CACHE_TTL", _config["backend_cache_ttl_seconds"]))
_availability_cache = {
    "ts": 0.0,
    "data": None,
}



# Create FastMCP instance
mcp = FastMCP("websearch-scraper")

# Usage statistics
stats = {
    "total_requests": 0,
    "successful_searches": 0,
    "failed_searches": 0,
    "successful_fetches": 0,
    "failed_fetches": 0,
    "filter_calls": 0,
    "start_time": datetime.now().isoformat()
}

def log_stats():
    """Log usage statistics."""
    logger.info(f"=== Server statistics ===")
    logger.info(f"Started at: {stats['start_time']}")
    logger.info(f"Total requests: {stats['total_requests']}")
    logger.info(f"Successful searches: {stats['successful_searches']}")
    logger.info(f"Failed searches: {stats['failed_searches']}")
    logger.info(f"Successful URL fetches: {stats['successful_fetches']}")
    logger.info(f"Failed URL fetches: {stats['failed_fetches']}")
    logger.info(f"URL filtering calls: {stats.get('filter_calls',0)}")
    logger.info(f"===============================")


def _get_ddgs_available() -> tuple[bool, str]:
    if DDGS is None:
        return False, f"ddgs_fork not installed ({DDGS_IMPORT_ERROR})"
    return True, "ok"


async def _get_searxng_available(timeout_seconds: float = None) -> tuple[bool, str]:
    if timeout_seconds is None:
        timeout_seconds = SEARXNG_HEALTH_TIMEOUT
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                SEARXNG_HEALTH_URL,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as response:
                if response.status < 500:
                    return True, "ok"
                return False, f"HTTP {response.status}"
    except Exception as e:
        return False, str(e)


async def _get_backend_availability(force_refresh: bool = False) -> dict:
    now = time.time()
    if not force_refresh and _availability_cache["data"] and now - _availability_cache["ts"] < BACKEND_CACHE_TTL_SECONDS:
        return _availability_cache["data"]

    searx_ok, searx_reason = await _get_searxng_available()
    ddgs_ok, ddgs_reason = _get_ddgs_available()
    data = {
        "searxng": {"available": searx_ok, "reason": searx_reason},
        "ddgs": {"available": ddgs_ok, "reason": ddgs_reason},
    }
    _availability_cache["ts"] = now
    _availability_cache["data"] = data
    return data


def _normalize_search_result(result: dict, backend: str) -> dict:
    if backend == "searxng":
        href = str(result.get("url", ""))
        title = str(result.get("title", "Untitled"))
        description = str(result.get("content", "No description")).strip(',')
    else:
        href = str(result.get("href") or result.get("url") or "")
        title = str(result.get("title") or "Untitled")
        description = str(result.get("body") or result.get("content") or "No description").strip(',')

    return {
        "title": title,
        "href": href,
        "description": description,
    }


def _filter_accessible_results(results: list[dict], backend: str, max_results: int) -> list[dict]:
    formatted_results = []
    for result in results:
        normalized = _normalize_search_result(result, backend)
        try:
            test = requests.get(normalized["href"], timeout=1)
            logger.info(f"⚠️⚠️⚠️ {normalized['href']} : {test.status_code}")
            if test.status_code == 200:
                formatted_results.append(normalized)
            if len(formatted_results) >= max_results:
                break
        except Exception as e:
            logger.warning(f"⚠️ url not accessible: '{e}'")
    return formatted_results


async def _search_with_searxng(query: str) -> list[dict]:
    search_payload = {"q": query, "format": "json"}
    response = requests.get(SEARXNG_SEARCH_URL, params=search_payload, timeout=8)
    response.raise_for_status()
    payload = response.json()
    return payload.get("results", [])


async def _search_with_ddgs(query: str, max_results: int) -> list[dict]:
    if DDGS is None:
        raise RuntimeError(f"ddgs_fork not installed ({DDGS_IMPORT_ERROR})")
    ddgs = DDGS()
    return list(
        ddgs.text(
            query,
            max_results=max_results,
            backend="duckduckgo, brave, google, bing",
            safesearch="off",
        )
    )


async def _resolve_backend(requested_backend: str) -> tuple[str, dict]:
    backend = (requested_backend or "auto").lower()
    if backend not in ("auto", "searxng", "ddgs"):
        raise ValueError("backend must be 'auto', 'searxng', or 'ddgs'")

    availability = await _get_backend_availability()
    if backend == "searxng":
        if availability["searxng"]["available"]:
            return "searxng", availability
        raise RuntimeError(f"searxng backend unavailable: {availability['searxng']['reason']}")
    if backend == "ddgs":
        if availability["ddgs"]["available"]:
            return "ddgs", availability
        raise RuntimeError(f"ddgs backend unavailable: {availability['ddgs']['reason']}")

    if availability["searxng"]["available"]:
        return "searxng", availability
    if availability["ddgs"]["available"]:
        return "ddgs", availability
    raise RuntimeError(
        "no backend available: searxng unavailable and ddgs not installed"
    )

@mcp.tool()
async def search_web(query: str, max_results: int = 5, backend: str = "auto") -> str:
    """Search for information on the web.
    
    Args:
        query: Search query
        max_results: Maximum number of results (default: 5)
        backend: Search backend ("auto", "searxng", or "ddgs")
    
    Returns:
        Search results as a JSON string
    """
    stats["total_requests"] += 1
    
    logger.info(f"🔧 Tool call received: search_web")
    logger.info(f"📥 Arguments: query='{query}', max_results={max_results}, backend='{backend}'")
    logger.info(f"🔎 Web search requested: '{query}' (max: {max_results} results, backend: {backend})")
    
    # Execute search logic
    results = await perform_web_search(query, max_results, backend)
    
    logger.info(f"✅ Search completed for '{query}'")
    log_stats()
    
    return str(results)

async def perform_web_search(query: str, max_results: int, backend: str = "auto") -> str:
    """Perform search and return formatted JSON results."""
    start_time = datetime.now()
    
    try:
        selected_backend, availability = await _resolve_backend(backend)
        logger.info(f"⏳ Starting search with backend '{selected_backend}' for: '{query}'")

        if selected_backend == "searxng":
            search_results = await _search_with_searxng(query)
        else:
            search_results = await _search_with_ddgs(query, max_results)

        if not search_results:
            logger.warning(f"⚠️ No results found for: '{query}'")
            stats["failed_searches"] += 1
            # Return a structured error so callers can distinguish failure from empty-success
            err = {
                "error": "no_results",
                "message": f"No results found for query: {query}",
                "query": query,
                "backend": selected_backend,
                "raw_count": 0,
            }
            return json.dumps(err, ensure_ascii=False)
        
        # Format as a structured list
        formatted_results = _filter_accessible_results(search_results, selected_backend, max_results)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        if not formatted_results:
            # If raw search returned hits but none were accessible/usable, consider it a failure
            logger.error(f"❌ No accessible URL after validation for: '{query}' (raw hits: {len(search_results)})")
            stats["failed_searches"] += 1
            sample_hrefs = [r.get('href') or r.get('url') for r in search_results[:5]]
            err = {
                "error": "no_accessible_urls",
                "message": f"No accessible URL after validation for query: {query}",
                "query": query,
                "backend": selected_backend,
                "raw_count": len(search_results),
                "sample_hrefs": sample_hrefs,
            }
            return json.dumps(err, ensure_ascii=False)

        logger.info(f"✅ {len(formatted_results)} results found in {elapsed_time:.2f}s")
        stats["successful_searches"] += 1
        logger.info(f"✅ Results: {formatted_results}")
        return json.dumps(formatted_results, ensure_ascii=False)

    except RatelimitException as e:
        logger.error(f"🚫 Rate limit reached for '{query}': {e}")
        stats["failed_searches"] += 1
        return json.dumps(
            {
                "error": "rate_limit",
                "message": "Rate limit reached. Please try again later.",
                "query": query,
            },
            ensure_ascii=False,
        )
    except TimeoutException as e:
        logger.error(f"⏱️ Timeout while searching '{query}': {e}")
        stats["failed_searches"] += 1
        return json.dumps(
            {
                "error": "timeout",
                "message": "Search timeout.",
                "query": query,
            },
            ensure_ascii=False,
        )
    except ValueError as e:
        logger.error(f"❌ Invalid parameter: {str(e)}")
        stats["failed_searches"] += 1
        return json.dumps(
            {
                "error": "invalid_backend",
                "message": str(e),
                "query": query,
            },
            ensure_ascii=False,
        )
    except RuntimeError as e:
        logger.error(f"❌ Backend unavailable for '{query}': {str(e)}")
        stats["failed_searches"] += 1
        availability = await _get_backend_availability(force_refresh=True)
        return json.dumps(
            {
                "error": "backend_unavailable",
                "message": str(e),
                "query": query,
                "backend_status": availability,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"❌ Error while searching '{query}': {str(e)}", exc_info=True)
        stats["failed_searches"] += 1
        return json.dumps(
            {
                "error": "search_error",
                "message": f"Search error: {str(e)}",
                "query": query,
            },
            ensure_ascii=False,
        )

@mcp.tool()
async def fetch_url_content(url: str, max_length: int = 10000) -> str:
    """Fetch and extract textual content from a URL.
    
    Args:
        url: URL of the web page to fetch
        max_length: Maximum content length to return (default: 10000 chars)
    
    Returns:
        Web page text content
    """
    stats["total_requests"] += 1
    
    logger.info(f"🔧 Tool call received: fetch_url_content")
    logger.info(f"📥 Arguments: url='{url}', max_length={max_length}")
    logger.info(f"🌐 Content fetch requested for: '{url}'")
    
    start_time = datetime.now()
    
    try:
        # Fetch page content
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, 
                timeout=aiohttp.ClientTimeout(total=10),
                headers={'User-Agent': 'Mozilla/5.0 (compatible; MCPBot/1.0)'}
            ) as response:
                if response.status != 200:
                    logger.error(f"❌ HTTP error {response.status} for '{url}'")
                    stats["failed_fetches"] += 1
                    return f"Error: HTTP {response.status}"
                
                html_content = await response.text()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove scripts and styles
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean text (remove multiple empty lines)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)
        
        # Trim length
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length] + "\n\n[... Content truncated ...]"
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Content fetched ({len(clean_text)} chars) in {elapsed_time:.2f}s")
        stats["successful_fetches"] += 1
        log_stats()
        
        # Return JSON result
        result = {
            "url": url,
            "content": clean_text,
            "length": len(clean_text),
            "truncated": len(clean_text) >= max_length
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except aiohttp.ClientError as e:
        logger.error(f"❌ Network error for '{url}': {e}")
        stats["failed_fetches"] += 1
        return f"Network error: {str(e)}"
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Timeout while fetching '{url}'")
        stats["failed_fetches"] += 1
        return "Error: Timeout while fetching page"
    except Exception as e:
        logger.error(f"❌ Error while fetching '{url}': {str(e)}", exc_info=True)
        stats["failed_fetches"] += 1
        return f"Error: {str(e)}"


@mcp.tool()
async def filter_search_results(search_results: any, query: str, top_k: int = 3) -> str:
    """Filter and rank search_web results (JSON string with list of dicts).
    Return the top_k most relevant results for the query."""
    if isinstance(search_results, str):
        print("search_results is a string")
        search_results = search_results.replace("'", '"')
    elif isinstance(search_results, list):
        print("search_results is a list")
        search_results = str(search_results).replace("'", '"')
    elif isinstance(search_results, dict):
        print("search_results is a dict")
        search_results = search_results['input_text']
        print(search_results)
    else: 
        print(f"search_results is a {type(search_results)}")
    stats["total_requests"] += 1
    stats["filter_calls"] = stats.get("filter_calls", 0) + 1
    logger.info(f"🔧 Appel d'outil reçu: filter_search_results (top_k={top_k})")
    #search_results = search_results.replace('"', "'")
    try:
        results = json.loads(search_results)
        if not isinstance(results, list):
            logger.warning("filter_search_results: input is not a list")
            return json.dumps([])
    except Exception as e:
        logger.error(f"filter_search_results: invalid input: {e}")
        return json.dumps([])

    # Simple scoring: keyword relevance and blacklist
    qtokens = [t.lower() for t in query.split() if len(t) > 2]
    blacklist = ("facebook.com", "twitter.com", "instagram.com", "youtube.com", "linkedin.com")

    def score_item(item):
        href = item.get('href', '')
        title = (item.get('title') or '').lower()
        desc = (item.get('description') or '').lower()
        from urllib.parse import urlparse
        parsed = urlparse(href)
        domain = parsed.netloc.lower()
        score = 0
        if parsed.scheme == 'https':
            score += 1
        for b in blacklist:
            if b in domain:
                score -= 5
        for t in qtokens:
            if t in title:
                score += 2
            if t in desc:
                score += 1
            if t in href.lower():
                score += 1
        return score

    scored = [(score_item(it), it) for it in results]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [it for score, it in scored[:top_k]]
    logger.info(f"filter_search_results: returning {len(top)} results")
    return json.dumps(top, ensure_ascii=False)

def main():
    """Run the server."""
    logger.info("=" * 60)
    logger.info("🚀 Starting MCP WebSearch server")
    logger.info(f"📅 Date: {datetime.now().isoformat()}")
    logger.info(f"🔧 Server name: websearch-scraper")
    logger.info("=" * 60)
    logger.info(f"🌐 SearXNG configuration:")
    logger.info(f"   Search URL: {SEARXNG_SEARCH_URL}")
    logger.info(f"   Health URL: {SEARXNG_HEALTH_URL}")
    logger.info(f"   Health check timeout: {SEARXNG_HEALTH_TIMEOUT}s")
    logger.info(f"⏱️  Backend cache TTL: {BACKEND_CACHE_TTL_SECONDS}s")
    logger.info("=" * 60)
    
    try:
        logger.info("✅ Server initialized and waiting for connections...")
        # mcp.run() manages its own event loop; no async wrapper needed.
        mcp.run()
    except Exception as e:
        logger.error(f"💥 Fatal server error: {e}", exc_info=True)
        raise
    finally:
        logger.info("=" * 60)
        logger.info("🛑 Stopping MCP WebSearch server")
        log_stats()
        logger.info("=" * 60)

if __name__ == "__main__":
    main()  # Direct call, no asyncio.run()