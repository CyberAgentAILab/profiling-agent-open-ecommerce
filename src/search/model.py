import json
import time

import requests
from ddgs import DDGS
from diskcache import Cache
from loguru import logger
from tqdm import tqdm

from common.io import save_as_json


class WebRetriever:
    def __init__(
        self,
        search_engine: str,
        region: str,
        backend: list[str],
        num_search: int,
        serp_url: str | None,
        serp_api_token: str | None,
        serper_url: str | None,
        serper_api_token: str | None,
        timeout: int = 60,
        search_cache_dir: str = "./search_cache",
        query_suffix: str = "",
    ):
        if search_engine == "serp":
            if serp_url is not None and serp_api_token is not None:
                logger.info("🟣 activate serp API")
                self.serp_url = serp_url
                self.serp_api_token = serp_api_token
                self.timeout = timeout
                self.searcher = self.serp_search
            else:
                logger.info("🟣 SERP_API_TOKEN is not set. Falling back to the DDGS (DuckDuckGo) scraper")
                self.searcher = self.ddgs_search
        elif search_engine == "serper":
            if serper_url is not None and serper_api_token is not None:
                logger.info("🟣 activate serper API")
                self.serper_url = serper_url
                self.serper_api_token = serper_api_token
                self.timeout = timeout
                self.searcher = self.serper_search
            else:
                logger.info("🟣 SERPER_API_TOKEN is not set. Falling back to the DDGS (DuckDuckGo) scraper")
                self.searcher = self.ddgs_search
        else:
            logger.info("🟣 activate DDGS (DuckDuckGo) scraper")
            self.searcher = self.ddgs_search

        self.region = region
        self.backend = backend
        self.num_search = num_search
        self.query_suffix = query_suffix
        if query_suffix:
            logger.info(f"🔖 query_suffix enabled: '{query_suffix}'")

        # Try to initialize the cache; disable caching on failure
        try:
            self.search_cache = Cache(
                directory=search_cache_dir,
                timeout=3600,
            )
            self.use_cache = True
            logger.info(f"✅ Search cache initialized: {search_cache_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize search cache: {e}")
            logger.info("🔍 Running without cache")
            self.search_cache = None
            self.use_cache = False

        return

    def serper_search(self, query: str) -> list[dict[str, str]]:
        try:
            payload = json.dumps({"q": query, "location": "United States", "gl": "us"})
            headers = {"X-API-KEY": self.serper_api_token, "Content-Type": "application/json"}
            response = requests.request("POST", self.serper_url, headers=headers, data=payload, timeout=self.timeout)
            response.raise_for_status()
            response_json = json.loads(response.text)
            if not isinstance(response_json, dict):
                logger.error(f"didn't parse the API response correctly: {response_json}")
                return []
            organic_results = response_json.get("organic", [])

            search_contexts = []
            for i, result in enumerate(organic_results, 1):
                search_context = ""
                title = result.get("title", "No title")
                body = result.get("snippet", "")
                href = result.get("link", "")

                search_context += f"### Search result {i}. {title}\n"
                search_context += f"   {body}\n"
                search_context += f"   URL: {href}\n\n"

                search_contexts.append(
                    {
                        "retrieval_context": search_context,
                        "title": title,
                        "body": body,
                        "url": href,
                    }
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Serper API request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse serper response as JSON: {e}")
            return []
        return search_contexts

    def serp_search(self, query: str) -> list[dict[str, str]]:
        try:
            params = {"engine": "google", "q": query, "api_key": self.serp_api_token}
            response = requests.get(self.serp_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            response_json = json.loads(response.text)
            if not isinstance(response_json, dict):
                logger.error(f"didn't parse the API response correctly: {response_json}")
                return []
            organic_results = response_json.get("organic_results", [])

            search_contexts = []
            for i, result in enumerate(organic_results, 1):
                search_context = ""
                title = result.get("title", "No title")
                body = result.get("snippet", "")
                href = result.get("link", "")

                search_context += f"### Search result {i}. {title}\n"
                search_context += f"   {body}\n"
                search_context += f"   URL: {href}\n\n"

                search_contexts.append(
                    {
                        "retrieval_context": search_context,
                        "title": title,
                        "body": body,
                        "url": href,
                    }
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Serp API request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse serp response as JSON: {e}")
            return []
        return search_contexts

    def ddgs_search(self, query: str) -> list[dict[str, str]]:
        logger.debug(query)
        self.search_engine = DDGS()
        try:
            results = list(
                self.search_engine.text(
                    query,
                    max_results=self.num_search,
                    region=self.region,
                    backend=self.backend,
                )
            )

            if not results:
                logger.debug(f"No hits at {query}")
                return []

            search_contexts = []
            for i, result in enumerate(results, 1):
                search_context = ""
                title = result.get("title", "No title")
                body = result.get("body", "")
                href = result.get("href", "")

                search_context += f"### Search result {i}. {title}\n"
                search_context += f"   {body}\n"
                search_context += f"   URL: {href}\n\n"

                search_contexts.append(
                    {
                        "retrieval_context": search_context,
                        "title": title,
                        "body": body,
                        "url": href,
                    }
                )
            return search_contexts

        except Exception as e:
            logger.info(f"❌ Couldn't get search results: {e}")
            return []

    def check_and_get_cache(self, query: str) -> tuple[bool, list[dict[str, str]] | None]:
        if not self.use_cache or self.search_cache is None:
            logger.debug(f"➡️ cache disabled: {query}. use search API")
            return False, None

        if query in self.search_cache:
            logger.info(f"✅ cache hit: {query}")
            return True, self.search_cache[query]
        else:
            logger.info(f"➡️ cache didn't hit: {query}. use search API")
            return False, None

    def save_to_cache(self, query: str, result: list[dict[str, str]]) -> None:
        if not self.use_cache or self.search_cache is None:
            logger.debug(f"💾 cache disabled: skipping save for {query}")
            return

        if result:
            try:
                self.search_cache[query] = result
                logger.debug(f"💾 saved to cache: {query}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save to cache: {e}")
                # Disable caching after a cache error
                self.use_cache = False
                self.search_cache = None

    def run_search(
        self,
        search_list: list[dict],
        out_dir: str,
        timestamp: str,
        job_id: str | None = None,
        num_debug: int | None = None,
    ) -> str:
        search_contents: list[dict] = []
        # Initial save keeps save_path defined (and the output file present)
        # even when search_list is empty; each step below overwrites it as a checkpoint.
        save_path = save_as_json(
            responses=search_contents,
            out_dir=out_dir,
            timestamp=timestamp,
            job_id=job_id,
        )

        num_api_usage = 0
        pbar = tqdm(enumerate(search_list), total=len(search_list))
        for step, need_search in pbar:
            # Debug mode: stop the search loop once the given count is reached
            if num_debug is not None and step >= num_debug:
                logger.info(f"🔍 Debug mode: Breaking search loop at step {step} (num_debug={num_debug})")
                break
            duration = 0.0

            # Only executes search when need_search=True
            if need_search["scan"]["need_search"]:
                query = need_search["resolved_query"] + self.query_suffix

                cache_hit, cached_result = self.check_and_get_cache(query)
                if cache_hit and cached_result is not None:
                    search_result = cached_result
                else:
                    num_api_usage += 1
                    begin = time.perf_counter()
                    search_result = self.searcher(query=query)
                    duration = time.perf_counter() - begin
                    self.save_to_cache(query, search_result)

                if len(search_result) < 1:
                    search_content = {
                        "searched": True,
                        "success_search": False,
                        "search_result": search_result,
                        "search_duration": duration,
                    }
                else:
                    search_content = {
                        "searched": True,
                        "success_search": True,
                        "search_result": search_result,
                        "search_duration": duration,
                    }
            else:
                search_content = {
                    "searched": False,
                    "success_search": False,
                    "search_result": [],
                    "search_duration": duration,
                }

            need_search.update({"search": search_content})
            search_contents.append(need_search)

            save_path = save_as_json(
                responses=search_contents,
                out_dir=out_dir,
                timestamp=timestamp,
                job_id=job_id,
            )
            logger.debug(f"saved timestamp: {timestamp}, job_id: {job_id}")
            logger.debug(f"{num_api_usage} search API called")

        logger.info(f"💾 search results: {len(search_contents)} records -> {save_path}")
        logger.info(f"🔍 {num_api_usage} search API calls in total")
        return save_path
