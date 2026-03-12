import logging
import requests
import time
import random
import openai
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel, Field
from decouple import config
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from requests.exceptions import HTTPError, Timeout

from exceptions import EmptyResponseError, ModelNotFoundException, GlobalTimeoutError, CooldownTimeoutError, UnauthorizedError, RequestFailedError

OPENROUTER_MAX_COOLDOWN = 3600  # 1 hour

load_dotenv("./service_config/files/.env")  # Load environment variables from .env file
load_dotenv("./config/models.env")
logger = logging.getLogger(__name__)

initial_models: list[str] = config("MODELS", default="").split(",")


class RateLimitedOpenAI(ChatOpenAI):
    """Extension of ChatOpenAI that handles rate limiting"""

    @retry(
        retry=retry_if_exception_type((
                HTTPError,
                Timeout,
                openai.RateLimitError,
                openai.APITimeoutError
        )),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limit hit, retrying in {retry_state.next_action.sleep} seconds (attempt {retry_state.attempt_number})"
        )
    )
    def _generate(self, *args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting RateLimitedOpenAI request (max retries: 5, max wait: 60s)")
        
        try:
            response = super()._generate(*args, **kwargs)
            total_time = time.time() - start_time
            logger.info(f"Generated response after {total_time:.1f}s total: {response}")
            return response
        except (HTTPError, openai.RateLimitError, openai.APITimeoutError) as e:
            headers = None
            status_code = None

            if isinstance(e, HTTPError) and hasattr(e, 'response'):
                headers = e.response.headers
                status_code = e.response.status_code
            elif hasattr(e, 'response'):
                headers = e.response.headers
                status_code = e.response.status_code

            if status_code == 429:
                sleep_time = None

                if headers and "X-RateLimit-Reset" in headers:
                    try:
                        reset_ts = int(headers["X-RateLimit-Reset"]) / 1000
                        current_ts = time.time()
                        sleep_time = max(0, int(reset_ts - current_ts)) + random.uniform(0.1, 1.0)
                        sleep_time = min(sleep_time, OPENROUTER_MAX_COOLDOWN)
                        logger.info(
                            f"Rate limited. Waiting {sleep_time:.1f}s using X-RateLimit-Reset (total elapsed: {time.time() - start_time:.1f}s)")
                    except (ValueError, TypeError) as ve:
                        logger.warning(f"Error parsing X-RateLimit-Reset: {ve}")

                if sleep_time is None and headers and "Retry-After" in headers:
                    try:
                        sleep_time = float(headers["Retry-After"]) + random.uniform(0.1, 1.0)
                        sleep_time = min(sleep_time, OPENROUTER_MAX_COOLDOWN)
                        logger.info(
                            f"Rate limited. Waiting {sleep_time:.1f}s using Retry-After (total elapsed: {time.time() - start_time:.1f}s)")
                    except (ValueError, TypeError) as ve:
                        logger.warning(f"Error parsing Retry-After: {ve}")

                if sleep_time:
                    time.sleep(sleep_time)
                    response_after_retry = super()._generate(*args, **kwargs)
                    total_time = time.time() - start_time
                    logger.info(f"Response after retry completed in {total_time:.1f}s total: {response_after_retry}")
                    return response_after_retry

            raise

class Models:
    """
    Collection of available models with their configurations.
    """

    models: List[str]

    def __init__(self):
        # Load from JSON
        self.models = initial_models
        self.refreshed_at = 0

    def list(self) -> List[str]:
        """
        List all available models.
        """
        return list(self.models)
    
    def list_initial(self) -> List[str]:
        """
        List initial models.
        """
        return list(initial_models)
    
    def set(self, models: List[str]):
        """
        Set the available models.
        """
        self.models = models

class Hawki2ChatModel(BaseChatModel, BaseModel):
    """
    Custom Hawki2 API client with timeout and rate limiting protection.
    """
    model: str = Field(default="gpt-4o")
    temperature: float = 0.7
    max_tokens: int = 32768
    top_p: float = 1.0
    base_backoff: float = 10.0
    connect_timeout: int = 20
    read_timeout: int = 20
    global_timeout: int = Field(default=config("GLOBAL_TIMEOUT", default=60, cast=int))
    max_cooldown: int = 30
    api_url: str = Field(default=config("HAWKI_API_URL"))
    api_key: str = Field(default=config("PRIMARY_API_KEY"))
    models: Models = Field(default_factory=Models)

    failures: int = Field(default=0)
    next_available: float = Field(default=0)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "hawki2"

    @staticmethod
    def _truncate_text(text: str, max_length: int = 300) -> str:
        """Truncate text for logging purposes"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + f"... [truncated {len(text) - max_length} chars]"

    def _set_cooldown(self, current_time: float) -> float:
        """Set exponential backoff cooldown with jitter and return the backoff duration."""
        self.failures += 1
        base_backoff = min(self.base_backoff * (2 ** (self.failures - 1)), self.max_cooldown)
        backoff = base_backoff * random.uniform(0.8, 1.2)
        self.next_available = current_time + backoff
        return backoff

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system"
        }
        return [
            {
                "role": role_map.get(message.type, "user"),
                "content": {"text": message.content}
            }
            for message in messages
            if message.type in role_map
        ]

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> str:
        start_time = time.time()
        logger.info(f"Starting Hawki2 request with {len(messages)} messages (global timeout: {self.global_timeout}s)")

        formatted_messages = self._convert_messages(messages)
        payload = {
            "payload": {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **({"stop": stop} if stop else {})
            }
        }

        for i, msg in enumerate(formatted_messages):
            logger.debug(f"Message {i + 1} ({msg['role']}): {self._truncate_text(msg['content']['text'])}")

        ratelimit:bool = False
        attempt = 0
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = self.global_timeout - elapsed

            if elapsed > self.global_timeout:
                logger.error(f"Global timeout ({self.global_timeout}s) exceeded after {attempt} attempt(s)")
                raise GlobalTimeoutError(f"Request timeout after {self.global_timeout} seconds, rate limit hit: {ratelimit}, attempts within global timeout: {attempt}")

            # Wait out cooldown if active   
            wait_time = self.next_available - current_time
            if wait_time > 0:
                if wait_time > remaining:
                    logger.error(f"Cooldown ({wait_time:.1f}s) exceeds remaining timeout ({remaining:.1f}s). Giving up.")
                    raise CooldownTimeoutError(f"Cooldown of {wait_time:.1f}s exceeds remaining timeout of {remaining:.1f}s, rate limit hit: {ratelimit}, attempts within global timeout: {attempt}")
                logger.warning(f"Waiting {wait_time:.1f}s for cooldown to expire (remaining timeout: {remaining:.1f}s)...")
                time.sleep(wait_time)
                continue

            attempt += 1
            remaining = self.global_timeout - (time.time() - start_time)
            logger.info(f"Attempt {attempt} — remaining timeout: {remaining:.1f}s")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            try:
                time.sleep(random.uniform(0.1, 0.5))

                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=(self.connect_timeout, self.read_timeout)
                )

                if not response.text.strip():
                    logger.warning(f"Empty response body (status {response.status_code}). Retrying (attempt {attempt + 1}), remaining timeout: {self.global_timeout - (time.time() - start_time):.1f}s")
                    self._set_cooldown(time.time())
                    continue

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except Exception as e:
                        logger.error(f"JSON parsing failed: {e}. Response: {self._truncate_text(response.text, 1000)}")
                        logger.warning(f"Retrying (attempt {attempt + 1}), remaining timeout: {self.global_timeout - (time.time() - start_time):.1f}s")
                        self._set_cooldown(time.time())
                        continue

                    if 'error' in data:
                        error_msg = data['error'].get('message', str(data['error']))
                        logger.error(f"Hawki2 API returned an error: {error_msg}")
                        logger.warning(f"Retrying (attempt {attempt + 1}), remaining timeout: {self.global_timeout - (time.time() - start_time):.1f}s")
                        self._set_cooldown(time.time())
                        continue

                    text = data.get("text") or data.get("content", {}).get("text") or ""
                    if text:
                        self.failures = 0
                        logger.info(f"Success after {time.time() - start_time:.1f}s (attempt {attempt})")
                        logger.debug(f"Extracted response: {self._truncate_text(text)}")
                        return text
                    else:
                        #logger.warning(f"Empty 'text' in response. Data: {self._truncate_text(str(data), 1000)}")
                        #logger.warning(f"Retrying (attempt {attempt + 1}), remaining timeout: {self.global_timeout - (time.time() - start_time):.1f}s")
                        #self._set_cooldown(time.time())
                        #continue
                        raise EmptyResponseError("Upstream API returned an empty response", status_code=522)

                response.raise_for_status()

            except requests.exceptions.Timeout as e:
                logger.warning(f"Request timed out: {e}")
                logger.warning(f"Retrying (attempt {attempt + 1}), remaining timeout: {self.global_timeout - (time.time() - start_time):.1f}s")
                self._set_cooldown(time.time())
                continue

            except requests.exceptions.RequestException as e:
                status_code = getattr(e.response, 'status_code', None)
                if status_code == 429:
                    backoff = self._set_cooldown(time.time())
                    logger.warning(f"Rate limit hit (429). Backing off for {backoff:.1f}s (failure #{self.failures}), remaining timeout: {self.global_timeout - (time.time() - start_time):.1f}s")
                    logger.warning(f"Retrying (attempt {attempt + 1}) after cooldown...")
                    ratelimit = True
                    continue
                elif status_code == 401:
                    logger.error(f"Unauthorized (401): {e}")
                    raise UnauthorizedError("Unauthorized access — check your API key", status_code=401)
                else:
                    logger.error(f"Request failed with status {status_code}: {e}")
                    raise RequestFailedError(f"Request failed with status code {status_code}: {e}", status_code=status_code)

        raise RequestFailedError(f"Unexpected exit from retry loop after {attempt} attempts")

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        logger.debug(f"Starting Hawki2 generation with {len(messages)} messages")
        text = self._call(messages, stop)
        generation = ChatGeneration(message=AIMessage(content=text))
        logger.debug(f"Completed Hawki2 generation")
        return ChatResult(generations=[generation])

    def setConfig(self, settings: dict):
        """
        Set configuration parameters for the Hawki2 client.
        """
        model:str = settings.get("model")
        if model not in self.models.list(): # Validate only if a model was provided
            if model not in self.models.list_initial():
                raise ModelNotFoundException(f"Model '{model}' not supported.", status_code=400)
            else:
                raise ModelNotFoundException(f"Model '{model}' currently not available. Send a GET-request to /health to check available models.", status_code=503)
        self.model = settings.get("model", self.model)
        self.temperature = settings.get("temperature", self.temperature)
        self.max_tokens = settings.get("max_tokens", self.max_tokens)
        self.top_p = settings.get("top_p", self.top_p)
        timeout = settings.get("timeout")
        self.global_timeout = timeout if timeout is not None else self.global_timeout
        if "api_key" in settings:
            self.setApiKey(settings["api_key"])
    
    def setApiKey(self, api_key: str):
        self.api_key = api_key