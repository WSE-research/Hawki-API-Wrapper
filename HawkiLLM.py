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

OPENROUTER_MAX_COOLDOWN = 60  # 1 minute

load_dotenv("./service_config/files/.env")  # Load environment variables from .env file
load_dotenv("./config/models.env")
logger = logging.getLogger(__name__)

supported_models: list[str] = config("MODELS", default="").split(",")


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
        self.models = supported_models

    def list(self) -> List[str]:
        """
        List all available models.
        """
        return list(self.models)

class Hawki2ChatModel(BaseChatModel, BaseModel):
    """
    Custom Hawki2 API client with comprehensive timeout and rate limiting protection.
    """
    model: str = Field(default="gpt-4o")
    temperature: float = 0.7
    max_tokens: int = 2048 # TODO: What's the usual max_tokens?
    top_p: float = 1.0 # TODO: Can this be used here? Otherwise, throw away
    base_backoff: float = 10.0
    connect_timeout: int = 30
    read_timeout: int = 60
    global_timeout: int = 60
    max_cooldown: int = 300
    api_url: str = Field(default=config("HAWKI_API_URL"))
    api_key: str = Field(default=config("PRIMARY_API_KEY"))
    secondary_api_key: str = Field(default=config("SECONDARY_API_KEY"))
    models: Models = Field(default_factory=Models)
    
    primary_failures: int = Field(default=0)
    primary_next_available: float = Field(default=0)
    secondary_failures: int = Field(default=0)
    secondary_next_available: float = Field(default=0)
    
    last_used_primary: bool = Field(default=False)
    request_count: int = Field(default=0)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self: str, **kwargs):
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

    def _get_available_key(self, current_time: float) -> tuple[str, bool, str]:
        """
        Returns the best available key using round-robin with cooldown awareness and occasional randomness.
        Returns: (api_key, is_secondary, key_type)
        """
        primary_available = current_time >= self.primary_next_available
        secondary_available = current_time >= self.secondary_next_available and self.secondary_api_key
        
        if primary_available and secondary_available:
            self.request_count += 1
            if random.random() < 0.1:
                use_secondary = random.choice([True, False])
                logger.debug("Using random key selection instead of round-robin")
            else:
                use_secondary = self.request_count % 2 == 0
            
            if use_secondary:
                return self.secondary_api_key, True, "secondary"
            else:
                return self.api_key, False, "primary"
        
        elif primary_available:
            return self.api_key, False, "primary"
        elif secondary_available:
            return self.secondary_api_key, True, "secondary"
        
        if not self.secondary_api_key:
            return self.api_key, False, "primary"
        
        if self.primary_next_available <= self.secondary_next_available:
            return self.api_key, False, "primary"
        else:
            return self.secondary_api_key, True, "secondary"

    def _set_key_cooldown(self, is_secondary: bool, current_time: float):
        """Set exponential backoff cooldown with jitter for the specified key"""
        if is_secondary:
            self.secondary_failures += 1
            base_backoff = min(self.base_backoff * (2 ** (self.secondary_failures - 1)), self.max_cooldown)
            jitter = random.uniform(0.8, 1.2)
            backoff = base_backoff * jitter
            self.secondary_next_available = current_time + backoff
            return backoff, self.secondary_failures
        else:
            self.primary_failures += 1
            base_backoff = min(self.base_backoff * (2 ** (self.primary_failures - 1)), self.max_cooldown)
            jitter = random.uniform(0.8, 1.2)
            backoff = base_backoff * jitter
            self.primary_next_available = current_time + backoff
            return backoff, self.primary_failures

    def _reset_key_failures(self, is_secondary: bool):
        """Reset failure count for successful key"""
        if is_secondary:
            self.secondary_failures = 0
        else:
            self.primary_failures = 0

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
        logger.info(f"Starting Hawki2 request with {len(messages)} messages (max cooldown: {self.max_cooldown/3600:.1f}h, global timeout: {self.global_timeout/3600:.1f}h)")

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

        # DEBUG-Logging of formatted messages
        for i, msg in enumerate(formatted_messages):
            role = msg["role"]
            content = self._truncate_text(msg["content"]["text"])
            logger.debug(f"Message {i + 1} ({role}): {content}")

        attempt = 0
        while True:
            current_time = time.time()
            
            # Timeout-Check
            if current_time - start_time > self.global_timeout: 
                logger.error(f"Global timeout ({self.global_timeout}s) exceeded for Hawki2 request")
                raise RuntimeError(f"Request timeout after {self.global_timeout} seconds")

            # Key selection
            current_key, using_secondary, key_type = self._get_available_key(current_time)
            
            wait_time = 0
            if using_secondary and current_time < self.secondary_next_available:
                wait_time = self.secondary_next_available - current_time
            elif not using_secondary and current_time < self.primary_next_available:
                wait_time = self.primary_next_available - current_time
                
            if wait_time > 0:
                if wait_time > self.max_cooldown:
                    logger.error(f"{key_type} key cooldown too long ({wait_time:.1f}s). Giving up.")
                    raise RuntimeError(f"{key_type} key cooldown exceeds maximum wait time")
                    
                logger.warning(f"{key_type} key in cooldown. Waiting {wait_time:.1f}s ({wait_time/3600:.1f}h)... (Total elapsed: {current_time - start_time:.1f}s)")
                time.sleep(wait_time)
                logger.info(f"{key_type} key cooldown completed after {wait_time:.1f}s. Resuming attempts...")
                continue
            
            attempt += 1

            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json"
            }

            try:
                logger.info(f"Attempt {attempt} with {key_type} key to Hawki2 API (unlimited retries with dual-key system)")
                
                pre_request_delay = random.uniform(0.1, 0.5)
                time.sleep(pre_request_delay)
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=(self.connect_timeout, self.read_timeout)
                )

                if not response.text.strip():
                    logger.warning(f"Empty response body from {key_type} key (status {response.status_code})")
                    backoff, failure_count = self._set_key_cooldown(using_secondary, time.time())
                    logger.warning(f"{key_type} key backing off for {backoff}s ({backoff/3600:.1f}h) - failure #{failure_count} (total elapsed: {time.time() - start_time:.1f}s)")
                    continue

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except Exception as e:
                        logger.error(f"JSON parsing failed with {key_type} key: {e}. Response: {self._truncate_text(response.text, 1000)}")
                        backoff, failure_count = self._set_key_cooldown(using_secondary, time.time())
                        logger.warning(f"{key_type} key backing off for {backoff}s ({backoff/3600:.1f}h) due to JSON parse error - failure #{failure_count} (total elapsed: {time.time() - start_time:.1f}s)")
                        continue

                    if 'error' in data:
                        error_msg = data['error'].get('message', str(data['error']))
                        logger.error(f"Hawki2 API error with {key_type} key: {error_msg}")
                        backoff, failure_count = self._set_key_cooldown(using_secondary, time.time())
                        logger.warning(f"{key_type} key backing off for {backoff}s ({backoff/3600:.1f}h) due to API error - failure #{failure_count} (total elapsed: {time.time() - start_time:.1f}s)")
                        continue

                    text = data.get("text") or data.get("content", {}).get("text") or ""
                    if text:
                        total_time = time.time() - start_time
                        logger.info(f"Successfully extracted response after {total_time:.1f}s total (attempt {attempt}, {key_type} key)")
                        logger.debug(f"Extracted response: {self._truncate_text(text)}")
                        self._reset_key_failures(using_secondary)
                        return text
                    # Empty 'text' in response from primary key. Data: {'success': True, 'content': {'text': ''}}
                    # primary key returning empty responses. Backing off for 34.828811101610285s (0.0h) - failure #3 (total elapsed: 36.2s)
                    # primary key in cooldown. Waiting 34.8s (0.0h)... (Total elapsed: 36.2s)
                    # There seems to be no concrete error for some cases and the 'text' field is just empty and the status code is 200
                    else:
                        logger.warning(f"Empty 'text' in response from {key_type} key. Data: {self._truncate_text(str(data), 1000)}")
                        backoff, failure_count = self._set_key_cooldown(using_secondary, time.time())
                        logger.warning(f"{key_type} key returning empty responses. Backing off for {backoff}s ({backoff/3600:.1f}h) - failure #{failure_count} (total elapsed: {time.time() - start_time:.1f}s)")
                        continue

                response.raise_for_status()

            except requests.exceptions.Timeout as e:
                logger.warning(f"Hawki2 API timeout with {key_type} key: {str(e)}")
                backoff, failure_count = self._set_key_cooldown(using_secondary, time.time())
                logger.warning(f"{key_type} key backing off for {backoff}s ({backoff/3600:.1f}h) due to timeout - failure #{failure_count} (total elapsed: {time.time() - start_time:.1f}s)")
                continue

            except requests.exceptions.RequestException as e:
                status_code = getattr(e.response, 'status_code', None)
                if status_code == 429:
                    logger.warning(f"Rate limit error (429) on {key_type} key")
                    # Set cooldown for this key and try the other
                    backoff, failure_count = self._set_key_cooldown(using_secondary, time.time())
                    logger.warning(f"{key_type} key rate limited (429). Backing off for {backoff}s ({backoff/3600:.1f}h) - failure #{failure_count} (total elapsed: {time.time() - start_time:.1f}s)")
                    continue
                else:
                    logger.error(f"Hawki2 API request failed with {key_type} key: {str(e)}")
                    # Set cooldown for this key and try the other
                    backoff, failure_count = self._set_key_cooldown(using_secondary, time.time())
                    logger.warning(f"{key_type} key backing off for {backoff}s ({backoff/3600:.1f}h) due to request error - failure #{failure_count} (total elapsed: {time.time() - start_time:.1f}s)")
                    continue

        # This should never be reached due to the infinite loop, but just in case
        raise RuntimeError(f"Unexpected exit from retry loop after {attempt} attempts")

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
        self.model = settings.get("model", self.model)
        self.temperature = settings.get("temperature", self.temperature)
        self.max_tokens = settings.get("max_tokens", self.max_tokens)
        self.top_p = settings.get("top_p", self.top_p)
        self.connect_timeout = settings.get("connect_timeout", self.connect_timeout)
        self.read_timeout = settings.get("read_timeout", self.read_timeout)
        self.global_timeout = settings.get("global_timeout", self.global_timeout)
        self.max_cooldown = settings.get("max_cooldown", self.max_cooldown)

    def test_passed_model(model: str) -> bool:
        """
        Tests if the given model is supported by Hawki2.
        """
        if model in supported_models:
            return True
        else:
            raise ValueError(f"Model {model} is not supported by Hawki2. Supported models: {supported_models}")
