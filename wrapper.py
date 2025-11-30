from fastapi import FastAPI, Request
import uvicorn
from fastapi import responses as fastapi_responses
from datetime import datetime
import json
from decouple import config
from logger_config import logger
from cache import LRUCache
from helpers import pretty_print_json
from httpx import AsyncClient
from fastapi import Request
from fastapi.responses import StreamingResponse
from HawkiLLM import Hawki2ChatModel
from dotenv import load_dotenv
from datetime import datetime
from cachetools import TTLCache, cached
from pprint import pformat
from langchain.schema import BaseMessage
from langchain_core.messages.utils import convert_to_messages
import time

load_dotenv('./service_config/files/.env')

ALLOWED_KEYS: list[str] = config("ALLOWED_KEYS", default="").split(",")
PORT = config('PORT', default=8000)
LRU_CACHE_CAPACITY = config('LRU_CACHE_CAPACITY', default=10000)
HAWKI_API_URL = config(
    'HAWKI_API_URL', default='https://hawki2.htwk-leipzig.de')
HEALTH_CHECK_PROMPT = "Health check test. Response with 'OK' if you are operational."

logger.warning(f"ALLOWED_KEYS: {ALLOWED_KEYS}")
logger.warning(f"Number of ALLOWED_KEYS: {len(ALLOWED_KEYS)}")
logger.warning(f"HAWKI_API_URL: {HAWKI_API_URL}")
app = FastAPI(docs_url="/swagger-ui", redoc_url=None)
hawkiClient: Hawki2ChatModel = Hawki2ChatModel()

completion_cache: LRUCache = LRUCache(capacity=LRU_CACHE_CAPACITY)
client_cache = TTLCache(maxsize=100, ttl=600)  # 10 minutes TTL

HTTP_SERVER = AsyncClient()


@app.get("/")
async def root():
    """
    API Information endpoint
    """
    return {
        "name": "Hawki API Wrapper",
        "version": "0.1.0",
        "endpoints": {
            "/v1/chat/completions": "Chat completions endpoint",
            "/health": "Health check endpoint",
            "/v1/models": "List available models"
        },
        "documentation": "/docs",
        "status": "operational"
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """FastAPI endpoint wrapper: parse request and delegate to core processor."""
    body = await request.json()
    auth_header = request.headers.get("Authorization")
    return await process_chat_request(body, auth_header, request)


async def process_chat_request(body: dict, auth_header: str | None, request_obj: Request | None = None):
    """
    Process a chat request given a plain dict `body` and `auth_header` string.
    If `request_obj` is provided, it will be used for logging.
    Returns a FastAPI `JSONResponse`.
    """
    # Optional logging
    if request_obj is not None:
        try:
            log_request(request_obj)
        except Exception:
            pass

    # pretty print the request body for logs
    logger.info(f"chat completions processing started for {pretty_print_json(body)}")

    # extract API key from header
    request_api_key = None
    if auth_header:
        request_api_key = auth_header.replace("Bearer ", "")

    model = body.get("model")
    raw_messages = body.get("messages", [])
    messages = convert_to_messages(raw_messages)
    temperature = body.get("temperature", None)
    max_tokens = body.get("max_tokens", None)
    top_p = body.get("top_p", None)
    stream: bool = body.get("stream", False)

    # Log the request details
    logger.info(
        f"chat completions request received - (Shared) API Key: {request_api_key}, Model: {model}, Temperature: {temperature}, Max Tokens: {max_tokens}, Top P: {top_p}, Stream: {stream}")
    logger.info(f"Messages: {messages}")

    # create a key from all the request details, and the current week number and year
    cache_key = f"{datetime.now().year}-{datetime.now().isocalendar()[1]}-{json.dumps(body, sort_keys=True)}"

    try:
        client = setClient(request_api_key)
    except ValueError:
        logger.warning(f"Unauthorized API key: {request_api_key}")
        # send a 401 error
        return fastapi_responses.JSONResponse(
            status_code=401,
            content={"error": "Unauthorized"}
        )

    # check the input data is contained in the cache holding the last 10000 input_data results
    completion_result = completion_cache.get(cache_key)
    if completion_result is not None:
        logger.warning(f"Completion result found in cache for input: {messages} ({cache_key})")
        return fastapi_responses.JSONResponse(
            content=completion_result,
            headers={"X-Cache-Hit": "true"}
        )

    # Set up the client
    client.setConfig({
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    })

    if stream:
        async def stream_response():
            for chunk in client.stream(messages, {"stream": stream}):
                content = chunk.text() or ""
                yield content.encode("utf-8")
                logger.info(f"Streaming chunk: {content}")

        return StreamingResponse(stream_response(), media_type="text/plain")

    # Can be deleted with custom stream implementation
        # TODO: Handle caching for streaming -> completion_cache.put(cache_key, response) -> Get full response
    else:
        try:
            response: BaseMessage = client.invoke(messages)
        except Exception as e:
            logger.error(f"Error: {str(e)}")

            return fastapi_responses.JSONResponse(
                content={"error": "Request error"},
                status_code=500
            )

        # log the response
        logger.info(f"Response completion: {pformat(response.model_dump(), indent=4)}")

        # For non-cached responses, explicitly set cache header to false
        try:
            response_json = json.loads(response.model_dump_json())
            response_content = response_json.get("content")
            completion_cache.put(cache_key, response_content)
            logger.info(f"Response added to cache: {cache_key}")
            return fastapi_responses.JSONResponse(
                content=response_content,
                headers={"X-Cache-Hit": "false"},
                status_code=200
            )
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            logger.error(f"Response: {pformat(response.model_dump(), indent=4)}")
            return fastapi_responses.JSONResponse(
                content={"error": "Internal Server Error"},
                status_code=500
            )


@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    AUTH = f"Bearer {ALLOWED_KEYS[0]}"
    modelCheckJson = {
        "models": {}
    }
    
    # Run model tests
    for model in hawkiClient.models.list():

        # First attempt
        result1 = await health_check_model(model)

        # Second attempt
        result2 = await health_check_model(model)
            
        # Add model entry if not exists

        modelCheckJson["models"][f"{model}"] = {}
        modelCheckJson["models"][f"{model}"]["requests"] = [result1, result2]
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "completion_cache_size": len(completion_cache.cache),
        "model_check": modelCheckJson
    }

async def health_check_model(model: str):
    """
    Health check for a specific model
    """

    request = {
        "model": model,
        "messages": [
            {"role": "user", "content": HEALTH_CHECK_PROMPT}
        ]
    }

    result = {}

    # Record both a human-readable timestamp and a precise start time for runtime measurement
    request_start_time = datetime.now()
    start_epoch = time.time()
    response = await process_chat_request(request, f"Bearer {ALLOWED_KEYS[0]}")
    # process_chat_request returns a FastAPI JSONResponse; extract JSON body and headers
    response_body = json.loads(response.body.decode())
    result["started_at"] = request_start_time.isoformat()
    # Measure runtime based on epoch seconds captured before the request
    result["runtime_in_ms"] = (time.time() - start_epoch) * 1000
    result["prompt"] = HEALTH_CHECK_PROMPT
    result["response"] = response_body
    result["status"] = "available" if response.status_code == 200 else "unavailable"
    result["cached"] = response.headers.get("X-Cache-Hit", "false") == "true"

    return result


@app.get("/test")
async def test():
    """
    Test endpoint that returns a simple chat completion to test the wrapper with a live API request 
    """
    test_request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Say hello!"}]
    }

    # Create a mock request with the default API key
    shadow_api_key = ALLOWED_KEYS[0]  # TODO: Restrict to avoid misuse?
    mock_headers = {"Authorization": f"Bearer {shadow_api_key}"}
    mock_request = Request(scope={
        "type": "http",
        "headers": [[b"authorization", mock_headers["Authorization"].encode()]]
    })
    mock_request._json = test_request

    logger.warning(f"Test Request Details - API Key: {shadow_api_key}")
    logger.warning(f"Test Request Details - Request Body: {test_request}")

    response = await chat_completions(mock_request)

    # get JSON object from JSONResponse and parse it as a JSON object
    json_response = json.loads(response.body.decode())  # TODO: Correct access
    # TODO: Correct access
    logger.warning(f"Test Response Details - Raw Response: {json_response}")

    json_response = json.loads(json_response)
    # pretty print the JSON response
    logger.warning(
        f"Test Response Details - Parsed Response: {json.dumps(json_response, indent=4)}")

    model = json_response.get("model")
    choices = json_response.get("choices")
    usage = json_response.get("usage")
    logger.warning(f"Test Response Details - Model: {model}")
    logger.warning(f"Test Response Details - Choices: {choices}")
    logger.warning(f"Test Response Details - Usage: {usage}")

    return json.loads(json.loads(response.body.decode()))


@app.get('/v1/models')
async def list_models(request: Request):
    logger.info("List models request received")
    log_request(request)

#    # get the API key from the request body or return a BAD REQUEST error
#    response400 = fastapi_responses.JSONResponse(
#        status_code=400,
#        content={"error": "Bad Request: Missing API key"}
#    )
#    try:
#        api_key = request.headers.get("Authorization").replace("Bearer ", "")
#    except Exception as e:
#        logger.error(f"Error getting API key: {e}")
#        return response400
#
#    if api_key is None or api_key == "":
#        return response400
#
#    # If key is not a proxy key or a valid Hawki Web UI key, then unauthorized
#    if api_key not in ALLOWED_KEYS and not is_api_key_working(api_key):
#        logger.warning(f"Unauthorized API key: {api_key}")
#        return fastapi_responses.JSONResponse(
#            status_code=401,
#            content={"error": "Unauthorized"}
#        )

    try:
        # Forward the request to OpenAI API using the client
        model_list = hawkiClient.models.list()

        # models: pretty print the JSON response
        logger.info(
            f"Models: {json.dumps(model_list, indent=4)}")

        # Return the models data
        return fastapi_responses.JSONResponse(
            content=model_list,
            status_code=200
        )

    except Exception as e:
        # Handle OpenAI specific errors
        content = {
            'error': {
                'message': str(e),
                'type': type(e).__name__,
                'code': getattr(e, 'code', None)
            }
        }
        logger.error(f"Error 500: {content}")

        return fastapi_responses.JSONResponse(
            content=content,
            status_code=getattr(e, 'http_status', 500)
        )

# Use this method to check and test the API key whether it is a proxy key or a Hawki Web UI key (for outside users)


def is_api_key_working(api_key: str) -> bool:
    """
    Check if the API key is valid by making a test request to the Hawki API, when it is not contained in the ALLOWED_KEYS list.
    """
    # Test the API key by making a simple request
    test_client = Hawki2ChatModel()
    test_client.setConfig({
        "api_key": api_key,
        "model": "gpt-4o"
    })

    try:
        test_client.invoke([
            {"role": "user", "content": "Hello, are you there?"}
        ])
        logger.info(
            f"API key is of type Hawki Web UI key and is valid: {api_key[:8]}...{api_key[-4:]}")
        return True
    except Exception as e:
        logger.error(
            f"API key test failed for key: {api_key[:8]}...{api_key[-4:]} with error: {e}")
        return False

# Maybe cache the clients for each API key to avoid re-creating them each time; set low deletion timer


@cached(client_cache)
def setClient(api_key: str) -> Hawki2ChatModel:
    client = Hawki2ChatModel()
    if api_key in ALLOWED_KEYS:  # Use primary shared API key
        return client
    # Check if user provided API key (for Hawki Web UI) is valid
    elif api_key and is_api_key_working(api_key):
        client.setConfig({
            "api_key": api_key
        })
        return client
    else:  # Not a valid API key
        raise ValueError("Invalid API Key")


def log_request(request: Request):
    """
    Log the request details
    """
    logger.info(f"Request received: {request.method} {request.url}")
    logger.info(f"Headers: {request.headers}")
    logger.info(f"Client: {request.client.host}")

    date_str = datetime.now().strftime('%Y-%m-%d')
    log_filename = f'logs/request_log_{date_str}.txt'

    # Write log to file
    with open(log_filename, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {request.method} {request.url}\n")
        f.write(f"Headers: {request.headers}\n")
        f.write(f"Client: {request.client.host}\n\n")


# main function
if __name__ == "__main__":
    logger.info(f"Starting the wrapper on port {PORT}")
    logger.info(f"Completion cache size: {len(completion_cache.cache)}")
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
