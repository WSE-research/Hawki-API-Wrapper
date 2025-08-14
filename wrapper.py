# define a FAST API wrapper for the OpenAI API

from fastapi import FastAPI, Request
import openai
import uvicorn
from fastapi import responses as fastapi_responses
from datetime import datetime
import json
from decouple import config
from logger_config import logger
from cache import LRUCache
from helpers import pretty_print_json, get_pretty_printed_json_response_body
import httpx
from httpx import AsyncClient
from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import sys
from HawkiLLM import Hawki2ChatModel
from dotenv import load_dotenv

from decouple import Config, Csv
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.utils import convert_to_messages

load_dotenv('./service_config/files/.env')

ALLOWED_KEYS : list[str] = config("ALLOWED_KEYS", default="").split(",")
PORT = config('PORT', default=8000)
LRU_CACHE_CAPACITY = config('LRU_CACHE_CAPACITY', default=10000)


app = FastAPI()
hawkiClient:Hawki2ChatModel = Hawki2ChatModel()

completion_cache: LRUCache = LRUCache(capacity=LRU_CACHE_CAPACITY)

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
    """
    Generate a chat completion using the OpenAI API 
    define a route for the wrapper: /v1/chat/completions, cf. https://platform.openai.com/docs/guides/text-generation
    input: a list of messages
    model: a string, default is "gpt-4o"
    output: a chat completion
    """

    # get the request body and parse as JSON
    body = await request.json()

    log_request(request)

    # pretty print the request body
    logger.info(
        f"chat completions processing started for {pretty_print_json(body)}")

    # get the API key from the request body
    request_api_key = request.headers.get("Authorization").replace("Bearer ", "")
    model = body.get("model")
    raw_messages = body.get("messages", [])
    messages = convert_to_messages(raw_messages)
    temperature = body.get("temperature", None)
    max_tokens = body.get("max_tokens", None)
    top_p = body.get("top_p", None)
    stream : bool = body.get("stream", False)

    # Log the request details
    logger.info(
        f"chat completions request received - Shared API Key: {request_api_key}, Model: {model}")
    logger.info(f"Messages: {messages}")

    # create a key from all the request details, and the current week number and year
    cache_key = f"{datetime.now().year}-{datetime.now().isocalendar()[1]}-{json.dumps(body, sort_keys=True)}"

    # if the API key is not allowed, return a 401 error
    if request_api_key not in ALLOWED_KEYS:
        logger.warning(f"Unauthorized API key: {request_api_key}")
        # send a 401 error
        return fastapi_responses.JSONResponse(
            status_code=401,
            content={"error": "Unauthorized"}
        )

    # check the input data is contained in the cache holding the last 10000 input_data results
    completion_result = completion_cache.get(cache_key)
    if completion_result is not None:
        logger.warning(
            f"Completion result found in cache for input: {messages} ({cache_key})")
        return fastapi_responses.JSONResponse(
            content=completion_result.text(),
            headers={"X-Cache-Hit": "true"}
        )

    # Set up the client
    client = hawkiClient
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
        response : BaseMessage = client.invoke(messages)

        # json_response = json.loads(response.model_dump_json())

        # log pretty print JSON response
        #logger.warning(
        #    f"Response: {pretty_print_json(json_response)}")

        # add the result to the cache
        completion_cache.put(cache_key, response)

        # For non-cached responses, explicitly set cache header to false
        return fastapi_responses.JSONResponse(
            content=response.text(),
            headers={"X-Cache-Hit": "false"}
        )


@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "completion_cache_size": len(completion_cache.cache)
    }


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
    shadow_api_key = ALLOWED_KEYS[0]
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
    json_response = json.loads(response.body.decode()) # TODO: Correct access
    logger.warning(f"Test Response Details - Raw Response: {json_response}") # TODO: Correct access

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


@app.route('/v1/models', methods=['GET'])
async def list_models(request: Request):
    logger.info("List models request received")
    log_request(request)

    # get the API key from the request body or return a BAD REQUEST error
    response400 = fastapi_responses.JSONResponse(
        status_code=400,
        content={"error": "Bad Request: Missing API key"}
    )
    try:
        api_key = request.headers.get("Authorization").replace("Bearer ", "")
    except Exception as e:
        logger.error(f"Error getting API key: {e}")
        return response400

    if api_key is None or api_key == "":
        return response400

    # if the API key is not allowed, return a 401 error
    if api_key not in ALLOWED_KEYS:
        return fastapi_responses.JSONResponse(
            status_code=401,
            content={"error": "Unauthorized"}
        )

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

def log_request(request: Request):
    """
    Log the request details
    """
    logger.info(f"Request received: {request.method} {request.url}")
    logger.info(f"Headers: {request.headers}")
    logger.info(f"Client: {request.client.host}")

    # Write log to file
    with open('logs/request_log.txt', 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {request.method} {request.url}\n")
        f.write(f"Headers: {request.headers}\n")
        f.write(f"Client: {request.client.host}\n\n")



# main function
if __name__ == "__main__":
    logger.info(f"Starting the wrapper on port {PORT}")
    logger.info(f"Allowed keys: {ALLOWED_KEYS}")
    logger.info(f"Completion cache size: {len(completion_cache.cache)}")
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
