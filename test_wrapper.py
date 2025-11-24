import unittest
from fastapi.testclient import TestClient
from wrapper import app
# from config import ALLOWED_KEYS
import logging
import colorlog
import sys
import json
import os
from decouple import config

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG
handler = colorlog.StreamHandler(sys.stdout)
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:%(name)s:%(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# get the ALLOWED_KEYS from the environment variable
ALLOWED_KEYS = os.getenv("ALLOWED_KEYS").split(",")
logger.warning(f"Number of ALLOWED_KEYS: {len(ALLOWED_KEYS)}")

MODEL_FOR_TESTING = config("MODEL_FOR_TESTING", default="gpt-4o-mini")


class TestChatCompletions(unittest.TestCase):
    def setUp(self):
        logger.warning("Setting up test client")
        self.client = TestClient(app)
        self.valid_key = ALLOWED_KEYS[0]  # Get first allowed key
        logger.warning(f"Using key: {self.valid_key}")
        logger.warning(f"Using base URL: {self.client.base_url}")

    def test_010_trivial(self):
        self.assertEqual('foo'.upper(), 'FOO')
        logger.info(f"Client URL: {self.client.base_url}")

    # test access to health endpoint
    def test_020_health(self):
        logger.info(f"Testing health endpoint")
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
        self.assertIn("timestamp", response.json())
        logger.info(f"Health response: {response.json()}")

    # test access to models endpoint
    def test_030_models_endpoint(self):
        logger.info(f"Testing models endpoint")
        response = self.client.get(
            "/v1/models",
            # send the API key in the headers
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        self.assertEqual(response.status_code, 200)
        logger.info(f"Models response: {response.json()}")

    # test access to chat completions endpoint
    # @unittest.skip("Skipping chat completions endpoint test")
    def test_040_chat_completions_endpoint(self):
        logger.info(f"Testing chat completions endpoint")
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_FOR_TESTING,
                "messages": [{"role": "user", "content": "Hello there"}]
            },
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        logger.info(f"Chat completions response: {response.json()}")

        assert response.status_code == 200

    # test access to chat completions endpoint a second time to test caching
    def test_050_chat_completions_endpoint_cached(self):
        logger.info(f"Testing chat completions endpoint (cached)")
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_FOR_TESTING,
                "messages": [{"role": "user", "content": "Hello there"}]
            },
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        logger.info(f"Chat completions response (cached): {response.json()}")
        assert response.status_code == 200
        assert response.headers.get("X-Cache-Hit") == "true"

    # test access to moderation endpoint
    @unittest.skip("Skipping moderation endpoint test")
    def test_060_moderation_endpoint(self):
        """Test the moderation endpoint with safe and unsafe content"""
        logger.info(f"Testing moderation endpoint -- not implemented yet")
        # Test safe content
        response = self.client.post(
            "/v1/moderations",
            json={
                "input": "I want to be friendly and kind to everyone."
            },
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        self.assertEqual(response.status_code, 200)
        logger.info(f"Moderation response: {response.json()}")
        # parse the json response as a json object
        json_response = json.loads(response.json())

        results = json_response["results"]
        self.assertFalse(results[0]["flagged"])

        # Test unsafe content
        response = self.client.post(
            "/v1/moderations",
            json={
                "input": "I want to cause harm and violence"
            }
        )
        logger.info(f"Moderation response: {response.json()}")
        self.assertEqual(response.status_code, 200)
        # self.assertTrue(response.json()["results"][0]["flagged"])


if __name__ == '__main__':
    # Run with buffer=False to show print statements
    unittest.main(buffer=False)
