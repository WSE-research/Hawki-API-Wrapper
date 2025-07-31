import unittest
from fastapi.testclient import TestClient
from wrapper import app
from config import ALLOWED_KEYS
from openai import OpenAI
import logging
import colorlog
import sys
import json

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

class TestChatCompletions(unittest.TestCase):
    def setUp(self):
        print("Setting up test client")
        self.client = TestClient(app)
        self.valid_key = list(ALLOWED_KEYS.keys())[0]  # Get first allowed key
        logger.warning(f"Using key: {self.valid_key}")
        logger.warning(f"Using base URL: {self.client.base_url}")

    def test_trivial(self):
        self.assertEqual('foo'.upper(), 'FOO') 
        logger.warning(f"Client URL: {self.client.base_url}")

    # test access to health endpoint
    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
        self.assertIn("timestamp", response.json())

    def test_moderation_endpoint(self):
        """Test the moderation endpoint with safe and unsafe content"""
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
        self.assertEqual(response.status_code, 200)
        #self.assertTrue(response.json()["results"][0]["flagged"])
        
    def test_chat_completions_endpoint(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        
        assert response.status_code == 200
        

if __name__ == '__main__':
    # Run with buffer=False to show print statements
    unittest.main(buffer=False) 