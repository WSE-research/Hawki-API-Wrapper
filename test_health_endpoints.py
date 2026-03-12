"""
Functional test suite for /health and /health/details endpoints.

This test suite focuses on core functionality:
1. Endpoints are accessible and return success
2. Core health indicators work correctly
3. Model diagnostics function properly
4. Cache behavior works as expected
5. Authorization affects usage tracking

Note: /health/details tests use mocked API responses to avoid real API calls.
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
import logging
import colorlog
import sys
import os
from datetime import datetime
from decouple import config
import time
from wrapper import app, add_model_usage, hawkiClient

# Setup colored logging
logger = logging.getLogger(__name__)
logger.level = logging.INFO
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
logger.setLevel(logging.INFO)

# Get allowed keys from environment
ALLOWED_KEYS = os.getenv("ALLOWED_KEYS", "test-key-1,test-key-2").split(",")
MODEL_FOR_TESTING = config("MODEL_FOR_TESTING", default="gpt-4o")


def create_mock_health_check_result(model: str, use_cache: bool = False, status: str = "available"):
    """Helper to create a mock health check result"""
    return {
        "started_at": datetime.now().isoformat(),
        "runtime_in_ms": 50.0 if use_cache else 200.0,
        "prompt": "Health check test. Response with 'OK' if you are operational.",
        "response": {
            "id": f"mock-{model}",
            "choices": [{"message": {"content": "OK"}}]
        },
        "status": status,
        "cached": use_cache
    }


class TestHealthEndpoint(unittest.TestCase):
    """Test suite for the /health endpoint"""

    def setUp(self):
        """Set up test client before each test"""
        logger.info("Setting up test client for /health endpoint tests")
        self.client = TestClient(app)

    def test_health_endpoint_accessible(self):
        """Test that /health endpoint is accessible without authentication"""
        logger.info("Testing /health endpoint accessibility")
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Basic sanity checks - endpoint should indicate it's healthy
        self.assertIsNotNone(data)
        self.assertEqual(data.get("status"), "healthy")        
        
        logger.info(f"✓ /health endpoint is accessible and healthy")

    def test_health_endpoint_reports_cache_info(self):
        """Test that /health endpoint reports cache information"""
        logger.info("Testing /health endpoint cache reporting")
        response = self.client.get("/health")
        data = response.json()
        
        # Should report some cache size (could be 0 or positive)
        cache_size = data.get("completion_cache_size")
        self.assertIsNotNone(cache_size, "Cache size should be reported")
        self.assertGreaterEqual(cache_size, 0, "Cache size should be non-negative")
        
        logger.info(f"✓ Cache size reported: {cache_size}")

    def test_health_endpoint_reports_models(self):
        """Test that /health endpoint reports available models"""
        logger.info("Testing /health endpoint model reporting")
        response = self.client.get("/health")
        data = response.json()
        
        # Should report initial models
        models = data.get("initial_models")
        self.assertIsNotNone(models, "Models list should be present")
        self.assertGreater(len(models), 0, "Should have at least one model")
        
        logger.info(f"✓ Reported {len(models)} models")


class TestHealthDetailsEndpoint(unittest.TestCase):
    """Test suite for the /health/details endpoint"""

    def setUp(self):
        """Set up test client before each test and apply mocking"""
        logger.info("Setting up test client for /health/details endpoint tests")
        
        # Mock process_chat_request to avoid real API calls
        self.process_chat_patcher = patch('wrapper.process_chat_request')
        self.mock_process_chat = self.process_chat_patcher.start()

        # Mock health_check_model to avoid real API calls in health diagnostics
        self.health_check_patcher = patch('wrapper.health_check_model')
        self.mock_health_check = self.health_check_patcher.start()

        async def default_health_check(model: str, api_key: str, use_cache: bool = False):
            return create_mock_health_check_result(model, use_cache)

        self.mock_health_check.side_effect = default_health_check
        
        # Configure mock to return proper async responses
        async def mock_chat_process(body: dict, header: dict | None, request_obj=None, use_cache: bool = True):
            # Track usage so health/details usage assertions work
            auth_header = header.get("Authorization") if header else None
            if auth_header:
                request_api_key = auth_header.replace("Bearer ", "")
                model = body.get("model")
                if model and request_api_key in ALLOWED_KEYS:
                    add_model_usage(request_api_key, model)

            # Simulate response based on caching
            response_data = {
                "id": f"mock-response-{body.get('model')}",
                "choices": [{"message": {"content": "OK"}}],
                "model": body.get("model"),
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}
            }
            
            response = JSONResponse(content=response_data, status_code=200)
            # Add cache header if use_cache is True
            if use_cache:
                response.headers["X-Cache-Hit"] = "true"
            return response
        
        self.mock_process_chat.side_effect = mock_chat_process
        self.client = TestClient(app)
        self.valid_key = ALLOWED_KEYS[0]

    def tearDown(self):
        """Clean up mocking after each test"""
        self.process_chat_patcher.stop()
        self.health_check_patcher.stop()

    def test_health_details_endpoint_accessible(self):
        """Test that /health/details endpoint is accessible and requires an API key for model data"""
        logger.info("Testing /health/details endpoint accessibility")
        
        # Without auth: endpoint still returns 200 but model_check is an info message
        response = self.client.get("/health/details")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data.get("status"), "healthy")
        model_check = data.get("model_check")
        self.assertIsInstance(model_check, str, "Without auth, model_check should be an info message")
        self.assertIn("API key", model_check)
        
        # With a valid key: model_check should be a dict of model diagnostics
        response_auth = self.client.get(
            "/health/details",
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        self.assertEqual(response_auth.status_code, 200)
        data_auth = response_auth.json()
        self.assertEqual(data_auth.get("status"), "healthy")
        self.assertIsInstance(data_auth.get("model_check"), dict)
        
        logger.info(f"✓ /health/details endpoint is accessible and healthy")

    def test_health_details_performs_model_diagnostics(self):
        """Test that /health/details runs diagnostics on models when API key is provided"""
        logger.info("Testing /health/details model diagnostics")
        
        response = self.client.get(
            "/health/details",
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        data = response.json()
        
        # Should have model_check with diagnostics
        model_check = data.get("model_check")
        self.assertIsNotNone(model_check, "Model diagnostics should be present")
        self.assertGreater(len(model_check), 0, "Should have diagnostics for at least one model")
        
        # Check that diagnostics were actually run (should have requests)
        for model_name, model_info in model_check.items():
            requests = model_info.get("requests", [])
            self.assertEqual(len(requests), 2, 
                           f"Model '{model_name}' should have 2 health check requests")
            
            # Each request should have a status
            for req in requests:
                status = req.get("status")
                self.assertIn(status, ["available", "unavailable"],
                            f"Request status should be 'available' or 'unavailable'")
        
        logger.info(f"✓ Diagnostics completed for {len(model_check)} models")

    def test_health_details_cache_behavior(self):
        """Test that the second diagnostic request uses cache"""
        logger.info("Testing /health/details cache behavior")
        
        response = self.client.get(
            "/health/details",
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        data = response.json()
        
        model_check = data.get("model_check", {})
        
        # Check cache behavior for models that were successfully tested
        cached_count = 0
        for model_name, model_info in model_check.items():
            requests = model_info.get("requests", [])
            if len(requests) == 2:
                first_req = requests[0]
                second_req = requests[1]
                
                # If first request succeeded, second should be cached
                if first_req.get("status") == "available":
                    is_cached = second_req.get("cached", False)
                    if is_cached:
                        cached_count += 1
                        # Cached requests should typically be faster
                        if "runtime_in_ms" in first_req and "runtime_in_ms" in second_req:
                            logger.info(f"  {model_name}: uncached={first_req['runtime_in_ms']:.2f}ms, "
                                      f"cached={second_req['runtime_in_ms']:.2f}ms")
        
        self.assertGreater(cached_count, 0, "At least one model should use cache on second request")
        logger.info(f"✓ Cache working correctly ({cached_count} models cached)")

    def test_health_details_updates_available_models(self):
        """Test that /health/details updates the available models list"""
        logger.info("Testing /health/details updates available models")
        
        # Configure mock to make some models available and some unavailable
        call_count = [0]
        
        async def mock_check_with_failures(model: str, api_key: str, use_cache: bool = False):
            call_count[0] += 1
            # Make every other model unavailable (first call is uncached, second is cached)
            status = "available" if (call_count[0] // 2) % 2 == 0 else "unavailable"
            return create_mock_health_check_result(model, use_cache, status)
        
        self.mock_health_check.side_effect = mock_check_with_failures

        # Get current available models
        models_before = set(hawkiClient.models.list())
        logger.info(f"  Available models before: {len(models_before)}")
        
        # Run health/details with valid API key
        response = self.client.get(
            "/health/details",
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Get updated available models
        models_after = set(hawkiClient.models.list())
        logger.info(f"  Available models after: {len(models_after)}")
        
        # Count available models in diagnostics
        available_in_diagnostics = sum(
            1 for model_info in data.get("model_check", {}).values()
            if model_info.get("requests", [{}])[0].get("status") == "available"
        )
        
        # Available models should match the diagnostics results
        self.assertEqual(len(models_after), available_in_diagnostics,
                        "Available models should match health check results")
        
        logger.info(f"✓ Available models correctly updated to {len(models_after)} models")

    def test_health_details_usage_tracking_without_auth(self):
        """Test that /health/details returns an info message instead of model data without authorization"""
        logger.info("Testing /health/details without authorization")
        
        response = self.client.get("/health/details")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Without auth, model_check should be a string message, not model diagnostics
        model_check = data.get("model_check")
        self.assertIsInstance(model_check, str,
                              "model_check should be an info message string when no API key is provided")
        self.assertIn("API key", model_check)
        
        logger.info(f"✓ No model diagnostics returned without authorization, got message: {model_check}")

    def test_health_details_usage_tracking_with_auth(self):
        """Test that /health/details can include usage with valid authorization"""
        logger.info("Testing /health/details with authorization")
        
        # First, generate some usage by making a chat completion
        logger.info("  Generating usage data...")
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_FOR_TESTING,
                "messages": [{"role": "user", "content": "Test for usage tracking"}]
            },
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        
        # Now check health/details with auth
        response = self.client.get(
            "/health/details",
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        data = response.json()
        
        # With valid auth, usage data may be included for models that were used
        # Note: This depends on whether the model is in initial_models
        model_check = data.get("model_check", {})
        
        # Just verify the endpoint works with auth header
        self.assertEqual(response.status_code, 200)

        # With valid auth, usage should be included for the model that was used
        if MODEL_FOR_TESTING in model_check:
            self.assertIn("usage", model_check[MODEL_FOR_TESTING],
                  f"Model '{MODEL_FOR_TESTING}' should have usage data with valid auth")

    def test_health_details_handles_invalid_auth(self):
        """Test that /health/details handles invalid authorization gracefully"""
        logger.info("Testing /health/details with invalid authorization")
        
        response = self.client.get(
            "/health/details",
            headers={"Authorization": "Bearer invalid-key-12345"}
        )
        
        # Should still return 200 OK (auth is optional for this endpoint)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # With an invalid key, model_check should be an error/info string, not model diagnostics
        model_check = data.get("model_check")
        self.assertIsInstance(model_check, str,
                              "model_check should be an info/error message string for an invalid API key")
        self.assertIn("API key", model_check)
        
        logger.info(f"✓ Invalid authorization handled gracefully, got message: {model_check}")


class TestHealthEndpointsIntegration(unittest.TestCase):
    """Integration tests for health endpoints"""

    def setUp(self):
        """Set up test client before each test"""
        logger.info("Setting up test client for integration tests")
        
        # Mock health_check_model for these tests as well
        self.health_check_patcher = patch('wrapper.health_check_model')
        self.mock_health_check = self.health_check_patcher.start()
        
        async def mock_check(model: str, api_key: str, use_cache: bool = False):
            return create_mock_health_check_result(model, use_cache)
        
        self.mock_health_check.side_effect = mock_check
        self.client = TestClient(app)
        self.valid_key = ALLOWED_KEYS[0]

    def tearDown(self):
        """Clean up mocking after each test"""
        self.health_check_patcher.stop()

    def test_health_is_lightweight(self):
        """Test that /health is significantly faster than /health/details"""
        logger.info("Testing that /health is lightweight compared to /health/details")
        
        # Measure /health response time
        start = time.time()
        health_response = self.client.get("/health")
        health_time = time.time() - start
        
        # Measure /health/details response time (API key required to run diagnostics)
        start = time.time()
        details_response = self.client.get(
            "/health/details",
            headers={"Authorization": f"Bearer {self.valid_key}"}
        )
        details_time = time.time() - start
        
        # ASSERT: Both endpoints should return 200
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(details_response.status_code, 200)
        
        # ASSERT: /health should be faster than /health/details
        self.assertLess(health_time, details_time,
                        "/health should be faster than /health/details")

class SummaryTestResult(unittest.TextTestResult):
    """Custom TestResult that collects per-test status for a final summary."""

    STATUS_PASS    = "PASS"
    STATUS_FAIL    = "FAIL"
    STATUS_ERROR   = "ERROR"
    STATUS_SKIP    = "SKIP"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._summary: list[tuple[str, str, str | None]] = []  # (status, name, detail)

    def _short_name(self, test: unittest.TestCase) -> str:
        cls  = type(test).__name__
        meth = test._testMethodName
        return f"{cls}.{meth}"

    def addSuccess(self, test):
        super().addSuccess(test)
        self._summary.append((self.STATUS_PASS, self._short_name(test), None))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._summary.append((self.STATUS_FAIL, self._short_name(test), self._exc_info_to_string(err, test)))

    def addError(self, test, err):
        super().addError(test, err)
        self._summary.append((self.STATUS_ERROR, self._short_name(test), self._exc_info_to_string(err, test)))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._summary.append((self.STATUS_SKIP, self._short_name(test), reason))

    def print_summary(self):
        colors = {
            self.STATUS_PASS:  "\033[32m",   # green
            self.STATUS_FAIL:  "\033[31m",   # red
            self.STATUS_ERROR: "\033[31m",   # red
            self.STATUS_SKIP:  "\033[33m",   # yellow
        }
        reset = "\033[0m"
        width = max((len(name) for _, name, _ in self._summary), default=40)

        print("\n" + "=" * (width + 14))
        print(f"  {'TEST':<{width}}  STATUS")
        print("=" * (width + 14))
        for status, name, detail in self._summary:
            color = colors.get(status, "")
            print(f"  {name:<{width}}  {color}{status}{reset}")
        print("-" * (width + 14))

        counts = {s: sum(1 for st, *_ in self._summary if st == s) for s in colors}
        totals = "  ".join(f"{color}{s}: {counts[s]}{reset}" for s, color in colors.items() if counts[s])
        print(f"  Total: {len(self._summary)}   {totals}")
        print("=" * (width + 14) + "\n")


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir='.', pattern='test_health_endpoints.py')

    runner = unittest.TextTestRunner(
        resultclass=SummaryTestResult,
        verbosity=2,
        buffer=False,
    )
    result = runner.run(suite)
    result.print_summary()
