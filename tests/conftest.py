# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from wrapper import hawkiClient
from HawkiLLM import Models

@pytest.fixture
def configure_hawki_models():
    initial_models = ["gpt-4o", "gpt-4.1-mini"]
    available_models = ["gpt-4o"]

    # Instantiate Models with no arguments
    hawkiClient.models = Models()
    # Optionally override the initial list on the instance
    hawkiClient.models.models = list(initial_models)
    # Then set the currently available models
    hawkiClient.models.set(list(available_models))

    yield