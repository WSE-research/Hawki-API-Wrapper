import pytest

from HawkiLLM import Hawki2ChatModel, Models
from exceptions import ModelNotFoundException


class DummyModels(Models):
    """
    Helper Models implementation for tests:
    - list_initial() returns all initially configured models
    - list() returns the currently available models
    """
    def __init__(self, initial, available):
        self._initial = list(initial)
        self.models = list(available)

    def list(self):
        return list(self.models)

    def list_initial(self):
        return list(self._initial)


def make_client(initial_models, available_models):
    """
    Create a Hawki2ChatModel instance with a controlled Models object.
    """
    client = Hawki2ChatModel()
    client.models = DummyModels(initial_models, available_models)
    return client


def test_set_config_supported_model():
    """
    When the model is in the available list, setConfig should succeed
    and update the model on the client.
    """
    client = make_client(
        initial_models=["gpt-4o", "gpt-4.1"],
        available_models=["gpt-4o"]  # currently available
    )

    client.setConfig({"model": "gpt-4o", "temperature": 0.5})

    assert client.model == "gpt-4o"


def test_set_config_model_not_supported():
    """
    When the model is not in the initial (supported) list at all,
    setConfig should raise ModelNotFoundException with 'not supported' message.
    """
    client = make_client(
        initial_models=["gpt-4o", "gpt-4.1"],
        available_models=["gpt-4o"]
    )

    with pytest.raises(ModelNotFoundException) as excinfo:
        client.setConfig({"model": "some-random-not-supported-model"})

    msg = str(excinfo.value)
    assert "some-random-not-supported-model" in msg

def test_set_config_model_not_available():
    """
    When the model is supported (in initial_models) but not currently available,
    setConfig should raise ModelNotFoundException with 'currently not available' message.
    """
    client = make_client(
        initial_models=["gpt-4o", "gpt-4.1"],
        available_models=["gpt-4o"]  # gpt-4.1 supported but not available
    )

    with pytest.raises(ModelNotFoundException) as excinfo:
        client.setConfig({"model": "gpt-4.1"})

    msg = str(excinfo.value)
    assert "currently not available" in msg
    assert "gpt-4.1" in msg