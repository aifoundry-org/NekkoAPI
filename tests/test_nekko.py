import uuid
import openai
import pytest
from constant_data import ConstantData

"""
Data for specific tests
"""

CHAT_COMPLETION_BASIC = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE,
    "kwargs": {
        "max_completion_tokens": 200,
        "stop": ["4.", "sushi"],
    }
}

CHAT_COMPLETION_FREQUENCY_PENALTY = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE,
    "kwargs": {
        "max_completion_tokens": 200,
        "stop": ["4.", "sushi"],
        "frequency_penalty": 2.0,
    }
}

CHAT_COMPLETION_PRESENCE_PENALTY = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE,
    "kwargs": {
        "max_completion_tokens": 200,
        "stop": ["4.", "sushi"],
        "presence_penalty": 2.0,
    }
}


@pytest.fixture(scope="session")
def setup_openai_client():
    """Fixture to set up OpenAI client with the API key."""
    openai.api_key = str(uuid.uuid4())
    if not openai.api_key:
        pytest.fail("OPENAI_API_KEY environment variable is not set")


@pytest.mark.parametrize(
    "test_data",
    [
        CHAT_COMPLETION_BASIC,
        CHAT_COMPLETION_FREQUENCY_PENALTY,
        CHAT_COMPLETION_PRESENCE_PENALTY,
    ]
)
def test_openai_completion_message(setup_openai_client, test_data):
    """Test completion request and check the recived message."""
    url = "http://localhost:8000/v1/"

    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )
        # Make a basic completion request
        completion = client.chat.completions.create(
            model=test_data["model"],
            messages=test_data["messages"],
            **test_data["kwargs"]
        )

        # Assert the response is OK
        assert completion.choices[0].message is not None

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")
