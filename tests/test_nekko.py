import uuid
import openai
import pytest
from constant_data import ConstantData

"""
Data for specific tests
"""

CHAT_COMPLETION_BASIC = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE_BASIC,
    "kwargs": {
        "max_completion_tokens": 200
    }
}

CHAT_COMPLETION_FREQUENCY_PENALTY = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE_BASIC,
    "kwargs": {
        "max_completion_tokens": 200,
        "frequency_penalty": 2.0,
    }
}

CHAT_COMPLETION_PRESENCE_PENALTY = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE_BASIC,
    "kwargs": {
        "max_completion_tokens": 200,
        "presence_penalty": 2.0,
    }
}

CHAT_COMPLETION_LOGITBIAS = {
    "model": "models/Llama-3.2-1B-Instruct-Q5_K_S.gguf",
    "messages": ConstantData.MESSAGE_BASIC,
    "kwargs": {
        "max_completion_tokens": 200,
        "logit_bias": {976: -100, 48887: -100, 328: -100, 125280:-100}
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
        CHAT_COMPLETION_LOGITBIAS
    ]
)
def test_openai_completion_message(setup_openai_client, test_data):
    """Test completion request and check the received message."""
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


CHAT_COMPLETION_LOGPROBS_3 = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE_LOGPROBS,
    "logprobs": True,
    "kwargs": {
        "max_completion_tokens": 200,
        "top_logprobs": 3
    }
}

CHAT_COMPLETION_LOGPROBS_NULL = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE_LOGPROBS,
    "logprobs": False,
    "kwargs": {
        "max_completion_tokens": 200
    }
}


@pytest.mark.parametrize(
    "test_data",
    [
        CHAT_COMPLETION_LOGPROBS_3,
        CHAT_COMPLETION_LOGPROBS_NULL
    ]
)
def test_openai_completion_logpobs(setup_openai_client, test_data):
    """Test completion request and check the logprobs."""
    url = "http://localhost:8000/v1/"

    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )
        # Make a basic completion request
        completion = client.chat.completions.create(
            model=test_data["model"],
            messages=test_data["messages"],
            logprobs=test_data["logprobs"],
            **test_data["kwargs"]
        )
        logprobs = completion.choices[0].logprobs

        # Check the logprobs data
        if not test_data["logprobs"]:
            assert logprobs is None
        else:
            assert logprobs is not None
            for prob in logprobs.content:
                assert prob.token is not None
                assert type(prob.logprob) == float
                for top_prob in prob.top_logprobs:
                    assert top_prob.token is not None
                    assert type(top_prob.logprob) == float

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")
