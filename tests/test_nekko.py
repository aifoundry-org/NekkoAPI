import uuid
import openai
import pytest
from constant_data import ConstantData

"""
Data for specific tests
"""

CHAT_COMPLETION_BASIC = {
    "model": ConstantData.MODEL_NAME,
    "messages": ConstantData.MESSAGE_BASIC,
    "kwargs": {
        "max_completion_tokens": 200
    }
}

CHAT_COMPLETION_FREQUENCY_PENALTY = {
    "model": ConstantData.MODEL_NAME,
    "messages": ConstantData.MESSAGE_BASIC,
    "kwargs": {
        "max_completion_tokens": 200,
        "frequency_penalty": 2.0,
    }
}

CHAT_COMPLETION_PRESENCE_PENALTY = {
    "model": ConstantData.MODEL_NAME,
    "messages": ConstantData.MESSAGE_BASIC,
    "kwargs": {
        "max_completion_tokens": 200,
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
    "model": ConstantData.MODEL_NAME,
    "messages": ConstantData.MESSAGE_LOGPROBS,
    "logprobs": True,
    "kwargs": {
        "max_completion_tokens": 200,
        "top_logprobs": 3
    }
}

CHAT_COMPLETION_LOGPROBS_NULL = {
    "model": ConstantData.MODEL_NAME,
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


CHAT_COMPLETION_STREAM_OPTIONS_ON = {
    "model": ConstantData.MODEL_NAME,
    "messages": ConstantData.MESSAGE_STREAM_OPTIONS,
    "stream": True,
    "stream_options": {"include_usage": True},
}

CHAT_COMPLETION_STREAM_OPTIONS_OFF = {
    "model": ConstantData.MODEL_NAME,
    "messages": ConstantData.MESSAGE_STREAM_OPTIONS,
    "stream": True,
    "stream_options": None
}


@pytest.mark.parametrize(
    "test_data",
    [
        CHAT_COMPLETION_STREAM_OPTIONS_ON,
        CHAT_COMPLETION_STREAM_OPTIONS_OFF
    ]
)
def test_stream_options(setup_openai_client, test_data):
    """Test completion request and check the stream_options."""
    url = "http://localhost:8000/v1/"
    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )

        stream = client.chat.completions.create(
            model=test_data["model"],
            messages=test_data["messages"],
            stream=test_data['stream'],
            stream_options=test_data['stream_options']
        )

        last_usage = None
        for chunk in stream:
            last_usage = chunk.usage

        assert (last_usage is None) == (test_data['stream_options'] is None)

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")