import json
import uuid
import openai
import pytest
import datetime
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
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE_BASIC,
    "kwargs": {
        "max_completion_tokens": 200,
        # Variations of "time":
        "logit_bias": {655: -100, 1711: -100, 2256: -100}
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


CHAT_COMPLETION_TOOLS = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE_TOOLS,
    "tools": ConstantData.TOOLS_FUNCTION,
    "tool_choice": {
        "type": "function",
        "function": {"name": "display_weather"}
    },
    ""
    "kwargs": {
        "max_completion_tokens": 200

    }
}


@pytest.mark.parametrize(
    "test_data",
    [
        CHAT_COMPLETION_TOOLS
    ]
)
def test_openai_completion_tools(setup_openai_client, test_data):
    """Test completion request and check tools."""
    url = "http://localhost:8000/v1/"

    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )
        # Make a completion request
        completion = client.chat.completions.create(
            model=test_data["model"],
            messages=test_data["messages"],
            tools=test_data["tools"],
            tool_choice=test_data["tool_choice"],
            **test_data["kwargs"]
        )

        tools = completion.choices[0].message.tool_calls
        for tool in tools:
            assert tool.type == test_data["tools"][0]["type"]
            assert tool.function.name == test_data["tools"][0]["function"]["name"]

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")


CHAT_COMPLETION_STOP = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE_STOP,
    "stop": ["K", "k"],
    "kwargs": {
        "max_completion_tokens": 200
    }
}


@pytest.mark.parametrize(
    "test_data",
    [
        CHAT_COMPLETION_STOP
    ]
)
def test_openai_completion_stop(setup_openai_client, test_data):
    """Test completion request and check the stop function ."""
    url = "http://localhost:8000/v1/"

    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )
        # Make a basic completion request
        completion = client.chat.completions.create(
            model=test_data["model"],
            messages=test_data["messages"],
            stop=test_data["stop"],
            **test_data["kwargs"]
        )

        # Check the stop function
        assert completion.choices[0].message is not None
        for stop_string in test_data["stop"]:
            assert stop_string not in completion.choices[0].message

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")


def get_response(client, **kwargs) -> str:
    result = str()

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        x = chunk.choices[0].delta.content
        if x is None:
            continue
        result += str(x)

    return result


def test_seed(setup_openai_client, seed=1337):
    url = "http://localhost:8000/v1/"
    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )

        base_params = {
            "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
            "messages": ConstantData.MESSAGE_SEED,
            "stream": True
        }

        result1 = get_response(client=client, **base_params, seed=seed)
        result2 = get_response(client=client, **base_params, seed=seed)
        result3 = get_response(client=client, **base_params, seed=seed+1)

        assert result1 == result2
        assert result1 != result3

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")


def test_chat_max_tokens(setup_openai_client):
    model = "models/SmolLM2-135M-Instruct-Q6_K.gguf"
    url = "http://localhost:8000/v1/"
    client = openai.OpenAI(base_url=url, api_key=openai.api_key)

    completion_chunks = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Tell me a story"}],
        max_tokens=1,
        stream=True
    )

    # Stream returns chunks one token at a time.
    token_chunks = [chunk for chunk in completion_chunks
                    if chunk.choices[0].delta.content is not None]

    assert len(token_chunks) == 1


def test_chat_stream_options(setup_openai_client):
    model = "models/SmolLM2-135M-Instruct-Q6_K.gguf"
    url = "http://localhost:8000/v1/"
    client = openai.OpenAI(base_url=url, api_key=openai.api_key)

    completion_chunks = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Tell me a story"}],
        max_tokens=3,
        stream=True,
        stream_options={"include_usage": True}
    )

    chunks = list(completion_chunks)

    token_chunks = [chunk for chunk in chunks
                    if chunk.choices and chunk.choices[0].delta.content is not None]
    assert len(token_chunks) == 3
    for token_chunk in token_chunks:
        assert token_chunk.usage is None
    usage = chunks[-1].usage
    assert usage.prompt_tokens == 34
    assert usage.completion_tokens == 3
    assert usage.total_tokens == 37


def test_chat_stream_options_tools(setup_openai_client):
    model = "models/SmolLM2-135M-Instruct-Q6_K.gguf"
    url = "http://localhost:8000/v1/"
    client = openai.OpenAI(base_url=url, api_key=openai.api_key)
    tools = [
        {
            "type": "function",
            "function": {
                "description": "Set door status",
                "name": "set_door",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "open": {"type": "boolean"},
                    },
                    "required": [
                        "open",
                    ],
                    "additionalProperties": False
                }
            }
        }
    ]

    completion_chunks = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Close the door"}],
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "set_door"}
        },
        stream=True,
        stream_options={"include_usage": True}
    )

    chunks = list(completion_chunks)

    usage = chunks[-1].usage
    assert usage.prompt_tokens == 33
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


# TODO: refactor into separate tests?
def test_chat_response(setup_openai_client):
    model = "models/SmolLM2-135M-Instruct-Q6_K.gguf"
    url = "http://localhost:8000/v1/"
    client = openai.OpenAI(base_url=url, api_key=openai.api_key)

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_completion_tokens=1,
    )
    finish_reason = completion.choices[0].finish_reason
    assert finish_reason == "length"
    assert isinstance(completion.choices[0].message.content, str)
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].index == 0

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stop=["38634"],  # corresponds to `Paris` for smolm2
        max_completion_tokens=1000,
    )
    finish_reason = completion.choices[0].finish_reason
    assert finish_reason == "stop"

    tools = [
        {
            "type": "function",
            "function": {
                "description": "Set door status",
                "name": "set_door",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "open": {"type": "boolean"},
                    },
                    "required": [
                        "open",
                    ],
                    "additionalProperties": False
                }
            }
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Open the door"}],
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "set_door"}
        },
    )
    finish_reason = completion.choices[0].finish_reason
    assert finish_reason == "tool_calls"
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content is None
    tool_call = completion.choices[0].message.tool_calls[0]
    assert tool_call.type == "function"
    assert tool_call.function.name == "set_door"
    assert isinstance(json.loads(tool_call.function.arguments), dict)

    functions = [
        {
            "description": "Set door status",
            "name": "set_door",
            "parameters": {
                "type": "object",
                "properties": {
                    "open": {"type": "boolean"},
                },
                "required": [
                    "open",
                ],
                "additionalProperties": False
            }
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Open the door"}],
        functions=functions,
        function_call={"name": "set_door", },
    )

    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content is None
    function_call = completion.choices[0].message.function_call
    assert function_call.name == "set_door"
    assert isinstance(json.loads(function_call.arguments), dict)


SYSTEM_FINGEPRINT = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.SYSTEM_FINGERPRINT_MESSAGES,
    "stream": False,
    "kwargs": {
        "max_completion_tokens": 200
    }
}

SYSTEM_FINGEPRINT_STREAM = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.SYSTEM_FINGERPRINT_MESSAGES,
    "stream": True,
    "kwargs": {
        "max_completion_tokens": 200
    }
}

@pytest.mark.parametrize(
    "test_data",
    [
        SYSTEM_FINGEPRINT,
        SYSTEM_FINGEPRINT_STREAM
    ]
)
def test_system_fingerprint(setup_openai_client, test_data):
    """Test completion request and check the stop function ."""
    url = "http://localhost:8000/v1/"

    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )

        completion = client.chat.completions.create(
            model=test_data["model"],
            messages=test_data["messages"],
            stream=test_data['stream'],
            **test_data["kwargs"]
        )

        if test_data['stream']:
            for chunk in completion:
                assert chunk.system_fingerprint is not None
        else:
            assert completion.system_fingerprint is not None

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")


def test_model_name_from_response(setup_openai_client):
    """Test completion request and check model info exists"""
    url = "http://localhost:8000/v1/"
    model = "models/SmolLM2-135M-Instruct-Q6_K.gguf"
    messages = ConstantData.MODEL_MESSAGES

    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=200,
        )

        assert completion.model == model
    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")


SIMPLE_COMPLETITION = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE_BASIC,
    "stream": False,
    "kwargs": {
        "max_completion_tokens": 200
    }
}

SIMPLE_STREAM = {
    "model": "models/SmolLM2-135M-Instruct-Q6_K.gguf",
    "messages": ConstantData.MESSAGE_BASIC,
    "stream": True,
    "kwargs": {
        "max_completion_tokens": 200
    }
}


@pytest.mark.parametrize(
    "test_data",
    [
        SIMPLE_COMPLETITION,
        SIMPLE_STREAM,
    ]
)
def test_created_timestamp(setup_openai_client, test_data):
    """Test completion and stream requests and then checks that created timestamp is correct"""

    url = "http://localhost:8000/v1/"
    current_timestamp = datetime.datetime.now().timestamp()

    # Seconds between sent time and time when message was recieved
    max_delay = 5

    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )

        completion = client.chat.completions.create(
            model=test_data["model"],
            messages=test_data["messages"],
            stream=test_data['stream'],
            **test_data["kwargs"]
        )

        if test_data['stream']:
            for chunk in completion:
                assert chunk.created - current_timestamp < max_delay
        else:
            assert completion.created - current_timestamp < max_delay

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")


@pytest.mark.parametrize(
    "test_data",
    [
        SIMPLE_COMPLETITION,
        SIMPLE_STREAM,
    ]
)
def test_object_type(setup_openai_client, test_data):
    """Test completion and stream requests is it correct object type"""

    url = "http://localhost:8000/v1/"

    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )

        completion = client.chat.completions.create(
            model=test_data["model"],
            messages=test_data["messages"],
            stream=test_data['stream'],
            **test_data["kwargs"]
        )

        if test_data['stream']:
            for chunk in completion:
                assert chunk.object == 'chat.completion.chunk'
        else:
            assert completion.object == 'chat.completion'

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")

