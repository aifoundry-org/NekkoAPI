import json
import uuid
import openai
import pytest
import datetime


MODEL = "models/SmolLM2-135M-Instruct-Q6_K.gguf"


@pytest.fixture(scope="session")
def openai_client():
    """Fixture to set up OpenAI client with the API key."""
    openai.api_key = str(uuid.uuid4())
    if not openai.api_key:
        pytest.fail("OPENAI_API_KEY environment variable is not set")
    url = "http://localhost:8000/v1/"
    client = openai.OpenAI(base_url=url, api_key=openai.api_key)
    return client


# TODO: separate into individual tests and assert effects of parameters
@pytest.mark.parametrize(
    "test_data",
    [
        {},
        {"frequency_penalty": 2.0},
        {"presence_penalty": 2.0},
        # logit bias token ids correspond to variatios of "Time"
        # TODO: use alphabet example
        {"logit_bias": {655: -100, 1711: -100, 2256: -100}}
    ]
)
def test_chat_message(openai_client, test_data):
    """Test completion request and check the received message."""
    completion = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What's the capital of France?"}],
        max_completion_tokens=3,
        **test_data
    )

    # Assert the response is OK
    assert completion.choices[0].message


def test_chat_stop(openai_client):
    """Test completion request and check the stop function ."""
    stops = ["K", "k"]
    completion = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Write the English alphabet"}],
        stop=stops,
        max_completion_tokens=100,
    )

    # Check the stop function
    assert completion.choices[0].message is not None
    assert completion.choices[0].finish_reason == "stop"
    for stop_string in stops:
        assert stop_string not in completion.choices[0].message.content


def test_chat_seed(openai_client, seed=1337):
    params = {
        "model": MODEL,
        "max_completion_tokens": 10,
        "messages": [{"role": "user", "content": "1+1="}],
    }

    def completion(**kwargs):
        return openai_client.chat.completions.create(**kwargs).choices[0].message.content

    result1 = completion(**params, seed=seed)
    result2 = completion(**params, seed=seed)
    result3 = completion(**params, seed=seed+1)

    assert result1 == result2
    assert result1 != result3


def test_chat_max_tokens(openai_client):
    completion_chunks = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a story"}],
        max_tokens=1,
        stream=True
    )

    # Stream returns chunks one token at a time.
    token_chunks = [chunk for chunk in completion_chunks
                    if chunk.choices[0].delta.content is not None]

    assert len(token_chunks) == 1


def test_chat_stream_options(openai_client):
    completion_chunks = openai_client.chat.completions.create(
        model=MODEL,
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


def test_chat_tools(openai_client):
    """Test completion request and check tools."""
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

    result = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Close the door"}],
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "set_door"}
        },
    )

    assert result.choices[0].finish_reason == "tool_calls"

    assert result.choices[0].message.function_call  # deprecated
    assert result.choices[0].message.content is None
    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls
    assert tool_calls[0].id
    assert tool_calls[0].type == "function"
    function = tool_calls[0].function
    assert function
    assert function.name == "set_door"

    # Check function arguments schema is respected
    arguments_json = tool_calls[0].function.arguments
    arguments = json.loads(arguments_json)
    assert isinstance(arguments, dict)
    assert isinstance(arguments["open"], bool)


def test_chat_stream_tools(openai_client):
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

    stream = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Close the door"}],
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "set_door"}
        },
        stream=True,
        stream_options={"include_usage": True}
    )

    chunks = list(stream)

    # Some chunks (eg initial, stop, last) may not have content.
    tool_chunks = [chunk for chunk in chunks
                   if chunk.choices and chunk.choices[0].delta.tool_calls]

    assert len(tool_chunks) > 0

    call_id = tool_chunks[0].choices[0].delta.tool_calls[0].id
    assert call_id
    for chunk in tool_chunks:
        assert chunk.choices[0].delta.function_call  # deprecated
        tool_calls = chunk.choices[0].delta.tool_calls
        assert tool_calls
        # All chunks have same id
        assert tool_calls[0].id == call_id
        assert tool_calls[0].type == "function"
        function = tool_calls[0].function
        assert function
        assert function.name == "set_door"

    # Check function arguments schema is respected
    arguments_json = ''.join(chunk.choices[0].delta.tool_calls[0].function.arguments
                             for chunk in tool_chunks)
    arguments = json.loads(arguments_json)
    assert isinstance(arguments, dict)
    assert isinstance(arguments["open"], bool)

    # Tool calling has independent "usage" codepath, therefore this check.
    usage = chunks[-1].usage
    assert usage.prompt_tokens == 33
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


def chat_completion_finish_reason(openai_client):
    completion = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_completion_tokens=1,
    )

    finish_reason = completion.choices[0].finish_reason
    assert finish_reason == "length"
    assert isinstance(completion.choices[0].message.content, str)
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].index == 0


def test_chat_response(openai_client):
    top_logprobs = 3

    request_timestamp = datetime.datetime.now().timestamp()

    completion = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Give me a random letter."}],
        max_completion_tokens=1,
        logprobs=True,
        top_logprobs=top_logprobs,
    )

    # Sets id
    assert completion.id

    # Sets model
    assert completion.model == MODEL

    # Sets correct object type
    assert completion.object == 'chat.completion'

    # Sets creation timestamp
    assert completion.created - request_timestamp < 5  # seconds

    # Sets system fingerprint
    assert completion.system_fingerprint

    # Returns expected role.
    assert completion.choices[0].message.role == "assistant"

    # Choice has correct index set (multiple choices not supported yet).
    assert completion.choices[0].index == 0

    logprobs = completion.choices[0].logprobs
    for logprob in logprobs.content:
        assert isinstance(logprob.token, str)
        assert isinstance(logprob.logprob, float)
        assert len(logprob.top_logprobs) == top_logprobs
        top_logprob = logprob.top_logprobs[0]
        assert isinstance(top_logprob.token, str)
        assert isinstance(top_logprob.logprob, float)

    usage = completion.usage
    assert usage.prompt_tokens == 36
    assert usage.completion_tokens == 1
    assert usage.total_tokens == 37


def test_chat_streaming_response(openai_client):
    top_logprobs = 3

    request_timestamp = datetime.datetime.now().timestamp()

    stream = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Give me a random letter."}],
        max_completion_tokens=1,
        logprobs=True,
        top_logprobs=top_logprobs,
        stream=True,
        stream_options={
            "include_usage": True
        }
    )

    chunks = list(stream)

    # All chunks have the same id.
    completion_id = chunks[0].id
    assert completion_id
    for chunk in chunks:
        assert chunk.id == completion_id

    # Sets model name
    for chunk in chunks:
        assert chunk.model == MODEL

    # Sets correct object type
    for chunk in chunks:
        assert chunk.object == 'chat.completion.chunk'

    # Sets creation timestamp
    for chunk in chunks:
        assert chunk.created - request_timestamp < 5  # seconds

    # Sets system fingerprint
    for chunk in chunks:
        assert chunk.system_fingerprint

    # Returns expected role.
    assert chunks[0].choices[0].delta.role == "assistant"

    # Choice has correct index set (multiple choices not supported yet).
    assert chunks[0].choices[0].index == 0

    # Some chunks (eg initial, stop, last) may not have content.
    content_chunks = [chunk for chunk in chunks
                      if chunk.choices and chunk.choices[0].delta.content]

    assert len(content_chunks) > 0

    for chunk in content_chunks:
        logprobs = chunk.choices[0].logprobs
        for logprob in logprobs.content:
            assert isinstance(logprob.token, str)
            assert isinstance(logprob.logprob, float)
            assert len(logprob.top_logprobs) == top_logprobs
            top_logprob = logprob.top_logprobs[0]
            assert isinstance(top_logprob.token, str)
            assert isinstance(top_logprob.logprob, float)

    usage = chunks[-1].usage
    assert usage.prompt_tokens == 36
    assert usage.completion_tokens == 1
    assert usage.total_tokens == 37
    # Only the last chunk has usage information.
    for chunk in chunks[:-1]:
        assert chunk.usage is None


def test_chat_developer_role(openai_client):
    """Test completion request and check the received message."""
    # Only check that developer and name arguments are accepted.
    # `role` is transmitted to the model as is
    # `name` is ignored
    completion = openai_client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=10,
        messages=[
            {
                "role": "developer",
                "content": "The year is 1995.",
                "name": "Nekko",
            },
            {
                "role": "user",
                "content": "What year it is?",
                "name": "Pier",
            }
        ],
    )

    assert completion.choices[0].message.content
