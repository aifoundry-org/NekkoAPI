import uuid
import openai
import pytest
from constant_data import ConstantData

"""
Data for specific tests
"""
CHAT_COMPLETION_BASIC = (
    "models/SmolLM2-135M-Instruct-Q6_K.gguf",  # Model
    ConstantData.MESSAGE,  # Messages
    200,  # Max completion tokens
    ["4.", "sushi"],  # Stop tokens
    0.3,  # Top_p
    True,  # Stream option
    False, # frequency_penalty,
    False, # presence_penalty
    None,  # seed
)

CHAT_COMPLETION_FREQUENCY_PENALTY = (
    "models/SmolLM2-135M-Instruct-Q6_K.gguf",  # Model
    ConstantData.MESSAGE,  # Messages
    200,  # Max completion tokens
    ["4.", "sushi"],  # Stop tokens
    0.3,  # Top_p
    True,  # Stream option
    2.0,   # frequency_penalty
    False, # presence_penalty
    None,  # seed
)

CHAT_COMPLETION_PRESENCE_PENALTY = (
    "models/SmolLM2-135M-Instruct-Q6_K.gguf",  # Model
    ConstantData.MESSAGE,  # Messages
    200,  # Max completion tokens
    ["4.", "sushi"],  # Stop tokens
    0.3,  # Top_p
    True,  # Stream option
    False,  # frequency_penalty
    2.0,    # presence_penalty
    None,   # seed
)

CHAT_COMPLETION_SEED = (
    "models/SmolLM2-135M-Instruct-Q6_K.gguf",  # Model
    ConstantData.MESSAGE,  # Messages
    200,  # Max completion tokens
    ["4.", "sushi"],  # Stop tokens
    0.3,  # Top_p
    True,  # Stream option
    False,  # frequency_penalty
    False,  # presence_penalty
    1337,   # seed
)


@pytest.fixture(scope="session")
def setup_openai_client():
    """Fixture to set up OpenAI client with the API key."""
    openai.api_key = str(uuid.uuid4())
    if not openai.api_key:
        pytest.fail("OPENAI_API_KEY environment variable is not set")


@pytest.mark.parametrize(
    "model, messages, max_completion_tokens, stop, top_p, stream_option, frequency_penalty, presence_penalty, seed",
    [
        CHAT_COMPLETION_BASIC,
        CHAT_COMPLETION_FREQUENCY_PENALTY,
        CHAT_COMPLETION_PRESENCE_PENALTY,
        CHAT_COMPLETION_SEED
    ]
)
def test_openai_completion(setup_openai_client, model,
                           messages, max_completion_tokens,
                           stop, top_p, stream_option,
                           frequency_penalty, presence_penalty,
                           seed):
    """Test API call and check for 200 OK response."""
    url = "http://localhost:8000/v1/"

    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )
        # Make a basic completion request
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            top_p=top_p,
            stream=stream_option,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed
        )

        # Assert the response is OK
        assert stream.response.status_code == 200

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")
