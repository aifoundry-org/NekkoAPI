import uuid
import openai
import pytest
import constant_data


@pytest.fixture(scope="session")
def setup_openai_client():
    """Fixture to set up OpenAI client with the API key."""
    openai.api_key = str(uuid.uuid4())
    if not openai.api_key:
        pytest.fail("OPENAI_API_KEY environment variable is not set")


@pytest.mark.parametrize(
    "model, messages, max_completion_tokens, stop, top_p, stream_option, frequency_penalty",
    [
        constant_data.TestData.CHAT_COMPLETION_BASIC,
        constant_data.TestData.CHAT_COMPLETION_FREQUENCY_PENALTY
    ]
)
def test_openai_completion(model, messages, max_completion_tokens, stop, top_p, stream_option, frequency_penalty):
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
        )

        # Assert the response is OK
        assert stream.response.status_code == 200

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")
