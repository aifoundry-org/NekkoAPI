import uuid
import openai
import pytest


@pytest.fixture
def setup_openai_client():
    """Fixture to set up OpenAI client with the API key."""
    openai.api_key = str(uuid.uuid4())
    if not openai.api_key:
        pytest.fail("OPENAI_API_KEY environment variable is not set")


def test_openai_completion(setup_openai_client):
    """Test API call and check for 200 OK response."""
    url = "http://localhost:8000/v1/"
    model = "smolm2-135m"
    try:
        client = openai.OpenAI(
            base_url=url, api_key=openai.api_key
        )
        # Make a basic completion request
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assitant named Nekko. " \
                           "For some reason you like cats. " \
                           "You always answer in numbered lists, top 3 items only."
            },
            {
                "role": "user",
                "content": "What should I see in Japan? Thanks!"
            }
        ]

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=200,
            stop=["4.", "sushi"],
            top_p=0.3,
            stream=True
        )

        # Assert the response is OK
        assert stream.response.status_code == 200

    except openai.OpenAIError as e:
        pytest.fail(f"OpenAI API call failed: {e}")
