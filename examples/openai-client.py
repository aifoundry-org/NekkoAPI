import json
from openai import OpenAI

model = "llama"
# model = "smolm2-135m"
# model = "olmo-7b"
uri = "http://localhost:8000/v1/"

# model = "gpt-4o-mini"
# uri = "https://api.openai.com/v1/"

client = OpenAI(
    base_url=uri
)


def example_simple():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assitant named Nekko. "
                       "For some reason you like cats. "
                       "Answer in numbered list."
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
        # temperature=2.0,
        # Multiple forms of the word time: " time", "time", "Time" etc
        # Assumes GPT-4 tokenizer (works with llama models)
        # logit_bias={ 1712: -100, 3115: -100, 15487: -100, 892: -100, 1489: -100 },
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is None:
            continue
        print(chunk.choices[0].delta.content, end="")

    print('\n')


def example_tool_calls():
    weather = "Steady light rain this evening. Showers continuing overnight. Low 44F. Winds SSW at 10 to 20 mph. Chance of rain 80%."
    tools = [
        {
            "type": "function",
            "function": {
                "description": "Display weather forecast on a wall",
                "name": "display_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                      "temperature": {
                        "type": "number",
                      },
                      "wind_speed": {
                        "type": "number"
                      },
                      "rain": {
                        "type": "boolean"
                      }
                    },
                    "required": [
                      "temperature",
                      "wind_speed",
                      "rain"
                    ],
                    "additionalProperties": False
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assitant named Nekko." \
        },
        {
            "role": "user",
            "content": f"Please display weather forecast using `display_weather` function." +
                f"The input is bellow: {weather}"
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "display_weather"}
        },
    )

    message = completion.choices[0].message
    if message.content is not None:
        print(message.content)
    if message.tool_calls is not None:
        for tool_call in message.tool_calls:
            print(f"TOOL_CALL {tool_call.function.name}({tool_call.function.arguments})")


def example_structured_output():
    weather = """
    Steady light rain this evening.
    Showers continuing overnight.
    Low 44F. Winds SSW at 10 to 20 mph.
    Chance of rain 80%.
    """
    schema = {
      "name": "weather_forecast",
      "schema": {
        "type": "object",
        "properties": {
          "description": {
            "type": "string",
            "maxLength": 20
          },
          "temperature": {
            "type": "number",
          },
          "wind_speed": {
            "type": "number"
          },
          "wind_direction": {
            "type": "string"
          },
          "rain": {
            "type": "boolean"
          }
        },
        "required": [
          "description",
          "temperature",
          "wind_speed",
          "wind_direction",
          "rain"
        ],
        "additionalProperties": False
      },
      "strict": True
    }
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assitant named Nekko."
        },
        {
            "role": "user",
            "content": "Please output weather forecast as JSON. " +
                       f"The input is bellow: {weather}"
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": schema
        }
    )

    message = completion.choices[0].message
    if message.content is not None:
        print(message.content)


def example_logprobs():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assitant named Nekko."
        },
        {
            "role": "user",
            "content": "What is the highest mountain? Answer with a single word."
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=20,
        logprobs=True,
        top_logprobs=3
    )

    logprobs = completion.choices[0].logprobs
    print(json.dumps(logprobs.model_dump()))


def main():
    example_logprobs()


if __name__ == "__main__":
    main()
