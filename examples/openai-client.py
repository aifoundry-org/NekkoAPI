from openai import OpenAI

# model = "llama"
model = "smolm2-135m"
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
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is None:
            continue
        print(chunk.choices[0].delta.content, end="")

    print('\n')


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


def main():
    example_structured_output()


if __name__ == "__main__":
    main()
