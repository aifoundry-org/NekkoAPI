class ConstantData:
    """
    Contains all the constant data that can be used across the framework
    """
    MESSAGE_BASIC = [
        {
            "role": "user",
            "content": "What should I see in Japan? Thanks!"
        }
    ]

    MESSAGE_LOGPROBS = [
        {
            "role": "user",
            "content": "What is the highest mountain? Answer with a single word."
        }
    ]

    MESSAGE_SEED = [
        {
            "role": "user",
            "content": "What is result of 1+1?"
        }
    ]

    MESSAGE_TOOLS = [
        {
            "role": "user",
            "content": "Please display weather forecast using `display_weather` function."
                "The input is bellow: Steady light rain this evening. Showers continuing overnight. Low 44F. Winds "
                "SSW at 10 to 20 mph. Chance of rain 80%."
        }
    ]

    MESSAGE_STOP = [
        {
            "role": "user",
            "content": "Write the English alphabet"
        }
    ]

    TOOLS_FUNCTION = [
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
