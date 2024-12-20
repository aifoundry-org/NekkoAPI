class ConstantData:
    """
    Contains all the constant data that can be used across the framework
    """
    MESSAGE = [
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


class TestData:
    """
    Contains data for specific tests
    """
    CHAT_COMPLETION_BASIC = (
        "models/SmolLM2-135M-Instruct-Q6_K.gguf",  # Model
        ConstantData.MESSAGE,  # Messages
        200,  # Max completion tokens
        ["4.", "sushi"],  # Stop tokens
        0.3,  # Top_p
        True,  # Stream option
        None  # frequency_penalty
    )

    CHAT_COMPLETION_FREQUENCY_PENALTY = (
        "models/SmolLM2-135M-Instruct-Q6_K.gguf",  # Model
        ConstantData.MESSAGE,  # Messages
        200,  # Max completion tokens
        ["4.", "sushi"],  # Stop tokens
        0.3,  # Top_p
        True,  # Stream option
        2.0  # frequency_penalty
    )

