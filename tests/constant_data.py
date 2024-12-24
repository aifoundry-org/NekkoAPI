class ConstantData:
    """
    Contains all the constant data that can be used across the framework
    """
    MESSAGE_BASIC = [
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

    MESSAGE_LOGPROBS = [
        {
            "role": "system",
            "content": "You are a helpful AI assitant named Nekko."
        },
        {
            "role": "user",
            "content": "What is the highest mountain? Answer with a single word."
        }
    ]