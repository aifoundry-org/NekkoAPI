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
	# temperature=2.0,
	stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is None:
    	continue
    print(chunk.choices[0].delta.content, end="")

print('\n')
