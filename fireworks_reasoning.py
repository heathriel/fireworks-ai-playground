from fireworks import Fireworks

client = Fireworks()

response = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v3p2",
    messages=[
        {"role": "user", "content": "What is 25 * 37? Show your work."}
    ],
    reasoning_effort="medium",
)

msg = response.choices[0].message
if msg.reasoning_content:
    print("Reasoning:", msg.reasoning_content)
print("Answer:", msg.content)
