from fireworks import Fireworks

client = Fireworks()

response = client.chat.completions.create(
  model="accounts/fireworks/models/llama-v3p3-70b-instruct",
  messages=[
    {
      "role": "user",
      "content": "Extract the name and age from: John is 30 years old",
    }
  ],
  response_format={
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "schema": {
        "type": "object",
        "properties": {
          "name": { "type": "string" },
          "age": { "type": "number" }
        },
        "required": ["name", "age"],
      },
    },
  },
)

print(response.choices[0].message.content)
