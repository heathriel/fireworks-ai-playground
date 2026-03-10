from fireworks import Fireworks

client = Fireworks()

response = client.chat.completions.create(
  model="accounts/fireworks/models/kimi-k2p5",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://storage.googleapis.com/fireworks-public/image_assets/fireworks-ai-wordmark-color-dark.png"
          },
        },
      ],
    }
  ],
)

print(response.choices[0].message.content)
