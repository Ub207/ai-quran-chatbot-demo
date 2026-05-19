from groq import Groq

client = Groq(api_key="PASTE_YOUR_KEY")

res = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "hello"}]
)

print(res.choices[0].message.content)