import os

from openai import OpenAI

# Utwórz klienta OpenAI z tokenem DeepInfra
openai = OpenAI(
    api_key=os.getenv("LLM_API_KEY", ""),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai"),
)

# Wywołaj model Kimi-K2-Instruct
chat_completion = openai.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[
        {"role": "system", "content": "Jesteś pomocnym asystentem AI, który mówi po polsku."},
        {"role": "user", "content": ("Przedstaw się i powiedz co potrafisz zrobić w 3 zdaniach.")},
    ],
    temperature=0.7,
    max_tokens=500,
)

# Wydrukuj odpowiedź
print(chat_completion.choices[0].message.content)

# Wydrukuj pełną odpowiedź jako obiekt JSON
print("\n--- Pełna odpowiedź ---")
print(chat_completion)
