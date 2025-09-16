import os

from openai import OpenAI


# Klient OpenAI dla DeepInfra
def create_kimi_client():
    return OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai"),
    )


# Wywołanie modelu Kimi-K2-Instruct
def kimi_chat(messages, max_tokens=500, temperature=0.7) -> str:
    client = create_kimi_client()
    chat_completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return chat_completion.choices[0].message.content or ""
