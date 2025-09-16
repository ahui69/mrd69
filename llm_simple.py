import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = os.getenv("LLM_BASE_URL", "https://api.deepinfra.com/v1/openai")
KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")


def chat(
    user_text: str,
    system_text: str = "You are Mordzix.",
    max_tokens: int = 600,
) -> str:
    if not KEY:
        raise ValueError("LLM_API_KEY not found in .env file")
    r = requests.post(
        BASE + "/chat/completions",
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    return (j.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
