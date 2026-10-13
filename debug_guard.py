
import os

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIStatusError,
    APIConnectionError,
    APITimeoutError,
)

from app import make_guard_prompt, parse_guard

load_dotenv()

model = os.getenv(
    "GUARD_MODEL",
    "meta-llama/llama-guard-4-12b:free"
)

client = OpenAI(
    base_url=os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    ),
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=30,
    max_retries=0,
)

print("Testing model:", model)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": make_guard_prompt(
                    "How long does standard shipping take?"
                )
            }
        ],
        temperature=0,
        max_tokens=64,
    )

    raw = response.choices[0].message.content or ""
    decision = parse_guard(raw)

    print("API request: SUCCESS")
    print("Model returned:", response.model)
    print("Raw classification:", repr(raw))
    print("Allowed:", decision.allowed)
    print("Reason:", decision.reason)

except APIStatusError as error:
    print("API request: FAILED")
    print("HTTP status:", error.status_code)
    print("Error type:", type(error).__name__)

except (APIConnectionError, APITimeoutError) as error:
    print("Connection failure:", type(error).__name__)
