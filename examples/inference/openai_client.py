"""Call a Basilica hosted model through the direct Envoy serving host.

Install the OpenAI Python client first:

    pip install openai

Then set:

    export BASILICA_INFERENCE_BASE_URL="https://tiny-gpt2.deployments.basilica.ai/v1"
    export BASILICA_API_KEY="basilica_..."
    export BASILICA_MODEL="tiny-gpt2"
"""

from __future__ import annotations

import os

from openai import OpenAI


def main() -> None:
    client = OpenAI(
        base_url=os.environ["BASILICA_INFERENCE_BASE_URL"],
        api_key=os.environ["BASILICA_API_KEY"],
    )
    response = client.chat.completions.create(
        model=os.environ.get("BASILICA_MODEL", "tiny-gpt2"),
        messages=[{"role": "user", "content": "Write one sentence about Basilica."}],
        max_tokens=32,
    )
    print(response.choices[0].message.content)
    print(f"request_id={response.id}")
    print(f"usage={response.usage}")


if __name__ == "__main__":
    main()
