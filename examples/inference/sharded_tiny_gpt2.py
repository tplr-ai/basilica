"""Deploy a Basilica sharded inference pipeline using a GPT-2-style smoke model.

Requires a logged-in Basilica CLI or BASILICA_API_TOKEN for the CLI process.
The resulting URL is OpenAI-compatible:

    {url}/v1/chat/completions
"""

import subprocess


def main() -> None:
    subprocess.run(
        [
            "basilica",
            "deploy",
            "sharded",
            "sshleifer/tiny-gpt2",
            "--stages",
            "2",
            "--gpu-model",
            "A100",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
