#!/usr/bin/env python3
"""
Miner Model Evaluation - Deploy evaluation environments to test LLMs on Chutes.

This example deploys evaluation containers that test miner models by:
  1. Generating verifiable challenges (math, code, reasoning)
  2. Sending them to the miner's model via Chutes API
  3. Scoring the responses objectively

Two evaluation suites are available:

  AFFINE (bignickeye/affine:latest):
    - SAT: Boolean satisfiability problems
    - ABD: Reverse-engineer program inputs from outputs
    - DED: Generate correct Python code from requirements

  AGENTGYM (bignickeye/agentgym:{env}):
    - webshop: E-commerce web navigation
    - alfworld: Text-based household tasks
    - babyai: Grid-world language instructions
    - sciworld: Scientific reasoning tasks

Usage:
    export BASILICA_API_TOKEN="your-token"
    export CHUTES_API_KEY="your-chutes-api-key"
    python3 11_agentgym.py [affine|agentgym]
"""
import os
import sys

from basilica import BasilicaClient, Deployment


# =============================================================================
# AFFINE EVALUATOR - SAT/ABD/DED reasoning tasks
# =============================================================================


def deploy_affine_evaluator(
    client: BasilicaClient,
    api_key: str | None = None,
) -> Deployment:
    """
    Deploy Affine evaluator for reasoning tasks.

    Evaluates miner models on:
      - SAT: Find satisfying assignments for k-SAT formulas
      - ABD: Reverse-engineer inputs from program outputs
      - DED: Generate working Python code from requirements

    Args:
        client: Basilica client instance
        api_key: Chutes API key (optional, can be passed per-request)

    Returns:
        Deployment instance
    """
    print("Deploying Affine evaluator (SAT/ABD/DED tasks)...")

    env_vars = {}
    if api_key:
        env_vars["CHUTES_API_KEY"] = api_key

    deployment = client.deploy(
        name="affine-evaluator",
        image="bignickeye/affine:latest",
        port=8000,
        env=env_vars,
        cpu="500m",
        memory="1Gi",
        ttl_seconds=3600,
        timeout=180,
    )

    print(f"  URL: {deployment.url}")
    return deployment


# =============================================================================
# AGENTGYM EVALUATOR - Interactive agent tasks
# =============================================================================


def deploy_agentgym_evaluator(
    client: BasilicaClient,
    env_name: str = "webshop",
    api_key: str | None = None,
) -> Deployment:
    """
    Deploy AgentGym evaluator for interactive agent tasks.

    Available environments:
      - webshop: E-commerce web navigation
      - alfworld: Text-based household tasks
      - babyai: Grid-world language instructions
      - sciworld: Scientific reasoning tasks
      - textcraft: Interactive fiction games

    Args:
        client: Basilica client instance
        env_name: AgentGym environment name
        api_key: Chutes API key (optional, can be passed per-request)

    Returns:
        Deployment instance
    """
    print(f"Deploying AgentGym evaluator ({env_name})...")

    env_vars = {"ENV_NAME": env_name}
    if api_key:
        env_vars["CHUTES_API_KEY"] = api_key

    deployment = client.deploy(
        name=f"agentgym-{env_name}",
        image=f"bignickeye/agentgym:{env_name}",
        port=8000,
        env=env_vars,
        cpu="500m",
        memory="1Gi",
        ttl_seconds=3600,
        timeout=180,
    )

    print(f"  URL: {deployment.url}")
    return deployment


# =============================================================================
# USAGE EXAMPLES
# =============================================================================


def print_affine_usage(url: str, api_key: str | None):
    """Print curl examples for Affine evaluator."""
    api_key_str = api_key or "YOUR_CHUTES_API_KEY"

    print(
        f"""
Affine Evaluator API:
---------------------
# Health check
curl {url}/health

# List available methods
curl {url}/methods

# Evaluate SAT task (boolean satisfiability)
curl -X POST {url}/call \\
  -H "Content-Type: application/json" \\
  -d '{{
    "method": "evaluate",
    "kwargs": {{
      "task_type": "sat",
      "model": "deepseek-ai/DeepSeek-V3",
      "base_url": "https://llm.chutes.ai/v1",
      "num_samples": 5,
      "api_key": "{api_key_str}"
    }}
  }}'

# Evaluate ABD task (reverse-engineer inputs)
curl -X POST {url}/call \\
  -H "Content-Type: application/json" \\
  -d '{{
    "method": "evaluate",
    "kwargs": {{
      "task_type": "abd",
      "model": "Qwen/Qwen2.5-72B-Instruct",
      "base_url": "https://llm.chutes.ai/v1",
      "num_samples": 3,
      "api_key": "{api_key_str}"
    }}
  }}'

# Evaluate DED task (code generation)
curl -X POST {url}/call \\
  -H "Content-Type: application/json" \\
  -d '{{
    "method": "evaluate",
    "kwargs": {{
      "task_type": "ded",
      "model": "deepseek-ai/DeepSeek-V3",
      "base_url": "https://llm.chutes.ai/v1",
      "num_samples": 3,
      "api_key": "{api_key_str}"
    }}
  }}'
"""
    )


def print_agentgym_usage(url: str, env_name: str, api_key: str | None):
    """Print curl examples for AgentGym evaluator."""
    api_key_str = api_key or "YOUR_CHUTES_API_KEY"

    print(
        f"""
AgentGym Evaluator API ({env_name}):
------------------------------------
# Health check
curl {url}/health

# Evaluate task
curl -X POST {url}/evaluate \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "base_url": "https://llm.chutes.ai/v1",
    "task_id": 0,
    "max_round": 10,
    "api_key": "{api_key_str}"
  }}'
"""
    )


def main():
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("Warning: CHUTES_API_KEY not set")
        print("  You'll need to pass api_key in each request")
        print()

    # Parse CLI args
    evaluator_type = "affine"
    env_name = "webshop"

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("affine", "sat", "abd", "ded"):
            evaluator_type = "affine"
        elif arg in (
            "agentgym",
            "webshop",
            "alfworld",
            "babyai",
            "sciworld",
            "textcraft",
        ):
            evaluator_type = "agentgym"
            if arg != "agentgym":
                env_name = arg
        else:
            print(f"Unknown evaluator: {arg}")
            print(
                "Usage: python3 11_agentgym.py [affine|agentgym|webshop|alfworld|...]"
            )
            sys.exit(1)

    client = BasilicaClient()

    print("=" * 60)
    print("Miner Model Evaluation")
    print("=" * 60)

    if evaluator_type == "affine":
        deployment = deploy_affine_evaluator(client, api_key)
        print_affine_usage(deployment.url, api_key)
    else:
        deployment = deploy_agentgym_evaluator(client, env_name, api_key)
        print_agentgym_usage(deployment.url, env_name, api_key)

    print("=" * 60)
    print(f"Deployment URL: {deployment.url}")
    print(f"TTL: 1 hour (auto-cleanup)")
    print("=" * 60)


if __name__ == "__main__":
    main()
