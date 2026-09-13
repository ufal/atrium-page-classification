"""Repo-local declarations for tests/test_env_contract.py (atrium-project#60).

Never vendored, never in para-drift, never in docs/templates/ruff.toml's [format]
exclude — unlike test_env_contract.py itself, this file's SHAPE is per-repo by
design. See the canonical test's module docstring for the full rationale.
"""

from __future__ import annotations

# Read by shipped code but deliberately absent from .env.example, each with a
# reason. Empty here: page-classification's service layer reads nothing beyond the
# shared set (service/inference.py, service/document_json.py, service/api_client.py
# have zero environment reads), and the batch CLI is argparse/config.txt only.
NOT_PUBLISHED: dict[str, str] = {}

# In .env.example but read by no Python in this repo — each with a reason.
CONSUMED_ELSEWHERE: dict[str, str] = {
    "ATRIUM_VERSION": "read only by docker-compose.yml to pick the image tag; no Python here reads it",
    "HF_HOME": "read by huggingface_hub itself, set by the Dockerfile and docker-compose.yml",
    "HF_TOKEN": "read by huggingface_hub itself for gated models/push_to_hub; no Python in this repo reads it directly",
}

# service/README.md or .env.example cells whose value is prose rather than a literal
# the code-default resolver can compare against.
PROSE_DEFAULTS: dict[str, str] = {}
