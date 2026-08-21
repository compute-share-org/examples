#!/usr/bin/env bash
set -euo pipefail

apt-get update -qq && apt-get install -y -qq curl && rm -rf /var/lib/apt/lists/*
curl -LsSf https://astral.sh/uv/install.sh | sh

source "$HOME/.local/bin/env"

uv sync
