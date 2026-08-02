#!/usr/bin/env bash
# Copy from ~/.config/llm-router if present, or clone the repository into ./llm-router.
set -euo pipefail

if [ -d "llm-router" ] && [ -f "llm-router/server.py" ]; then
  echo "[fetch_llm_router] llm-router already present in workdir"
  exit 0
fi

if [ -d "$HOME/.config/llm-router" ] && [ -f "$HOME/.config/llm-router/server.py" ]; then
  echo "[fetch_llm_router] copying from $HOME/.config/llm-router..."
  mkdir -p llm-router
  cp -R "$HOME/.config/llm-router/"* llm-router/
  echo "[fetch_llm_router] copied llm-router from local config"
  exit 0
fi

echo "[fetch_llm_router] cloning llm-router repository..."
git clone --depth 1 https://github.com/mallochio/llm-router.git llm-router
echo "[fetch_llm_router] llm-router ready"
