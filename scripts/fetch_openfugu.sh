#!/usr/bin/env bash
# Clone the pinned OpenFugu commit if the submodule is not present.
set -euo pipefail

COMMIT="7ad7ccf977c1b5f38bbd07ba33d86fe655c17be8"

if [ -d "OpenFugu/openfugu" ]; then
  echo "[fetch_openfugu] OpenFugu already present"
  exit 0
fi

if [ -d ".git" ] && [ -f ".gitmodules" ]; then
  echo "[fetch_openfugu] initializing OpenFugu submodule..."
  git submodule update --init --recursive
  if [ -d "OpenFugu/openfugu" ]; then
    exit 0
  fi
fi

rm -rf OpenFugu
mkdir -p OpenFugu
cd OpenFugu
git init --quiet
git remote add origin https://github.com/trotsky1997/OpenFugu.git
git fetch --depth 1 origin "$COMMIT"
git checkout --quiet "$COMMIT"
cd ..
echo "[fetch_openfugu] checked out OpenFugu at $COMMIT"
