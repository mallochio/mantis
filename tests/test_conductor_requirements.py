"""Regression tests pinning requirements-conductor.txt to its generator.

requirements-conductor.txt is a generated artifact of pyproject.toml's
`train` extra plus a curated additions list in
scripts/generate_conductor_requirements.py. These tests fail loudly if
someone edits the checked-in file by hand or the two sources drift apart.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GENERATOR = REPO / "scripts" / "generate_conductor_requirements.py"
CHECKED_IN = REPO / "requirements-conductor.txt"

EXPECTED_PACKAGES = {
    "accelerate",
    "bitsandbytes",
    "boto3",
    "cma",
    "datasets",
    "huggingface-hub",
    "hydra-core",
    "litellm",
    "llm-blender",
    "math-verify",
    "mergekit",
    "numpy",
    "omegaconf",
    "peft",
    "safetensors",
    "torch",
    "transformers",
    "trl",
}


def _package_name(line: str) -> str:
    for index, char in enumerate(line):
        if not (char.isalnum() or char in "-_."):
            return line[:index].replace("_", "-").lower()
    return line.replace("_", "-").lower()


def _package_names(path: Path) -> set[str]:
    return {
        _package_name(line)
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    }


def test_generated_file_is_current(tmp_path: Path) -> None:
    output = tmp_path / "requirements.txt"
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(output)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == CHECKED_IN.read_bytes()


def test_package_set() -> None:
    names = _package_names(CHECKED_IN)
    assert names == EXPECTED_PACKAGES


def test_header_marks_generated() -> None:
    first_line = CHECKED_IN.read_text().splitlines()[0]
    assert "do not edit by hand" in first_line
