"""Tests for Conductor device/dtype auto-detection in openfugu-patch/serve.py."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_serve_module() -> ModuleType:
    """Load serve.py as a module despite the hyphen in the directory name."""
    spec = importlib.util.spec_from_file_location(
        "fugu_serve", REPO_ROOT / "openfugu-patch" / "serve.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # serve.py needs to find upstream OpenFugu packages.
    sys.path.insert(0, str(REPO_ROOT / "OpenFugu" / "openfugu"))
    sys.path.insert(0, str(REPO_ROOT / "openfugu-patch"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    return module


def _mock_torch(*, mps: bool = False, cuda: bool = False) -> MagicMock:
    torch = MagicMock()
    torch.backends.mps.is_available.return_value = mps
    torch.cuda.is_available.return_value = cuda
    torch.bfloat16 = "bf16"
    torch.float32 = "f32"
    torch.float16 = "f16"
    return torch


def test_choose_device_prefers_mps():
    serve = _load_serve_module()
    torch = _mock_torch(mps=True, cuda=True)
    assert serve.choose_conductor_device(torch) == "mps"


def test_choose_device_prefers_cuda_when_no_mps():
    serve = _load_serve_module()
    torch = _mock_torch(mps=False, cuda=True)
    assert serve.choose_conductor_device(torch) == "cuda:0"


def test_choose_device_falls_back_to_cpu():
    serve = _load_serve_module()
    torch = _mock_torch(mps=False, cuda=False)
    assert serve.choose_conductor_device(torch) == "cpu"


def test_choose_device_explicit_env_override():
    serve = _load_serve_module()
    torch = _mock_torch(mps=True, cuda=True)
    assert serve.choose_conductor_device(torch, env_device="cpu") == "cpu"


def test_choose_device_auto_does_not_override():
    serve = _load_serve_module()
    torch = _mock_torch(mps=False, cuda=True)
    assert serve.choose_conductor_device(torch, env_device="auto") == "cuda:0"


def test_choose_dtype_defaults():
    serve = _load_serve_module()
    torch = _mock_torch()
    assert serve.choose_conductor_dtype("mps", torch) is torch.bfloat16
    assert serve.choose_conductor_dtype("cuda:0", torch) is torch.bfloat16
    assert serve.choose_conductor_dtype("cpu", torch) is torch.float32


def test_choose_dtype_env_override():
    serve = _load_serve_module()
    torch = _mock_torch()
    assert serve.choose_conductor_dtype("mps", torch, env_dtype="float32") is torch.float32
    assert serve.choose_conductor_dtype("cpu", torch, env_dtype="bfloat16") is torch.bfloat16


def test_env_vars_are_respected(monkeypatch: pytest.MonkeyPatch):
    """EnvLocalConductor propagates MANTIS_CONDUCTOR_DEVICE and MANTIS_CONDUCTOR_DTYPE."""
    serve = _load_serve_module()
    monkeypatch.setenv("MANTIS_CONDUCTOR_DEVICE", "cuda:0")
    monkeypatch.setenv("MANTIS_CONDUCTOR_DTYPE", "float16")

    torch = _mock_torch(mps=False, cuda=False)
    env_device = os.environ.get("MANTIS_CONDUCTOR_DEVICE")
    env_dtype = os.environ.get("MANTIS_CONDUCTOR_DTYPE")
    device = serve.choose_conductor_device(torch, env_device)
    dtype = serve.choose_conductor_dtype(device, torch, env_dtype)
    assert device == "cuda:0"
    assert dtype is torch.float16
