#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: OpenAI-compatible serving layer for the OpenFugu TRINITY + Conductor coordinators.
"""
serve.py — Fugu as a single OpenAI-compatible model endpoint.

A client POSTs to /v1/chat/completions as if calling one model; internally the
requested coordinator ("trinity" or "conductor") runs the full loop. The
model field in the request selects the coordinator.

stdlib http.server only — no FastAPI/uvicorn.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# When running from the repo's openfugu-patch overlay, also find upstream OpenFugu.
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))


import sys
import types

import conductor
import providers
import serve_config
import trinity
import utils
from conductor import *
from providers import *
from serve_config import *
from trinity import *
from utils import *

import runs
from runs import *


class ServeProxy(types.ModuleType):
    def __getattr__(self, name):
        for mod in (providers, runs, trinity, conductor, utils, serve_config):
            if hasattr(mod, name):
                return getattr(mod, name)
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        for mod in (providers, runs, trinity, conductor, utils, serve_config):
            if hasattr(mod, name):
                setattr(mod, name, value)
        super().__setattr__(name, value)


try:
    sys.modules[__name__].__class__ = ServeProxy
except KeyError:
    pass


def __getattr__(name):
    for mod in (providers, runs, trinity, conductor, utils, serve_config):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
