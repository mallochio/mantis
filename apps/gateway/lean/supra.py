"""Supra model loading and complexity scoring."""

from __future__ import annotations

import re
import time
from functools import lru_cache

_supra_model = None
_supra_tokenizer = None
SUPRA_FALLBACK_COUNT = 0


def _load_supra():
    global _supra_model, _supra_tokenizer
    if _supra_model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("SupraLabs/Supra-Router-51M")
    model = AutoModelForCausalLM.from_pretrained("SupraLabs/Supra-Router-51M", torch_dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.eval()
    _supra_tokenizer, _supra_model = tok, model


@lru_cache(maxsize=512)
def _supra_complexity(prompt: str) -> tuple[int, int]:
    from transformers import StoppingCriteria

    model, tokenizer = _supra_model, _supra_tokenizer

    class _Seen(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return (
                re.search(r"Complexity:\s*\d", tokenizer.decode(input_ids[0][-24:], skip_special_tokens=True))
                is not None
            )

    import torch

    inputs = tokenizer(
        f"Task: {prompt}\nAnalysis: ", return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length
    )
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=[_Seen()],
        )
    ms = int((time.time() - t0) * 1000)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    c = 0
    for part in gen.split("|"):
        m = re.search(r"Complexity:\s*(\d)", part, re.I)
        if m:
            c = int(m.group(1))
            break
    return c, ms
