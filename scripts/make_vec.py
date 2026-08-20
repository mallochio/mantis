import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

VEC_LEN = 19456
SVF_LEN = 9216


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("head-only", "full"), default="head-only")
    parser.add_argument(
        "--input",
        default=os.environ.get("MANTIS_ROUTER_HEAD", "artifacts/router_head.safetensors"),
    )
    parser.add_argument(
        "--output", default=os.environ.get("MANTIS_VECTOR_OUT", "artifacts/model_iter_60.npy")
    )
    args = parser.parse_args()
    if not args.input or not Path(args.input).exists():
        parser.error(f"--input or MANTIS_ROUTER_HEAD file not found: {args.input}")
    if args.mode == "head-only":
        with safe_open(args.input, framework="pt") as handle:
            head = handle.get_tensor("trinity_router_head")
        print("head shape", head.shape, head.dtype, flush=True)
        head = head.to(torch.float32).numpy().reshape(-1)
        if head.shape[0] != VEC_LEN - SVF_LEN:
            raise ValueError(f"head length {head.shape[0]} != {VEC_LEN - SVF_LEN}")
        vec = np.zeros(VEC_LEN, dtype=np.float64)
        vec[SVF_LEN:] = head.astype(np.float64)
    else:
        vec = np.load(args.input).astype(np.float64)
        if vec.shape != (VEC_LEN,):
            raise ValueError(f"full vector must be {VEC_LEN} floats, got {vec.shape}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vec)
    provenance = {
        "mode": args.mode,
        "input": str(args.input),
        "vector_length": VEC_LEN,
        "svf_length": SVF_LEN,
        "head_length": VEC_LEN - SVF_LEN,
        "svf_offsets_all_zero": bool(np.all(vec[:SVF_LEN] == 0)),
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("wrote", out, vec.shape, vec.dtype, "mode", args.mode, flush=True)


if __name__ == "__main__":
    main()
