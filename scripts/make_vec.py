import os

import numpy as np
import torch
from safetensors import safe_open

VEC_LEN = 19456
SVF_LEN = 9216

input_path = os.environ.get("FUGU_ROUTER_HEAD", "artifacts/router_head.safetensors")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"router head not found: {input_path}")

with safe_open(input_path, framework='pt') as f:
    head = f.get_tensor('trinity_router_head')

print('head shape', head.shape, head.dtype, flush=True)
head = head.to(torch.float32).numpy().reshape(-1)
if head.shape[0] != VEC_LEN - SVF_LEN:
    raise ValueError(f'head length {head.shape[0]} != {VEC_LEN - SVF_LEN}')

vec = np.zeros(VEC_LEN, dtype=np.float64)
vec[SVF_LEN:] = head.astype(np.float64)

out = os.environ.get("FUGU_VECTOR_OUT", "artifacts/model_iter_60.npy")
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
np.save(out, vec)
print('wrote', out, vec.shape, vec.dtype, flush=True)
