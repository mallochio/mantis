import torch, numpy as np
from safetensors import safe_open

VEC_LEN = 19456
SVF_LEN = 9216

path = '/app/artifacts/router_head.safetensors'
with safe_open(path, framework='pt') as f:
    head = f.get_tensor('trinity_router_head')
print('head shape', head.shape, head.dtype, flush=True)
head = head.to(torch.float32).numpy().reshape(-1)
if head.shape[0] != VEC_LEN - SVF_LEN:
    raise ValueError(f'head length {head.shape[0]} != {VEC_LEN - SVF_LEN}')
vec = np.zeros(VEC_LEN, dtype=np.float64)
vec[SVF_LEN:] = head.astype(np.float64)
out = '/app/model_iter_60.npy'
np.save(out, vec)
print('wrote', out, vec.shape, vec.dtype, flush=True)
