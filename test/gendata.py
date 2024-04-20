from itertools import product
from safetensors.numpy import save_file, load_file
import numpy as np


shapes = [[3],[3,5],[3,5,7]]
dtypes = [np.float16, np.float32, np.float64, bool, np.uint8, np.int8, np.int16, np.int32, np.int64]

tensors = {}
for (shape, dtype) in product(shapes, dtypes):

    name = dtype.__name__
    s = ''.join(map(str,shape))
    key = f"{name}_{s}"
    if dtype == bool:
        tensor = np.arange(np.prod(shape)).reshape(shape)
        tensor = (tensor % 2).astype(bool)
    else:
        tensor = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    tensors[key] = tensor

save_file(tensors, "./model.safetensors")
loaded = load_file("./model.safetensors")

import torch
from safetensors.torch import save_file as th_save_file
from safetensors.torch import load_file as th_load_file

th_dtypes = [
    torch.bool, torch.uint8, torch.int8, torch.float8_e5m2, torch.float8_e4m3fn,
    torch.int16, #torch.uint16,
    torch.float16, torch.bfloat16,
    torch.int32, #torch.uint32,
    torch.float32,
    torch.int64, #torch.uint64,
    torch.float64,
]
th_tensors = {}
for (shape, dtype) in product(shapes, th_dtypes):
    name = dtype.__repr__().split('.')[1].lower()
    s = ''.join(map(str,shape))
    key = f"{name}_{s}"
    if key.startswith("bool"):
        tensor = torch.randint(0, 1, shape, dtype=dtype)
    elif key.startswith("int") or key.startswith("uint"):
        tensor = torch.randint(0, 100, shape, dtype=dtype)
    elif key.startswith("float8"):
        tensor = torch.randn(shape, dtype=torch.float32).type(dtype)
    else:
        tensor = torch.randn(shape, dtype=dtype).type(dtype)
    tensors[key] = tensor

th_save_file(tensors, "./torch.safetensors")
loaded = th_load_file("./torch.safetensors")
th_save_file(tensors, "./torch_metadata.safetensors", {"test":"metadata", "version":"2.2"})
loaded = th_load_file("./torch_metadata.safetensors")
