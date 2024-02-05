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