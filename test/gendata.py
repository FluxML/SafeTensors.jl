from itertools import product
from safetensors.numpy import save_file, load_file
import numpy as np
import os


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

os.mkdir("./non_sharded")
save_file(tensors, "./non_sharded/model.numpy.safetensors")
loaded = load_file("./non_sharded/model.numpy.safetensors")

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

th_save_file(tensors, "./non_sharded/model.safetensors")
loaded = th_load_file("./non_sharded/model.safetensors")
th_save_file(tensors, "./non_sharded/model.with_metadata.safetensors", {"test":"metadata", "version":"2.2"})
loaded = th_load_file("./non_sharded/model.with_metadata.safetensors")

# sharded tensors
# borrowed from https://github.com/huggingface/huggingface_hub/blob/dc5e893556ce46fd82d16ddf5d7db2df6963e4fb/src/huggingface_hub/serialization/_torch.py#L344-L368
import json
from huggingface_hub import split_torch_state_dict_into_shards

state_dict = tensors
state_dict_split = split_torch_state_dict_into_shards(state_dict, max_shard_size=1000)
save_directory = "./sharded"
os.mkdir(save_directory)
assert state_dict_split.is_sharded

for filename, tensors in state_dict_split.filename_to_tensors.items():
    shard = {tensor: state_dict[tensor] for tensor in tensors}
    th_save_file(
        shard,
        os.path.join(save_directory, filename),
        metadata={"format": "pt"},
    )
index = {
    "metadata": state_dict_split.metadata,
    "weight_map": state_dict_split.tensor_to_filename,
}
with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
    f.write(json.dumps(index, indent=2))