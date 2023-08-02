import torch
from llama.model import Transformer

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from llama.model import ModelArgs, Transformer, GenLoop
from llama.tokenizer import Tokenizer

ckpt_dir = 'llama-2-7b-chat'
tokenizer_path = 'tokenizer.model'

max_seq_len = 128
max_batch_size = 1

checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
ckpt_path = checkpoints[0]
print('started loading...')
start_time = time.time()
checkpoint = torch.load(ckpt_path)
print(f"Loaded in {time.time() - start_time:.2f} seconds")
with open(Path(ckpt_dir) / "params.json", "r") as f:
    params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
print('started tokenizer ...')
start_time = time.time()
tokenizer = Tokenizer(model_path=tokenizer_path)
print(f"toked in {time.time() - start_time:.2f} seconds")
model_args.vocab_size = tokenizer.n_words
# torch.set_default_tensor_type(torch.cuda.HalfTensor)

print('model init...')
start_time = time.time()
model = model2.Transformer(model_args)
print(f"model init in {time.time() - start_time:.2f} seconds")
print('load state dict ...')
start_time = time.time()
model.load_state_dict(checkpoint, strict=False)
model = model.eval()
print(f"load state dict in {time.time() - start_time:.2f} seconds")

args = (torch.randint(0, tokenizer.n_words, (1, 30)),
        torch.arange(0, 30), None)
model(*args)
from torch_xla.experimental.stablehlo_saved_model import save_as_stablehlo
from torch._export import export
print(f"exporting")
exported = export(model, args)
op_set = set()
for node in exported.graph_module.graph.nodes:
    if node.op == 'call_function':
        op_set.add(node.target)
print(op_set)
sys.exit(0)
print(f"... in {time.time() - start_time:.2f} seconds")

print(f"saving")
start_time = time.time()
save_as_stablehlo(exported, (args, ), 'llama_exported')
print(f"saving in {time.time() - start_time:.2f} seconds")

