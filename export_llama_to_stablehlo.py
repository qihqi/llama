import torch
from torch_xla import stablehlo

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from llama.model import ModelArgs, Transformer, GenLoop
from llama import model_exportable_unbatched
from llama.tokenizer import Tokenizer

ckpt_dir = 'llama-2-7b-chat'
tokenizer_path = 'tokenizer.model'


model_args: ModelArgs = ModelArgs(
    dim = 32,
    n_layers = 2,
    n_heads = 2,
    n_kv_heads = None,
    vocab_size = 1000,  # defined later by tokenizer,
    multiple_of = 256,  # make SwiGLU hidden layer size multiple of large power of 2,
    ffn_dim_multiplier = None,
    norm_eps = 1e-5,
    max_batch_size = 2,
    max_seq_len = 32,
)
def make_cache(args):
    n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    n_local_heads = args.n_heads
    n_local_kv_heads = n_kv_heads
    n_rep = n_local_heads // n_local_kv_heads
    head_dim = args.dim // args.n_heads
    res = []
    for i in range(args.n_layers):
        res.append((torch.zeros (
                #max_batch_size,
                args.max_seq_len,
                n_local_kv_heads,
                head_dim,
            ), torch.zeros (
                #max_batch_size,
                args.max_seq_len,
                n_local_kv_heads,
                head_dim,
            )))
    return res


def verify_cache(caches, model1):
    for i, layer in enumerate(model1.layers):
        k1, v1 = caches[i]
        k2 = layer.attention.cache_k
        v2 = layer.attention.cache_v
        if not torch.allclose(k1, k2):
            print('key dont match for', i)
        if not torch.allclose(v1, v2):
            print('value dont match for', i)

# 7b 
# {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}
# 13b
# {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": -1}
# 70b
# {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": -1} 

def get_arg(param_size, seqlen):
    if param_size == 'tiny':
        data = {"dim": 128, "multiple_of": 32, "n_heads": 2, "n_layers": 3, "norm_eps": 1e-05, "vocab_size": -1}
    elif param_size == '7b':
        data = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}
    elif param_size == '13b':
        data = {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": -1}
    elif param_size == '70b':
        data = {
            "dim": 8192, 
            "multiple_of": 4096,
            "ffn_dim_multiplier": 1.3,
            "n_heads": 64,
            "n_kv_heads": 8,
            "n_layers": 80,
            "norm_eps": 1e-05,
            "vocab_size": -1
        } 
    return ModelArgs(
        max_seq_len=seqlen,
        max_batch_size=2,
        **data,
    )


def export_llama2_to_stablehlo(
    param_size: str,
    context_length: int,
    infer_length: int,
    path_prefix: str 
):
    print('start')
    tokenizer = Tokenizer(model_path=tokenizer_path)
    assert param_size in ('tiny', '7b', '13b', '70b'), param_size
    max_input_seq_length = context_length + infer_length

    model_arg = get_arg(param_size, max_input_seq_length)
    model_arg.vocab_size = tokenizer.n_words

    start = time.time()
    m = model_exportable_unbatched.Transformer(model_arg)
    end = time.time()
    print('Model init took', end - start, 'seconds')
    caches = make_cache(model_arg)

    sample_input_prefill = (
        torch.randint(0, 1000, (context_length, )),  # len seq length
        torch.arange(0, context_length), # input indexes
        torch.arange(0, context_length), # context indexes
        caches, # caches
        True, # prefil
    )
    # m(*sample_input_prefill)

    sample_input_decode = (
        torch.randint(0, 1000, (1, )),  # len = 1
        torch.arange(context_length, context_length + 1), # input indexes
        torch.arange(0, context_length + 1), # context indexes
        caches,
        False # prefill
    )

    stablehlo.save_torch_model_as_stablehlo(m, sample_input_prefill, os.path.join(path_prefix, 'prefill'))
    stablehlo.save_torch_model_as_stablehlo(m, sample_input_decode, os.path.join(path_prefix, 'decode'))
    print('done')


def main2(): 
    start_time = time.time() 
    m = model2.Transformer(model_args)
    print(f"model init in {time.time() - start_time:.2f} seconds")
    model1 = model.Transformer(model_args)
    model1.load_state_dict(m.state_dict())
    caches = make_cache(2, 2, 32, 2, 32 // 2)
    m = m.eval()

    print('before inf')
    #verify_cache(caches, model1)

    args = torch.randint(0, 1000, (1, ))
    input_tokens_pos = torch.arange(2, 3)
    cache_inputs = torch.arange(0, 3)

    inputs = (args, input_tokens_pos, cache_inputs, caches, False)
    print('eval model', args.shape, input_tokens_pos)
    model2_res, caches = m.forward(*inputs)
    exported = torch.export.export(m, inputs)

    print(exported.graph_module.code)
    print(stablehlo.exported_program_to_stablehlo(exported).get_stablehlo_text('forward'))




if __name__ == '__main__':
    import fire
    fire.Fire(export_llama2_to_stablehlo)
