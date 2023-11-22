import copy
import functools
import time
import collections
import torch
from torch._functorch.make_functional import make_functional_with_buffers
import jax
# from torch_xla import stablehlo
import numpy as np
import jax.random as jrandom

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map, tree_map_only

from llama.model import ModelArgs
from llama import model3
from llama import model
from llama.tokenizer import Tokenizer
from jax import numpy as jnp
from llama.jax_integration import *

tokenizer_path = 'tokenizer.model'

jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

from torch.utils import dlpack as torch_dlpack
from jax import dlpack as jax_dlpack

def j2t(x_jax):
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    return x_torch

def t2j(x_torch):
    x_torch = x_torch.contiguous()
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax


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

def make_cache_jax(args, max_batch_size):
    n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    n_local_heads = args.n_heads
    n_local_kv_heads = n_kv_heads
    n_rep = n_local_heads // n_local_kv_heads
    head_dim = args.dim // args.n_heads
    res = []
    for i in range(args.n_layers):
        res.append((jnp.zeros (
                (max_batch_size,
                args.max_seq_len,
                n_local_kv_heads,
                head_dim,)
            ), jnp.zeros (
                (max_batch_size,
                args.max_seq_len,
                n_local_kv_heads,
                head_dim,)
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
        data = {"dim": 128, "multiple_of": 32, "n_heads": 4, "n_layers": 3, "norm_eps": 1e-05, "vocab_size": -1}
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

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding


def compact_weights(state_dict, m_jax, n_layers):
    to_be_stacked = collections.defaultdict(lambda: [None] * n_layers)
    new_state_dict = copy.copy(m_jax.state_dict())
    new_state_dict.update(state_dict)
    for k, v in new_state_dict.items():
        if k.startswith('layers'):
            pieces = k.split('.')
            pos = int(pieces[1])
            name = '___'.join(pieces[2:])
            to_be_stacked[name][pos] = v
        else:
            new_state_dict[k] = JaxTensor(t2j(v.detach()))
    
    for k, v in to_be_stacked.items():
        val = torch.stack(v)
        val_jax = t2j(val.detach())
        new_state_dict[k] = JaxTensor(val_jax)
    return new_state_dict


def transform_input(inputs, mask, freqs, context_length):
    seqlen = inputs[0].shape[1]
    pos = inputs[1]
    return (t2j(inputs[0].detach().cpu()),
        jnp.arange(pos, seqlen + pos),
        jnp.arange(pos + seqlen - context_length, seqlen + pos),
        mask, 
        freqs
    )

def run_llama2_test_jit():
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
    param_size = 'tiny'
    context_length = 128
    infer_length = 10
    model3.use_jax_loop = True
    print('start')

    test_data = torch.load('tiny_test_bf.pt')
    assert param_size in ('tiny', '7b', '13b', '70b'), param_size
    max_input_seq_length = context_length + infer_length

    model_arg = test_data['model_arg']
    start = time.time()
    m_jax = model3.Transformer(model_arg)
    m_jax.eval()

    import ipdb; ipdb.set_trace()

    m_jax_func, weights, buffers = make_functional_with_buffers(m_jax)
    state_dict = compact_weights(test_data['state_dict'], m_jax, model_arg.n_layers)

    weights = [state_dict[name]._elem for name in m_jax_func.param_names]
    buffers = [state_dict[name]._elem.astype(jnp.bfloat16) if 'cache' in name else 
               state_dict[name]._elem
               for name in m_jax_func.buffer_names]

    end = time.time()

    mask = torch.full(
        (1, 1, context_length, context_length), float("-inf"),
    )
    mask = torch.triu(mask, diagonal=1)

    mask = jnp.array(mask.cpu().detach().numpy())
    freqs = jnp.array(m_jax.freqs_cis.cpu().detach().numpy())


    def run_jax_model(weights, buffers, *args):
        weights = tree_map(JaxTensor, weights)
        buffers = tree_map(JaxTensor, buffers)
        args = tree_map_only(jnp.ndarray, JaxTensor, args)
        res = m_jax_func(weights, buffers, *args)
        unwrap_buffers = tuple(buf._elem for buf in buffers)
        if isinstance(res, tuple):
            return tuple(r._elem if isinstance(r, JaxTensor) else r for r in res)
        else:
            return res._elem

    run_func = jax.jit(run_jax_model)

    cache_k = jnp.zeros((
        model_arg.n_layers,
        1,
        max_input_seq_length,
        m_jax._layer.attention.n_local_kv_heads,
        m_jax._layer.attention.head_dim
    )).astype(jnp.bfloat16)
    cache_v = jnp.zeros((
        model_arg.n_layers,
        1,
        max_input_seq_length,
        m_jax._layer.attention.n_local_kv_heads,
        m_jax._layer.attention.head_dim
    )).astype(jnp.bfloat16)

    def make_compact_caches(caches):
        cache_k = torch.stack([c[0][0].reshape((1, *c[0][0].shape)) for c in caches])
        cache_v = torch.stack([c[1][0].reshape((1, *c[1][0].shape)) for c in caches])
        return t2j(cache_k), t2j(cache_v)
        

    one_row = test_data['data'][0]
    for testcase in one_row:
        inputs = testcase['input']
        inputs = transform_input(inputs, mask, freqs, context_length)
        mask = None
        # buffers get updated
        res, xk, xv = run_func(weights, buffers, *(inputs + (cache_k, cache_v)))
        official_caches = make_compact_caches(testcase['caches'])
        print(m_jax_func.buffer_names[0])
        print(m_jax_func.buffer_names[1])
        res_torch = j2t(res)
        cache_k, cache_v = official_caches
        print('MAX', torch.max(torch.abs(res_torch - testcase['output'])))
        res_top5 = set(t.item() for t in torch.topk(res_torch[:, -1], 5, dim=-1).indices[0])
        orig_top5 = set(t.item() for t in torch.topk(testcase['output'][:, -1], 5, dim=-1).indices[0])
        print('top5 intesect', len(res_top5 & orig_top5))


    print('done')




if __name__ == '__main__':
    print('test')
    run_llama2_test_jit()
