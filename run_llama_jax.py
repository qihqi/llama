import functools
import time
import collections
import torch
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
from torch.utils._pytree import tree_map

from llama.model import ModelArgs
from llama import model2
from llama import model
from llama.tokenizer import Tokenizer
from jax import numpy as jnp
from llama.jax_integration import *

tokenizer_path = 'tokenizer.model'

jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)


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
sharding = PositionalSharding(mesh_utils.create_device_mesh((4, )))
shardings = [sharding.reshape(4, 1), sharding.reshape(1, 4)]


def wrap_weights_sharded(rev_state_dict, module):
    for k, v in module.__dict__.items():
        if isinstance(v, torch.Tensor) and not isinstance(v, JaxTensor):
            name = rev_state_dict.get(id(v))
            print(id(v), name, k, module)
            axis = _shard_axis(name)
            jax_arr = jnp.array(v.detach().cpu().numpy()).astype(jnp.bfloat16)
            if axis is not None:
                print('shared', name, 'at', axis)
                jax_arr = jax.device_put(jax_arr, shardings[axis])
            setattr(module, k, JaxTensor(jax_arr))

def wrap_weights(module):
    for k, v in module.__dict__.items():
        if isinstance(v, torch.Tensor) and not isinstance(v, JaxTensor):
            jax_arr = jnp.array(v.detach().cpu().numpy())
            setattr(module, k, JaxTensor(jax_arr))


def run_llama2_test_jit():
    param_size = 'tiny'
    context_length = 100
    infer_length = 10
    print('start')

    tokenizer = Tokenizer(model_path=tokenizer_path)
    assert param_size in ('tiny', '7b', '13b', '70b'), param_size
    max_input_seq_length = context_length + infer_length

    model_arg = get_arg(param_size, max_input_seq_length)
    model_arg.vocab_size = tokenizer.n_words

    start = time.time()
    m_jax = model2.Transformer(model_arg)
    m_torch = model2.Transformer(model_arg)
    m_jax.eval()
    m_torch.eval()

    m_torch.load_state_dict(m_jax.state_dict())  # same state dict

    m_jax.apply(wrap_weights)



    end = time.time()
    caches = make_cache(model_arg)
     

    mask = torch.full(
        (1, 1, context_length, context_length), float("-inf"),
    )
    mask = torch.triu(mask, diagonal=1)

    sample_input_prefill = (
        torch.randint(0, 1000, (2, context_length, )),  # len seq length
        torch.arange(0, context_length),   #indexes
        torch.arange(0, context_length),   #cache indexes
        mask,
    )
    with JaxMode():
        result_jax = m_jax(*sample_input_prefill)

    #with VerifyMode():
    result_torch = m_torch(*sample_input_prefill)

    print('DIFF', (result_torch - torch.from_numpy(np.array(result_jax._elem))).norm())

    def run_jax_model(*args):
        with JaxMode():
            jt = tuple(JaxTensor(a) for a in args)
            return m_jax(*jt)._elem

    jax_input = tree_map(make_jax_array, sample_input_prefill)
    jax_jit_ans = jax.jit(run_jax_model, backend='cpu')(*jax_input)

    print('DIFF JITTED', (result_torch - torch.from_numpy(np.array(jax_jit_ans))).norm())

    # stablehlo.save_torch_model_as_stablehlo(m, sample_input_prefill, os.path.join(path_prefix, 'prefill'))
    # stablehlo.save_torch_model_as_stablehlo(m, sample_input_decode, os.path.join(path_prefix, 'decode'))
    print('done')

def run_llama2_test_eager():
    param_size = 'tiny'
    context_length = 100
    infer_length = 10
    print('start')

    tokenizer = Tokenizer(model_path=tokenizer_path)
    assert param_size in ('tiny', '7b', '13b', '70b'), param_size
    max_input_seq_length = context_length + infer_length

    model_arg = get_arg(param_size, max_input_seq_length)
    model_arg.vocab_size = tokenizer.n_words

    start = time.time()
    m_jax = model.Transformer(model_arg)
    m_torch = model.Transformer(model_arg)
    m_jax.eval()
    m_torch.eval()

    m_torch.load_state_dict(m_jax.state_dict())  # same state dict

    m_jax.apply(wrap_weights)



    end = time.time()
    caches = make_cache(model_arg)

    sample_input_prefill = (
        torch.randint(0, 1000, (2, context_length, )),  # len seq length
        0,
    )
    with JaxMode():
        result_jax = m_jax(*sample_input_prefill)

    result_torch = m_torch(*sample_input_prefill)

    print('DIFF', (result_torch - torch.from_numpy(np.array(result_jax._elem))).norm())

_State = collections.namedtuple(
    'State',
    [
        'res',
        'caches',
        'input_tokens',
        'index',
        'cache_index',
    ],
)

def _shard_axis(name):
    if name is None:
        return None
    if 'tok_embeddings' in name:
        return 0
    if 'attention.' in name:
      if 'wo' in name:
          return 0
        #position_to_sharding[i] = (_NUM_OF_PARTITIONS.value, 1)
      else:
          return 1
       # position_to_sharding[i] = (1, _NUM_OF_PARTITIONS.value)
    if 'feed_forward.' in name:
      if 'w2' in name:
          return 0
       # position_to_sharding[i] = (_NUM_OF_PARTITIONS.value, 1)
      else:
          return 1
      #  position_to_sharding[i] = (1, _NUM_OF_PARTITIONS.value)
    if 'output' in name:
        return 1
      #position_to_sharding[i] = (1, _NUM_OF_PARTITIONS.value)


def run_llama_benchmark():
    param_size = '7b'
    context_length = 2048
    infer_length = 128

    model2.use_jax_loop = False

    print('start')

    tokenizer = Tokenizer(model_path=tokenizer_path)
    assert param_size in ('tiny', '7b', '13b', '70b'), param_size
    max_input_seq_length = context_length + infer_length

    model_arg = get_arg(param_size, max_input_seq_length)
    model_arg.vocab_size = tokenizer.n_words

    # model_arg.n_layers = 4

    start = time.time()
    m_jax = model2.Transformer(model_arg)
    end = time.time()
    print('instantiated model', end - start)
    start = time.time()
    m_jax.eval()

    state_dict = m_jax.state_dict()
    weights = {}

    for k, v in state_dict.items():
        print(k, v.shape)
        if isinstance(v, torch.Tensor) and not isinstance(v, JaxTensor):
            axis = _shard_axis(k)
            jax_arr = jnp.array(v.detach().cpu().numpy()).astype(jnp.bfloat16)
            if axis is not None:
                print('shared', k, 'at', axis)
                jax_arr = jax.device_put(jax_arr, shardings[axis])
            weights[k] = jax_arr

    if model2.use_jax_loop:
        new_layer_weights = []
        for k, v in zip(m_jax._keys, m_jax._layer_weights):
            if isinstance(v, torch.Tensor) and not isinstance(v, JaxTensor):
                axis = _shard_axis(k)
                jax_arr = jnp.array(v.detach().cpu().numpy()).astype(jnp.bfloat16)
                if axis is not None:
                    print('shared', k, 'at', axis)
                    local_sharding = shardings[axis]
                    local_sharding = local_sharding.reshape((1, ) + local_sharding.shape)
                    jax_arr = jax.device_put(jax_arr, local_sharding)
                new_layer_weights.append(jax_arr)
        layer_weights = new_layer_weights
    else:
        layer_weights = []

    caches = make_cache_jax(model_arg, 1)
                # manually shard the kv cache
                
    if model2.use_jax_loop:
        args = model_arg
        n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        n_local_heads = args.n_heads
        n_local_kv_heads = n_kv_heads
        n_rep = n_local_heads // n_local_kv_heads
        head_dim = args.dim // args.n_heads
        caches = (jnp.zeros (
                (model_arg.n_layers, 
                    1,
                args.max_seq_len,
                n_local_kv_heads,
                head_dim,)
            ), jnp.zeros(
                (model_arg.n_layers, 
                1,
                args.max_seq_len,
                n_local_kv_heads,
                head_dim,)
            ))
        sharded_caches = (
                jax.device_put(caches[0], sharding.reshape(1, 1, 1, 4, 1)),
                jax.device_put(caches[1], sharding.reshape(1, 1, 1, 4, 1)),
            )
    else:
        sharded_caches = []
        for ck, cv in caches:
            sharded_caches.append((
                jax.device_put(ck, sharding.reshape(1,1,4,1)),
                jax.device_put(cv, sharding.reshape(1,1,4,1))
            ))
        del caches

    mask = jnp.full((1, 1, context_length, context_length), -jnp.inf)
    mask = jnp.triu(mask, k=1)
    print('set up things', end - start)

    # takes jax array, returns jax array
    def run_jax_model(args, caches, weights, layer_weights):
        if model2.use_jax_loop:
            weights = tree_map(JaxTensor, weights)
            layer_weights = tree_map(JaxTensor, layer_weights)
            m_jax.load_state_dict(weights, False, True)
            m_jax._layer_weights = layer_weights
        else:
            for (cache_k, cache_v), layer in zip(caches, m_jax.layers):
                layer.attention.cache_k = JaxTensor(cache_k)
                layer.attention.cache_v = JaxTensor(cache_v)

        jt = tuple(JaxTensor(a) if isinstance(a, jnp.ndarray) else a for a in args )
        res = m_jax(*jt)._elem
        if model2.use_jax_loop:
            pass
        else:
            caches = [(layer.attention.cache_k._elem, 
                       layer.attention.cache_v._elem) for layer in m_jax.layers]
        return res, caches


    def run_loop(orig_inputs, caches, freqs, weights, layer_weights):
        prefill_input = (
            orig_inputs,
            jnp.arange(0, context_length),
            jnp.arange(0, context_length),
            mask,
            freqs)

        logits, caches = run_jax_model(prefill_input, caches, weights, layer_weights)
        next_token = jnp.argmax(logits[0][-1]).reshape((1,))

        def body(state):
            decode_input = (
                state.input_tokens.reshape( (1, 1)),
                state.index,
                state.cache_index,
                None,
                freqs,
            )
            logits, caches = run_jax_model(decode_input, state.caches, weights, layer_weights)
            next_token = jnp.argmax(logits[0][-1]).reshape((1,))
            res = state.res.at[state.index - context_length].set(next_token[0])
            return _State(res, caches, next_token, state.index + 1, state.cache_index + 1)

        def condition(states):
            return states.index[-1] < infer_length  + context_length

        results = jnp.zeros((infer_length, )).astype(jnp.int64)
        results = results.at[0].set(next_token[0])
        start = _State(
            results,
            caches,
            next_token,
            jnp.arange(1 + context_length, context_length + 2),
            jnp.arange(2, 2 + context_length),
        )

        return jax.lax.while_loop(condition, body, start).res

    orig_inputs = jnp

    key = jrandom.PRNGKey(0)
    jit_func = jax.jit(run_loop, in_shardings=None, out_shardings=None)

    random_integers = jrandom.randint(key, (1, 2048,), 0, 32000)
    freqs = jnp.array(m_jax.freqs_cis.cpu().detach().numpy())
   # print(jit_func.lower(random_integers, sharded_caches, freqs, weights, layer_weights).as_text())


    for i in range(3):
        print('Iteration start', i)
        random_integers = jrandom.randint(key, (1, 2048,), 0, 32000)
        start = time.time()
        res = jit_func(random_integers, sharded_caches, freqs, weights, layer_weights)
        end = time.time()
        print('Iteration ', i, end - start)
    print(res)


if __name__ == '__main__':
    '''
    print('--- eager and jit ---')
    run_llama2_test_jit()
    print('--- eager only ---')
    run_llama2_test_eager()
    '''
    print('benchmark')
    run_llama_benchmark()
