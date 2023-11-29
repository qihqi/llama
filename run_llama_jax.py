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
from llama import model2, model3
from llama import model
from llama.tokenizer import Tokenizer
from jax import numpy as jnp
from llama.jax_integration import *

tokenizer_path = 'tokenizer.model'

#jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)


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
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
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
    mask._elem = mask._elem.astype(jnp.bfloat16)

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
        'cache_k',
        'cache_v',
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
    if 'attention_' in name:
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


def run_llama_benchmark(
    param_size = '7b',
    context_length = 2048,
    infer_length = 128,
    use_jax_loop = True,
    use_jax_mha = False,
    print_stablehlo = False,
):
    model3.use_jax_loop = use_jax_loop
    model3.use_jax_mha = use_jax_mha
    use_spmd = True

    print('start')
    print("""
    param_size = {},
    context_length = {},
    infer_length = {},
    use_jax_loop = {},
    print_stablehlo= {},
    use_jax_mha = {},""".format(
        param_size ,
        context_length ,
        infer_length ,
        use_jax_loop ,
        print_stablehlo,
        use_jax_mha ,
    ))

    tokenizer = Tokenizer(model_path=tokenizer_path)
    assert param_size in ('tiny', '7b', '13b', '70b'), param_size
    max_input_seq_length = context_length + infer_length

    model_arg = get_arg(param_size, max_input_seq_length)
    model_arg.vocab_size = tokenizer.n_words

    # model_arg.n_layers = 10

    start = time.time()
    m_jax_orig = model3.Transformer(model_arg)
    m_jax_orig.eval()

    freqs = jnp.array(m_jax_orig.freqs_cis.cpu().detach().numpy())
    fr = jnp.real(freqs).astype(jnp.bfloat16)
    fi = jnp.imag(freqs).astype(jnp.bfloat16)
    freqs = (fr, fi)


    m_jax, weights, buffers = make_functional_with_buffers(m_jax_orig)
    m_jax.eval()

    def make_model_jax(model):

        def jax_callable(weights, buffers, args):
            weights = tree_map(JaxTensor, weights)
            buffers = tree_map(JaxTensor, buffers)
            args = tree_map_only(jnp.ndarray, JaxTensor, args)
            res = model(weights, buffers, *args)
            if isinstance(res, tuple):
                return tuple(r._elem if isinstance(r, JaxTensor) else r for r in res)
            else:
                return res._elem
        return jax_callable

    def make_jax_array(t):
        if isinstance(t, torch.Tensor):
            return jnp.array(t.detach().cpu().numpy()).astype(jnp.bfloat16)
        return t



    m_jax_func = make_model_jax(m_jax)
    jitted = jax.jit(m_jax_func)
    jax_weights = tree_map(make_jax_array, weights)
    jax_buffers = tree_map(make_jax_array, buffers)

    mask = jnp.full((1, 1, context_length, context_length), -jnp.inf, dtype=jnp.bfloat16)
    mask = jnp.triu(mask, k=1)
    cache_k = jnp.zeros((
        model_arg.n_layers,
        1,
        max_input_seq_length,
        m_jax_orig._layer.attention.n_local_kv_heads,
        m_jax_orig._layer.attention.head_dim
    )).astype(jnp.bfloat16)
    cache_v = jnp.zeros((
        model_arg.n_layers,
        1,
        max_input_seq_length,
        m_jax_orig._layer.attention.n_local_kv_heads,
        m_jax_orig._layer.attention.head_dim
    )).astype(jnp.bfloat16)
    args = (
        jnp.arange(0, context_length).reshape((1, context_length)),
        jnp.arange(0, context_length),
        jnp.arange(0, context_length),
        mask,
        freqs,
        cache_k,
        cache_v,
    )

    if print_stablehlo:
        print(jitted.lower(jax_weights, jax_buffers, args).as_text())


    if use_spmd:
        jax_weights_new = []
        jax_buffers_new = []
        num_devices = len(jax.devices())
        for name, jax_arr in zip(m_jax.param_names, jax_weights):
            axis = _shard_axis(name)
            if axis is not None:
                this_sharding = shardings[axis]
                if len(jax_arr.shape) > len(this_sharding.shape):
                    this_sharding = this_sharding.reshape((1, *this_sharding.shape))
                jax_arr = jax.device_put(jax_arr, this_sharding) 
            jax_weights_new.append(jax_arr)
        for name, jax_arr in zip(m_jax.buffer_names, jax_buffers):
            axis = _shard_axis(name)
            if axis is not None:
                this_sharding = shardings[axis]
                if len(jax_arr.shape) > len(this_sharding.shape):
                    this_sharding = this_sharding.reshape((1, *this_sharding.shape))
                jax_arr = jax.device_put(jax_arr, this_sharding) 
            jax_buffers_new.append(jax_arr)

        cache_k = jax.device_put(cache_k, sharding.reshape(1, 1, 1, 4, 1))
        cache_v = jax.device_put(cache_v, sharding.reshape(1, 1, 1, 4, 1))
            
        jax_weights = jax_weights_new
        del jax_weights_new
        jax_buffer = jax_buffers_new
        del jax_buffers_new


    def run_loop(orig_inputs, freqs, mask, jax_weights, jax_buffers, cache_k, cache_v):
        prefill_input = (
            orig_inputs,
            jnp.arange(0, context_length),
            jnp.arange(0, context_length),
            mask,
            freqs,
            cache_k, cache_v
        )

        (logits, updatek, updatev) = m_jax_func(jax_weights, jax_buffers, prefill_input)
        cache_k = cache_k.at[:, :, jnp.arange(0, context_length)].set(updatek)
        cache_v = cache_v.at[:, :, jnp.arange(0, context_length)].set(updatev)
        next_token = jnp.argmax(logits[0][-1]).reshape((1,))

        def body(state):
            decode_input = (
                state.input_tokens.reshape( (1, 1)),
                state.index,
                state.cache_index,
                None,
                freqs,
                state.cache_k,
                state.cache_v,
            )
            (logits, updatek, updatev) = m_jax_func(jax_weights, jax_buffers, decode_input)
            cache_k = state.cache_k.at[:, :, state.index].set(updatek)
            cache_v = state.cache_v.at[:, :, state.index].set(updatev)
            next_token = jnp.argmax(logits[0][-1]).reshape((1,))
            res = state.res.at[state.index - context_length].set(next_token[0])
            return _State(res, cache_k, cache_v, next_token, state.index + 1, state.cache_index + 1)

        def condition(states):
            return states.index[-1] < infer_length  + context_length

        results = jnp.zeros((infer_length, )).astype(jnp.int64)
        results = results.at[0].set(next_token[0])
        start = _State(
            results,
            cache_k, 
            cache_v, 
            next_token,
            jnp.arange(context_length, context_length + 1),
            jnp.arange(1, 1 + context_length),
        )

        return jax.lax.while_loop(condition, body, start).res

    orig_inputs = jnp

    key = jrandom.PRNGKey(0)
    jit_func = jax.jit(run_loop, in_shardings=None, out_shardings=None)

    random_integers = jrandom.randint(key, (1, context_length,), 0, 32000)



    jax.profiler.start_trace('jax_trace.pb')
    for i in range(3):
        print('Iteration start', i)
        random_integers = jrandom.randint(key, (1, context_length,), 0, 32000)
        start = time.time()
        res = jax.block_until_ready(
            jit_func(
                random_integers, freqs, mask, 
                jax_weights, jax_buffers, cache_k, cache_v)
        )
        end = time.time()
        print('Iteration ', i, end - start)
    jax.profiler.stop_trace()
    print(res)


if __name__ == '__main__':
    '''
    print('--- eager and jit ---')
    run_llama2_test_jit()
    print('--- eager only ---')
    run_llama2_test_eager()
    '''
    print('benchmark')
    import fire
    fire.Fire(run_llama_benchmark)
