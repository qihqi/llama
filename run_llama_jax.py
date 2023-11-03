import torch
import jax
# from torch_xla import stablehlo
import numpy as np

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
_lowerings = {}
def register(name):
    def inner(func):
        global _lowerings
        _lowerings[name] = func
        return func
    return inner


def _get_numpy_dtype(dtype):
  return {
      torch.double: jnp.double,
      torch.float32: jnp.float32,
      #torch.half: jnp.half,
      torch.long: jnp.int64,
      torch.int32: jnp.int32,
      torch.int16: jnp.int16,
      torch.bool: jnp.bool_,
  }.get(dtype)

@register("aten::view")
@register("aten::_unsafe_view")
def _aten_unsafe_view(x, shape):
    return jnp.reshape(x, shape)

@register("aten::add")
def _aten_add(x, y):
    return x + y

@register("aten::copy_")
def _aten_copy(x, y, memory_format=None):
    return jnp.copy(y)

@register("aten::clone")
def _aten_clone(x, memory_format=None):
    return jnp.copy(x)

@register("aten::full")
def _aten_full(size, value, **kwargs):
    return jnp.full(size, value)

@register("aten::index_copy")
def _aten_index_copy(x, dim, indexes, source):
    # return jax.lax.scatter(x, index, dim)
    dims = []
    for i in range(len(x.shape)):
        if i == dim:
            dims.append(indexes)
        else:
            dims.append(slice(None, None, None))
    return x.at[dim].set(source)

@register("aten::index_select")
def _aten_index_select(x, dim, indexes):
    """
    slice_sizes = list(x.shape)
    slice_sizes[dim] = 1
    indexes = jnp.append(indexes, 1)
    offset_dims = [i for i in range(len(x.shape)) if i != dim]
    gather_dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(dim, ),
        collapsed_slice_dims=(dim, ),
        start_index_map=(dim, ),
    )
    return jax.lax.gather(x, indexes, gather_dnums, tuple(slice_sizes))
    """
    dims = []
    for i in range(len(x.shape)):
        if i == dim:
            dims.append(indexes)
        else:
            dims.append(slice(None, None, None))
    return x[tuple(dims)]

@register("aten::mean")
def _aten_mean(x, dim, keepdim):
    return jnp.mean(x, dim, keepdims=keepdim)

@register("aten::mm")
def _aten_mm(x, y):
    return x @ y

@register("aten::mul")
def _aten_mul(x, y):
    return x * y

@register("aten::silu")
def _aten_silu(x):
    return jax.nn.silu(x)

@register("aten::t")
def _aten_t(x):
    return jnp.transpose(x)
    
@register("aten::transpose")
def _aten_transpose(x, dim0, dim1):
    shape = list(range(len(x.shape)))
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    return jnp.transpose(x, shape)

@register("aten::triu")
def _aten_triu(m, k):
    return jnp.triu(m, k)

@register("aten::slice")
def _aten_slice(self, dim=0, start=None, end=None, step=1):
    sl = slice(start, end, step)
    dims = []
    for i in range(len(self.shape)):
        if i == dim:
            dims.append(sl)
        else:
            dims.append(slice(None, None, None))
    return self[tuple(dims)]

@register("aten::detach")
def _aten_detach(self):
    return self



@register("aten::view_as_real")
def _aten_view_as_real(x):
    real = jnp.real(x)
    im = jnp.imag(x)
    res = jnp.stack([real, im], -1)
    return res


@register('aten::_softmax')
def _aten_softmax(x, dim, halftofloat):
    return jax.nn.softmax(x, dim)


@register('aten::pow')
def _aten_pow(x, y):
  if isinstance(y, int):
    y = float(y)
  if isinstance(y, jnp.ndarray):
    y = y.astype(jnp.float32)
  return jax.lax.pow(x, y)

@register('aten::view_as_complex')
def _aten_view_as_complex(input):
  x, y = input[..., 0], input[..., 1]
  return jax.lax.complex(x, y)

@register('aten::div')
def _aten_div(x, y):
  return x / y

@register('aten::bmm')
def _aten_bmm(x, y):
  return jnp.einsum('bnm,bmk->bnk', x, y)

@register('aten::embedding')
def _aten_embedding(a, w):
  return jnp.take(a, w, axis=0)

@register('aten::rsqrt')
def _aten_rsqrt(x):
  return jax.lax.rsqrt(x)

@register('aten::expand')
def _aten_expand(x, dims):
    return jnp.broadcast_to(x, dims)

@register('aten::dot')
def _aten_dot(x, y):
    return jnp.dot(x, y)

@register('aten::_to_copy')
def _aten__to_copy(tensor, **kwargs):
    dtype = _get_numpy_dtype(kwargs['dtype'])
    if dtype != tensor.dtype:
        return tensor.astype(dtype)
    return jnp.copy(tensor)


@register('aten::empty')
def _aten_empty(sizes, **kwargs):
    return jnp.zeros(sizes)

@register('aten::index_put_')
def _aten_index_put(self, indexes, values):
    indexes = [
        slice(None, None, None) if i is None
        else i
        for i in indexes
    ]
    indexes = tuple(indexes)
    return self.at[indexes].set(values)

@register('aten::index')
def _aten_index(self, indexes):
    indexes = [
        slice(None, None, None) if i is None
        else i
        for i in indexes
    ]
    indexes = tuple(indexes)
    return self[indexes]




def get_function(name):
    name = name.split('.')[0]
    return _lowerings[name]


from torch.utils._python_dispatch import TorchDispatchMode

def make_jax_array(tensor):
    if isinstance(tensor, JaxTensor):
        return tensor._elem
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach()
        if hasattr(tensor, '_elem'):
            return tensor._elem
        return jnp.array(tensor.numpy())
    return tensor
    # TODO use dlpack


import torch

# All of the tensor examples in this zoo inherit from BaseTensor.  Ideally,
# however, they would inherit directly from Tensor.  This is just our staging
# ground for applying behavior that hasn't yet made it into core but that
# we would like to apply by default.
class JaxTensor(torch.Tensor):
    # See https://github.com/pytorch/pytorch/pull/73727 ; this is necessary
    # to ensure that super().__new__ can cooperate with each other
    @staticmethod
    def __new__(cls, elem):
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(elem.shape, dtype=torch.float32, device="meta"),
            require_grad=False,
        )

    def __init__(self, elem):
        super().__init__()
        self._elem = elem

    def __str__(self):
        return 'Tensor backed by Jax ' + str(self._elem)


    @property
    def shape(self):
        return self._elem.shape

    @property
    def ndim(self):
        return len(self._elem.shape)

    def flatten(self, start_dim=0, end_dim=-1):
        if end_dim == -1:
            end_dim = self.ndim
        new_shape = self.shape[:start_dim] + (-1,) + self.shape[end_dim:]
        return torch.reshape(self, new_shape)

    def __setitem__(self, key, val):
        key = tuple(make_jax_array(k) if isinstance(k, torch.Tensor) else k for k in key)
        self._elem = self._elem.at[key].set(val._elem)

    __torch_function__ = torch._C._disabled_torch_function_impl


class PrintingMode(TorchDispatchMode):
  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    print(func.name())
    return func(*args, **kwargs)

class VerifyMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        jax_args = tree_map(make_jax_array, args)
        jax_kwargs = tree_map(make_jax_array, kwargs)
        jax_res = get_function(func.name())(*jax_args, **jax_kwargs)
        torch_res = func(*args, **kwargs)
        print(func.name())
        jax_res_t = torch.from_numpy(np.array(jax_res))
        if not torch.allclose(torch_res, torch.from_numpy(np.array(jax_res)), atol=1e-4):
            print('FAIL', func.name(), (torch_res - jax_res_t).norm())
        else:
            print('... passes')
        return torch_res


def run_torch_and_diff(func, args, kwargs, res):
    def to_torch(x):
        if isinstance(x, jnp.ndarray):
            return torch.from_numpy(np.array(x))
        return x
    args = tree_map(to_torch, args)
    kwargs = tree_map(to_torch, kwargs)
    if func.name() == 'aten::view':
        torch_res = args[0].reshape(args[1])
    else:
        torch_res = func(*args, **kwargs)
    res = to_torch(res)
    print(func.name(), end=' ')
    if torch.allclose(torch_res, res, atol=1e-3):
        print('pass')
    else:
        print('fail')
    


class JaxMode(TorchDispatchMode):
  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    # print('running...', func.name())
    args = tree_map(make_jax_array, args)
    kwargs = tree_map(make_jax_array, kwargs)
    res = get_function(func.name())(*args, **kwargs)
    # run_torch_and_diff(func, args, kwargs, res)
    if func.name() == 'aten::copy_':
        args[0]._elem = res
        return args[0]
    return JaxTensor(res)

def wrap_weights(module):
    for k, v in module.__dict__.items():
        if isinstance(v, torch.Tensor) and not isinstance(v, JaxTensor):
            setattr(module, k, JaxTensor(jnp.array(v.detach().cpu().numpy())))


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


if __name__ == '__main__':
    print('--- eager and jit ---')
    run_llama2_test_jit()
    print('--- eager only ---')
    run_llama2_test_eager()
