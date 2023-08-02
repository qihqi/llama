import copy
import json
import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
import torch
from torch_xla import stablehlo
from torch_xla import tf_saved_model_integration
from copy import deepcopy
from pathlib import Path
import os
from llama.model import ModelArgs
from llama.model import precompute_freqs_cis
from tensorflow.compiler.tf2xla.python import xla as tfxla

from torch.fx import _pytree as fx_pytree
from jax.tree_util import register_pytree_node
from torch.utils import _pytree as pytree
import numpy as np

CONTEXT_LENGTH = 2048
max_input_seq_length = CONTEXT_LENGTH + 256

def _flatten_bundle(bundle):
    funcs = [ json.dumps(f.meta, cls=stablehlo.StableHLOJSONSerializer) for f in bundle.stablehlo_funcs ]
    return (bundle.state_dict, bundle.additional_constants, funcs), None

def _unflatten_bundle(aux_data, children):
    state_dict, consts, funcs = children
    funcs = [stablehlo.StableHLOFunc(
        json.load(f, object_hook=stablehlo.stablehlo_obj_hook), None, None)
        for f in funcs]
    return stablehlo.StableHLOModelBundle(
      stablehlo_funcs=funcs,
      additional_constants=consts,
      state_dict=state_dict
    )

register_pytree_node(
    stablehlo.StableHLOModelBundle,
    _flatten_bundle,
    _unflatten_bundle
)


def merge_bundle(**kwargs):
    first = next(iter(kwargs.values()))
    new_bundle = stablehlo.StableHLOModelBundle(
        state_dict=first.state_dict,
        additional_constants=[],
        stablehlo_funcs=[]
    )

    for name, bundle in kwargs.items():
        const_offset = len(new_bundle.additional_constants)
        for func in bundle.stablehlo_funcs:
            func.meta.name = name + '_'  + func.meta.name
            for loc in func.meta.input_locations:
                if loc.type_ == stablehlo.VariableType.CONSTANT:
                    loc.position += const_offset
            new_bundle.stablehlo_funcs.append(func)
        new_bundle.additional_constants.extend(bundle.additional_constants)
    return new_bundle
    

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


def flatten_state_dict(state_dict):
  i = 0
  res = []
  name_to_pos = {}
  for name, val in state_dict.items():
    res.append(val)
    name_to_pos[name] = i
    i += 1
  return res, name_to_pos 


def compute_args_mappings(name_to_pos, meta):
  res = []
  for loc in meta.input_locations:
    if loc.type_ == stablehlo.VariableType.PARAMETER:
      res.append((0, name_to_pos[loc.name]))
    elif loc.type_ == stablehlo.VariableType.CONSTANT:
      res.append((0, loc.position + len(name_to_pos)))
    else:
      res.append((1, loc.position))
  return res


def wrap_as_jax_func(func, mappings):
  touts = [sig.dtype for sig in func.meta.output_signature]
  souts = [sig.shape for sig in func.meta.output_signature]
  def inner(weights, args):
    full_args = (weights, args)  #(296), (1)
    call_args = [full_args[type_][index] for type_, index in mappings]
    return tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=touts,  # dtype information
        Sout=souts,  # Shape information
        function_list=[],
        module=func.bytecode,
    )
  return jax2tf.call_tf(inner)


def tensor_to_jax_array(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    #if isinstance(tensor, np.ndarray):
    #    tensor = jnp.asarray(tensor)
    return tensor


def _fill_freqs_cis(state_dict, model_args):
    state_dict['L__fn___freqs_cis'] = precompute_freqs_cis(
            model_args.dim // model_args.n_heads, model_args.max_seq_len * 2
    )


def make_prefill_input(caches, tokens_int):
    # todo: use tokens str to generate input
    # NOTE prefill input size has to be same as context length
    input_prefill = (
        torch.randint(0, 1000, (CONTEXT_LENGTH, )).to(torch.int64),  # len seq length
        torch.arange(0, CONTEXT_LENGTH).to(torch.int64), # input indexes
        torch.arange(0, CONTEXT_LENGTH).to(torch.int64), # context indexes
        caches, # caches
        True, # prefil
    )
    return input_prefill

def make_decode_input(caches, tokens_str, position):
    # todo: use tokens str to generate input
    # NOTE decode input size has to be 1
    # NOTE possition > CONTEXT_LENGTH
    input_prefill = (
        torch.randint(0, 1000, (1, )).to(torch.int64),  # len seq length
        torch.arange(position, position + 1), # input indexes
        torch.arange(position - CONTEXT_LENGTH, position), # context indexes
        caches, # caches
        False, # prefil
    )
    return input_prefill
class Llama2StableHLO:

    def __init__(self, state_dict, shlo):
        self.state_dict = state_dict
        self._prefill_name = 'prefill_forward'
        self._decode_name= 'decode_forward'
        self._shlo_bundle = shlo._bundle
        self._shlo_bundle.state_dict = state_dict
        self._name_to_stablehlo = {
            func.meta.name: func for func in self._shlo_bundle.stablehlo_funcs
        }

        self._flattened_state_dict, self._name_to_pos = flatten_state_dict(state_dict)
        self._flattened_state_dict.extend(self._shlo_bundle.additional_constants)
        self._flattened_state_dict = pytree.tree_map(tensor_to_jax_array, self._flattened_state_dict)

        pref_func = self._name_to_stablehlo[self._prefill_name]
        dec_func = self._name_to_stablehlo[self._decode_name]
        self._prefill_jax_function = wrap_as_jax_func(pref_func, compute_args_mappings(self._name_to_pos, pref_func.meta))
        self._decode_jax_function = wrap_as_jax_func(dec_func, compute_args_mappings(self._name_to_pos, dec_func.meta))

    @property
    def jax_weights(self):
        return self._flattened_state_dict

    def prefill_jax_function(self):
        return self._prefill_jax_function

    def decode_jax_function(self):
        return self._decode_jax_function

    def call_prefill(self, tokens, caches):
        func = self._name_to_stablehlo[self._prefill_name]
        inputs = make_prefill_input(caches, tokens)
        inputs = pytree.tree_map(tensor_to_jax_array, inputs)
        in_spec = pytree.treespec_loads(func.meta.input_pytree_spec)
        out_spec = pytree.treespec_loads(func.meta.output_pytree_spec)
        inputs, _ = pytree.tree_flatten(inputs)
        output = self._prefill_jax_function(self.jax_weights, inputs)
        return pytree.tree_unflatten(output, out_spec)

    def call_decode(self, tokens, caches):
        func = self._name_to_stablehlo[self._prefill_name]
        inputs = make_decode_input(caches, tokens, 2049)
        inputs = pytree.tree_map(tensor_to_jax_array, inputs)
        in_spec = pytree.treespec_loads(func.meta.input_pytree_spec)
        out_spec = pytree.treespec_loads(func.meta.output_pytree_spec)
        inputs, _ = pytree.tree_flatten(inputs)

        output = self._decode_jax_function(self.jax_weights, inputs)
        return pytree.tree_unflatten(output, out_spec)




def make_random_checkpoint(bundle):
    res = {}
    for func in bundle.stablehlo_funcs:
        for sig, loc in zip(func.meta.input_signature, func.meta.input_locations):
            if loc.type_ == stablehlo.VariableType.PARAMETER:
                res[loc.name] = torch.ones(sig.shape, dtype=getattr(torch, sig.dtype))
    return res

def load_stablehlo_model(checkpoint_dir, model_dir):
    with open(os.path.join(model_dir, 'METADATA.json')) as f:
        metadata = json.load(f)
    param_size = metadata['param_size']
    model_arg = get_arg(metadata['param_size'], metadata['context_length'] + metadata['infer_length'])
    shlo = stablehlo.StableHLOGraphModule.load(os.path.join(model_dir))
    if checkpoint_dir is not None:
        checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
        assert len(checkpoints) == 1, 'currently only support one file'
        # TODO: To support 13 and 70 B, we need to implement the ability to 
        #  merge several checkpoint file into one
        checkpoint = torch.load(checkpoints[0])
        _fill_freqs_cis(checkpoint, model_arg)
    else:
        if shlo._bundle.state_dict:
            checkpoint = shlo._bundle.state_dict
        else:
            checkpoint = make_random_checkpoint(shlo._bundle)
    caches = make_cache(model_arg)
    return Llama2StableHLO(checkpoint, shlo), caches