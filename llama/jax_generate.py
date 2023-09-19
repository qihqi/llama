import make_tf_function
import sys
import collections

import time
import json

from pathlib import Path
import jax.config
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

import tensorflow as tf

import json
from absl import app, flags
from torch.utils import _pytree as pytree
from torch.fx import _pytree as fx_pytree
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np
from jax import lax
from typing import Sequence

from jax.experimental import jax2tf

from tensorflow.compiler.tf2xla.python import xla as tfxla
import dataclasses

_INPUT_PATH = '/mnt/hanq/llama2_spmd_full/'
_OUTPUT_PATH =  '/mnt/hanq/tfllama2_jax'


States = collections.namedtuple('States', [
'res',
'caches_k',
'caches_v',
'input_tokens',
'pos',
])

@dataclasses.dataclass
class MockTpuDevice:
  """Mock TPU device for testing."""
  id: int
  platform: str
  device_kind: str
  process_index: int
  coords: Sequence[int]
  core_on_chip: int
  slice_index: int = 0
  client = None
  
  def __hash__(self):
    return self.id
  
'''  
mesh = np.asarray([
  MockTpuDevice(id=0, platform='cpu', device_kind='TPU v4', process_index=0, coords=[0, 0, 0], core_on_chip=2),
  MockTpuDevice(id=1, platform='cpu', device_kind='TPU v4', process_index=0, coords=[1, 0, 0], core_on_chip=2),
  MockTpuDevice(id=2, platform='cpu', device_kind='TPU v4', process_index=0, coords=[0, 1, 0], core_on_chip=2),
  MockTpuDevice(id=3, platform='cpu', device_kind='TPU v4', process_index=0, coords=[1, 1, 0], core_on_chip=2)
])
'''

def wrap_for_loop(orig_callable, model_args, in_spec, out_spec, id_to_sharding):

  sharding = PositionalSharding(mesh_utils.create_device_mesh((4,), jax.devices()))
  def func(weights, input_tokens):
    # weightws 
    print('inside of func, weights', len(weights))
    # sharding
    for i, w in enumerate(weights):
      if len(w.shape) != 2:
        continue
      if (sharding_tuple := id_to_sharding.get(i)) is not None:
        weights[i] = lax.with_sharding_constraint(w, sharding.reshape(sharding_tuple))
      
    max_gen_len = 1024
    section_len = 128

    caches_k = []
    caches_v = []
    n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
    head_dim = model_args.dim // model_args.n_heads
    for _ in range(model_args.n_layers):
        cache_v = jnp.zeros((
            model_args.max_batch_size,
            model_args.max_seq_len,
            n_kv_heads,
            head_dim,
        ), dtype=jnp.float32)
        # self.strategy.experimental_split_to_logical_devices(cache_v, [1, 1, 4, 1])
        cache_v = lax.with_sharding_constraint(cache_v, sharding.reshape((1,1,4,1)))
        cache_k = jnp.zeros((
            model_args.max_batch_size,
            model_args.max_seq_len,
            n_kv_heads,
            head_dim,
        ), dtype=jnp.float32)
        # self.strategy.experimental_split_to_logical_devices(cache_k, [1, 1, 4, 1])   
        cache_k = lax.with_sharding_constraint(cache_k, sharding.reshape((1,1,4,1)))
        caches_k.append(cache_k)
        caches_v.append(cache_v)
    caches_k = tuple(caches_k)
    caches_v = tuple(caches_v)
      
    def body(states):
      input_pos_tensor = states.pos
      cur_pos_tensor = input_pos_tensor[-1] + 1
      output_pos_tensor = cur_pos_tensor - 1
      
      print('Han innput tokens ', states.input_tokens.dtype)

      # add batch dim
      input_tokens_pt = jnp.reshape(states.input_tokens, ( (1, ) + states.input_tokens.shape))
      inputs = fx_pytree.tree_flatten_spec( (
          input_tokens_pt, input_pos_tensor, output_pos_tensor, states.caches_k, states.caches_v), 
                                            in_spec)

      result = orig_callable(weights, inputs)
      print('HAN', result)
      logits, caches_k, caches_v = pytree.tree_unflatten(result, out_spec)
  
      next_token = jnp.argmax(logits[0][-1]).reshape((1, ))
      print('Han cur pos', cur_pos_tensor.shape)
      print('Han next', next_token.shape)
      res = states.res.at[cur_pos_tensor.reshape((1, ))].set(next_token)
      input_tokens = jnp.concatenate(
          (states.input_tokens[1:], next_token),
          axis=0,
      )
      return States(
          res,
          caches_k,
          caches_v,
          input_tokens,
          states.pos + 1
      )

    initial_result = jnp.zeros((max_gen_len, ), dtype=jnp.int64)
    states = States(
      initial_result,
      caches_k,
      caches_v,
      input_tokens,
      jnp.arange(0, section_len, dtype=jnp.int64),
    )
    
    def condition(states):
      return states.pos[-1] < max_gen_len

    return lax.while_loop(condition, body, states).res
  return func

class Params:
  dim = 4096
  multiple_of = 256
  n_heads = 32
  n_layers = 32
  norm_eps = 1e-06
  vocab_size = -1
  n_kv_heads: Optional[int] = None
  multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
  ffn_dim_multiplier: Optional[float] = None
  max_batch_size: int = 1
  max_seq_len: int = 1024
  quant: bool = False
  
def flatten_state_dict(state_dict):
  i = 0
  res = []
  mapping = {}
  for name, val in state_dict.items():
    res.append(val)
    mapping[name] = i
    i += 1
  return res, mapping


def compute_args_correspondence(name_to_pos, meta):
  res = []
  for loc in meta.input_locations:
    if loc.type_ == make_tf_function.VariableType.PARAMETER:
      res.append((0, name_to_pos[loc.name]))
    elif loc.type_ == make_tf_function.VariableType.CONSTANT:
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
  return jax2tf.call_tf(inner,
          has_side_effects=False,
          ordered=False,
          output_shape_dtype=func.meta.output_signature,
          call_tf_graph=False)


def shard_variables(name_to_pos):
  position_to_sharding = {}
  for name, i in name_to_pos.items():
    if 'tok_embeddings' in name:
      position_to_sharding[i] = (4, 1)
    if 'attention_' in name:
        if 'wo' in name:
          position_to_sharding[i] = (4, 1)
        else:
          position_to_sharding[i] = (1, 4)
    if 'feed_forward_' in name:
        if 'w2' in name:
          position_to_sharding[i] = (4, 1)
        else:
          position_to_sharding[i] = (1, 4)
    if 'output' in name:
        position_to_sharding[i] = (1, 4)
  return position_to_sharding


def main(argv) -> None:
  del argv
  jax.config.update('jax_dynamic_shapes', True)
  jax.config.update('jax_enable_x64', True)
  print('Start to load bundle', _INPUT_PATH)
  shlo_model = make_tf_function.load_program_bundle(_INPUT_PATH)
  print('done to load bundle', _INPUT_PATH)
  param = Params()
  max_seq_len = 1024
  max_batch_size = 1

  state_dict_flattened, mapping = flatten_state_dict(shlo_model.state_dict)
  id_to_sharding = shard_variables(mapping)
  mappings = compute_args_correspondence(mapping, shlo_model.stablehlo_funcs[0].meta)
  
  weights = state_dict_flattened + list(shlo_model.additional_constants)
  

  bundle = shlo_model

  in_spec_like = (5, 5, None, tuple([9 for i in range(param.n_layers)]), tuple([9 for i in range(param.n_layers)]))
  out_spec_like = (4, tuple([9 for i in range(param.n_layers)]), tuple([9 for i in range(param.n_layers)]))
  _, in_spec = pytree.tree_flatten(in_spec_like)
  _, out_spec = pytree.tree_flatten(out_spec_like)

  jax_model = wrap_as_jax_func(shlo_model.stablehlo_funcs[0], mappings)
  meta  = shlo_model.stablehlo_funcs[0].meta
  jax_model_2 = wrap_for_loop(jax_model, param, in_spec, out_spec, id_to_sharding)

  input_tokens = jnp.arange(128, dtype=jnp.int64)
  
  print('len weights = ', len(weights))
  
  # jax_model_2(weights, input_tokens)

  f_tf = jax2tf.convert(jax_model_2)
  prediction_f = lambda inputs: f_tf(weights, inputs)
  
  # print(jax.jit(jax_model_2).lower(weights, input_tokens).as_text())

  tfm = tf.Module()
  weights = tf.nest.map_structure(tf.Variable, weights) 
  tfm._variables = tf.nest.flatten(weights)
  tfm.inference = tf.function(prediction_f, jit_compile=True, autograph=False)

  signatures = {}
  signatures[tf.saved_model.
            DEFAULT_SERVING_SIGNATURE_DEF_KEY] = tfm.inference.get_concrete_function(
    tf.TensorSpec(shape=(128,), dtype=tf.int64))
  save_options = tf.saved_model.SaveOptions(function_aliases={
    'func_on_gpu1': tfm.inference,
  })


  tf.saved_model.save(
    tfm,
    _OUTPUT_PATH,
    signatures=signatures,
    options=save_options,
  )
  sys.exit(0)
  
  converted_dir = _OUTPUT_PATH.value + '_converted'

 # print('Running converter...')
 # convert_to_tpu(_OUTPUT_PATH.value, converted_dir, enable_spmd_xla_partitioning=True, 
 #                 num_cores_per_replica=4)
  

  
def convert_to_tpu(
    export_dir_cpu: str,
    export_dir_tpu: str,
    enable_spmd_xla_partitioning: bool = False,
    num_cores_per_replica: int = -1,
    topology: bytes = b'',
    device_assigment: Optional[list[int]] = None,
):
  """Converts the model in `export_dir_cpu` to TPU, and writes it to `export_dir_tpu`."""
  converter_options = tpu_converter_options_pb2.ConverterOptionsV2(
      disable_default_optimizations=True)
  converter_options.tpu_functions.add().jit_compile_functions = True
  if enable_spmd_xla_partitioning:
    xla_sharding_options = converter_options.xla_sharding_options
    xla_sharding_options.num_cores_per_replica = num_cores_per_replica
    xla_sharding_options.topology = topology
    if device_assigment is not None:
      xla_sharding_options.device_assignment.extend(device_assigment)
  tpu_converter.ConvertSavedModel(
      export_dir_cpu, export_dir_tpu, options=converter_options)


if __name__ == '__main__':
    main('')
