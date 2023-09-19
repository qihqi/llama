import dataclasses
from dataclasses import dataclass
import enum
import enum
import json
import json
import os
import os
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from jax.experimental import jax2tf

from tensorflow.compiler.tf2xla.python import xla as tfxla




def _extract_call_parameters(args, meta, bundle):
  call_args = []
  # if meta.input_pytree_spec is not None:
  #   args, _ = pytree.tree_flatten(args)
  for loc in meta.input_locations:
    if loc.type_ == VariableType.PARAMETER:
      call_args.append(bundle.state_dict[loc.name])
    elif loc.type_ == VariableType.CONSTANT:
      call_args.append(bundle.additional_constants[loc.position])
    else:
      call_args.append(args[loc.position])
  return call_args


class VariableType(enum.Enum):
  INPUT_ARG = 'input_arg'
  PARAMETER = 'parameter'
  CONSTANT = 'constant'


@dataclass
class VariableSignature:  # either argument or parameters
  shape: List[int]
  dtype: str


@dataclass
class InputLocation:
  type_: VariableType
  position: int = -1
  name: str = ''

  @classmethod
  def parameter(cls, name: str):
    return cls(type_=VariableType.PARAMETER, name=name)

  @classmethod
  def input_arg(cls, position: int):
    return cls(type_=VariableType.INPUT_ARG, position=position)

  @classmethod
  def constant(cls, position):
    return cls(type_=VariableType.CONSTANT, position=position)


@dataclass
class StableHLOFunctionMeta:
  # name of the callable.
  name: str
  # version.
  stablehlo_version: str
  # order is the order accepted by the stablehlo
  input_signature: List[VariableSignature]
  output_signature: List[VariableSignature]
  # An input to the underlying stable callable can come from
  # the arguments the user supplied, OR a parameter, OR a constant
  input_locations: List[InputLocation]

  # input_pytree_spec
  input_pytree_spec: Optional[str] = None
  output_pytree_spec: Optional[str] = None


class StableHLOJSONSerializer(json.JSONEncoder):

  def default(self, obj):
    if dataclasses.is_dataclass(obj):
      return dataclasses.asdict(obj)
    if isinstance(obj, VariableType):
      return obj.value
    return super().default(obj)


def stablehlo_obj_hook(dct):
  targets = [
      StableHLOFunctionMeta,
      StableHLOFunc,
      VariableSignature,
      InputLocation,
      VariableSignature,
  ]

  def _match_field(clazz):
    # A dataclass become a dict on serialization then,
    # it the dict must have values for all the fields in that dataclass.
    return set(f.name for f in dataclasses.fields(clazz)) == dct.keys()

  def _try_convert_as_enum(v):
    try:
      return VariableType(v)
    except:
      return v

  for clazz in targets:
    if _match_field(clazz):
      new_dict = {k: _try_convert_as_enum(v) for k, v in dct.items()}
      return clazz(**new_dict)


@dataclass
class StableHLOFunc:
  meta: StableHLOFunctionMeta
  bytecode: bytes
  text: Optional[str]


@dataclass
class StableHLOModelBundle:
  # original state dict; but torch.Tensor's converted to np.array
  state_dict: Dict[str, Any]
  # Additional constants that we decide to hardcode.
  additional_constants: List[np.ndarray]
  # can support the case of multiple callable of the same model.
  stablehlo_funcs: List[StableHLOFunc]


def _iter_dir(path: str):
  for name in os.listdir(path):
    with tf.io.gfile.GFile(os.path.join(path, name), 'rb') as f:
      yield name, f


def load_program_bundle(stablehlo_dir: str) -> StableHLOModelBundle:
  state_dict = {}
  for name, f in _iter_dir(os.path.join(stablehlo_dir, 'data')):
    state_dict[name] = np.load(f, allow_pickle=True)

  constants = []
  for name, f in _iter_dir(os.path.join(stablehlo_dir, 'constants')):
    # name of constants are ints
    constants.append((int(name), np.load(f, allow_pickle=True)))
  constants = [v for k, v in sorted(constants)]

  metas = []
  name_to_bytecode = {}
  name_to_text = {}
  stablehlo_funcs = []
  for name, f in _iter_dir(os.path.join(stablehlo_dir, 'functions')):
    if name.endswith('.meta'):
      metas.append(json.load(f, object_hook=stablehlo_obj_hook))
    elif name.endswith('.bytecode'):
      name_to_bytecode[os.path.splitext(name)[0]] = f.read()
    elif name.endswith('.mlir'):
      name_to_text[os.path.splitext(name)[0]] = f.read()

  for meta in metas:
    stablehlo_funcs.append(
        StableHLOFunc(
            meta, name_to_bytecode[meta.name], name_to_text.get(meta.name)
        )
    )

  return StableHLOModelBundle(
      stablehlo_funcs=stablehlo_funcs,
      additional_constants=constants,
      state_dict=state_dict,
  )


def make_input_signatures(meta: StableHLOFunctionMeta) -> List[tf.TensorSpec]:
  input_pos_to_spec = {
      loc.position: spec
      for loc, spec in zip(meta.input_locations, meta.input_signature)
      if loc.type_ == VariableType.INPUT_ARG
  }
  for i in range(len(input_pos_to_spec)):
    spec = input_pos_to_spec[i]
    yield tf.TensorSpec(
        shape=spec.shape, dtype=getattr(tf, spec.dtype), name=f'input{i}'
    )


def _wrap_as_tf_func(func, bundle, call_before=None):
  def inner(*args):
    if call_before is not None: 
      call_before(bundle)
    touts = [sig.dtype for sig in func.meta.output_signature]
    souts = [sig.shape for sig in func.meta.output_signature]
    call_args = _extract_call_parameters(args, func.meta, bundle)
    return tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=touts,  # dtype information
        Sout=souts,  # Shape information
        function_list=[],
        module=func.bytecode,
    )

  return inner


def make_tf_function(bundle: StableHLOModelBundle, call_before=None):
  return _wrap_as_tf_func(bundle.stablehlo_funcs[0], bundle, call_before)

def make_jax_function(bundle: StableHLOModelBundle):
  return wrap_as_jax_func(bundle.stablehlo_funcs[0])

