import time
import fire
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import enum
import json
import os
from typing import Any, Dict, List, Tuple
from pathlib import Path

from absl import app, flags
import numpy as np
import tensorflow as tf
from llama.model import ModelArgs, Transformer, GenLoop
from llama import model2
from llama.tokenizer import Tokenizer

from tensorflow.compiler.tf2xla.python import xla as tfxla
from torch_xla.experimental.stablehlo_saved_model import *
from torch_xla.experimental.stablehlo_saved_model import _load_program_bundle

ckpt_dir = 'llama-2-7b-chat'
tokenizer_path = 'tokenizer.model'


def _make_input_signatures(meta: StableHLOFunctionMeta) -> List[tf.TensorSpec]:
  input_pos_to_spec = {
      loc.position: spec
      for loc, spec in zip(meta.input_locations, meta.input_signature)
      if loc.type_ == VariableType.INPUT_ARG
  }
  for i in range(len(input_pos_to_spec)):
    spec = input_pos_to_spec[i]
    yield tf.TensorSpec(
        shape=spec.shape, dtype=getattr(tf, spec.dtype), name=f'args_{i}')


def make_tf_function(bundle: StableHLOModelBundle):
  assert len(bundle.stablehlo_funcs) == 1
  meta, bytecode = bundle.stablehlo_funcs[0]
  assert len(meta.output_signature) == 1
  output_sig = meta.output_signature[0]

  def func(*args):
    call_args = []
    for loc in meta.input_locations:
      if loc.type_ == VariableType.PARAMETER:
        call_args.append(bundle.state_dict[loc.name])
      elif loc.type_ == VariableType.CONSTANT:
        call_args.append(bundle.additional_constants[loc.position])
      else:
        call_args.append(args[loc.position])
    return tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=[output_sig.dtype],  # dtype information
        Sout=[output_sig.shape],  # Shape information
        function_list=[],
        platforms=(),
        module=bytecode,
    )

  return func


class MyModel(tf.keras.Model):

    def __init__(self, bundle):
        super().__init__()
        self._func = make_tf_function(bundle)

    @tf.function(
      input_signature=(
        tf.TensorSpec(shape=(32,), dtype=tf.int64),
      )
    )
    def inference(self, input_tokens):
        max_gen_len = 128
        section_len = 32

        decoding_start_time = time.time()
        prev_pos = 0
        cur_pos = 0
        input_tokens = tf.reshape(input_tokens, ( (1, ) + input_tokens.shape))

        res = []

        for i in range(max_gen_len - section_len):
            cur_pos_tensor = tf.constant(i + section_len)
            input_pos_tensor = tf.cast(tf.range(i, i + section_len), tf.int64)
            output_pos_tensor = cur_pos_tensor - 1

            # add batch dim
            logits = self._func(input_tokens, input_pos_tensor)

            next_token = tf.math.argmax(logits[0][0][-1])
            res.append(next_token)
            input_tokens = tf.concat(
                [   tf.slice(input_tokens, (0, 1), (1, input_tokens.shape[1] - 1)),
                    tf.reshape(next_token, (1, 1))],
                axis=1,
            )

        return tf.stack(res)


def main(input_path: str, output_path: str) -> None:
    bundle = _load_program_bundle(input_path)

    tfm = MyModel(bundle)

    bundle.state_dict = {
      k: tf.Variable(v, trainable=False) for k, v in bundle.state_dict.items()
    }
    bundle.additional_constants = [ 
      tf.Variable(v, trainable=False) for v in bundle.additional_constants
    ]
    signatures = {}
    tfm._variables = (
      list(bundle.state_dict.values()) + bundle.additional_constants)
    signatures[tf.saved_model.
             DEFAULT_SERVING_SIGNATURE_DEF_KEY] = tfm.inference
    save_options = tf.saved_model.SaveOptions(function_aliases={
      'func_on_gpu1': tfm.inference,
    })

    max_seq_len = 128
    max_batch_size = 1

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

    print('model init...')
    start_time = time.time()
    print(f"model init in {time.time() - start_time:.2f} seconds")

    sentence = 'Hello, please tell me where has best weather ever'

    tokenized = tokenizer.encode(sentence, bos=True, eos=False)

    if len(tokenized) < 32:
        tokenized = [tokenizer.pad_id] * (32 - len(tokenized)) + tokenized

    tokens = tf.cast(tf.constant(tokenized), tf.int64)
    print(tfm.inference(tokens))

    tf.saved_model.save(
      tfm,
      output_path,
      signatures=signatures,
      options=save_options,
    )



if __name__ == '__main__':
    fire.Fire(main)
