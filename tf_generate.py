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
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

from tensorflow.compiler.tf2xla.python import xla as tfxla
from torch_xla import stablehlo
from torch_xla import tf_saved_model_integration
from torch.utils import _pytree as pytree
import torch.fx._pytree as fx_pytree

ckpt_dir = 'checkpointing/7B'
tokenizer_path = 'spiece.model'
pytorch_shlo_path = '../llama2_spmd_full'


class MyModel(tf.keras.Model):

    def __init__(self, func, args, in_spec, out_spec, strategy):
        super().__init__()
        self._func = tf.function(func)
        self.model_args = args
        self.in_spec = in_spec
        self.out_spec = out_spec
        self.strategy = strategy

    @tf.function(
        autograph=False,
        jit_compile=True,
      input_signature=(
        tf.TensorSpec(shape=(128,), dtype=tf.int64),
      )
    )
    def inference(self, input_tokens):
        max_gen_len = 1024
        section_len = 128

        decoding_start_time = time.time()
        prev_pos = 0
        cur_pos = 0

        # input_tokens = tf.reshape(input_tokens, ( (1, ) + input_tokens.shape))

        res = []

        caches_k = []
        caches_v = []
        model_args = self.model_args
        n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        head_dim = model_args.dim // model_args.n_heads
        for i in range(self.model_args.n_layers):
            cache_v = tf.zeros((
                self.model_args.max_batch_size,
                self.model_args.max_seq_len,
                n_kv_heads,
                head_dim,
            ))
            # self.strategy.experimental_split_to_logical_devices(cache_v, [1, 1, 4, 1])
            cache_k = tf.zeros((
                self.model_args.max_batch_size,
                self.model_args.max_seq_len,
                n_kv_heads,
                head_dim,
            ))
            # self.strategy.experimental_split_to_logical_devices(cache_k, [1, 1, 4, 1])   
            caches_k.append(cache_k)
            caches_v.append(cache_v)

        def condition(cur_position_tensor, *args):
            return tf.less(cur_position_tensor, max_gen_len)

        def body(cur_pos_tensor, input_tokens, caches_k, caches_v, res):
            input_pos_tensor = tf.cast(tf.range(i, i + section_len), tf.int64)
            output_pos_tensor = cur_pos_tensor - 1

            # add batch dim
            input_tokens_pt = tf.reshape(input_tokens, ( (1, ) + input_tokens.shape))
            inputs = fx_pytree.tree_flatten_spec( (
                input_tokens_pt, input_pos_tensor, None, caches_k, caches_v), self.in_spec)

            result = self._func(*inputs)
            logits, caches_k, caches_v = pytree.tree_unflatten(result, self.out_spec)
            print('HAN', logits.shape)

            next_token = tf.reshape(tf.math.argmax(logits[0][-1]), (1, ))
            print('Han res ', res.shape)
            print('Han cur pos', cur_position_tensor.shape)
            print('Han next', next_token.shape)
            res = tf.tensor_scatter_nd_update(res, [[cur_position_tensor + 1]], next_token) 
            input_tokens = tf.concat(
                [   tf.slice(input_tokens, (1, ), (input_tokens.shape[0] - 1, )),
                    next_token],
                axis=0,
            )
            return cur_pos_tensor + 1, input_tokens, caches_k, caches_v, res

        initial_result = tf.zeros((max_gen_len, ), dtype=tf.int64)

        initial_result = tf.tensor_scatter_nd_update(
            initial_result, tf.reshape(tf.range(section_len), (section_len, 1)), input_tokens
        )
        cur_position_tensor = tf.constant(section_len)
        res = tf.while_loop(condition, body, (cur_position_tensor, input_tokens, caches_k, caches_v, initial_result))[-1]
        return res


def main(input_path: str, output_path: str) -> None:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu='local')
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    print(topology)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        computation_shape=[2, 2, 1, 1],
        num_replicas=1)
    tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver, device_assignment,True)

    with tpu_strategy.scope():
        shlo_model = stablehlo.StableHLOGraphModule.load(input_path)
        tf_model = tf_saved_model_integration.make_tf_function(shlo_model)

        max_seq_len = 1024
        max_batch_size = 1

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
            model_args: ModelArgs = ModelArgs(
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                **params,
            )

        bundle = shlo_model._bundle
        shlo_model._bundle.state_dict = {
          k: tf.Variable(v, trainable=False) for k, v in shlo_model._bundle.state_dict.items()
        }
        shlo_model._bundle.additional_constants = [ 
            tf.Variable(v, trainable=False) for v in shlo_model._bundle.additional_constants
        ]
        in_spec = pytree.treespec_loads(bundle.stablehlo_funcs[0].meta.input_pytree_spec)
        out_spec = pytree.treespec_loads(bundle.stablehlo_funcs[0].meta.output_pytree_spec)
        tfm = MyModel(tf_model, model_args, in_spec, out_spec, tpu_strategy)
        signatures = {}
        tfm._variables = (
          list(bundle.state_dict.values()) + bundle.additional_constants)
        signatures[tf.saved_model.
                 DEFAULT_SERVING_SIGNATURE_DEF_KEY] = tfm.inference
        save_options = tf.saved_model.SaveOptions(function_aliases={
          'func_on_gpu1': tfm._func,
        })

        print('started tokenizer ...')
        start_time = time.time()
        tokenizer = Tokenizer(model_path=tokenizer_path)
        print(f"toked in {time.time() - start_time:.2f} seconds")

        print('model init...')
        start_time = time.time()
        print(f"model init in {time.time() - start_time:.2f} seconds")

        sentence = """
        Paul Erdős (Hungarian: Erdős Pál [ˈɛrdøːʃ ˈpaːl]; 26 March 1913 – 20 September 1996) was a Hungarian mathematician. He was one of the most prolific mathematicians and producers of mathematical conjectures[2] of the 20th century.[3] Erdős pursued and proposed problems in discrete mathematics, graph theory, number theory, mathematical analysis, approximation theory, set theory, and probability theory.[4] Much of his work centered around discrete mathematics, cracking many previously unsolved problems in the field. He championed and contributed to Ramsey theory, which studies the conditions in which order necessarily appears. Overall, his work leaned towards solving previously open problems, rather than developing or exploring new areas of mathematics."""
        sentence = 'Hello, please tell me where has best weather ever'

        tokenized = tokenizer.encode(sentence, bos=True, eos=False)

        if len(tokenized) < 128:
            tokenized = [tokenizer.pad_id] * (128 - len(tokenized)) + tokenized
        if len(tokenized) > 128:
            tokenized = tokenized[-128:]

        print(tf.autograph.to_code(tfm.inference.python_function))
        tokens = tf.cast(tf.constant(tokenized), tf.int64)
        tpu_strategy.experimental_replicate_to_logical_devices(tokens)
        print('HANQ inf result:', tpu_strategy.run(tfm.inference, args=(tokens,)))

        tf.saved_model.save(
          tfm,
          output_path,
          signatures=signatures,
          options=save_options,
        )


if __name__ == '__main__':
    fire.Fire(main)
