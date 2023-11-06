import os
import json
import time
import jax
import fire
import torch
import numpy as np
from jax import numpy as jnp
from llama.stablehlo_model import load_stablehlo_model, get_arg, make_cache
from llama.tokenizer import Tokenizer
from torch.utils import _pytree as pytree

CONTEXT_LENGTH = 2048
max_input_seq_length = CONTEXT_LENGTH + 256
tokenizer_path = 'tokenizer.model'

def make_prefill_input(caches, tokens_str):
    # todo: use tokens str to generate input
    # NOTE prefill input size has to be same as context length
    input_prefill = (
        torch.randint(0, 1000, (CONTEXT_LENGTH, )),  # len seq length
        torch.arange(0, CONTEXT_LENGTH), # input indexes
        torch.arange(0, CONTEXT_LENGTH), # context indexes
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


def tensor_to_jax_array(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        tensor = jnp.asarray(tensor)
    return tensor
        

def main(model_dir, checkpoint_dir=None):
    jax.config.update('jax_dynamic_shapes', True)
    jax.config.update('jax_enable_x64', True)
    start = time.time()
    print('start', 0)
    with open(os.path.join(model_dir, 'METADATA.json')) as f:
        metadata = json.load(f)
    m, caches = load_stablehlo_model(checkpoint_dir, model_dir)
    print('loaded grah', time.time() - start)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    print('loaded tokenizer', time.time() - start)
    print('prepare input', time.time() - start)
    if metadata['batch_size']:
        prefill_shape = (metadata['batch_size'], metadata['context_length'])
        decode_shape = (metadata['batch_size'], 1)
    else:
        prefill_shape = (metadata['context_length'], )
        decode_shape = (1, )
    print(m.call_prefill(torch.randint(0, 32000, prefill_shape), caches)[0].shape)
    print('one prefill', time.time() - start)
    print(m.call_decode(torch.randint(0, 32000, decode_shape), caches)[0].shape)
    print('one decode ', time.time() - start)


if __name__ == '__main__':
    fire.Fire(main)


