import torch
from torch_xla import stablehlo

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
# from torch.export.graph_signature import InputKind

from llama.model import ModelArgs, Transformer, GenLoop
from llama.stablehlo_model import get_arg, make_cache, merge_bundle
from llama import model_exportable_unbatched
from llama import model_exportable
from llama.tokenizer import Tokenizer

ckpt_dir = 'llama-2-7b-chat'
tokenizer_path = 'tokenizer.model'


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

def make_exported_to_use_orig_names(nn_module, exported):
    nn_state_dict = nn_module.state_dict()
    reverse_name = {val : key for key, val in nn_state_dict.items()}
    keys_to_replace = exported.state_dict.keys()
    new_state_dict = {}
    old_name_to_new_name = {}
    for key, val in exported.state_dict.items():
        if val in reverse_name:
            new_name = reverse_name[val]
            new_state_dict[new_name] = val
            old_name_to_new_name[key] = new_name
        else:
            new_state_dict[key] = val
            old_name_to_new_name[key] = key
    exported._state_dict = new_state_dict
    for param in exported.graph_signature.input_specs:
        if param.kind.name in ('PARAMETER', 'BUFFER'):
            param.target = old_name_to_new_name[param.target]


@torch.no_grad()
def export_llama2_to_stablehlo(
    path_prefix: str,
    checkpoint_dir: Optional[str] = None,
    batch_size: Optional[int] = 2,
    param_size: str = 'tiny',
    context_length: int = 2048,
    infer_length: int = 256,
    write_meta: bool = True,
    bf16_enable: bool = False,
):
    
    print('start')
    tokenizer = Tokenizer(model_path=tokenizer_path)
    assert param_size in ('tiny', '7b', '13b', '70b'), param_size
    max_input_seq_length = context_length + infer_length

    model_arg = get_arg(param_size, max_input_seq_length)
    model_arg.vocab_size = tokenizer.n_words
    model_arg.bf16_enable = bf16_enable

    start = time.time()
    if batch_size is None:
        m = model_exportable_unbatched.Transformer(model_arg)
    else:
        m = model_exportable.Transformer(model_arg)

    if bf16_enable:
      m = m.to(torch.bfloat16)

    end = time.time()
    print('Model init took', end - start, 'seconds')
    prefill_caches = make_cache(model_arg, 1)
    decode_caches = make_cache(model_arg, batch_size)
    if checkpoint_dir:
        checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
        assert len(checkpoints) == 1, 'currently only support one file'
        # TODO: To support 13 and 70 B, we need to implement the ability to 
        #  merge several checkpoint file into one
        checkpoint = torch.load(checkpoints[0])
        m.load_state_dict(checkpoint, strict=False)

    if batch_size is not None:
        #input_shape_prefill = (batch_size, context_length)
        input_shape_prefill = (1, context_length)
        input_shape_decode = (batch_size, 1)
    else:
        input_shape_prefill = (context_length, )
        input_shape_decode = (1,)


    sample_input_prefill = (
        torch.randint(0, 1000, input_shape_prefill,dtype=torch.int32),  # len seq length
        torch.arange(0, context_length, dtype=torch.int32), # input indexes
        torch.arange(0, context_length, dtype=torch.int32), # context indexes
        prefill_caches,
        True, # prefil
    )
    #m(*sample_input_prefill)
    
    sample_input_decode = (
        torch.randint(0, 1000, input_shape_decode, dtype=torch.int32),  # len = 1
        torch.tensor([0], dtype=torch.int32),
        torch.roll(torch.arange(context_length, dtype=torch.int32),1,0),
        decode_caches,
        False # prefill
    )
    #m(*sample_input_decode)

    print('exporting to stablehlo use original name')
    exported_prefill = torch.export.export(m, sample_input_prefill)
    make_exported_to_use_orig_names(m, exported_prefill)
    exported_decode = torch.export.export(m, sample_input_decode)
    make_exported_to_use_orig_names(m, exported_decode)

    print('exporting to stablehlo')
    shlo_prefill = stablehlo.exported_program_to_stablehlo(exported_prefill, stablehlo.StableHLOExportOptions(save_weights=False))
    shlo_decode = stablehlo.exported_program_to_stablehlo(exported_decode, stablehlo.StableHLOExportOptions(save_weights=False))

    print('merge bundle')
    merged = merge_bundle(prefill=shlo_prefill._bundle, decode=shlo_decode._bundle)
    stablehlo._save_program_bundle(merged, path_prefix)

    print('write meta')
    if write_meta:
        with open(os.path.join(path_prefix, 'META.json'), 'w') as f:
            json.dump({
                'context_length': context_length, 
                'infer_length': infer_length, 
                'param_size': param_size,
                'batch_size': batch_size
            }, f)
    print('done')


if __name__ == '__main__':
    import fire
    fire.Fire(export_llama2_to_stablehlo)
