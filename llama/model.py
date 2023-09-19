# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.utils import divide_and_check_no_remainder

from .xla_model_parallel import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
    get_model_parallel_group,
    get_model_parallel_world_size,
    get_model_parallel_rank,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    quant: bool = False

    def print_values(self):
        print(f'[WONJOO] ModelArgs')
        print(f'[WONJOO] dim={self.dim}')
        print(f'[WONJOO] n_layers={self.n_layers}')
        print(f'[WONJOO] n_heads={self.n_heads}')
        print(f'[WONJOO] max_batch_size={self.max_batch_size}')
        print(f'[WONJOO] max_seq_len={self.max_seq_len}')
        print(f'[WONJOO] quant={self.quant}')


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(
        xq.transpose(1, 2).reshape(-1, xq.shape[1], int(xq.shape[-1] / 2),
                                   2).float())
    xk_ = torch.view_as_complex(
        xk.transpose(1, 2).reshape(-1, xq.shape[1], int(xq.shape[-1] / 2),
                                   2).float())
    xq_out = torch.view_as_real(xq_ * freqs_cis)
    xk_out = torch.view_as_real(xk_ * freqs_cis)
    xq_out = xq_out.reshape(xq.shape[0], xq.shape[2], xq.shape[1],
                            xq.shape[3]).transpose(1, 2)
    xk_out = xk_out.reshape(xk.shape[0], xk.shape[2], xk.shape[1],
                            xk.shape[3]).transpose(1, 2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self,
                 args: ModelArgs,
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 groups: Optional[List] = None):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # this should be 1 if args.n_kv_heads is None
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        input_indexes: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        cache_k = cache_k.index_copy(1, input_indexes, xk)
        cache_v = cache_v.index_copy(1, input_indexes, xv)

        keys = cache_k[:, :]
        values = cache_v[:, :]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        #scores = torch.einsum('ijkl,ijml->ijkm', xq, keys)
        scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        #output = torch.einsum('ijkl,ijlm->ijkm', scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)

        # # Activation output sharding
        # # TODO(yeounoh) remove this after activation sharding support is enabled.
        # import torch_xla.core.xla_model as xm
        # import torch_xla.experimental.xla_sharding as xs
        # import torch_xla.runtime as xr
        # num_devices = xr.global_runtime_device_count()
        # device_ids = torch.arange(num_devices)
        # model = 8
        # data = num_devices // model
        # data_model_mesh = xs.HybridMesh(ici_mesh_shape=(data, 1, model))
        # xs.mark_sharding(output, data_model_mesh, (0, 1, 2))

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        groups: Optional[List] = None,
        quant: bool = False,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        if world_size is None:
            groups = get_model_parallel_group()
            world_size = get_model_parallel_world_size()
            rank = get_model_parallel_rank()

        init_method = lambda x: x

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self,
                 layer_id: int,
                 args: ModelArgs,
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 groups: Optional[List] = None):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        if world_size is None:
            groups = get_model_parallel_group()
            world_size = get_model_parallel_world_size()
            rank = get_model_parallel_rank()

        self.attention = Attention(
            args,
            world_size=world_size,
            rank=rank,
            groups=groups,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            world_size=world_size,
            rank=rank,
            groups=groups,
            quant=args.quant,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        input_indexes: torch.Tensor,
        cache_k, cache_v
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), freqs_cis, mask, input_indexes,
            cache_k, cache_v
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 world_size: Optional[int] = None,
                 rank: Optional[int] = None,
                 groups: Optional[List] = None):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        if world_size is None:
            groups = get_model_parallel_group()
            world_size = get_model_parallel_world_size()
            rank = get_model_parallel_rank()

        init_method = lambda x: x

        self.tok_embeddings = nn.Embedding(
            params.vocab_size,
            params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(
                layer_id,
                params,
                world_size=world_size,
                rank=rank,
                groups=groups,
            ))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.register_buffer("freqs_cis", freqs_cis)

        mask = torch.full(
            (1, 1, self.params.max_seq_len, self.params.max_seq_len),
            float("-inf")).to(torch.float)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, input_indexes: torch.Tensor, output_index: Optional[torch.Tensor], caches_k, caches_v):
        _bsz, seqlen = tokens.shape
        assert _bsz == self.params.max_batch_size
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.index_select(0, input_indexes)

        mask = self.mask.index_select(2, input_indexes)
        
        for i, layer in enumerate(self.layers):
            cache_k = caches_k[i]
            cache_v = caches_v[i]
            h = layer(h, freqs_cis, mask, input_indexes, cache_k, cache_v)

        h = self.norm(h)
        if output_index is not None:
            h = h.index_select(1, output_index - input_indexes[0]).squeeze(dim=1)
        output = self.output(h).float()
        return output, caches_k, caches_v
