# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import jax.numpy as jnp
import jax
from llama.jax_integration import JaxTensor
from jax.experimental.pallas.ops import attention
from jax.experimental.pallas.ops.tpu import flash_attention

from torch.utils._pytree import tree_map_only
from torch._functorch.make_functional import make_functional_with_buffers

use_jax_loop = True
use_jax_mha = False



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


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        output._elem = output._elem.astype(jnp.bfloat16)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (x.shape, freqs_cis.shape)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # TODO convert type using torch proper
    #return xq_out.type_as(xq), xk_out.type_as(xk)
    xq_out._elem = xq_out._elem.astype(jnp.bfloat16)
    xk_out._elem = xk_out._elem.astype(jnp.bfloat16)
    return xq_out, xk_out

def apply_rotary_emb2(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: tuple, #torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq = xq.reshape(*xq.shape[:-1], -1, 2)
    xk = xk.reshape(*xk.shape[:-1], -1, 2)
    xq_r = xq[..., 0]
    xq_i = xq[..., 1]
    xk_r = xk[..., 0]
    xk_i = xk[..., 1]
    fr, fi = freqs_cis
    fr = reshape_for_broadcast(fr, xq_r)
    fi = reshape_for_broadcast(fi, xq_r)

    def mul(r, i, fr, fi):
        return (r * fr - fi * i), (r*fi + i*fr)

    xq_r, xq_i = mul(xq_r, xq_i, fr, fi)
    xk_r, xk_i = mul(xk_r, xk_i, fr, fi)

    xq_out = torch.stack([xq_r, xq_i], -1).flatten(3)
    xk_out = torch.stack([xk_r, xk_i], -1).flatten(3)

    #xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    #xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # TODO convert type using torch proper
    #return xq_out.type_as(xq), xk_out.type_as(xk)
    #xq_out._elem = xq_out._elem.astype(jnp.bfloat16)
    #xk_out._elem = xk_out._elem.astype(jnp.bfloat16)
    return xq_out, xk_out


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
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
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
        indexes, cache_indexes,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_k,  # whole cache
        cache_v,  # whole cache
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb2(xq, xk, freqs_cis=freqs_cis)

        cache_k[:, indexes] = xk
        cache_v[:, indexes] = xv

        keys = cache_k[:, cache_indexes]
        values = cache_v[:, cache_indexes]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        if use_jax_mha:
            # JaxTensor -> jax.ndarry
            jxq, jkeys, jvals = xq._elem, keys._elem, values._elem
            causal = (mask is not None)
            output = flash_attention.flash_attention(jxq, jkeys, jvals, segment_ids=None, causal=causal)
            # output = jax.jit(attention.mha)(jxq, jkeys, jvals, segment_ids=None, causal=causal)
            output = JaxTensor(output)
        else:
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), xk, xv


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False,
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        indexes, 
        cache_indexes,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_k, cache_v
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        attn, xk, xv =  self.attention.forward(
            self.attention_norm(x), indexes, cache_indexes, freqs_cis, mask,
            cache_k, cache_v,
        )
        h = x + attn
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out, xk, xv

class JittedModule:

    def __init__(self, mod):
        self.func_mod, self.weights, self.buffer = make_functional_with_buffers(mod)
        def call_as_jax(weights, buffer, args):
            args = tree_map_only(jnp.ndarray, JaxTensor, args)
            weights = tree_map_only(jnp.ndarray, JaxTensor, weights)
            buffer = tree_map_only(jnp.ndarray, JaxTensor, buffer)
            res = self.func_mod(weights, buffer, *args)
            if isinstance(res, tuple):
                return tuple(r._elem if isinstance(r, JaxTensor) else r for r in res)
            else:
                return res._elem
        self.func_mod_jax = jax.jit(call_as_jax)

    def __call__(self, weights, buffer, *args):
        args = tree_map_only(JaxTensor, lambda x: x._elem, args)
        weights = tree_map_only(JaxTensor, lambda x: x._elem, weights)
        buffer = tree_map_only(JaxTensor, lambda x: x._elem, buffer)
        res = self.func_mod_jax(weights, buffer, args)
        if isinstance(res, tuple):
            return tree_map_only(jnp.ndarray, JaxTensor, res)
        else:
            return JaxTensor(res)


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False,
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self._layer = TransformerBlock(0, params)
        if use_jax_loop:
            # make one layer
            # but make the weights replicated
            self._layer.eval()
            state_dict = self._layer.state_dict()
            self._keys = list(state_dict.keys())
            for k in self._keys:
                weights = torch.stack([state_dict[k]] * params.n_layers)
                self.register_buffer(k.replace('.', '___'), weights)
        else:
            self.layers = torch.nn.ModuleList()
            self._layers2 = []
            for layer_id in range(params.n_layers):
                m = TransformerBlock(layer_id, params)
                self.layers.append(m)
                self._layers2.append(JittedModule(m))

    def _call_one_layer(self, i, args, state_dict, new_cache_k, new_cache_v):
        state_dict = {key: JaxTensor(state_dict[key][i])
                      for key in self._keys}
        state_dict = {key: torch.nn.Parameter(val) if 'cache' not in key else val 
                      for key, val in state_dict.items()}
        self._layer.load_state_dict(state_dict, False, True)
        cache_k, cache_v = args[-2:]
        cache_v = JaxTensor(cache_v._elem[i])
        cache_k = JaxTensor(cache_k._elem[i])
        res, xk, xv = self._layer(*args[:-2], cache_k, cache_v)
        new_cache_k.at[i].set(xk._elem)
        new_cache_v.at[i].set(xv._elem)
        return res, new_cache_k, new_cache_v

    def _call_one_layer2(self, inputs, weights):
        cache_k, cache_v = weights[-2:]
        cache_k = JaxTensor(cache_k)
        cache_v = JaxTensor(cache_v)
        state_dict = {key: JaxTensor(w) for key, w in zip(self._keys, weights[:-2])}
        state_dict = {key: torch.nn.Parameter(val) if 'cache' not in key else val 
                      for key, val in state_dict.items()}
        self._layer.load_state_dict(state_dict, False, True)
        h, indexes, cache_indexes, freqs_cis, mask = tree_map_only(jnp.ndarray,
            lambda x: JaxTensor(x), inputs)
        res, xk, xv = self._layer(
            h, indexes, cache_indexes, freqs_cis, mask, cache_k, cache_v)
        return tree_map_only(JaxTensor, 
            lambda x: x._elem,
            (res, indexes, cache_indexes, freqs_cis, mask)), (xk._elem, xv._elem)

    def forward(self, tokens: torch.Tensor, indexes, cache_indexes, mask, freqs_cis, cache_k, cache_v):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        # self.freqs_cis = self.freqs_cis.to(h.device)
        # freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        # indexes = torch.arange(start_pos, start_pos + seqlen)
        fr, fi = freqs_cis
        fr = torch.index_select(fr, 0, indexes)
        fi = torch.index_select(fi, 0, indexes)
        freqs_cis = (fr, fi)

        if use_jax_loop:
            print('I am called')

            state_dict = [
                getattr(self, key.replace('.', '___'))._elem
                for key in self._keys
            ]
            state_dict.append(cache_k._elem)
            state_dict.append(cache_v._elem)
            h._elem = h._elem.astype(jnp.bfloat16)
            
            carry = tree_map_only(torch.Tensor, 
                lambda x: x._elem,
                (h, indexes, cache_indexes, freqs_cis, mask))
            carry, stacked_caches = jax.lax.scan(self._call_one_layer2, carry, state_dict)
            h = carry[0]
            h = JaxTensor(h)
        else:
            for layer, layer_jit in zip(self.layers, self._layers2):
                state_dict = layer.state_dict()
                states = [state_dict[name] for name in layer_jit.func_mod.param_names]
                buffers = [state_dict[name] for name in layer_jit.func_mod.buffer_names]
                h, xk, xv = layer_jit(
                    states, buffers, h, indexes, cache_indexes, 
                    freqs_cis, mask, cache_k, cache_v)
            stacked_caches = [cache_k, cache_v]

        h = self.norm(h)
        output = self.output(h).float()
        return output, stacked_caches[0], stacked_caches[1]
