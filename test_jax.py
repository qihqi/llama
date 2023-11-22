import jax
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu import flash_attention

bs = 2
seqlen = 1024
n_heads = 32
dim = 128
rng = jax.random.PRNGKey(0)
xq = jax.random.normal(rng, (bs, n_heads, seqlen, dim), dtype=jnp.bfloat16)
xk = jax.random.normal(rng, (bs, n_heads, seqlen, dim), dtype=jnp.bfloat16)
xv = jax.random.normal(rng, (bs, n_heads, seqlen, dim), dtype=jnp.bfloat16)

res = flash_attention.flash_attention(xq, xk, xv)

print(res.shape)
print(res.dtype)
