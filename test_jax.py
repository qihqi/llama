import time
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import flash_attention

bs = 2
seqlen = 1024
n_heads = 1024
dim = 512
rng = jax.random.PRNGKey(0)
xq = jax.random.normal(rng, (bs, n_heads, seqlen, dim))
xk = jax.random.normal(rng, (bs, n_heads, seqlen, dim))
xv = jax.random.normal(rng, (bs, n_heads, seqlen, dim))

block_sizes = flash_attention.BlockSizes(
    block_q=1024,
    block_k_major=1024,
    block_k=1024,
    block_b=1,
    block_q_major_dkv=256,
    block_k_major_dkv=256,
    block_k_dkv=256,
    block_q_dkv=256,
    block_k_major_dq=256,
    block_k_dq=256,
    block_q_dq=256,
)


print('real kernel')
mha_real = jax.jit(flash_attention.flash_attention)
for _ in range(4):
    xq = jax.random.normal(rng, (bs, n_heads, seqlen, dim))
    xk = jax.random.normal(rng, (bs, n_heads, seqlen, dim))
    xv = jax.random.normal(rng, (bs, n_heads, seqlen, dim))
    start = time.time()
    res = jax.block_until_ready(flash_attention.flash_attention(xq, xk, xv, None, block_sizes=block_sizes))
    end = time.time()
    print(end - start)

print('reference')
mha_ref = jax.jit(flash_attention.mha_reference)
for _ in range(4):
    xq = jax.random.normal(rng, (bs, n_heads, seqlen, dim))
    xk = jax.random.normal(rng, (bs, n_heads, seqlen, dim))
    xv = jax.random.normal(rng, (bs, n_heads, seqlen, dim))
    start = time.time()
    res = jax.block_until_ready(mha_ref(xq, xk, xv, None))
    end = time.time()
    print(end - start)
print(res.shape)
del res

print(' diff ')
res1 = jax.block_until_ready(mha_ref(xq, xk, xv, None))
res2 = jax.block_until_ready(mha_real(xq, xk, xv, None))
print('max diff', jnp.max(jnp.abs(res1 - res2)))
