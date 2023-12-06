import math
import mlx.core as mx
import mlx.nn as nn


def exists(val):
    return val is not None


class FlashAttention(nn.Module):
    """Flash attention module.

    Args:
        dim: Dimension of the attention.
        heads: Number of attention heads.
        bias: Whether to use bias.
        mask: Attention mask.
        cache: Cache of previous attention keys and values.


    Example:
        >>> attn = FlashAttention(512, 8)
        >>> q = mx.randn((4, 128, 512))
        >>> k = mx.randn((4, 128, 512))
        >>> v = mx.randn((4, 128, 512))
        >>> out, cache = attn(q, k, v)
        >>> out.shape
        (4, 128, 512)
        >>> cache[0].shape
        (4, 8, 128, 64)
        >>> cache[1].shape
        (4, 8, 128, 64)
        >>> out, cache = attn(q, k, v, cache=cache)
        >>> out.shape
        (4, 128, 512)
        >>> cache[0].shape
        (4, 8, 128, 128)
        >>> cache[1].shape
        (4, 8, 128, 128)
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        bias: bool = False,
        mask=None,
        cache=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.bias = bias
        self.mask = mask
        self.cache = cache

        self.rope = nn.Rope(dim // heads, traditional=True)
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)

    def __call__(self, q, k, v, *args, **kwargs):
        """computes the forward pass.

        Args:
            q (_type_): _description_
            k (_type_): _description_
            v (_type_): _description_

        Returns:
            _type_: _description_
        """
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Extract some shapes
        heads = self.heads
        B, L, D = q.shape

        # Prepare q,k,v and for attn
        q = q.reshape(B, L, heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, heads, -1).transpose(0, 2, 1, 3)

        # Add rope to the q, k and combine them with a cache
        if exists(self.cache):
            k_cache, v_cache = self.cache
            q = self.rope(q, offset=k_cache.shape[2])
            k = self.rop(k, offset=k_cache.shape[2])
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)

        else:
            q = self.rope(q)
            k = self.rope(k)

        # Finally perform the attn
        scale = math.sqrt(1 / q.shape[-1])
        scores = (q * scale) @ k.transpose(0, 1, 3, 2)
        if self.mask is not None:
            scores = scores + self.mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

        # We return the keys and values to used as a cache
        return self.out_proj(values_hat), (k, v)
