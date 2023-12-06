import math
import time
import mlx
import datasets
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


class TransformerLM(nn.Module):
    """Transformer language model.

    Args:
        vocab_size: Size of the vocabulary.
        depth: Number of transformer encoder layers.
        dim: Dimension of the transformer.
        heads: Number of attention heads.
        
    Example:
        >>> model = TransformerLM(1000, 3, 512, 8)
        >>> x = mx.randn((4, 128)).astype(np.int32)
        >>> logits = model(x)
        >>> logits.shape
        (4, 128, 1000)
        >>> model.loss(x, x)
        array(6.9077554, dtype=float32)
    """
    def __init__(
        self,
        vocab_size: int,
        depth: int,
        dim: int,
        heads: int
    ):
        super().__init__()
        # Embed the tokens
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # input: (batch, seq_len, dim) -> (batch, seq_len, dim)
        self.transformer = nn.TransformerEncoder(depth, dim, heads)
        
        
        # input: (batch, seq_len, dim) -> (batch, seq_len, vocab_size)
        self.out_proj = nn.Linear(dim, vocab_size)
        
    def __call__(
        self, x
    ):
        """Computes the forward pass.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        mask = nn.MultiHeadAttention.create_additive_causal_mask(
            x.shape[1]
        )
        x = self.embedding(x)
        x = self.transformer(x, mask)
        return self.out_proj(x)
    
    def loss(
        self,
        x,
        y,
        reduce=True
    ):
        """Loss function.

        Args:
            x (_type_): _description_
            y (_type_): _description_
            reduce (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        logits = self(x)
        losses = nn.losses.cross_entropy(logits, y)
        mx.simpligy(losses)
        
        return mx.mean(losses) if reduce else mx.mean(
            losses,
            axis=(-1, -2)
        )
        
        
