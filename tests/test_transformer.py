import unittest
import numpy as np
import mlx.core as mx
from mlx_transformer import Transformer


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.model = Transformer(1000, 3, 512, 8)
        self.x = mx.randn((4, 128)).astype(np.int32)

    def test_transformer_output_shape(self):
        logits = self.model(self.x)
        self.assertEqual(logits.shape, (4, 128, 1000))

    def test_transformer_loss(self):
        loss = self.model.loss(self.x, self.x)
        self.assertIsInstance(loss, np.ndarray)
        self.assertEqual(loss.dtype, np.float32)

    # Add more tests here...
    # You can test different aspects of the Transformer class, such as:
    # - The output shape with different input shapes
    # - The loss with different inputs and targets
    # - The gradients after backpropagation
    # - The weights after training
    # - The output with different model configurations (e.g., different vocab sizes, depths, dimensions, and number of heads)
    # - The output before and after calling model.eval() and model.train()
    # - The output with different types of input data (e.g., all zeros, all ones, random numbers)
    # - The output with different batch sizes
    # - The output with different sequence lengths
    # - The output with different padding lengths
    # - The output with and without a mask
    # - The output with and without positional encoding
    # - The output with and without layer normalization
    # - The output with and without dropout
    # - The output with and without a learnable positional encoding
    # - The output with and without a learnable embedding
    # - The output with and without a learnable output projection
    # - The output with and without a learnable input projection
    # - The output with and without a learnable feed-forward network
    # - The output with and without a learnable multi-head attention mechanism
    # - The output with and without a learnable self-attention mechanism
    # - The output with and without a learnable cross-attention mechanism
    # - The output with and without a learnable encoder-decoder attention mechanism


if __name__ == "__main__":
    unittest.main()
