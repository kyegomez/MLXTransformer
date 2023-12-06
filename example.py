from mlx_transformer.main import Transformer
from mlx_transformer.flash_attn import FlashAttention
from mlx.core.random.randint import randint


model = Transformer(1000, 3, 512, 8)

# Define the lower and upper bounds of the interval
low = 0
high = 10

# Generate a single random integer within the interval [low, high)
rand_int = randint(low, high)

print(rand_int)  # Output: a random integer between 0 and 9

# Generate a random array of integers within the interval [low, high)
shape = [3, 2]  # Shape of the output array
rand_array = randint(low, high, shape)

print(rand_array)
# Output: a 2D array with shape (3, 2) containing random integers between 0 and 9

