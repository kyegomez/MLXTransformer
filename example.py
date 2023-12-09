from mlx.core.random.randint import randint

from mlx_transformer.main import Transformer

model = Transformer(
    vocab_size=10000,
    depth=12,
    dim=512,
    heads=8,
)

# Define the lower and upper bounds of the interval
low = 0
high = 10

# Generate a single random integer within the interval [low, high)
rand_int = randint(low, high)

print(rand_int)  # Output: a random integer between 0 and 9

# Generate a random array of integers within the interval [low, high)
shape = [1, 10000, 512]  # Shape of the output array
q = randint(low, high, shape)
k = randint(low, high, shape)
v = randint(low, high, shape)

# Use the random array to perform a forward pass through the model
output = model(q, k, v)

print(output.shape)
