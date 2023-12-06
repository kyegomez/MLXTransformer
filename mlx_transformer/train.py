import numpy as np


def to_samples(
    context_size,
    dataset
):
    """Transforms a dataset into samples.

    Args:
        context_size (_type_): _description_
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    tokens = dataset.size
    window_size = context_size + 1
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize)
    )
    return X[:, :-1], X[:, 1:]

def iterate_batches(
    batch_size,
    context_size,
    dataset
):
    """Iterates over batches of samples.

    Args:
        batch_size (_type_): _description_
        context_size (_type_): _description_
        dataset (_type_): _description_

    Yields:
        _type_: _description_
    """
    inputs, target = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], target[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0
