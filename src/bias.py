"""Support biases for MSDE (the ``bias_fn`` of ``MeanShiftDensityEnhancement.forward``).

Every bias has the signature
``bias(dist, density, stability, h, learning_rate, iteration, n_iters, memory) -> (bias, memory)``, where ``dist``
is the (s, n, d) difference ``support[j] - x[i]`` and ``bias`` is the (n, d) vector added to every point's step.
"""


def default(dist, density, stability, h, learning_rate, iteration, n_iters, memory):
    """A step of ``learning_rate`` towards the mean of the support points: ``learning_rate * dist.mean(0)``.

    The bias is added outside ``learning_rate``, so without it every point would jump onto the support mean.
    """
    return learning_rate * dist.mean(0), memory
