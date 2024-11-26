from jax import numpy as jnp
from jaxtyping import *

import jax

from typing import Tuple
from math import prod

import numpy as np
from scipy.stats import norm

from jaxtyping import Array, Float
from typing import *

l2 = lambda x: jnp.linalg.norm(x, axis=-1)

from finitediffx import difference


def derivative(t: Float[Array, "T"], x: Float[Array, "*T"], axis: int = 0, accuracy: int = 2) -> Float[Array, "*T"]:
    """
    Estimates the derivative of x with a central finite difference method.
    """

    return difference(x, step_size=t, axis=axis, accuracy=accuracy)


def time_gaussian_filter(times, samples, timescale=1.0):
    """
    Apply Gaussian filtering based on time distances

    Parameters:
    - times: 1D array of time points
    - samples: 2D array of samples (n_samples, n_features)
    - bandwidth: width of Gaussian kernel

    Returns:
    - Filtered samples
    """
    filtered_samples = np.zeros_like(samples)

    for i in range(len(times)):
        # Compute time-based weights
        time_distances = times - times[i]
        weights = norm.pdf(time_distances, loc=0, scale=timescale)
        weights /= weights.sum()  # Normalize weights

        # Apply weighted average
        filtered_samples[i] = np.average(samples, axis=0, weights=weights)

    return filtered_samples


def double_center(x: Float[Array, "K K"]) -> Float[Array, "K K"]:
    x = x - x.mean(axis=0, keepdims=True)

    return x - x.mean(axis=1, keepdims=True)


def build_adjacency_matrix(data: Float[Array, "K N"], distance_fn=l2) -> Float[Array, "K K"]:
    g = data[jnp.newaxis] - data[:, jnp.newaxis]

    g = distance_fn(g)

    g = double_center(g)

    return g


def knn(graph: Float[Array, "K K"], k: int) -> Tuple[Float[Array, "K k"], Float[Array, "K k"]]:
    ids = jnp.argsort(graph, axis=-1)[:, :k]

    values = jnp.take_along_axis(graph, ids, axis=-1)

    return ids, values


def local_dynamics_(x, dx_dt, number_of_neighbors=10):
    x_adjacency_graph = build_adjacency_matrix(x)
    x_neighbors_ids, x_neighbors_dist = knn(x_adjacency_graph, number_of_neighbors)  # + jnp.eye(x.shape[0])*jnp.inf
    dx_dt_neighbors = dx_dt[x_neighbors_ids]

    return dx_dt_neighbors


def compute_batch_size(x, memory=8 * 10 ** 9):

    return int(memory / (prod(x.shape) * 64 / 8))


def local_dynamics(x: Float[Array, "T N"], dx_dt: Float[Array, "T N"], number_of_neighbors: int = 10, distance_fn=l2, unit_length=True, distance_scaling=None):
    """
    :param x: data points
    :param dx_dt: vector field evaluated at data points
    :param number_of_neighbors: number of neighbors of each data point
    :param distance_fn: how to compute the distance between points
    :param unit_length: whether to normalise the dynamics to 1
    :param distance_scaling: if not none, rescales the dynamics according to an exponential with coef distance_scaling
    :return: the local dynamics
    """

    @jax.jit
    def f(y):
        dis = distance_fn(y - x)
        ids = jnp.argsort(dis)[1:number_of_neighbors + 1]
        dis = jnp.take_along_axis(dis, ids, axis=-1)

        return ids, dis

    neighbors_x, neighbors_dis = jax.lax.map(f, x, batch_size=1)

    # dx_dt = dx_dt / (jnp.linalg.norm(dx_dt, axis=-1, keepdims=True)+10**-8)

    neighbors_dx_dt = dx_dt[neighbors_x]  # x[neighbors_x] - x[:, jnp.newaxis]

    if unit_length:
        neighbors_dx_dt = neighbors_dx_dt / (jnp.linalg.norm(neighbors_dx_dt, axis=-1, keepdims=True) + 10**-6)

    if distance_scaling is not None:
        neighbors_dx_dt = neighbors_dx_dt*jnp.exp(-neighbors_dis[..., jnp.newaxis]*distance_scaling)

    return neighbors_dx_dt