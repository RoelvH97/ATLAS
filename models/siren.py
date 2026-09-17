"""Sine layers and initialization used by both Functa models."""

import flax.linen as nn
import jax
import jax.numpy as jnp


def input_initializer(key, shape, dtype=jnp.float32):
    bound = 1 / shape[0]
    return jax.random.uniform(key, shape, dtype, -bound, bound)


def hidden_initializer(frequency):
    def initialize(key, shape, dtype=jnp.float32):
        bound = jnp.sqrt(6 / shape[0]) / frequency
        return jax.random.uniform(key, shape, dtype, -bound, bound)

    return initialize


def sine_network(coords, shifts, width, layers, outputs, frequency, modulate_input):
    value = nn.Dense(width, kernel_init=input_initializer, name="input_layer")(coords)
    if modulate_input:
        value = value + shifts[..., 0, :]
    value = jnp.sin(frequency * value)
    for index in range(layers - 2):
        value = nn.Dense(width, kernel_init=hidden_initializer(frequency), name=f"hidden_{index}")(
            value
        )
        value = value + shifts[..., index + int(modulate_input), :]
        value = jnp.sin(frequency * value)
    return nn.Dense(outputs, kernel_init=hidden_initializer(frequency), name="output_layer")(value)
