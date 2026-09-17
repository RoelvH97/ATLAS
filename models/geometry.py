"""Relative geometry for translation and rigid-motion equivariance."""

import jax
import jax.numpy as jnp


def pose_dimension(group, dimension):
    if group == "translation":
        return dimension
    if group == "roto_translation" and dimension == 2:
        return 3
    if group == "roto_translation_3d" and dimension == 3:
        return 9
    raise ValueError(f"Unsupported group/dimension: {group}/{dimension}")


def bound_pose(pose, group):
    if group == "translation":
        return jnp.tanh(pose)
    dimension = {"roto_translation": 2, "roto_translation_3d": 3}.get(group, pose.shape[-1])
    return jnp.concatenate((jnp.tanh(pose[..., :dimension]), pose[..., dimension:]), axis=-1)


def frame_matrix(frame, epsilon=1e-8):
    first, second = frame[..., :3], frame[..., 3:]
    norm = jnp.sum(first * first, axis=-1, keepdims=True)
    identity = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], frame.dtype), first.shape)
    first = jnp.where(norm > epsilon**2, first / jnp.sqrt(jnp.maximum(norm, epsilon**2)), identity)
    second = second - jnp.sum(first * second, axis=-1, keepdims=True) * first
    norm = jnp.sum(second * second, axis=-1, keepdims=True)
    axis = jax.nn.one_hot(jnp.argmin(jnp.abs(first), axis=-1), 3, dtype=frame.dtype)
    fallback = axis - jnp.sum(first * axis, axis=-1, keepdims=True) * first
    fallback /= jnp.linalg.norm(fallback, axis=-1, keepdims=True)
    second = jnp.where(
        norm > epsilon**2, second / jnp.sqrt(jnp.maximum(norm, epsilon**2)), fallback
    )
    return jnp.stack((first, second, jnp.cross(first, second)), axis=-1)


def relative_coordinates(query, key, group):
    """Express each query relative to each key pose.

    Translation uses x_q - p_k; rigid motion uses R_k.T @ (x_q - p_k).
    Posed queries also contribute relative orientation. Applying the same group
    action to both inputs leaves these attributes unchanged (bi-invariance).
    """
    if group == "translation":
        dimension = min(query.shape[-1], key.shape[-1])
        return query[:, None, :dimension] - key[None, :, :dimension]
    if group == "roto_translation":
        delta = query[:, None, :2] - key[None, :, :2]
        angle = key[None, :, 2]
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        components = [
            delta[..., 0] * cosine + delta[..., 1] * sine,
            -delta[..., 0] * sine + delta[..., 1] * cosine,
        ]
        if query.shape[-1] == 3:
            angle = query[:, None, 2] - key[None, :, 2]
            components += [jnp.cos(angle), jnp.sin(angle)]
        return jnp.stack(components, axis=-1)
    if group == "roto_translation_3d":
        rotation = frame_matrix(key[:, 3:9])
        delta = query[:, None, :3] - key[None, :, :3]
        relative = jnp.einsum("kab,qka->qkb", rotation, delta)
        if query.shape[-1] == 9:
            orientation = jnp.einsum("kab,qac->qkbc", rotation, frame_matrix(query[:, 3:9]))
            orientation = jnp.swapaxes(orientation[..., :2], -1, -2).reshape(
                query.shape[0], key.shape[0], 6
            )
            relative = jnp.concatenate((relative, orientation), axis=-1)
        return relative
    raise ValueError(f"Unknown group: {group}")
