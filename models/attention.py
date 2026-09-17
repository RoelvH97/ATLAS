"""Geometry-conditioned attention shared by the neural fields."""

import flax.linen as nn
import jax.numpy as jnp


def geglu(value):
    value, gate = jnp.split(value, 2, axis=-1)
    return value * nn.gelu(gate)


class FourierEmbedding(nn.Module):
    features: int
    frequency: float = 1.0

    @nn.compact
    def __call__(self, coordinates):
        phase = nn.Dense(
            self.features // 2, use_bias=False, kernel_init=nn.initializers.normal(self.frequency)
        )(jnp.pi * (coordinates + 1))
        features = jnp.sin(jnp.concatenate((coordinates, phase, phase + jnp.pi / 2), axis=-1))
        return nn.Dense(self.features)(features)


class SpatialAttention(nn.Module):
    """Attention over sender tokens, conditioned on invariant pairwise geometry."""

    n_heads: int
    freq_q: float = 1.0
    freq_v: float = 3.0
    nearest_k: int | None = None
    gaussian_window: bool = True
    coord_dim: int = 2
    content_query: bool = False
    film_base: str = "geometry"
    window: tuple | None = None

    @nn.compact
    def __call__(self, relative, content, query_content=None):
        # relative has shape (receivers, senders, geometric attributes); content
        # belongs to senders, while query_content belongs to receivers.
        n_queries, n_tokens = relative.shape[:2]
        width = content.shape[-1]
        if width % self.n_heads:
            raise ValueError("Attention width must be divisible by n_heads")
        head_width = width // self.n_heads
        if self.nearest_k is not None and self.nearest_k < n_tokens:
            # The first attributes are displacement in each sender's frame.
            # Its Euclidean norm makes neighbor selection independent of global pose.
            distance = (relative[..., : self.coord_dim] ** 2).sum(-1)
            indices = jnp.argsort(distance, axis=-1)[:, : self.nearest_k]
            relative = relative[jnp.arange(n_queries)[:, None], indices]
            selected = content[indices]
            keys = nn.Dense(width, use_bias=False, name="wk")(selected)
            values = nn.Dense(width, use_bias=False, name="wv")(selected)
        else:
            indices = jnp.broadcast_to(jnp.arange(n_tokens), (n_queries, n_tokens))
            keys = nn.Dense(width, use_bias=False, name="wk")(content)[None]
            values = nn.Dense(width, use_bias=False, name="wv")(content)[None]
        count = relative.shape[1]
        keys = keys.reshape(keys.shape[0], count, self.n_heads, head_width)
        if self.content_query:
            if query_content is None:
                raise ValueError("Content queries require query_content")
            query = nn.Dense(width, use_bias=False, name="wq")(query_content)
            query = query.reshape(n_queries, self.n_heads, head_width)
            geometry = FourierEmbedding(width, self.freq_q, name="pos_emb_q")(relative)
            # Default FiLM: key_jk = geometry_jk * (1 + gamma(content_k))
            #                     + beta(content_k). Keys depend on both tokens' poses.
            if self.film_base == "geometry":
                base = geometry.reshape(n_queries, count, self.n_heads, head_width)
                modulation = keys.reshape(keys.shape[0], count, width)
            elif self.film_base == "content":
                base, modulation = keys, geometry
            else:
                raise ValueError("film_base must be geometry or content")
            scale = nn.Dense(
                width, use_bias=False, name="k_film_gamma", kernel_init=nn.initializers.zeros
            )(modulation)
            shift = nn.Dense(width, use_bias=False, name="k_film_beta")(modulation)
            shape = (modulation.shape[0], count, self.n_heads, head_width)
            keys = base * (1 + scale.reshape(shape)) + shift.reshape(shape)
            scores = (query[:, None] * keys).sum(-1) * head_width**-0.5
        else:
            # Coordinate queries have no content; relative geometry supplies the query.
            query = FourierEmbedding(width, self.freq_q, name="pos_emb_q")(relative)
            query = nn.Dense(width, use_bias=False, name="wq")(query)
            query = query.reshape(n_queries, count, self.n_heads, head_width)
            scores = (query * keys).sum(-1) * head_width**-0.5
        # A Gaussian distance bias favors nearby senders before neighbor softmax.
        if self.gaussian_window:
            if self.window is None:
                window = self.coord_dim / n_tokens ** (1 / self.coord_dim)
                distance = (relative[..., : self.coord_dim] ** 2).sum(-1)
                scores = scores - distance[..., None] / window**2
            else:
                inverse = jnp.asarray([1 / value**2 for value in self.window])
                distance = (relative[..., : self.coord_dim] ** 2 * inverse).sum(-1)
                scores = scores - distance[..., None]
        weights = nn.softmax(scores, axis=1)
        self.sow("intermediates", "attention_weights", weights)
        self.sow("intermediates", "attention_indices", indices)
        # Geometry also modulates values: one sender can deliver different messages
        # to different receivers, using the same invariant relative attributes.
        scale, shift = jnp.split(
            FourierEmbedding(2 * width, self.freq_v, name="pos_emb_v")(relative), 2, axis=-1
        )
        values = (values * (1 + scale) + shift).reshape(n_queries, count, self.n_heads, head_width)
        output = jnp.einsum("qkh,qkhd->qhd", weights, values).reshape(n_queries, width)
        return nn.Dense(width, name="wo")(output)
