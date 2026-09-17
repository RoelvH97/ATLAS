"""ATLAS: adaptable posed tokens with geometry-conditioned self-attention."""

import flax.linen as nn

from .attention import SpatialAttention, geglu
from .geometry import bound_pose, pose_dimension, relative_coordinates


class ATLAS(nn.Module):
    """Decode coordinates from tokens with a pose and scalar content features.

    Self-attention exchanges content between tokens; cross-attention reads the
    resulting features at query coordinates. Both use relative pose attributes.

    Args:
        output_dim: Number of predicted channels per query coordinate, such as
            three RGB channels or image channels plus a segmentation logit.
        n_latents: Number of spatial tokens in the adapted representation.
        latent_channels: Number of adapted content features per token, before
            projection to model_dim.
        model_dim: Internal feature width for attention and residual blocks.
            Must be divisible by n_heads.
        n_self_attn: Number of token self-attention blocks before decoding.
            Zero skips the self-attention stack.
        n_heads: Number of self-attention heads. Cross-attention uses one head.
        content_query: Use receiver content as the self-attention query. If
            False, construct pair-specific queries from relative geometry.
        sa_film_base: Base for self-attention keys when content_query is True.
            "geometry" lets sender content scale and shift geometry embeddings;
            "content" lets geometry scale and shift content keys. Ignored when
            content_query is False.
        freq_q: Initialization scale of learned Fourier frequencies used for
            attention scores, in both self-attention and cross-attention.
        freq_v: Initialization scale of learned Fourier frequencies used to
            modulate attention values. Larger scales start with faster spatial
            variation; the Fourier projections remain trainable.
        nearest_k: Maximum nearest tokens per coordinate query in cross-attention.
            None or a value at least n_latents uses all tokens.
        sa_nearest_k: Maximum nearest tokens per self-attention query; the token
            itself is eligible. None or a value at least n_latents uses all tokens.
        gaussian_window: Add a Gaussian distance bias to cross-attention scores,
            favoring nearby tokens before softmax.
        sa_gaussian_window: Apply the same distance bias in self-attention.
        decoder_ff: Add a residual feed-forward block after cross-attention.
            Self-attention blocks always include their own feed-forward layers.
        bi_invariant: Relative geometry: "translation" for translations,
            "roto_translation" for SE(2) with coord_dim=2, or
            "roto_translation_3d" for SE(3) with coord_dim=3.
        coord_dim: Number of spatial coordinate dimensions, usually 2 for images
            or 3 for volumes. This excludes pose orientation parameters.
        latent_grid: Optional token counts along each spatial axis, used by
            ModelFactory for initialization and by attention for Gaussian widths.
            Supply coord_dim counts whose product is n_latents. None uses an
            equal-sided grid, requiring a square token count in 2D or a cube in 3D.
        bounded_pose: Map position parameters through tanh into [-1, 1].
            Orientation parameters are unaffected.
    """

    output_dim: int = 3
    n_latents: int = 25
    latent_channels: int = 64
    model_dim: int = 128
    n_self_attn: int = 4
    n_heads: int = 4
    content_query: bool = True
    sa_film_base: str = "geometry"
    freq_q: float = 1.0
    freq_v: float = 3.0
    nearest_k: int | None = 4
    sa_nearest_k: int | None = 5
    gaussian_window: bool = True
    sa_gaussian_window: bool = True
    decoder_ff: bool = True
    bi_invariant: str = "translation"
    coord_dim: int = 2
    latent_grid: tuple | None = None
    bounded_pose: bool = True

    @property
    def pose_dim(self):
        return pose_dimension(self.bi_invariant, self.coord_dim)

    @property
    def window(self):
        return (
            tuple(self.coord_dim / size for size in self.latent_grid)
            if self.latent_grid is not None
            else None
        )

    def positions(self, latent):
        """Map optimization parameters to the physical poses used by attention."""
        # Spatial transformations act on these poses, after the optional tanh map.
        pose = latent["pose"].reshape(self.n_latents, self.pose_dim)
        return bound_pose(pose, self.bi_invariant) if self.bounded_pose else pose

    @nn.compact
    def token_features(self, latent, pose=None):
        """Update token content at fixed poses; the inner optimizer adapts poses."""
        pose = self.positions(latent) if pose is None else pose
        content = nn.Dense(self.model_dim, name="content_proj")(
            latent["ctx"].reshape(self.n_latents, self.latent_channels)
        )
        # relative[j, k] describes receiver j in sender k's local frame. Joint
        # translations (and rotations for SE(n)) leave these attributes unchanged.
        relative = relative_coordinates(pose, pose, self.bi_invariant)
        for index in range(self.n_self_attn):
            normalized = nn.LayerNorm(name=f"sa_k_norm_{index}")(content)
            # Receiver content supplies the query. Sender content and relative
            # geometry supply pair-specific keys and values over spatial neighbors.
            content = content + SpatialAttention(
                self.n_heads,
                self.freq_q,
                self.freq_v,
                self.sa_nearest_k,
                self.sa_gaussian_window,
                self.coord_dim,
                self.content_query,
                self.sa_film_base,
                self.window,
                name=f"sa_{index}",
            )(relative, normalized, normalized if self.content_query else None)
            # A shared, tokenwise feed-forward residual preserves this symmetry.
            residual = nn.LayerNorm(name=f"sa_norm2_{index}")(content)
            residual = nn.Dense(self.model_dim * 8, use_bias=False, name=f"sa_ff1_{index}")(
                residual
            )
            residual = geglu(residual)
            content = content + nn.Dense(self.model_dim, name=f"sa_ff2_{index}")(residual)
        return content

    @nn.compact
    def __call__(self, coords, latent):
        pose = self.positions(latent)
        content = self.token_features(latent, pose)
        self.sow("intermediates", "sa_content", content)
        # Cross-attention uses spatial coordinates as queries. Transforming these
        # coordinates together with token poses preserves the decoded scalar values.
        relative = relative_coordinates(coords, pose, self.bi_invariant)
        content = nn.LayerNorm(name="ca_k_norm")(content)
        output = SpatialAttention(
            1,
            self.freq_q,
            self.freq_v,
            self.nearest_k,
            self.gaussian_window,
            self.coord_dim,
            window=self.window,
            name="ca",
        )(relative, content)
        if self.decoder_ff:
            residual = nn.LayerNorm(name="ca_ff_norm")(output)
            residual = nn.Dense(self.model_dim * 8, use_bias=False, name="ca_ff1")(residual)
            residual = geglu(residual)
            output = output + nn.Dense(self.model_dim, name="ca_ff2")(residual)
        return nn.Dense(
            self.output_dim, name="output_layer", kernel_init=nn.initializers.normal(1e-3)
        )(output)
