from collections.abc import Callable
from functools import partial
from typing import NamedTuple

from torch import Tensor, nn

__all__ = [
    "Mixer1D",
    "MixerBlock1D",
    "MixerBlockConfig",
    "Mlp",
    "PreNormResidual",
]


class PreNormResidual[T: nn.Module](nn.Module):
    def __init__(
        self,
        dim: int,
        func: T,
        *,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.func = func

    def forward(self, x: Tensor) -> Tensor:
        return self.func(self.norm(x)) + x


class Mlp(nn.Sequential):
    def __init__(
        self,
        dim: int,
        latent_dim: int,
        dropout: float = 0.0,
        *,
        act_fn: Callable[[], nn.Module] = nn.GELU,
        dense_layer: Callable[[int, int], nn.Module] = nn.Linear,
    ) -> None:
        super().__init__(
            dense_layer(dim, latent_dim),
            act_fn(),
            nn.Dropout(dropout),
            dense_layer(latent_dim, dim),
            nn.Dropout(dropout),
        )


class MixerBlockConfig(NamedTuple):
    expansion: float = 4.0
    compression: float = 0.5
    dropout: float = 0.0
    norm_layer: Callable[[int], nn.Module] = nn.LayerNorm


class MixerBlock1D(nn.Sequential):
    def __init__(
        self,
        dim: int,
        tokens: int,
        config: MixerBlockConfig,
    ) -> None:
        self.dim = dim
        self.tokens = tokens
        expansion_dim = round(config.expansion * dim)
        compression_dim = round(config.compression * dim)
        # Conv over the feature dimension = spatial mixing
        spatial_mix_layer = partial(nn.Conv1d, kernel_size=1)
        super().__init__(
            # Spatial mixing block
            PreNormResidual(
                dim,
                Mlp(
                    tokens,
                    expansion_dim,
                    dropout=config.dropout,
                    dense_layer=spatial_mix_layer,
                ),
            ),
            # Feature mixing block
            PreNormResidual(
                dim,
                Mlp(
                    dim,
                    compression_dim,
                    dropout=config.dropout,
                ),
            ),
        )


class Mixer1D(nn.Sequential):
    """A 1D (Batch * Sequence * Dim) MLP Mixer model"""
    def __init__(
        self,
        dim: int,
        tokens: int,
        depth: int,
        block_config: MixerBlockConfig | None = None,
        *,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        block_config = block_config or MixerBlockConfig(norm_layer=norm_layer)
        super().__init__(
            *[MixerBlock1D(dim, tokens, block_config) for _ in range(depth)],
            norm_layer(dim),
        )

    @property
    def dim(self) -> int:
        return self[0].dim  # type: ignore[return-value]

    @property
    def tokens(self) -> int:
        return self[0].tokens  # type: ignore[return-value]
