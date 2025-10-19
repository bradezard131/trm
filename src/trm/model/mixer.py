from collections.abc import Callable
from typing import NamedTuple

from einops.layers.torch import Rearrange
from torch import Tensor, nn


class PreNormResidual(nn.Module):
    def __init__(
        self,
        dim: int,
        func: nn.Module,
        *,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
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
    ) -> None:
        super().__init__(
            nn.Linear(dim, latent_dim),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, dim),
            nn.Dropout(dropout),
        )


class MixerBlockConfig(NamedTuple):
    expansion: float = 4.0
    compression: float = 0.5
    dropout: float = 0.0


class MixerBlock1D(nn.Sequential):
    def __init__(
        self,
        dim: int,
        config: MixerBlockConfig,
    ) -> None:
        expansion_dim = round(config.expansion * dim)
        compression_dim = round(config.compression * dim)
        super().__init__(
            PreNormResidual(dim, Mlp(dim, expansion_dim, dropout=config.dropout)),
            Rearrange("batch seq dim -> batch dim seq"),
            PreNormResidual(dim, Mlp(dim, compression_dim, dropout=config.dropout)),
            Rearrange("batch dim seq -> batch seq dim"),
        )


class Mixer1D(nn.Sequential):
    def __init__(
        self,
        dim: int,
        depth: int,
        block_config: MixerBlockConfig | None = None,
        *,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        block_config = block_config or MixerBlockConfig()
        super().__init__(
            *[MixerBlock1D(dim, block_config) for _ in range(depth)],
            norm_layer(dim),
        )
