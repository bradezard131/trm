from collections.abc import Callable

from torch import nn

from trm.model import mixer, trm

__all__ = ["make_mlp_tiny_recursive_model", "mixer", "trm"]


def make_mlp_tiny_recursive_model(  # noqa: PLR0913
    dim: int,
    num_tokens: int,
    max_seq_len: int,
    inner_depth: int = 2,
    num_register_tokens: int = 0,
    refinement_iters: int = 3,
    latent_refinement_iters: int = 6,
    *,
    block_config: mixer.MixerBlockConfig | None = None,
    norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
) -> trm.TinyRecursiveModel:
    embeddings = trm.TRMEmbeddings(dim, num_tokens, max_seq_len, num_register_tokens)
    inner_model = mixer.Mixer1D(
        embeddings.dim,
        embeddings.max_seq_len,
        inner_depth,
        block_config,
        norm_layer=norm_layer,
    )
    return trm.TinyRecursiveModel(
        embeddings,
        inner_model,
        refinement_iters=refinement_iters,
        latent_refinement_iters=latent_refinement_iters,
    )
