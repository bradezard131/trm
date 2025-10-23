from __future__ import annotations

from typing import NamedTuple

import torch
from einops import repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, nn

__all__ = [
    "TRMEmbeddings",
    "TRMPrediction",
    "TRMState",
    "TRMTrainPrediction",
    "TinyRecursiveModel",
]


class TRMState(NamedTuple):
    outputs: Tensor
    latents: Tensor

    def masked(self, mask: Tensor) -> TRMState:
        return TRMState(outputs=self.outputs[mask], latents=self.latents[mask])


class TRMEmbeddings(nn.Module):
    """Learnable embeddings for the Tiny Recursive Model:
    - Input token embeddings
    - Output initialisation
    - Latent initialisation
    - Registers
    - Position embeddings"""

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        max_seq_len: int,
        num_register_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.input_embedding = nn.Embedding(num_tokens, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.output_init_embed = nn.Parameter(torch.empty(dim))
        self.latent_init_embed = nn.Parameter(torch.empty(dim))
        self.register_tokens = nn.Parameter(torch.empty(num_register_tokens, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.input_embedding.reset_parameters()
        self.position_embedding.reset_parameters()

        def _init(tensor: Tensor) -> None:
            nn.init.normal_(tensor.data, mean=0.0, std=0.02)

        _init(self.output_init_embed)
        _init(self.latent_init_embed)
        _init(self.register_tokens)

    @property
    def num_tokens(self) -> int:
        return self.input_embedding.num_embeddings

    @property
    def dim(self) -> int:
        return self.input_embedding.embedding_dim

    @property
    def max_seq_len(self) -> int:
        return self.position_embedding.num_embeddings

    def generate_position_embeddings(self, x: Tensor) -> Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, seq)
        return self.position_embedding(positions)  # (1, seq, dim)

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        x = self.input_embedding(tokens) + self.generate_position_embeddings(tokens)
        registers = repeat(
            self.register_tokens,
            "registers dim -> batch registers dim",
            batch=x.size(0),
        )
        return torch.cat((registers, x), dim=1)

    def strip_registers(self, x: Tensor) -> Tensor:
        return x[:, self.register_tokens.size(0) :, :]

    def get_init_state(self, x: Tensor) -> TRMState:
        def _expand(p: Tensor) -> Tensor:
            return repeat(
                p,
                "dim -> batch seq dim",
                batch=x.size(0),
                seq=x.size(1),
            )

        return TRMState(
            outputs=_expand(self.output_init_embed),
            latents=_expand(self.latent_init_embed),
        )


class TRMTrainPrediction(NamedTuple):
    logits: Tensor
    halt_logits: Tensor
    state: TRMState


class TRMPrediction(NamedTuple):
    predictions: Tensor
    exit_steps: Tensor


class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        embeddings: TRMEmbeddings,
        inner_model: nn.Module,
        *,
        refinement_iters: int = 3,
        latent_refinement_iters: int = 6,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.inner_model = inner_model
        self.token_pred = nn.Linear(
            embeddings.dim, self.embeddings.num_tokens, bias=False
        )
        self.halt_pred = nn.Sequential(
            Reduce("batch seq dim -> batch dim", "mean"),
            nn.Linear(embeddings.dim, 1, bias=False),
            Rearrange("batch 1 -> batch"),
        )
        self.refinement_iters = refinement_iters
        self.latent_refinement_iters = latent_refinement_iters

    @property
    def no_grad_refinement_iters(self) -> int:
        return self.refinement_iters - 1

    def _refine_step(self, inputs: Tensor, state: TRMState) -> TRMState:
        outputs, latents = state
        for _step in range(self.latent_refinement_iters):
            latents = self.inner_model(outputs + latents + inputs)
        outputs = self.inner_model(latents + outputs)
        return TRMState(outputs=outputs, latents=latents)

    def deep_refinement(self, inputs: Tensor, state: TRMState) -> TRMState:
        with torch.no_grad():
            for _step in range(self.no_grad_refinement_iters):
                state = self._refine_step(inputs, state)
        # Final step receives gradients
        return self._refine_step(inputs, state)

    @torch.inference_mode()
    def predict(
        self,
        x: Tensor,
        *,
        max_steps: int = 12,
        halt_thresh: float = 0.5,
    ) -> TRMPrediction:
        x = self.embeddings.embed_tokens(x)
        active_idxs = torch.arange(x.size(0), device=x.device)
        state = self.embeddings.get_init_state(x)

        preds = []
        exit_steps = []
        batch_idx_halt_order = []
        for step in range(max_steps):
            state = self.deep_refinement(x, state)

            if step < max_steps - 1:
                halt_prob = self.halt_pred(state.outputs).sigmoid_()
                halting = halt_prob >= halt_thresh
            else:
                halting = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

            halting_outputs = state.outputs[halting]
            halting_preds = self.token_pred(
                self.embeddings.strip_registers(halting_outputs)
            )

            preds.append(halting_preds)
            exit_steps.extend([step] * halting_preds.size(0))
            batch_idx_halt_order.extend(active_idxs[halting].tolist())

            if halting.all():
                break
            x = x[~halting]
            state = state.masked(~halting)
            active_idxs = active_idxs[~halting]

        token_preds = torch.cat(preds, dim=0).argmax(dim=-1)
        exit_steps_t = torch.as_tensor(exit_steps)
        sort_indices = torch.as_tensor(batch_idx_halt_order).argsort()
        return TRMPrediction(
            predictions=token_preds[sort_indices], exit_steps=exit_steps_t[sort_indices]
        )

    def forward(self, x: Tensor, state: TRMState | None = None) -> TRMTrainPrediction:
        x = self.embeddings.embed_tokens(x)
        if state is None:
            state = self.embeddings.get_init_state(x)
        state = self.deep_refinement(x, state)
        outputs = self.embeddings.strip_registers(state.outputs)
        token_logits = self.token_pred(outputs)
        halt_logits = self.halt_pred(state.outputs)

        return TRMTrainPrediction(
            logits=token_logits, halt_logits=halt_logits, state=state
        )

    # for type-checking benefits only
    def __call__(self, x: Tensor, state: TRMState | None = None) -> TRMTrainPrediction:
        return super().__call__(x, state=state)
