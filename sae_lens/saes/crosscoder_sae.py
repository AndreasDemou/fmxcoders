"""Crosscoder Sparse Autoencoder mixin.

A crosscoder extends any SAE to operate on stacked activations from multiple
layers simultaneously. Instead of encoding a single layer's activations
[batch, seq, d_in], the crosscoder encodes stacked multi-layer activations
[batch, seq, n_layers, d_in].

The key difference from a standard SAE is that the encoder and decoder weights
are three-dimensional:

    W_enc: [n_layers, d_in, d_sae]   — per-layer encoder projections
    W_dec: [d_sae, n_layers, d_in]   — per-layer decoder projections
    b_dec: [n_layers, d_in]           — per-layer decoder biases

Encoding aggregates across layers:
    hidden_pre = einsum('...ld, lds -> ...s', sae_in, W_enc) + b_enc

Decoding reconstructs all layers:
    sae_out = einsum('...s, sld -> ...ld', feature_acts, W_dec) + b_dec

The activation function (TopK, BatchTopK, JumpReLU, etc.) operates on
hidden_pre which is still [batch, seq, d_sae] — unchanged from standard SAEs.

This module provides:
- CrosscoderConfigMixin: Adds n_layers config field to any SAE config
- CrosscoderMixin: Overrides encode/decode to handle 3D weights
- Cross* config and SAE classes for each architecture
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from typing_extensions import override

from sae_lens.saes.sae import SAEConfig, TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import (
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
)
from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.jumprelu_sae import (
    JumpReLUSAE,
    JumpReLUSAEConfig,
)
from sae_lens.saes.matryoshka_batchtopk_sae import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)


# ============================================================================
# Crosscoder Config Mixin
# ============================================================================


@dataclass
class CrosscoderConfigMixin:
    """Adds crosscoder configuration to any SAE config.

    n_layers: number of layers whose activations are stacked along a new
              dimension before being fed to the SAE.
    layer_masking: per-token probability of zeroing out each layer's input
                   during training (0.0 disables). The target (sae_in) is
                   left unmasked, so the SAE learns to reconstruct missing
                   layers from the others.
    use_cross_fact: enable tensor factorization of the 3D crosscoder weights
                    (W_enc, W_dec). Factorized weights replace the full 3D
                    tensors with a low-parameter decomposition.
    cross_fact_type: factorization type. Currently supports "CP" (CANDECOMP /
                     PARAFAC). Future: "TR" (Tensor Ring).
    cross_fact_ranks: rank R of the factorization (applies to both encoder
                      and decoder).
    """

    n_layers: int = 1
    layer_masking: float = 0.0
    use_cross_fact: bool = False
    cross_fact_type: str = "CP"
    # Tuple of ints. CP requires exactly 1 value (the rank R). TR requires
    # exactly 3 values, interpreted as (r1, r2, r3) bordering each dimension:
    #   n_layers borders (r1, r2), d_in borders (r2, r3), d_sae borders (r3, r1).
    cross_fact_ranks: tuple[int, ...] = (32,)


# ============================================================================
# Crosscoder Mixin
# ============================================================================


class CrosscoderMixin:
    """Mixin that turns any SAE into a crosscoder.

    Overrides weight initialisation, encode, and decode so that the SAE
    operates on stacked multi-layer activations [batch, seq, n_layers, d_in].

    Usage (combined with a base training SAE via multiple inheritance):

        class CrossBatchTopKTrainingSAE(CrosscoderMixin, BatchTopKTrainingSAE):
            pass

    The mixin must come *first* in the MRO so its overrides take priority.
    """

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_crosscoder_weights(self) -> None:
        """Replace the standard 2-D weights with crosscoder weights.

        Called from initialize_weights() after super() has created the
        standard W_enc [d_in, d_sae], W_dec [d_sae, d_in], b_dec [d_in].

        Two modes:
        - Full (default): create 3-D W_enc [n_layers, d_in, d_sae] and
          W_dec [d_sae, n_layers, d_in].
        - CP factorized (use_cross_fact=True, cross_fact_type="CP"): create
          factor matrices instead of the full 3-D tensors.
              W_enc[l, d, s] = Σ_r A_enc[l,r] * B_enc[d,r] * C_enc[s,r]
              W_dec[s, l, d] = Σ_r A_dec[s,r] * B_dec[l,r] * C_dec[d,r]
          The full W_enc/W_dec are NEVER materialised in the forward pass;
          we use efficient einsum directly on the factors.

        b_dec [n_layers, d_in] is identical in both modes.
        """
        n_layers: int = self.cfg.n_layers  # type: ignore[attr-defined]
        d_in: int = self.cfg.d_in  # type: ignore[attr-defined]
        d_sae: int = self.cfg.d_sae  # type: ignore[attr-defined]

        # --- decoder bias: [n_layers, d_in] ---
        self.b_dec = nn.Parameter(
            torch.zeros(n_layers, d_in, dtype=self.dtype, device=self.device)  # type: ignore[attr-defined]
        )

        if self._is_cp_factored() or self._is_tr_factored():
            # The base-class initialize_weights has already registered stale
            # 2-D W_dec [d_sae, d_in] and W_enc [d_in, d_sae] parameters that
            # the factored forward pass never touches. Drop them so they
            # don't waste optimizer state or clutter state_dicts.
            if hasattr(self, "W_dec"):
                del self.W_dec
            if hasattr(self, "W_enc"):
                del self.W_enc

        if self._is_cp_factored():
            self._init_cp_factors(n_layers, d_in, d_sae)
            return
        if self._is_tr_factored():
            self._init_tr_cores(n_layers, d_in, d_sae)
            return

        # --- Full 3-D weights ---
        # decoder weights: [d_sae, n_layers, d_in]
        w_dec_data = torch.empty(
            d_sae, n_layers, d_in, dtype=self.dtype, device=self.device  # type: ignore[attr-defined]
        )
        for l in range(n_layers):
            nn.init.kaiming_uniform_(w_dec_data[:, l, :])
        self.W_dec = nn.Parameter(w_dec_data)

        # encoder weights: [n_layers, d_in, d_sae] (transpose of W_dec, per layer)
        w_enc_data = self.W_dec.data.permute(1, 2, 0).clone().detach().contiguous()
        self.W_enc = nn.Parameter(w_enc_data)

    # ------------------------------------------------------------------
    # CP factorization helpers
    # ------------------------------------------------------------------

    def _is_cp_factored(self) -> bool:
        """True iff weights are stored as CP factors instead of full 3-D tensors."""
        cfg = self.cfg  # type: ignore[attr-defined]
        return (
            getattr(cfg, "use_cross_fact", False)
            and getattr(cfg, "cross_fact_type", "CP") == "CP"
        )

    def _is_tr_factored(self) -> bool:
        """True iff weights are stored as TR cores instead of full 3-D tensors."""
        cfg = self.cfg  # type: ignore[attr-defined]
        return (
            getattr(cfg, "use_cross_fact", False)
            and getattr(cfg, "cross_fact_type", "CP") == "TR"
        )

    def _is_factored(self) -> bool:
        """True iff weights are stored in any factorized form (CP or TR)."""
        return self._is_cp_factored() or self._is_tr_factored()

    def _init_cp_factors(self, n_layers: int, d_in: int, d_sae: int) -> None:
        """Create CP factor matrices for encoder and decoder.

        Init scheme: per-factor i.i.d. Normal with std chosen so that the
        materialised W_enc[l,d,s] (sum of R rank-1 terms) has variance
        ≈ 6/(d_in + d_sae) — matching Kaiming-uniform on a per-layer slice.

        Var(W) = R * Var(A) * Var(B) * Var(C). With equal variance per
        factor, std_factor = (target_var / R) ** (1/6).
        """
        cfg = self.cfg  # type: ignore[attr-defined]
        if cfg.cross_fact_type != "CP":
            raise RuntimeError(
                f"_init_cp_factors called but cross_fact_type={cfg.cross_fact_type!r}"
            )
        ranks = tuple(cfg.cross_fact_ranks)
        if len(ranks) != 1:
            raise ValueError(
                f"CP requires exactly 1 rank, got cross_fact_ranks={ranks}"
            )
        R: int = ranks[0]
        if R <= 0:
            raise ValueError(f"CP rank must be positive, got {R}")

        # Stds (variance-matching to Kaiming-uniform on a 2-D slice)
        target_var_enc = 6.0 / (d_in + d_sae)
        std_enc = (target_var_enc / R) ** (1.0 / 6.0)
        target_var_dec = 6.0 / (n_layers * d_in + d_sae)
        std_dec = (target_var_dec / R) ** (1.0 / 6.0)

        dtype = self.dtype  # type: ignore[attr-defined]
        device = self.device  # type: ignore[attr-defined]

        # Encoder factors: W_enc[l, d, s] = Σ_r A_enc[l,r] * B_enc[d,r] * C_enc[s,r]
        self.A_enc = nn.Parameter(torch.empty(n_layers, R, dtype=dtype, device=device))
        self.B_enc = nn.Parameter(torch.empty(d_in, R, dtype=dtype, device=device))
        self.C_enc = nn.Parameter(torch.empty(d_sae, R, dtype=dtype, device=device))
        nn.init.normal_(self.A_enc, std=std_enc)
        nn.init.normal_(self.B_enc, std=std_enc)
        nn.init.normal_(self.C_enc, std=std_enc)

        # Decoder factors: W_dec[s, l, d] = Σ_r A_dec[s,r] * B_dec[l,r] * C_dec[d,r]
        self.A_dec = nn.Parameter(torch.empty(d_sae, R, dtype=dtype, device=device))
        self.B_dec = nn.Parameter(torch.empty(n_layers, R, dtype=dtype, device=device))
        self.C_dec = nn.Parameter(torch.empty(d_in, R, dtype=dtype, device=device))
        nn.init.normal_(self.A_dec, std=std_dec)
        nn.init.normal_(self.B_dec, std=std_dec)
        nn.init.normal_(self.C_dec, std=std_dec)

    def _init_tr_cores(self, n_layers: int, d_in: int, d_sae: int) -> None:
        """Create TR cores for encoder and decoder.

        Three ranks (r1, r2, r3) border the dimensions:
            n_layers borders (r1, r2),  d_in borders (r2, r3),  d_sae borders (r3, r1).

        Each tensor has 3 cores; encoder and decoder share core SHAPES per
        dimension (so the rank associated with each dim is identical between
        encoder and decoder), but the parameters are independent.

        Encoder W_enc[l, d, s] = einsum('alb, bdc, csa -> lds', G_enc_n, G_enc_d, G_enc_s)
        Decoder W_dec[s, l, d] = einsum('xsy, ylz, zdx -> sld', G_dec_s, G_dec_n, G_dec_d)
            with x=r3, y=r1, z=r2 — so cores share shapes:
              G_dec_s [r3, d_sae, r1]  (matches G_enc_s)
              G_dec_n [r1, n_layers, r2] (matches G_enc_n)
              G_dec_d [r2, d_in, r3]   (matches G_enc_d)

        Init scheme: Var(W) = (r1*r2*r3) * Var(G_n) * Var(G_d) * Var(G_s).
        Equal stds give std = (target_var / (r1*r2*r3)) ** (1/6).
        """
        cfg = self.cfg  # type: ignore[attr-defined]
        if cfg.cross_fact_type != "TR":
            raise RuntimeError(
                f"_init_tr_cores called but cross_fact_type={cfg.cross_fact_type!r}"
            )
        ranks = tuple(cfg.cross_fact_ranks)
        if len(ranks) != 3:
            raise ValueError(
                f"TR requires exactly 3 ranks (r1, r2, r3), got cross_fact_ranks={ranks}"
            )
        r1, r2, r3 = ranks
        if min(r1, r2, r3) <= 0:
            raise ValueError(f"TR ranks must be positive, got {ranks}")

        # Variance-matching init
        rank_prod = r1 * r2 * r3
        target_var_enc = 6.0 / (d_in + d_sae)
        std_enc = (target_var_enc / rank_prod) ** (1.0 / 6.0)
        target_var_dec = 6.0 / (n_layers * d_in + d_sae)
        std_dec = (target_var_dec / rank_prod) ** (1.0 / 6.0)

        dtype = self.dtype  # type: ignore[attr-defined]
        device = self.device  # type: ignore[attr-defined]

        # Encoder cores
        self.G_enc_n = nn.Parameter(torch.empty(r1, n_layers, r2, dtype=dtype, device=device))
        self.G_enc_d = nn.Parameter(torch.empty(r2, d_in, r3, dtype=dtype, device=device))
        self.G_enc_s = nn.Parameter(torch.empty(r3, d_sae, r1, dtype=dtype, device=device))
        nn.init.normal_(self.G_enc_n, std=std_enc)
        nn.init.normal_(self.G_enc_d, std=std_enc)
        nn.init.normal_(self.G_enc_s, std=std_enc)

        # Decoder cores (same shapes per-dimension as encoder, independent params)
        self.G_dec_n = nn.Parameter(torch.empty(r1, n_layers, r2, dtype=dtype, device=device))
        self.G_dec_d = nn.Parameter(torch.empty(r2, d_in, r3, dtype=dtype, device=device))
        self.G_dec_s = nn.Parameter(torch.empty(r3, d_sae, r1, dtype=dtype, device=device))
        nn.init.normal_(self.G_dec_n, std=std_dec)
        nn.init.normal_(self.G_dec_d, std=std_dec)
        nn.init.normal_(self.G_dec_s, std=std_dec)

    # ------------------------------------------------------------------
    # process_sae_in — subtract b_dec from multi-layer input
    # ------------------------------------------------------------------

    def process_sae_in(self, sae_in: torch.Tensor) -> torch.Tensor:
        """Preprocess input: cast, hook, subtract b_dec.

        Input shape:  [..., n_layers, d_in]
        Output shape: [..., n_layers, d_in]
        """
        sae_in = sae_in.to(self.dtype)  # type: ignore[attr-defined]
        sae_in = self.reshape_fn_in(sae_in)  # type: ignore[attr-defined]
        sae_in = self.hook_sae_input(sae_in)  # type: ignore[attr-defined]
        sae_in = self.run_time_activation_norm_fn_in(sae_in)  # type: ignore[attr-defined]

        bias_term = self.b_dec * self.cfg.apply_b_dec_to_input  # type: ignore[attr-defined]
        return sae_in - bias_term

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def _crosscoder_encode_hidden_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoding logic: x -> hidden_pre.

        x:          [..., n_layers, d_in]
        W_enc:      [n_layers, d_in, d_sae]
        hidden_pre: [..., d_sae]

        During training with layer_masking > 0, each token position gets its
        own Bernoulli mask over layers: each layer is zeroed independently
        with probability `layer_masking`.
        """
        sae_in = self.process_sae_in(x)

        p_mask = getattr(self.cfg, "layer_masking", 0.0)  # type: ignore[attr-defined]
        if self.training and p_mask > 0.0:  # type: ignore[attr-defined]
            # Per-token mask: shape [..., n_layers], broadcast over d_in
            keep_shape = sae_in.shape[:-1]  # [..., n_layers]
            keep_mask = torch.bernoulli(
                torch.full(keep_shape, 1.0 - p_mask, device=sae_in.device, dtype=sae_in.dtype)
            )
            sae_in = sae_in * keep_mask.unsqueeze(-1)

        if self._is_cp_factored():
            # Efficient CP forward (never materialise W_enc):
            # hidden_pre[..., s] = Σ_l Σ_d sae_in[..., l, d] * Σ_r A[l,r] B[d,r] C[s,r]
            hidden_pre = torch.einsum(
                "...ld, lr, dr, sr -> ...s",
                sae_in, self.A_enc, self.B_enc, self.C_enc,
            ) + self.b_enc  # type: ignore[attr-defined]
        elif self._is_tr_factored():
            # Efficient TR forward (never materialise W_enc):
            #   W_enc[l, d, s] = Σ_{a,b,c} G_enc_n[a,l,b] G_enc_d[b,d,c] G_enc_s[c,s,a]
            #   hidden_pre[..., s] = Σ_{l,d} sae_in[..., l, d] * W_enc[l, d, s]
            hidden_pre = torch.einsum(
                "...ld, alb, bdc, csa -> ...s",
                sae_in, self.G_enc_n, self.G_enc_d, self.G_enc_s,
            ) + self.b_enc  # type: ignore[attr-defined]
        else:
            hidden_pre = torch.einsum("...ld, lds -> ...s", sae_in, self.W_enc) + self.b_enc  # type: ignore[attr-defined]
        return self.hook_sae_acts_pre(hidden_pre)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def _crosscoder_decode_pre(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Shared decoding logic: feature_acts -> sae_out (before hooks).

        feature_acts: [..., d_sae]
        W_dec:        [d_sae, n_layers, d_in]
        output:       [..., n_layers, d_in]
        """
        if self._is_cp_factored():
            return torch.einsum(
                "...s, sr, lr, dr -> ...ld",
                feature_acts, self.A_dec, self.B_dec, self.C_dec,
            ) + self.b_dec  # type: ignore[attr-defined]
        if self._is_tr_factored():
            # W_dec[s, l, d] = Σ_{x,y,z} G_dec_s[x,s,y] G_dec_n[y,l,z] G_dec_d[z,d,x]
            #   (x=r3, y=r1, z=r2; matches encoder's per-dim ranks)
            return torch.einsum(
                "...s, xsy, ylz, zdx -> ...ld",
                feature_acts, self.G_dec_s, self.G_dec_n, self.G_dec_d,
            ) + self.b_dec  # type: ignore[attr-defined]
        return torch.einsum("...s, sld -> ...ld", feature_acts, self.W_dec) + self.b_dec  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # decoder norms (for rescale_acts_by_decoder_norm)
    # ------------------------------------------------------------------

    def _crosscoder_decoder_norms(self) -> torch.Tensor:
        """Per-feature decoder Frobenius norm.

        Standard mode: ||W_dec[s, :, :]||_F by reshape+norm.
        CP mode: compute via Gram matrices without materialising W_dec.

        For CP, W_dec[s,l,d] = Σ_r A_dec[s,r] B_dec[l,r] C_dec[d,r], so:
            ||W_dec[s, :, :]||_F²
              = <Σ_r A_dec[s,r] (B_dec[:,r] ⊗ C_dec[:,r]), Σ_r' ...>
              = Σ_r Σ_r' A_dec[s,r] A_dec[s,r'] (B_dec^T B_dec)[r,r'] (C_dec^T C_dec)[r,r']
              = A_dec[s] @ M @ A_dec[s]^T   with M = (B_dec^T B_dec) ⊙ (C_dec^T C_dec)

        Cost: O(d_sae * R²) instead of O(d_sae * n_layers * d_in).

        Returns: [d_sae]
        """
        if self._is_cp_factored():
            G_B = self.B_dec.T @ self.B_dec  # [R, R]
            G_C = self.C_dec.T @ self.C_dec  # [R, R]
            M = G_B * G_C  # elementwise (Hadamard)
            # ||W_dec[s, :, :]||² = sum_r sum_r' A_dec[s,r] A_dec[s,r'] M[r,r']
            #                     = ((A_dec @ M) * A_dec).sum(-1)
            norms_sq = (self.A_dec @ M * self.A_dec).sum(dim=-1)
            return norms_sq.clamp(min=1e-16).sqrt()
        if self._is_tr_factored():
            # Cheap proxy: ||G_dec_s[:, s, :]||_F.
            # G_dec_s has shape [r3, d_sae, r1]; sum over r3 and r1 dims.
            # This is NOT the true Frobenius norm of W_dec[s,:,:] (which would
            # depend on G_dec_n and G_dec_d too), but it's:
            #   - per-feature monotonic in the s-indexed core's magnitude
            #   - cheap: O(d_sae * r3 * r1)
            #   - internally consistent with fold_W_dec_norm (see that method).
            norms_sq = self.G_dec_s.pow(2).sum(dim=(0, 2))  # [d_sae]
            return norms_sq.clamp(min=1e-16).sqrt()
        flat = self.W_dec.reshape(self.W_dec.shape[0], -1)  # type: ignore[attr-defined]
        return flat.norm(dim=-1).clamp(min=1e-8)

    # ------------------------------------------------------------------
    # log_histograms — override to report true 3-D decoder norms
    # ------------------------------------------------------------------

    @torch.no_grad()
    def log_histograms(self) -> dict[str, NDArray[Any]]:
        """Return per-feature ||W_dec[s, :, :]||_F for wandb logging.

        The base SAE.log_histograms computes `self.W_dec.norm(dim=1)` which
        assumes a 2-D decoder. In crosscoder mode that tensor is either
        (a) a stale 2-D ghost parameter left over from the base class init
        when CP/TR factorization is active, or (b) a 3-D tensor whose dim=1
        norm is a 2-D matrix — neither is meaningful. Use the factor-aware
        helper instead so histograms reflect the actual trainable decoder.
        """
        norms = self._crosscoder_decoder_norms().detach().float().cpu().numpy()
        return {"weights/W_dec_norms": norms}

    # ------------------------------------------------------------------
    # fold_W_dec_norm
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        """Fold per-feature decoder norms into encoder weights.

        Standard mode: divide W_dec[s,:,:] by norms[s], scale W_enc[:,:,s] by norms[s].

        CP mode: divide A_dec[s,:] by norms[s] (this exactly normalises
        W_dec[s,:,:] since it appears as a scalar coefficient in the CP
        decomposition); scale C_enc[s,:] by norms[s] (this exactly scales
        W_enc[:,:,s]). The factorisation is preserved.
        """
        norms = self._crosscoder_decoder_norms()  # [d_sae]

        if self._is_cp_factored():
            self.A_dec.data = self.A_dec.data / norms[:, None]
            self.C_enc.data = self.C_enc.data * norms[:, None]
        elif self._is_tr_factored():
            # Both d_sae cores have shape [r3, d_sae, r1]. Per-feature scaling of
            # the d_sae axis preserves the factorisation:
            #   W_dec[s,:,:] is scaled by 1/norms[s] when G_dec_s[:,s,:] /= norms[s]
            #   W_enc[:,:,s] is scaled by norms[s]   when G_enc_s[:,s,:] *= norms[s]
            self.G_dec_s.data = self.G_dec_s.data / norms[None, :, None]
            self.G_enc_s.data = self.G_enc_s.data * norms[None, :, None]
        else:
            self.W_dec.data = self.W_dec.data / norms[:, None, None]  # type: ignore[attr-defined]
            self.W_enc.data = self.W_enc.data * norms[None, None, :]  # type: ignore[attr-defined]

        if hasattr(self, "b_enc") and isinstance(self.b_enc, nn.Parameter):
            self.b_enc.data = self.b_enc.data * norms  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # process_state_dict_for_saving_inference
    # ------------------------------------------------------------------

    def _crosscoder_fold_norm_state_dict(self, state_dict: dict) -> None:
        """Fold per-feature decoder norms into encoder weights in a state_dict.

        Standard mode: 3D W_dec / W_enc tensors.
        CP mode: factor matrices A_dec / C_enc (mirrors fold_W_dec_norm).
        """
        if self._is_cp_factored():
            A_dec = state_dict["A_dec"]
            B_dec = state_dict["B_dec"]
            C_dec = state_dict["C_dec"]
            G_B = B_dec.T @ B_dec
            G_C = C_dec.T @ C_dec
            M = G_B * G_C
            norms_sq = (A_dec @ M * A_dec).sum(dim=-1)
            norms = norms_sq.clamp(min=1e-16).sqrt()  # [d_sae]

            state_dict["A_dec"] = A_dec / norms[:, None]
            state_dict["C_enc"] = state_dict["C_enc"] * norms[:, None]
            if "b_enc" in state_dict:
                state_dict["b_enc"] = state_dict["b_enc"] * norms
            return

        if self._is_tr_factored():
            # Cheap proxy norm: ||G_dec_s[:, s, :]||_F (matches _crosscoder_decoder_norms).
            G_dec_s = state_dict["G_dec_s"]  # [r3, d_sae, r1]
            norms_sq = G_dec_s.pow(2).sum(dim=(0, 2))
            norms = norms_sq.clamp(min=1e-16).sqrt()  # [d_sae]

            state_dict["G_dec_s"] = G_dec_s / norms[None, :, None]
            state_dict["G_enc_s"] = state_dict["G_enc_s"] * norms[None, :, None]
            if "b_enc" in state_dict:
                state_dict["b_enc"] = state_dict["b_enc"] * norms
            return

        W_dec = state_dict["W_dec"]  # [d_sae, n_layers, d_in]
        flat = W_dec.reshape(W_dec.shape[0], -1)
        norms = flat.norm(dim=-1).clamp(min=1e-8)  # [d_sae]

        state_dict["W_dec"] = W_dec / norms[:, None, None]
        state_dict["W_enc"] = state_dict["W_enc"] * norms[None, None, :]
        if "b_enc" in state_dict:
            state_dict["b_enc"] = state_dict["b_enc"] * norms

    def _crosscoder_process_state_dict_for_saving_inference(
        self, state_dict: dict
    ) -> None:
        """Override for crosscoder: skip the base class norm folding (which
        assumes 2D W_dec) and use our 3D-aware version instead.

        Calls the grandparent TrainingSAE method for common logic, then
        applies crosscoder-specific norm folding if rescale_acts_by_decoder_norm.
        """
        from sae_lens.saes.sae import TrainingSAE
        TrainingSAE.process_state_dict_for_saving_inference(self, state_dict)

        if self.cfg.rescale_acts_by_decoder_norm:  # type: ignore[attr-defined]
            self._crosscoder_fold_norm_state_dict(state_dict)

    # ------------------------------------------------------------------
    # get_decoder_directions (for SAEBench compatibility)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_decoder_directions(self) -> torch.Tensor:
        """Return decoder directions for SAEBench compatibility.

        SAEBench expects shape [d_sae, d_in].  For crosscoders we return
        the mean across layers of the per-layer decoder directions,
        normalised to unit norm.

        CP mode: mean_l W_dec[s,l,d] = Σ_r A_dec[s,r] * mean_l(B_dec[l,r]) * C_dec[d,r]
                                     = (A_dec * B_dec.mean(0)) @ C_dec.T
        """
        if self._is_cp_factored():
            B_mean = self.B_dec.mean(dim=0)  # [R]
            mean_dec = (self.A_dec * B_mean) @ self.C_dec.T  # [d_sae, d_in]
        elif self._is_tr_factored():
            # mean_l W_dec[s,l,d] = einsum over (x,y,z) of
            #   G_dec_s[x,s,y] * (mean_l G_dec_n[y,l,z]) * G_dec_d[z,d,x]
            G_dec_n_mean = self.G_dec_n.mean(dim=1)  # [r1, r2]
            mean_dec = torch.einsum(
                "xsy, yz, zdx -> sd",
                self.G_dec_s, G_dec_n_mean, self.G_dec_d,
            )
        else:
            mean_dec = self.W_dec.mean(dim=1)  # type: ignore[attr-defined]
        return mean_dec / mean_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Single-layer slice (for SAEBench RAVEL / absorption evals)
    # ------------------------------------------------------------------

    def _slice_unfactored(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_enc_2d = self.W_enc[layer_idx].clone()  # type: ignore[attr-defined]
        W_dec_2d = self.W_dec[:, layer_idx, :].clone()  # type: ignore[attr-defined]
        b_dec_2d = self.b_dec[layer_idx].clone()  # type: ignore[attr-defined]
        return W_enc_2d, W_dec_2d, b_dec_2d

    def _slice_cp(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # W_enc[l,d,s] = Σ_r A_enc[l,r] B_enc[d,r] C_enc[s,r]
        # → W_enc_2d[d,s] = ((B_enc * A_enc[layer_idx]) @ C_enc.T)
        W_enc_2d = (self.B_enc * self.A_enc[layer_idx]) @ self.C_enc.T
        # W_dec[s,l,d] = Σ_r A_dec[s,r] B_dec[l,r] C_dec[d,r]
        # → W_dec_2d[s,d] = (A_dec * B_dec[layer_idx]) @ C_dec.T
        W_dec_2d = (self.A_dec * self.B_dec[layer_idx]) @ self.C_dec.T
        b_dec_2d = self.b_dec[layer_idx].clone()  # type: ignore[attr-defined]
        return W_enc_2d, W_dec_2d, b_dec_2d

    def _slice_tr(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # W_enc[l,d,s] = einsum("alb, bdc, csa -> lds", G_enc_n, G_enc_d, G_enc_s)
        # → W_enc_2d[d,s] = einsum("ab, bdc, csa -> ds", G_enc_n[:,l,:], G_enc_d, G_enc_s)
        W_enc_2d = torch.einsum(
            "ab, bdc, csa -> ds",
            self.G_enc_n[:, layer_idx, :], self.G_enc_d, self.G_enc_s,
        )
        # W_dec[s,l,d] = einsum("xsy, ylz, zdx -> sld", G_dec_s, G_dec_n, G_dec_d)
        # → W_dec_2d[s,d] = einsum("xsy, yz, zdx -> sd", G_dec_s, G_dec_n[:,l,:], G_dec_d)
        W_dec_2d = torch.einsum(
            "xsy, yz, zdx -> sd",
            self.G_dec_s, self.G_dec_n[:, layer_idx, :], self.G_dec_d,
        )
        b_dec_2d = self.b_dec[layer_idx].clone()  # type: ignore[attr-defined]
        return W_enc_2d, W_dec_2d, b_dec_2d

    def _slice_2d_weights(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to the correct slice helper based on factorization mode.

        Returns (W_enc_2d, W_dec_2d, b_dec_2d) with shapes
        ([d_in, d_sae], [d_sae, d_in], [d_in]). Never materializes the full
        3-D weights for factored modes.
        """
        if self._is_cp_factored():
            return self._slice_cp(layer_idx)
        if self._is_tr_factored():
            return self._slice_tr(layer_idx)
        return self._slice_unfactored(layer_idx)

    @torch.no_grad()
    def as_single_layer_view(self, layer_idx: int) -> "SAE":  # noqa: F821
        """Return a 2-D SAE whose weights are the layer-`layer_idx` slice of this crosscoder.

        Use this to feed crosscoders into single-layer-SAE evals such as
        SAEBench's RAVEL and feature-absorption: both factored and unfactored
        crosscoders go through the same slicing, so comparisons are
        apples-to-apples (and also comparable to single-layer SAE baselines).

        The returned SAE has architecture "topk" (when source is "cross_topk")
        or "jumprelu" (otherwise) and its `metadata.hook_name` is set to
        `metadata.hook_names[layer_idx]` if available, so SAEBench's
        `_standardize_sae_cfg` can derive the correct `hook_layer`.

        Note: this degrades the crosscoder to a single-layer SAE — the encoder
        sees only one layer's input, not the multi-layer input it was trained
        on. That degradation is the cost of the apples-to-apples comparison.
        """
        n_layers: int = self.cfg.n_layers  # type: ignore[attr-defined]
        if not 0 <= layer_idx < n_layers:
            raise ValueError(
                f"layer_idx={layer_idx} out of range for n_layers={n_layers}"
            )

        W_enc_2d, W_dec_2d, b_dec_2d = self._slice_2d_weights(layer_idx)

        arch = self.cfg.architecture()  # type: ignore[attr-defined]
        if arch == "cross_topk":
            target_arch = "topk"
        elif arch in ("cross_jumprelu", "cross_batchtopk", "cross_matryoshka"):
            target_arch = "jumprelu"
        else:
            raise ValueError(
                f"as_single_layer_view: unsupported source architecture {arch!r}"
            )

        # Build a 2-D config from this SAE's cfg, dropping crosscoder-specific
        # fields and updating architecture + metadata.hook_name.
        cfg_dict = self.cfg.to_dict()  # type: ignore[attr-defined]
        for key in (
            "n_layers",
            "layer_masking",
            "use_cross_fact",
            "cross_fact_type",
            "cross_fact_ranks",
        ):
            cfg_dict.pop(key, None)
        cfg_dict["architecture"] = target_arch

        metadata = dict(cfg_dict.get("metadata") or {})
        hook_names = metadata.get("hook_names")
        if hook_names is not None and len(hook_names) > layer_idx:
            metadata["hook_name"] = hook_names[layer_idx]
        cfg_dict["metadata"] = metadata

        from sae_lens.registry import get_sae_class
        sae_cls, cfg_cls = get_sae_class(target_arch)
        cfg = cfg_cls.from_dict(cfg_dict)
        view = sae_cls(cfg)

        # Inject the sliced weights and copy over per-feature parameters.
        view.W_enc.data = W_enc_2d.contiguous().to(view.dtype).to(view.device)
        view.W_dec.data = W_dec_2d.contiguous().to(view.dtype).to(view.device)
        view.b_dec.data = b_dec_2d.contiguous().to(view.dtype).to(view.device)
        if (
            hasattr(view, "b_enc")
            and isinstance(getattr(view, "b_enc", None), nn.Parameter)
            and hasattr(self, "b_enc")
            and isinstance(getattr(self, "b_enc", None), nn.Parameter)
        ):
            view.b_enc.data = self.b_enc.data.clone().to(view.dtype).to(view.device)  # type: ignore[attr-defined]
        if (
            hasattr(view, "threshold")
            and isinstance(getattr(view, "threshold", None), nn.Parameter)
            and hasattr(self, "threshold")
            and isinstance(getattr(self, "threshold", None), nn.Parameter)
        ):
            view.threshold.data = self.threshold.data.clone().to(view.dtype).to(view.device)  # type: ignore[attr-defined]

        return view


# ============================================================================
# Cross JumpReLU SAE (inference target for Cross BatchTopK)
# ============================================================================


@dataclass
class CrossJumpReLUSAEConfig(CrosscoderConfigMixin, JumpReLUSAEConfig):
    """Configuration for crosscoder with JumpReLU activation (inference)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "cross_jumprelu"


class CrossJumpReLUSAE(CrosscoderMixin, JumpReLUSAE):
    """Inference-only crosscoder with JumpReLU activation."""

    cfg: CrossJumpReLUSAEConfig

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_crosscoder_weights()

    @override
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # JumpReLUSAE.encode does a 2-D matmul (sae_in @ W_enc). For the
        # crosscoder we need the multi-layer einsum via the helper.
        hidden_pre = self._crosscoder_encode_hidden_pre(x)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        sae_out_pre = self._crosscoder_decode_pre(feature_acts)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return sae_out_pre


# ============================================================================
# Cross BatchTopK SAE
# ============================================================================


@dataclass
class CrossBatchTopKTrainingSAEConfig(CrosscoderConfigMixin, BatchTopKTrainingSAEConfig):
    """Configuration for crosscoder with BatchTopK activation (training)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "cross_batchtopk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return CrossJumpReLUSAEConfig


class CrossBatchTopKTrainingSAE(CrosscoderMixin, BatchTopKTrainingSAE):
    """Training crosscoder with BatchTopK activation."""

    cfg: CrossBatchTopKTrainingSAEConfig  # type: ignore[assignment]

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_crosscoder_weights()

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict
    ) -> None:
        # Skip TopK/BatchTopK norm folding (assumes 2D W_dec).
        # Use crosscoder 3D-aware version instead.
        self._crosscoder_process_state_dict_for_saving_inference(state_dict)
        # Convert topk_threshold → threshold (same as BatchTopK)
        topk_threshold = state_dict.pop("topk_threshold").item()
        state_dict["threshold"] = torch.ones_like(self.b_enc) * topk_threshold

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_pre = self._crosscoder_encode_hidden_pre(x)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self._crosscoder_decoder_norms()

        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        z = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            z = z * (1 / self._crosscoder_decoder_norms())

        sae_out_pre = self._crosscoder_decode_pre(z)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return sae_out_pre


# ============================================================================
# Cross TopK SAE
# ============================================================================


@dataclass
class CrossTopKSAEConfig(CrosscoderConfigMixin, TopKSAEConfig):
    """Configuration for crosscoder with TopK activation (inference)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "cross_topk"


class CrossTopKSAE(CrosscoderMixin, TopKSAE):
    """Inference-only crosscoder with TopK activation."""

    cfg: CrossTopKSAEConfig

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_crosscoder_weights()

    @override
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden_pre = self._crosscoder_encode_hidden_pre(x)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self._crosscoder_decoder_norms()

        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        z = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            z = z / self._crosscoder_decoder_norms()

        sae_out_pre = self._crosscoder_decode_pre(z)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return sae_out_pre


@dataclass
class CrossTopKTrainingSAEConfig(CrosscoderConfigMixin, TopKTrainingSAEConfig):
    """Configuration for crosscoder with TopK activation (training)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "cross_topk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return CrossTopKSAEConfig


class CrossTopKTrainingSAE(CrosscoderMixin, TopKTrainingSAE):
    """Training crosscoder with TopK activation."""

    cfg: CrossTopKTrainingSAEConfig  # type: ignore[assignment]

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_crosscoder_weights()

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict
    ) -> None:
        self._crosscoder_process_state_dict_for_saving_inference(state_dict)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_pre = self._crosscoder_encode_hidden_pre(x)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self._crosscoder_decoder_norms()

        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        z = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            z = z * (1 / self._crosscoder_decoder_norms())

        sae_out_pre = self._crosscoder_decode_pre(z)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return sae_out_pre


# ============================================================================
# Cross Matryoshka BatchTopK SAE
# ============================================================================


@dataclass
class CrossMatryoshkaBatchTopKTrainingSAEConfig(CrosscoderConfigMixin, MatryoshkaBatchTopKTrainingSAEConfig):
    """Configuration for crosscoder with Matryoshka BatchTopK activation (training)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "cross_matryoshka"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return CrossJumpReLUSAEConfig


class CrossMatryoshkaBatchTopKTrainingSAE(CrosscoderMixin, MatryoshkaBatchTopKTrainingSAE):
    """Training crosscoder with Matryoshka BatchTopK activation.

    The matryoshka inner-level decoding uses slices of W_dec which is now 3D
    [d_sae, n_layers, d_in], so _decode_matryoshka_level is overridden to use
    einsum instead of @ for the inner reconstruction.
    """

    cfg: CrossMatryoshkaBatchTopKTrainingSAEConfig  # type: ignore[assignment]

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_crosscoder_weights()

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict
    ) -> None:
        self._crosscoder_process_state_dict_for_saving_inference(state_dict)
        topk_threshold = state_dict.pop("topk_threshold").item()
        state_dict["threshold"] = torch.ones_like(self.b_enc) * topk_threshold

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_pre = self._crosscoder_encode_hidden_pre(x)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self._crosscoder_decoder_norms()

        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        z = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            z = z * (1 / self._crosscoder_decoder_norms())

        sae_out_pre = self._crosscoder_decode_pre(z)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return sae_out_pre

    @override
    def _decode_matryoshka_level(
        self,
        feature_acts: torch.Tensor,
        width: int,
        inv_W_dec_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a matryoshka inner level with 3D crosscoder weights.

        Standard mode: slice W_dec[:width] (3D) and einsum.
        CP mode: slice A_dec[:width] (B_dec, C_dec stay full).
        """
        inner_feature_acts = feature_acts[..., :width]
        if self.cfg.rescale_acts_by_decoder_norm:
            inner_feature_acts = inner_feature_acts * inv_W_dec_norm[:width]

        if self._is_cp_factored():
            sae_out_pre = torch.einsum(
                "...s, sr, lr, dr -> ...ld",
                inner_feature_acts, self.A_dec[:width], self.B_dec, self.C_dec,
            ) + self.b_dec
        elif self._is_tr_factored():
            # Slice the d_sae axis of G_dec_s; other cores stay full.
            sae_out_pre = torch.einsum(
                "...s, xsy, ylz, zdx -> ...ld",
                inner_feature_acts, self.G_dec_s[:, :width, :], self.G_dec_n, self.G_dec_d,
            ) + self.b_dec
        else:
            sae_out_pre = torch.einsum(
                "...s, sld -> ...ld", inner_feature_acts, self.W_dec[:width]
            ) + self.b_dec
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return sae_out_pre
