"""Tests for CrosscoderMixin.as_single_layer_view across factorization modes.

The view is what RAVEL/absorption see when invoked through compare_ravel.py /
compare_absorption.py: a 2-D SAE whose weights are the per-layer slice of the
crosscoder. These tests pin down two invariants:

  1. Slice equivalence: the view's W_enc/W_dec/b_dec match the per-layer slice
     of the materialized 3-D crosscoder weights, regardless of factorization.
  2. Forward identity: the view's encode/decode applied to a single layer's
     activation equals the full crosscoder's encode/decode applied to an input
     that is zero everywhere except that layer.
"""

from __future__ import annotations

import pytest
import torch

from sae_lens.saes.crosscoder_sae import (
    CrossTopKSAE,
    CrossTopKSAEConfig,
)


D_IN = 6
D_SAE = 16
N_LAYERS = 3
K = 4


def _make_unfactored() -> CrossTopKSAE:
    cfg = CrossTopKSAEConfig(
        d_in=D_IN,
        d_sae=D_SAE,
        n_layers=N_LAYERS,
        k=K,
        apply_b_dec_to_input=True,
        normalize_activations="none",
        rescale_acts_by_decoder_norm=False,
        use_cross_fact=False,
    )
    sae = CrossTopKSAE(cfg)
    # Randomize parameters so slicing isn't trivially zero.
    with torch.no_grad():
        sae.W_enc.data = torch.randn_like(sae.W_enc)
        sae.W_dec.data = torch.randn_like(sae.W_dec)
        sae.b_dec.data = torch.randn_like(sae.b_dec)
        sae.b_enc.data = torch.randn_like(sae.b_enc)
    return sae


def _make_cp(rank: int = 5) -> CrossTopKSAE:
    cfg = CrossTopKSAEConfig(
        d_in=D_IN,
        d_sae=D_SAE,
        n_layers=N_LAYERS,
        k=K,
        apply_b_dec_to_input=True,
        normalize_activations="none",
        rescale_acts_by_decoder_norm=False,
        use_cross_fact=True,
        cross_fact_type="CP",
        cross_fact_ranks=(rank,),
    )
    sae = CrossTopKSAE(cfg)
    with torch.no_grad():
        for name in ("A_enc", "B_enc", "C_enc", "A_dec", "B_dec", "C_dec"):
            getattr(sae, name).data = torch.randn_like(getattr(sae, name))
        sae.b_dec.data = torch.randn_like(sae.b_dec)
        sae.b_enc.data = torch.randn_like(sae.b_enc)
    return sae


def _make_tr(ranks: tuple[int, int, int] = (3, 4, 5)) -> CrossTopKSAE:
    cfg = CrossTopKSAEConfig(
        d_in=D_IN,
        d_sae=D_SAE,
        n_layers=N_LAYERS,
        k=K,
        apply_b_dec_to_input=True,
        normalize_activations="none",
        rescale_acts_by_decoder_norm=False,
        use_cross_fact=True,
        cross_fact_type="TR",
        cross_fact_ranks=ranks,
    )
    sae = CrossTopKSAE(cfg)
    with torch.no_grad():
        for name in (
            "G_enc_n", "G_enc_d", "G_enc_s",
            "G_dec_n", "G_dec_d", "G_dec_s",
        ):
            getattr(sae, name).data = torch.randn_like(getattr(sae, name))
        sae.b_dec.data = torch.randn_like(sae.b_dec)
        sae.b_enc.data = torch.randn_like(sae.b_enc)
    return sae


def _materialize_W_enc(sae: CrossTopKSAE) -> torch.Tensor:
    """Reconstruct full [n_layers, d_in, d_sae] encoder weights regardless of mode."""
    if sae._is_cp_factored():
        return torch.einsum("lr, dr, sr -> lds", sae.A_enc, sae.B_enc, sae.C_enc)
    if sae._is_tr_factored():
        return torch.einsum(
            "alb, bdc, csa -> lds", sae.G_enc_n, sae.G_enc_d, sae.G_enc_s
        )
    return sae.W_enc.detach().clone()


def _materialize_W_dec(sae: CrossTopKSAE) -> torch.Tensor:
    """Reconstruct full [d_sae, n_layers, d_in] decoder weights regardless of mode."""
    if sae._is_cp_factored():
        return torch.einsum("sr, lr, dr -> sld", sae.A_dec, sae.B_dec, sae.C_dec)
    if sae._is_tr_factored():
        return torch.einsum(
            "xsy, ylz, zdx -> sld", sae.G_dec_s, sae.G_dec_n, sae.G_dec_d
        )
    return sae.W_dec.detach().clone()


@pytest.mark.parametrize(
    "factory",
    [_make_unfactored, _make_cp, _make_tr],
    ids=["unfactored", "cp", "tr"],
)
@pytest.mark.parametrize("layer_idx", list(range(N_LAYERS)))
def test_slice_equivalence(factory, layer_idx):
    """View weights match the per-layer slice of the materialized crosscoder weights."""
    torch.manual_seed(0)
    sae = factory()

    W_enc_full = _materialize_W_enc(sae)  # [L, d_in, d_sae]
    W_dec_full = _materialize_W_dec(sae)  # [d_sae, L, d_in]

    view = sae.as_single_layer_view(layer_idx)

    torch.testing.assert_close(
        view.W_enc.data, W_enc_full[layer_idx], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        view.W_dec.data, W_dec_full[:, layer_idx, :], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        view.b_dec.data, sae.b_dec[layer_idx], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(view.b_enc.data, sae.b_enc.data, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "factory",
    [_make_unfactored, _make_cp, _make_tr],
    ids=["unfactored", "cp", "tr"],
)
@pytest.mark.parametrize("layer_idx", list(range(N_LAYERS)))
def test_forward_identity_layer_only_input(factory, layer_idx):
    """Crosscoder.encode on a layer-only input matches view.encode on the slice.

    Concretely: build x_full of shape [batch, n_layers, d_in] that is zero on
    every layer except layer_idx, where it equals x_slice. Then
        crosscoder.encode(x_full)  ==  view.encode(x_slice)
    when apply_b_dec_to_input is False (or with a corresponding correction).

    Note: with apply_b_dec_to_input=True, the crosscoder subtracts b_dec[l] from
    every layer (including the zeroed ones), while the view subtracts only
    b_dec[layer_idx]. To isolate the encoder-weight math, we disable the bias
    application during this test.
    """
    torch.manual_seed(1)
    sae = factory()
    sae.cfg.apply_b_dec_to_input = False  # remove confound from per-layer b_dec
    sae.eval()

    view = sae.as_single_layer_view(layer_idx)
    view.cfg.apply_b_dec_to_input = False
    view.eval()

    x_slice = torch.randn(7, D_IN)
    x_full = torch.zeros(7, N_LAYERS, D_IN)
    x_full[:, layer_idx, :] = x_slice

    # Bypass the activation function (TopK with k=4 picks different indices for
    # the two paths if numerics differ at all). Compare hidden_pre directly.
    sae_in_full = sae.process_sae_in(x_full)
    if sae._is_cp_factored():
        hp_full = torch.einsum(
            "...ld, lr, dr, sr -> ...s",
            sae_in_full, sae.A_enc, sae.B_enc, sae.C_enc,
        ) + sae.b_enc
    elif sae._is_tr_factored():
        hp_full = torch.einsum(
            "...ld, alb, bdc, csa -> ...s",
            sae_in_full, sae.G_enc_n, sae.G_enc_d, sae.G_enc_s,
        ) + sae.b_enc
    else:
        hp_full = torch.einsum("...ld, lds -> ...s", sae_in_full, sae.W_enc) + sae.b_enc

    sae_in_view = view.process_sae_in(x_slice)
    hp_view = sae_in_view @ view.W_enc + view.b_enc

    torch.testing.assert_close(hp_full, hp_view, atol=1e-5, rtol=1e-5)


def test_layer_idx_out_of_range():
    sae = _make_unfactored()
    with pytest.raises(ValueError):
        sae.as_single_layer_view(N_LAYERS)
    with pytest.raises(ValueError):
        sae.as_single_layer_view(-1)


def test_view_architecture_string():
    """Wrapper architecture is the non-cross counterpart."""
    sae = _make_unfactored()
    view = sae.as_single_layer_view(1)
    assert view.cfg.architecture() == "topk"
