"""Polynomial Sparse Autoencoder (PolySAE) decoder mixin.

PolySAE extends any SAE by replacing the standard decoder with a polynomial decoder
that combines linear, quadratic, and cubic terms:

    y = b_dec + y₁ + λ₂·y₂ + λ₃·y₃

where:
    y₁ = z @ U⁽¹⁾ @ C⁽¹⁾ᵀ                           (linear term)
    y₂ = ((z @ U⁽¹⁾) * (z @ U⁽²⁾)) @ C⁽²⁾ᵀ          (quadratic term)
    y₃ = ((z @ U⁽¹⁾) * (z @ U⁽²⁾) * (z @ U⁽³⁾)) @ C⁽³⁾ᵀ  (cubic term)

This module provides:
- PolyDecoderMixin: A mixin class that can be combined with any TrainingSAE
- Poly* config and SAE classes for each architecture (TopK, BatchTopK, JumpReLU, Matryoshka)

Reference: ICLR 2026 submission "Polynomial Sparse Autoencoders"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from typing_extensions import override

# Import all base SAE classes at the top to avoid scattered imports
from sae_lens.saes.sae import SAEConfig, TrainingSAE
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
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
)
from sae_lens.saes.matryoshka_batchtopk_sae import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)


# ============================================================================
# Polynomial Decoder Mixin
# ============================================================================


class PolyDecoderMixin:
    """
    Mixin class that adds polynomial decoder functionality to any SAE.

    This mixin:
    1. Adds CP-structured tensors (U¹,²,³ and C¹,²,³)
    2. Adds learnable λ₂, λ₃ coefficients
    3. Overrides decode() to use polynomial reconstruction
    4. Provides orthogonalize_U() for Gram-Schmidt retraction
    5. Overrides fold_W_dec_norm() to prevent base class corruption

    Usage:
        class PolyTopKTrainingSAE(PolyDecoderMixin, TopKTrainingSAE):
            pass
    """

    # These will be initialized by _init_poly_decoder_weights
    U1: nn.Parameter  # [d_sae, R]
    U2: nn.Parameter  # [d_sae, R]
    U3: nn.Parameter  # [d_sae, R]
    C1: nn.Parameter  # [d_in, R]
    C2: nn.Parameter  # [d_in, R]
    C3: nn.Parameter  # [d_in, R]
    lambda2: nn.Parameter  # scalar
    lambda3: nn.Parameter  # scalar

    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        """
        CRITICAL: Override to prevent base class from corrupting PolySAE.
        
        PolySAE uses polynomial decoder (U, C matrices), NOT W_dec for reconstruction.
        The base class fold_W_dec_norm() modifies W_dec and scales W_enc accordingly,
        which would break the encoder-decoder relationship since our decode() uses
        U @ C.T, not W_dec.
        
        This is a NO-OP for PolySAE - the polynomial decoder doesn't need norm folding.
        W_dec is kept only for SAEBench compatibility (linear term approximation).
        """
        # Intentionally do nothing - PolySAE decoder is already correctly scaled
        pass

    def _init_poly_decoder_weights(
        self,
        poly_ranks: tuple[int, int, int],
        poly_order: int = 3,
        lambda2_init: float = -0.5,
        lambda3_init: float = 0.5,
        shared_u: bool = False,
    ) -> None:
        """
        Initialize polynomial decoder weights with potentially different ranks per order.

        Args:
            poly_ranks: Tuple (R1, R2, R3) of ranks for each polynomial order.
                        R1 >= R2 >= R3. Linear uses R1, quadratic uses R2, cubic uses R3.
            poly_order: 1=linear, 2=quadratic, 3=cubic
            lambda2_init: Initial value for λ₂
            lambda3_init: Initial value for λ₃
            shared_u: If True, use single U_shared matrix instead of separate U1/U2/U3
        """
        self._poly_order = poly_order  # Store for use in decode
        self._poly_ranks = poly_ranks  # Store ranks for decode/norm methods
        self._shared_u = shared_u  # Store shared_u mode flag
        d_sae = self.cfg.d_sae  # type: ignore
        d_in = self.cfg.d_in  # type: ignore
        R1, R2, R3 = poly_ranks

        if shared_u:
            # Shared-U mode: single U_shared matrix (d_sae x R1) with C matrices
            U_shared_data = torch.empty(d_sae, R1, dtype=self.dtype, device=self.device)  # type: ignore
            nn.init.orthogonal_(U_shared_data)
            self.U_shared = nn.Parameter(U_shared_data)

            # C1 for linear term (d_in x R1)
            C1_data = torch.empty(d_in, R1, dtype=self.dtype, device=self.device)  # type: ignore
            nn.init.kaiming_uniform_(C1_data)
            self.C1 = nn.Parameter(C1_data)

            # Register placeholder buffers for U1/U2/U3 (for checkpoint compatibility)
            self.register_buffer('U1', torch.zeros(d_sae, R1, dtype=self.dtype, device=self.device))  # type: ignore
            self.register_buffer('U2', torch.zeros(d_sae, R2, dtype=self.dtype, device=self.device))  # type: ignore
            self.register_buffer('U3', torch.zeros(d_sae, R3, dtype=self.dtype, device=self.device))  # type: ignore

            # C2/C3 and lambda2/lambda3 for quadratic/cubic terms
            if poly_order >= 2:
                C2_data = torch.empty(d_in, R2, dtype=self.dtype, device=self.device)  # type: ignore
                nn.init.kaiming_uniform_(C2_data)
                self.C2 = nn.Parameter(C2_data)
                self.lambda2 = nn.Parameter(
                    torch.tensor(lambda2_init, dtype=self.dtype, device=self.device)  # type: ignore
                )
            else:
                self.register_buffer('C2', torch.zeros(d_in, R2, dtype=self.dtype, device=self.device))  # type: ignore
                self.register_buffer('lambda2', torch.tensor(0.0, dtype=self.dtype, device=self.device))  # type: ignore

            if poly_order >= 3:
                C3_data = torch.empty(d_in, R3, dtype=self.dtype, device=self.device)  # type: ignore
                nn.init.kaiming_uniform_(C3_data)
                self.C3 = nn.Parameter(C3_data)
                self.lambda3 = nn.Parameter(
                    torch.tensor(lambda3_init, dtype=self.dtype, device=self.device)  # type: ignore
                )
            else:
                self.register_buffer('C3', torch.zeros(d_in, R3, dtype=self.dtype, device=self.device))  # type: ignore
                self.register_buffer('lambda3', torch.tensor(0.0, dtype=self.dtype, device=self.device))  # type: ignore

        else:
            # Default mode: separate U1/U2/U3 matrices
            # Initialize U1/C1 matrices (linear term, rank R1)
            U1_data = torch.empty(d_sae, R1, dtype=self.dtype, device=self.device)  # type: ignore
            nn.init.orthogonal_(U1_data)
            self.U1 = nn.Parameter(U1_data)

            C1_data = torch.empty(d_in, R1, dtype=self.dtype, device=self.device)  # type: ignore
            nn.init.kaiming_uniform_(C1_data)
            self.C1 = nn.Parameter(C1_data)

            # Initialize quadratic terms only if order >= 2
            if poly_order >= 2:
                # U2/C2 have rank R2
                U2_data = torch.empty(d_sae, R2, dtype=self.dtype, device=self.device)  # type: ignore
                nn.init.orthogonal_(U2_data)
                self.U2 = nn.Parameter(U2_data)
                C2_data = torch.empty(d_in, R2, dtype=self.dtype, device=self.device)  # type: ignore
                nn.init.kaiming_uniform_(C2_data)
                self.C2 = nn.Parameter(C2_data)
                self.lambda2 = nn.Parameter(
                    torch.tensor(lambda2_init, dtype=self.dtype, device=self.device)  # type: ignore
                )
            else:
                # Register placeholders for compatibility (but not as parameters)
                self.register_buffer('U2', torch.zeros(d_sae, R2, dtype=self.dtype, device=self.device))  # type: ignore
                self.register_buffer('C2', torch.zeros(d_in, R2, dtype=self.dtype, device=self.device))  # type: ignore
                self.register_buffer('lambda2', torch.tensor(0.0, dtype=self.dtype, device=self.device))  # type: ignore

            # Initialize cubic terms only if order >= 3
            if poly_order >= 3:
                # U3/C3 have rank R3
                U3_data = torch.empty(d_sae, R3, dtype=self.dtype, device=self.device)  # type: ignore
                nn.init.orthogonal_(U3_data)
                self.U3 = nn.Parameter(U3_data)
                C3_data = torch.empty(d_in, R3, dtype=self.dtype, device=self.device)  # type: ignore
                nn.init.kaiming_uniform_(C3_data)
                self.C3 = nn.Parameter(C3_data)
                self.lambda3 = nn.Parameter(
                    torch.tensor(lambda3_init, dtype=self.dtype, device=self.device)  # type: ignore
                )
            else:
                # Register placeholders for compatibility (but not as parameters)
                self.register_buffer('U3', torch.zeros(d_sae, R3, dtype=self.dtype, device=self.device))  # type: ignore
                self.register_buffer('C3', torch.zeros(d_in, R3, dtype=self.dtype, device=self.device))  # type: ignore
                self.register_buffer('lambda3', torch.tensor(0.0, dtype=self.dtype, device=self.device))  # type: ignore

    def _poly_decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Polynomial decoder: y = b_dec + y₁ + λ₂·y₂ + λ₃·y₃
        
        Default mode (shared_u=False, multi-rank R1 >= R2 >= R3):
        - y₁ = (z @ U1) @ C1.T                           [full R1]
        - y₂ = ((z @ U1[:,:R2]) * (z @ U2)) @ C2.T       [truncated to R2]
        - y₃ = ((z @ U1[:,:R3]) * (z @ U2[:,:R3]) * (z @ U3)) @ C3.T  [truncated to R3]
        
        Shared-U mode (shared_u=True):
        - a1 = z @ U_shared                              [batch x R1]
        - y₁ = a1 @ C1.T
        - y₂ = (a1[:, :R2] ⊙ a1[:, :R2]) @ C2.T          [element-wise square]
        - y₃ = (a1[:, :R3] ⊙ a1[:, :R3] ⊙ a1[:, :R3]) @ C3.T  [element-wise cube]

        Args:
            feature_acts: Sparse feature activations z [batch, seq, d_sae]

        Returns:
            Reconstructed activations [batch, seq, d_in]
        """
        z = feature_acts
        poly_order = getattr(self, '_poly_order', 3)
        poly_ranks = getattr(self, '_poly_ranks', (self.U1.shape[1], self.U2.shape[1], self.U3.shape[1]))
        R1, R2, R3 = poly_ranks
        shared_u = getattr(self, '_shared_u', False)

        # Convert sparse to dense if needed for polynomial operations
        if z.is_sparse:
            z = z.to_dense()

        if shared_u:
            # Shared-U mode: single projection, element-wise powers for higher orders
            a1 = z @ self.U_shared  # [..., R1]
            
            # Linear term: y₁ = a1 @ C1.T
            y1 = a1 @ self.C1.T  # [..., d_in]
            result = self.b_dec + y1  # type: ignore
            
            # Quadratic term: y₂ = (a1[:,:R2]²) @ C2.T
            if poly_order >= 2:
                a1_r2 = a1[..., :R2] if R2 < R1 else a1  # [..., R2]
                y2 = (a1_r2 * a1_r2) @ self.C2.T  # [..., d_in]
                result = result + self.lambda2 * y2
            
            # Cubic term: y₃ = (a1[:,:R3]³) @ C3.T
            if poly_order >= 3:
                a1_r3 = a1[..., :R3] if R3 < R1 else a1  # [..., R3]
                y3 = (a1_r3 * a1_r3 * a1_r3) @ self.C3.T  # [..., d_in]
                result = result + self.lambda3 * y3
        else:
            # Default mode: separate U1/U2/U3 projections
            # Compute full projection for linear term (rank R1)
            zU1_full = z @ self.U1  # [..., R1] - works for both 2D and 3D input

            # Linear term: y₁ = z @ U1 @ C1.T
            y1 = zU1_full @ self.C1.T  # [..., d_in]
            
            # Start with linear term
            result = self.b_dec + y1  # type: ignore

            # Add quadratic term if order >= 2
            if poly_order >= 2:
                # Use first R2 columns of U1's projection
                # Use [..., :R2] slicing which works for both 2D and 3D tensors
                zU1_r2 = zU1_full[..., :R2] if R2 < R1 else zU1_full  # [..., R2]
                zU2 = z @ self.U2  # [..., R2]
                y2 = (zU1_r2 * zU2) @ self.C2.T  # [..., d_in]
                result = result + self.lambda2 * y2

            # Add cubic term if order >= 3
            if poly_order >= 3:
                # Use first R3 columns of U1 and U2's projections
                zU1_r3 = zU1_full[..., :R3] if R3 < R1 else zU1_full  # [..., R3]
                zU2_r3 = (z @ self.U2[:, :R3]) if R3 < R2 else (z @ self.U2)  # [..., R3]
                zU3 = z @ self.U3  # [..., R3]
                y3 = (zU1_r3 * zU2_r3 * zU3) @ self.C3.T  # [..., d_in]
                result = result + self.lambda3 * y3

        return result

    def compute_poly_decoder_norms(self) -> torch.Tensor:
        """
        Compute full polynomial decoder norm for each feature.
        
        Default mode (R1 >= R2 >= R3):
        - Linear: D1 = U1 @ C1.T
        - Quadratic: D2 = (U1[:,:R2] * U2) @ C2.T  
        - Cubic: D3 = (U1[:,:R3] * U2[:,:R3] * U3) @ C3.T
        
        Shared-U mode:
        - Linear: D1 = U_shared @ C1.T
        - Quadratic: D2 = (U_shared[:,:R2]²) @ C2.T
        - Cubic: D3 = (U_shared[:,:R3]³) @ C3.T
        
        Returns:
            decoder_norms: [d_sae] tensor of decoder direction norms
        """
        poly_order = getattr(self, '_poly_order', 3)
        poly_ranks = getattr(self, '_poly_ranks', (self.U1.shape[1], self.U2.shape[1], self.U3.shape[1]))
        R1, R2, R3 = poly_ranks
        shared_u = getattr(self, '_shared_u', False)
        
        if shared_u:
            # Shared-U mode: use U_shared with element-wise powers
            # Linear term: [d_sae, d_in]
            D = self.U_shared @ self.C1.T
            
            # Add quadratic term if order >= 2
            if poly_order >= 2:
                U_r2 = self.U_shared[:, :R2] if R2 < R1 else self.U_shared
                D2 = (U_r2 * U_r2) @ self.C2.T
                D = D + self.lambda2 * D2
            
            # Add cubic term if order >= 3
            if poly_order >= 3:
                U_r3 = self.U_shared[:, :R3] if R3 < R1 else self.U_shared
                D3 = (U_r3 * U_r3 * U_r3) @ self.C3.T
                D = D + self.lambda3 * D3
        else:
            # Default mode: separate U matrices
            # Linear term: [d_sae, d_in]
            D = self.U1 @ self.C1.T
            
            # Add quadratic term if order >= 2
            if poly_order >= 2:
                # Use first R2 columns of U1
                U1_r2 = self.U1[:, :R2] if R2 < R1 else self.U1
                D2 = (U1_r2 * self.U2) @ self.C2.T
                D = D + self.lambda2 * D2
            
            # Add cubic term if order >= 3
            if poly_order >= 3:
                # Use first R3 columns of U1 and U2
                U1_r3 = self.U1[:, :R3] if R3 < R1 else self.U1
                U2_r3 = self.U2[:, :R3] if R3 < R2 else self.U2
                D3 = (U1_r3 * U2_r3 * self.U3) @ self.C3.T
                D = D + self.lambda3 * D3
        
        # Norm per feature
        return D.norm(dim=-1)  # [d_sae]

    def _qr_pos(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Reduced QR decomposition with positive diagonal on R for continuity.
        
        This ensures sign consistency across optimization steps.
        """
        Q, R = torch.linalg.qr(mat, mode='reduced')
        s = torch.sign(torch.diag(R))
        s[s == 0] = 1
        return Q * s  # Broadcasting s over rows of Q (columns of Q are scaled)

    def _robust_qr_update(
        self, 
        target_block: torch.Tensor, 
        original_block: torch.Tensor,
        eps_norm: float = 1e-6,
        eps_jitter: float = 1e-4,
        jitter_std: float = 1e-6
    ) -> torch.Tensor:
        """
        Robustly compute orthonormal basis for target_block.
        
        - Checks frobenius norm; if too small (< eps_norm), returns original_block (skips update).
        - If small but nonzero (< eps_jitter), adds small jitter for stability.
        - Applies _qr_pos to ensure sign continuity.
        """
        fro_norm = torch.norm(target_block, p='fro')
        
        # If the block has collapsed to zero (e.g. perfect projection), 
        # do not introduce noise or NaNs. Keep the previous valid weights.
        if fro_norm < eps_norm:
            return original_block
            
        # If norm is very small but not zero, add jitter to stabilize QR
        if fro_norm < eps_jitter:
            target_block = target_block + jitter_std * torch.randn_like(target_block)
            
        return self._qr_pos(target_block)

    @torch.no_grad()
    def orthogonalize_U(self) -> None:
        """
        Apply Gram-Schmidt orthogonality retraction to U matrices with robust handling.

        Shared-U mode (shared_u=True):
        - Only orthonormalize U_shared columns (no cross-matrix constraints)
        
        Default mode with multi-rank support (R1 >= R2 >= R3):
        - If all ranks are equal: Full cross-term orthogonalization
        - If ranks differ: Shared-prefix orthogonalization
          * The first R3 columns of U1, U2, U3 are mutually orthogonalized
          * Remaining columns are orthonormalized independently within each matrix

        Robustness:
        - Checks norm of projected blocks before QR to avoid instability.
        - Skips update if block is rank-deficient/zero.
        """
        shared_u = getattr(self, '_shared_u', False)
        
        # Shared-U mode: only orthonormalize U_shared
        if shared_u:
            Q = self._robust_qr_update(self.U_shared.data, self.U_shared.data)
            self.U_shared.data.copy_(Q)
            return
        
        # Default mode: separate U matrices
        poly_order = getattr(self, '_poly_order', 3)
        poly_ranks = getattr(self, '_poly_ranks', (self.U1.shape[1], self.U2.shape[1], self.U3.shape[1]))
        R1, R2, R3 = poly_ranks
        
        # NOTE: R1 >= R2 >= R3 constraint removed to allow any rank configuration
        assert self.U1.shape[1] == R1
        assert (poly_order < 2) or (self.U2.shape[1] == R2)
        assert (poly_order < 3) or (self.U3.shape[1] == R3)
        
        # Check if all ranks are equal (can do full cross-term orthogonalization)
        all_equal = (R1 == R2 == R3)
        
        if all_equal:
            # Original algorithm: full cross-term orthogonalization
            Q1 = self._robust_qr_update(self.U1.data, self.U1.data)
            self.U1.data.copy_(Q1)

            if poly_order >= 2:
                U2_t = self.U2.data - Q1 @ (Q1.T @ self.U2.data)
                Q2 = self._robust_qr_update(U2_t, self.U2.data)
                self.U2.data.copy_(Q2)

                if poly_order >= 3:
                    U3_t = self.U3.data - Q1 @ (Q1.T @ self.U3.data) - Q2 @ (Q2.T @ self.U3.data)
                    Q3 = self._robust_qr_update(U3_t, self.U3.data)
                    self.U3.data.copy_(Q3)
        else:
            # Shared-prefix orthogonalization for different ranks
            # Step 1: Orthonormalize U1[:,:R3] (the shared prefix)
            Q1_shared = self._robust_qr_update(self.U1.data[:, :R3], self.U1.data[:, :R3])
            self.U1.data[:, :R3].copy_(Q1_shared)
            
            if poly_order >= 2:
                # Step 2: Orthonormalize U2[:,:R3] and make it orthogonal to U1[:,:R3]
                U2_shared_proj = self.U2.data[:, :R3] - Q1_shared @ (Q1_shared.T @ self.U2.data[:, :R3])
                Q2_shared = self._robust_qr_update(U2_shared_proj, self.U2.data[:, :R3])
                self.U2.data[:, :R3].copy_(Q2_shared)
                
                # Orthonormalize U2[:,R3:] independently (if R2 > R3)
                if R2 > R3:
                    # Make U2[:,R3:] orthogonal to U2[:,:R3] (ensure self-orthogonality of U2)
                    U2_rest_proj = self.U2.data[:, R3:] - Q2_shared @ (Q2_shared.T @ self.U2.data[:, R3:])
                    Q2_rest = self._robust_qr_update(U2_rest_proj, self.U2.data[:, R3:])
                    self.U2.data[:, R3:].copy_(Q2_rest)
                
                if poly_order >= 3:
                    # Step 3: Orthonormalize U3 and make it orthogonal to both shared prefixes
                    U3_t = self.U3.data - Q1_shared @ (Q1_shared.T @ self.U3.data) - Q2_shared @ (Q2_shared.T @ self.U3.data)
                    Q3 = self._robust_qr_update(U3_t, self.U3.data)
                    self.U3.data.copy_(Q3)
            
            # Step 4: Orthonormalize remaining parts of U1
            # We must ensure U1 is fully self-orthogonal
            
            # Construct the current basis for U1 projection
            U1_basis = Q1_shared
            
            # Handle middle part (R3 to R2)
            mid = min(R2, R1)
            if mid > R3:
                U1_mid_proj = self.U1.data[:, R3:mid] - U1_basis @ (U1_basis.T @ self.U1.data[:, R3:mid])
                Q1_mid = self._robust_qr_update(U1_mid_proj, self.U1.data[:, R3:mid])
                self.U1.data[:, R3:mid].copy_(Q1_mid)
                # Update basis to include this part
                U1_basis = torch.cat([U1_basis, Q1_mid], dim=1)
            
            # Handle rest (R2 to R1)
            if R1 > R2:
                # Note: if R2 < R3 (not possible by assumption R1>=R2>=R3), logic still holds as ranges would be empty
                start_col = max(mid, R3) # Should be R2 if R2>R3
                if R1 > start_col:
                     U1_rest_proj = self.U1.data[:, start_col:] - U1_basis @ (U1_basis.T @ self.U1.data[:, start_col:])
                     Q1_rest = self._robust_qr_update(U1_rest_proj, self.U1.data[:, start_col:])
                     self.U1.data[:, start_col:].copy_(Q1_rest)

    @torch.no_grad()
    def get_decoder_directions(self) -> torch.Tensor:
        """Return decoder directions for the polynomial decoder (LINEAR TERM ONLY).
        
        The full polynomial decoder is: y = b_dec + y1 + λ2·y2 + λ3·y3
        where y1 is linear, y2 is quadratic, y3 is cubic.
        
        This method returns the LINEAR TERM (U @ C1.T) normalized to unit row norms,
        where U is U_shared (shared_u mode) or U1 (default mode).
        
        This is a meaningful approximation for SAEBench metrics that analyze decoder
        geometry (cosine similarity, absorption, meta-structure).
        
        Note: Metrics using these directions analyze linear-mode geometry only,
        not quadratic/cubic interaction geometry.
        
        Returns:
            Tensor of shape (n_features, d_model) = (d_sae, d_in) with unit-norm rows.
            Each row is the normalized linear decoder direction for a single feature.
            
        Contract:
            - Shape is always (d_sae, d_in)
            - Each row has L2 norm == 1.0 (within numerical precision)
            - Represents the linear term of the polynomial decoder
        """
        shared_u = getattr(self, '_shared_u', False)
        
        # Use appropriate U matrix based on mode
        U = self.U_shared if shared_u else self.U1
        
        # U is (d_sae, rank), C1 is (d_in, rank)
        # U @ C1.T gives (d_sae, d_in) - each row is a feature's linear decoder direction
        poly_dec = U @ self.C1.T
        directions = poly_dec / poly_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Shape assertion
        assert directions.shape == (self.cfg.d_sae, self.cfg.d_in), (
            f"get_decoder_directions() must return (d_sae, d_in) = ({self.cfg.d_sae}, {self.cfg.d_in}), "
            f"got {directions.shape}"
        )
        return directions

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float):
        """
        Fold activation normalization scaling into PolySAE weights.
        
        Math derivation:
        ---------------
        Before folding (with scaling s):
          sae_in = x*s - b_dec
          hidden_pre = sae_in @ W_enc + b_enc
          [if rescale: hidden_pre *= D, then feature_acts /= D in decode - these cancel]
          sae_out_scaled = _poly_decode(feature_acts)
          sae_out = sae_out_scaled / s
          
        After folding (without scaling):
          We want same sae_out for raw input x.
          
        Solution:
          W_enc_new = W_enc * s        → makes hidden_pre same as before
          C_new = C / s                → makes poly_decode output 1/s smaller
          b_dec_new = b_dec / s        → makes bias term 1/s smaller
          
        The decoder norm D = (U @ C.T).norm() scales by 1/s, but since
        we multiply and divide by D in encode/decode, this cancels out
        and doesn't affect TopK selection or final output.
        """
        s = scaling_factor
        shared_u = getattr(self, '_shared_u', False)
        
        # Scale encoder weights to compensate for missing input scaling
        self.W_enc.data *= s  # type: ignore
        
        # Scale decoder bias and C matrices by 1/s
        self.b_dec.data /= s  # type: ignore
        self.C1.data /= s
        self.C2.data /= s
        self.C3.data /= s
        
        # Sync W_dec with polynomial decoder (linear term) for SAEBench compatibility
        # SAEBench evaluations directly access W_dec for decoder analysis
        # Also normalize W_dec rows to unit norm (SAEBench requires this)
        U = self.U_shared if shared_u else self.U1
        poly_dec = U @ self.C1.T  # [d_sae, d_in]
        W_dec_norms = poly_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = poly_dec / W_dec_norms  # type: ignore
        
        # Update config to indicate scaling has been folded
        self.cfg.normalize_activations = "none"  # type: ignore


# ============================================================================
# Poly Config Mixin
# ============================================================================


@dataclass
class PolyConfigMixin:
    """Mixin that adds polynomial decoder configuration to any SAE config.
    
    Supports two modes for specifying ranks:
    1. Single rank: poly_rank sets the same rank for all terms
    2. Multi-rank: poly_ranks=(R1, R2, R3) sets separate ranks for each order
       where R1 >= R2 >= R3 (linear term has highest rank, cubic has lowest)
    
    Shared-U mode (shared_u=True):
    - Uses single U_shared matrix (d_sae x R1) for all polynomial orders
    - Quadratic/cubic terms use truncated columns: U_shared[:, :R2] / U_shared[:, :R3]
    - Only U_shared needs orthonormalization (no cross-matrix constraints)
    """

    poly_rank: int | None = None  # Single rank for all terms (defaults to d_in if None)
    poly_ranks: tuple[int, int, int] | None = None  # Separate ranks (R1, R2, R3)
    poly_order: int = 3  # 1=linear only, 2=linear+quadratic, 3=full polynomial
    shared_u: bool = False  # Use single shared U matrix for all polynomial orders
    lambda2_init: float = -0.5
    lambda3_init: float = 0.5

    def get_poly_ranks(self) -> tuple[int, int, int]:
        """Get the effective (R1, R2, R3) ranks for each polynomial order.
        
        Returns a tuple (R1, R2, R3) where R1 >= R2 >= R3.
        - R1: rank for linear term (largest)
        - R2: rank for quadratic term
        - R3: rank for cubic term (smallest)
        """
        if self.poly_ranks is not None:
            # Return ranks as-is (constraint R1 >= R2 >= R3 removed)
            return self.poly_ranks
        
        # Fall back to single rank for all terms
        r = self.poly_rank if self.poly_rank is not None else self.d_in  # type: ignore
        return (r, r, r)

    def get_poly_rank(self) -> int:
        """Get the effective poly_rank (for backward compatibility).
        
        Returns R1 (the largest rank) when using multi-rank mode.
        """
        return self.get_poly_ranks()[0]


# ============================================================================
# Poly TopK SAE
# ============================================================================


@dataclass
class PolyTopKSAEConfig(PolyConfigMixin, TopKSAEConfig):
    """Configuration for PolySAE with TopK activation (inference)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "poly_topk"


@dataclass
class PolyTopKTrainingSAEConfig(PolyConfigMixin, TopKTrainingSAEConfig):
    """Configuration for PolySAE with TopK activation (training)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "poly_topk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return PolyTopKSAEConfig


class PolyTopKSAE(PolyDecoderMixin, TopKSAE):
    """Inference-only PolySAE with TopK activation."""

    cfg: PolyTopKSAEConfig

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_poly_decoder_weights(
            self.cfg.get_poly_ranks(),
            self.cfg.poly_order,
            shared_u=self.cfg.shared_u,
        )

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        # Handle rescaling for TopK
        z = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            z = z / self.W_dec.norm(dim=-1)

        sae_out_pre = self._poly_decode(z)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)


class PolyTopKTrainingSAE(PolyDecoderMixin, TopKTrainingSAE):
    """Training PolySAE with TopK activation."""

    cfg: PolyTopKTrainingSAEConfig

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_poly_decoder_weights(
            self.cfg.get_poly_ranks(),
            self.cfg.poly_order,
            self.cfg.lambda2_init,
            self.cfg.lambda3_init,
            shared_u=self.cfg.shared_u,
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode with PolySAE-specific handling of rescale_acts_by_decoder_norm.
        
        Uses polynomial decoder norm (U1 @ C1.T) instead of W_dec.norm() for rescaling.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            # Use FULL polynomial decoder norm (linear + quadratic + cubic terms)
            decoder_norm = self.compute_poly_decoder_norms()
            hidden_pre = hidden_pre * decoder_norm

        # Apply the TopK activation function
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        z = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            # Use FULL polynomial decoder norm (linear + quadratic + cubic terms)
            decoder_norm = self.compute_poly_decoder_norms()
            z = z / decoder_norm

        # Convert sparse to dense if needed
        if z.is_sparse:
            z = z.to_dense()

        sae_out_pre = self._poly_decode(z)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)


# ============================================================================
# Poly BatchTopK SAE
# ============================================================================


@dataclass
class PolyBatchTopKTrainingSAEConfig(PolyConfigMixin, BatchTopKTrainingSAEConfig):
    """Configuration for PolySAE with BatchTopK activation (training)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "poly_batchtopk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        # BatchTopK saves as JumpReLU for inference, so we need PolyJumpReLU
        return PolyJumpReLUSAEConfig


class PolyBatchTopKTrainingSAE(PolyDecoderMixin, BatchTopKTrainingSAE):
    """Training PolySAE with BatchTopK activation."""

    cfg: PolyBatchTopKTrainingSAEConfig

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_poly_decoder_weights(
            self.cfg.get_poly_ranks(),
            self.cfg.poly_order,
            self.cfg.lambda2_init,
            self.cfg.lambda3_init,
            shared_u=self.cfg.shared_u,
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode with full polynomial decoder norm for rescaling."""
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            decoder_norm = self.compute_poly_decoder_norms()
            hidden_pre = hidden_pre * decoder_norm

        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        z = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            # Use FULL polynomial decoder norm (linear + quadratic + cubic terms)
            decoder_norm = self.compute_poly_decoder_norms()
            z = z / decoder_norm

        if z.is_sparse:
            z = z.to_dense()

        sae_out_pre = self._poly_decode(z)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        """
        Override to use polynomial decoder norms instead of W_dec norms.
        
        The parent class (TopKTrainingSAE) uses _fold_norm_topk() which scales
        W_enc by W_dec norms. But for PolySAE, W_dec is a placeholder - the real
        decoder is the polynomial U⊗C. Using W_dec norms would incorrectly scale
        W_enc (making it ~10x smaller), causing L0=0 at inference.
        
        This override:
        1. Skips the parent's _fold_norm_topk() by calling grandparent's method
        2. Uses polynomial decoder norms (from U1 @ C1.T) for correct scaling
        3. Converts topk_threshold to threshold (same as BatchTopK)
        """
        # Skip TopKTrainingSAE's process_state_dict_for_saving_inference
        # (which calls _fold_norm_topk with wrong W_dec norms)
        # Instead, call TrainingSAE's method directly
        TrainingSAE.process_state_dict_for_saving_inference(self, state_dict)
        
        # If rescale_acts_by_decoder_norm is True, we need to fold norms using
        # polynomial decoder norms, not W_dec norms
        if self.cfg.rescale_acts_by_decoder_norm:
            # Compute polynomial decoder norms from linear term
            shared_u = getattr(self, '_shared_u', False)
            U = self.U_shared if shared_u else self.U1
            poly_dec = U @ self.C1.T  # [d_sae, d_in]
            poly_dec_norm = poly_dec.norm(dim=-1).clamp(min=1e-8)
            
            # Scale W_enc and b_enc by polynomial norms
            state_dict["b_enc"] = state_dict["b_enc"] * poly_dec_norm
            state_dict["W_enc"] = state_dict["W_enc"] * poly_dec_norm.unsqueeze(0)
            
            # Normalize polynomial decoder (divide U rows by norms)
            poly_dec_norms = poly_dec_norm.unsqueeze(1)  # [d_sae, 1]
            if shared_u:
                state_dict["U_shared"] = state_dict["U_shared"] / poly_dec_norms
            else:
                state_dict["U1"] = state_dict["U1"] / poly_dec_norms
            
            # Update W_dec to match normalized linear term (for compatibility)
            U_normalized = state_dict["U_shared"] if shared_u else state_dict["U1"]
            state_dict["W_dec"] = U_normalized @ state_dict["C1"].T
        
        # Convert topk_threshold to threshold (same as BatchTopK parent)
        # If we rescaled W_enc/b_enc, we also need to rescale threshold
        topk_threshold = state_dict.pop("topk_threshold").item()
        if self.cfg.rescale_acts_by_decoder_norm:
            # Threshold needs to be scaled by same factor as hidden_pre
            # hidden_pre_new = hidden_pre_old * poly_dec_norm
            # So threshold_new = threshold_old * poly_dec_norm
            state_dict["threshold"] = torch.ones_like(self.b_enc) * topk_threshold * poly_dec_norm
        else:
            state_dict["threshold"] = torch.ones_like(self.b_enc) * topk_threshold


# ============================================================================
# Poly JumpReLU SAE
# ============================================================================


@dataclass
class PolyJumpReLUSAEConfig(PolyConfigMixin, JumpReLUSAEConfig):
    """Configuration for PolySAE with JumpReLU activation (inference)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "poly_jumprelu"


@dataclass
class PolyJumpReLUTrainingSAEConfig(PolyConfigMixin, JumpReLUTrainingSAEConfig):
    """Configuration for PolySAE with JumpReLU activation (training)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "poly_jumprelu"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return PolyJumpReLUSAEConfig


class PolyJumpReLUSAE(PolyDecoderMixin, JumpReLUSAE):
    """Inference-only PolySAE with JumpReLU activation.
    
    This combines:
    - JumpReLU encoder activation (threshold-based sparsity)
    - Polynomial decoder (U⊗C factorization with λ₂y₂ + λ₃y₃ terms)
    
    SAEBench uses get_decoder_directions() (from PolyDecoderMixin) which returns
    the normalized linear term (U1 @ C1.T). Actual decoding uses _poly_decode().
    """

    cfg: PolyJumpReLUSAEConfig

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_poly_decoder_weights(
            self.cfg.get_poly_ranks(),
            self.cfg.poly_order,
            shared_u=self.cfg.shared_u,
        )

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode using polynomial decoder."""
        sae_out_pre = self._poly_decode(feature_acts)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)


class PolyJumpReLUTrainingSAE(PolyDecoderMixin, JumpReLUTrainingSAE):
    """Training PolySAE with JumpReLU activation."""

    cfg: PolyJumpReLUTrainingSAEConfig

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_poly_decoder_weights(
            self.cfg.get_poly_ranks(),
            self.cfg.poly_order,
            self.cfg.lambda2_init,
            self.cfg.lambda3_init,
            shared_u=self.cfg.shared_u,
        )

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        sae_out_pre = self._poly_decode(feature_acts)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)


# ============================================================================
# Poly Matryoshka BatchTopK SAE
# ============================================================================


@dataclass
class PolyMatryoshkaBatchTopKTrainingSAEConfig(PolyConfigMixin, MatryoshkaBatchTopKTrainingSAEConfig):
    """Configuration for PolySAE with Matryoshka BatchTopK activation (training)."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "poly_matryoshka"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        # Matryoshka BatchTopK saves as JumpReLU for inference, so we need PolyJumpReLU
        return PolyJumpReLUSAEConfig


class PolyMatryoshkaBatchTopKTrainingSAE(PolyDecoderMixin, MatryoshkaBatchTopKTrainingSAE):
    """Training PolySAE with Matryoshka BatchTopK activation."""

    cfg: PolyMatryoshkaBatchTopKTrainingSAEConfig

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self._init_poly_decoder_weights(
            self.cfg.get_poly_ranks(),
            self.cfg.poly_order,
            self.cfg.lambda2_init,
            self.cfg.lambda3_init,
            shared_u=self.cfg.shared_u,
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode with full polynomial decoder norm for rescaling."""
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            decoder_norm = self.compute_poly_decoder_norms()
            hidden_pre = hidden_pre * decoder_norm

        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        z = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            # Use FULL polynomial decoder norm (linear + quadratic + cubic terms)
            decoder_norm = self.compute_poly_decoder_norms()
            z = z / decoder_norm

        if z.is_sparse:
            z = z.to_dense()

        sae_out_pre = self._poly_decode(z)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# For backward compatibility, alias the original names to TopK variants
PolySAE = PolyTopKSAE
PolySAEConfig = PolyTopKSAEConfig
PolyTrainingSAE = PolyTopKTrainingSAE
PolyTrainingSAEConfig = PolyTopKTrainingSAEConfig
