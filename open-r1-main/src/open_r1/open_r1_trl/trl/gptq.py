import math
import time

import torch
import torch.nn as nn
import transformers

from method import QuantMethod
from quant import *

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class GPTQ(QuantMethod):
    """
    GPTQ with robust H sanitization, adaptive damping, and multi-path factorization:
      1) Cholesky on GPU (fast path)
      2) CPU eigen (eigh) with eigenvalue clamp
      3) CPU SVD pseudo-inverse (pinv-like) with singular clamp
      4) Identity fallback (no cross-column error propagation)

    Added: per-layer logging of which path ran.
    """

    @torch.no_grad()
    def fasterquant(
        self,
        blocksize: int = 128,
        groupsize: int = -1,
        copy_H: bool = False,
        debug_equiv: bool = False,
        *,
        # adaptive damping
        percdamp: float = 0.01,
        max_percdamp: float = 1.0,
        growth: float = 3.0,
        # sanitation + fallback knobs
        jitter_rel: float = 1e-6,       # absolute jitter relative to mean(diag(H))
        eig_floor_scale: float = 1e-6,  # min eigenvalue = eig_floor_scale * mean(|eig|)
        svd_floor_scale: float = 1e-6,  # min singular value floor
        clip_mult: float = 1e6,         # clamp H entries to ±(clip_mult * mean|H|)
        use_cpu_fallbacks: bool = True,
        # NEW
        verbose: bool = True,           # print which path ran
    ):
        dev = self.layer.weight.device

        # Helper to name the layer nicely
        try:
            lname = getattr(self.layer, "name", None) or self.layer.__class__.__name__
        except Exception:
            lname = "Layer"

        # --- weight view -------------------------------------------------------
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        if not debug_equiv:
            W = W.float()

        rows, cols = W.shape
        full_W = W.clone()  # for error_compute

        tick = time.time()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H_src = self.H.data.clone() if copy_H else self.H

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        # ----------------- helpers -----------------
        def _symmetrize(H: torch.Tensor) -> torch.Tensor:
            return 0.5 * (H + H.T)

        def _sanitize_H_inplace(H: torch.Tensor, W_work: torch.Tensor) -> None:
            """
            Make H finite, reasonably scaled, and diagonally viable.
            Also zero any rows/cols that had non-finite data and set diag to 1.
            """
            # Replace NaN/±Inf with 0 (off-diagonals) before anything else
            torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0, out=H)

            # Clip extreme values relative to global mean magnitude
            mean_abs = H.abs().mean().clamp_min(1e-12)
            H.clamp_(-clip_mult * mean_abs, clip_mult * mean_abs)

            # Symmetrize
            H.copy_(_symmetrize(H))

            # Identify "dead" rows/cols (all zeros after cleaning) or nonpositive diag
            diag = torch.diag(H)
            row_zero = (H.abs().sum(dim=1) == 0)
            nonpos_diag = diag <= 0

            bad = row_zero | nonpos_diag
            if bad.any():
                idx = torch.where(bad)[0]
                # Zero rows/cols and set diag to 1
                H[idx, :] = 0
                H[:, idx] = 0
                H[idx, idx] = 1.0
                # Zero corresponding W columns: GPTQ will quantize them trivially
                W_work[:, idx] = 0

        def _add_damping(H: torch.Tensor, percdamp_val: float) -> float:
            md = torch.mean(torch.diag(H)).clamp_min(1e-12)
            lam = percdamp_val * md + jitter_rel * md
            i = torch.arange(H.size(0), device=H.device)
            H[i, i] += lam
            return float(lam)

        def _chol_build_U(H: torch.Tensor) -> torch.Tensor:
            # Returns U s.t. U^T U = H^{-1}; expects H cleaned/damped (float64)
            L, info = torch.linalg.cholesky_ex(H, upper=False)
            if info > 0:
                raise torch._C._LinAlgError(f"cholesky_ex failed at minor={int(info)}")
            Hinv = torch.cholesky_inverse(L, upper=False)
            # small ridge to keep next chol stable
            md = torch.mean(torch.diag(Hinv)).clamp_min(1e-12)
            i = torch.arange(Hinv.size(0), device=Hinv.device)
            Hinv[i, i] += 1e-12 * md
            U = torch.linalg.cholesky(Hinv, upper=True)
            return U

        def _eigh_build_U_cpu(H: torch.Tensor) -> torch.Tensor:
            # CPU eigen fallback: project to PD and factor H^{-1}
            Hc = H.to("cpu", dtype=torch.float64)
            evals, evecs = torch.linalg.eigh(Hc)  # symmetric
            ref = evals.abs().mean().clamp_min(1e-12)
            evals = torch.clamp(evals, min=eig_floor_scale * ref)
            inv = 1.0 / evals
            # H^{-1} = V diag(inv) V^T
            Hinv = (evecs * inv.unsqueeze(0)) @ evecs.T
            md = torch.mean(torch.diag(Hinv)).clamp_min(1e-12)
            i = torch.arange(Hinv.size(0))
            Hinv[i, i] += 1e-12 * md
            U = torch.linalg.cholesky(Hinv, upper=True)
            return U.to(H.device)

        def _svd_build_U_cpu(H: torch.Tensor) -> torch.Tensor:
            # CPU SVD fallback: pinv-like inverse then chol
            Hc = H.to("cpu", dtype=torch.float64)
            Uu, S, Vh = torch.linalg.svd(Hc, full_matrices=False)
            ref = S.abs().mean().clamp_min(1e-12)
            S = torch.clamp(S, min=svd_floor_scale * ref)
            invS = 1.0 / S
            Hinv = (Vh.T * invS.unsqueeze(0)) @ Uu.T
            Hinv = 0.5 * (Hinv + Hinv.T)  # force symmetry
            md = torch.mean(torch.diag(Hinv)).clamp_min(1e-12)
            i = torch.arange(Hinv.size(0))
            Hinv[i, i] += 1e-12 * md
            U = torch.linalg.cholesky(Hinv, upper=True)
            return U.to(H.device)

        def _identity_U(n: int, device, dtype=torch.float32) -> torch.Tensor:
            # U^T U = I  ->  U = I
            return torch.eye(n, device=device, dtype=dtype)

        # --------------------- adaptive attempts ----------------------
        attempt = 0
        cur_percdamp = percdamp

        # one-line header
        if verbose:
            print(f"[GPTQ] quantizing {lname} (rows={rows}, cols={cols})")

        while True:
            attempt += 1
            H_work = H_src.clone()
            W_work = W.clone()

            # Sanitize & damp
            _sanitize_H_inplace(H_work, W_work)
            H_work = H_work.double()
            lam_val = _add_damping(H_work, cur_percdamp)

            path_used = None
            try:
                # Fast path: GPU Cholesky
                U = _chol_build_U(H_work).float()
                path_used = "chol-gpu"

            except Exception as e1:
                # Try CPU eigh
                try:
                    if not use_cpu_fallbacks:
                        raise
                    U = _eigh_build_U_cpu(H_work).float()
                    path_used = "eigh-cpu"
                except Exception as e2:
                    # Try CPU SVD
                    try:
                        if not use_cpu_fallbacks:
                            raise
                        U = _svd_build_U_cpu(H_work).float()
                        path_used = "svd-cpu"
                    except Exception as e3:
                        # If we've maxed damping, fall back to identity
                        if cur_percdamp >= max_percdamp:
                            n = H_work.size(0)
                            U = _identity_U(n, device=H_work.device).float()
                            path_used = "identity"
                        else:
                            # escalate damping and retry
                            if verbose:
                                print(f"[GPTQ] {lname}: factorization failed on attempt {attempt} "
                                      f"(path=none, percdamp={cur_percdamp:.3g}, λ≈{lam_val:.3e}); retrying…")
                            cur_percdamp = min(cur_percdamp * growth, max_percdamp)
                            continue  # retry outer loop

            # At this point we have a path
            if verbose:
                print(f"[GPTQ] {lname}: attempt {attempt} -> path={path_used}, "
                      f"percdamp={cur_percdamp:.3g}, λ≈{lam_val:.3e}")

            # ------------------------- main GPTQ sweep ----------------------
            Q.zero_()
            Losses.zero_()

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                cnt = i2 - i1

                W1 = W_work[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                U11 = U[i1:i2, i1:i2]

                for i in range(cnt):
                    w = W1[:, i]
                    d = U11[i, i]  # sqrt of H^{-1} diag for this column

                    if groupsize != -1 and ((i1 + i) % groupsize == 0):
                        self.quantizer.find_params(
                            W_work[:, (i1 + i):(i1 + i + groupsize)], weight=True
                        )

                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q).pow(2) / (d * d)

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1) @ U11[i, i:].unsqueeze(0)
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2
                W_work[:, i2:] -= Err1 @ U[i1:i2, i2:]

                if DEBUG:
                    self.layer.weight.data[:, :i2] = Q[:, :i2]
                    self.layer.weight.data[:, i2:] = W_work[:, i2:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))

            # Success → commit and finish
            Q_out = Q.t() if isinstance(self.layer, transformers.Conv1D) else Q
            self.layer.weight.data = Q_out.reshape(self.layer.weight.shape).to(
                self.layer.weight.data.dtype
            )

            torch.cuda.synchronize()
            self.time = time.time() - tick

            if verbose:
                print(f"[GPTQ] {lname}: done in {self.time:.2f}s "
                      f"(total_loss≈{float(torch.sum(Losses)):.4g})")

            if DEBUG:
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

            self.postproc()
            self.error_compute(full_W, self.layer.weight.data)

            if not copy_H:
                del self.H
            return