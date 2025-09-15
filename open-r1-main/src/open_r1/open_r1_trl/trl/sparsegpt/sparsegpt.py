"""
Memory‑aware SparseGPT helper (FULL FP32).

Differences vs. the original version
------------------------------------
1. Signature of `add_batch` now accepts an optional **weights** tensor
   so pruning.py can pass per‑sample weights.
2. `fasterprune` frees large intermediates (`L`, `Hinv`) as soon as they
   are no longer needed to keep peak RAM lower, but *everything* stays
   in float32.
3. NEW: prints `nsamples` every `print_every` updates (default = 10 000).
"""

from __future__ import annotations
import math
import time
import torch
import torch.nn as nn
import transformers
from .quant import *

DEBUG = False
PRINT_EVERY = 10_000          # ← change this to customise the logging cadence

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:
    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.dev = layer.weight.device

        W = layer.weight.data
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()

        self.rows, self.columns = W.shape
        self.H = torch.zeros(
            (self.columns, self.columns), device=self.dev, dtype=torch.float32
        )
        self.nsamples = 0
        self._next_log = PRINT_EVERY  # next sample count at which to print

    # ------------------------------------------------------------------ #
    # collect activations -> running Hessian
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def add_batch(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,                      # kept for API compatibility
        *,
        weights: torch.Tensor | None = None,    # optional
    ) -> None:
        if len(inp.shape) == 2:                 # (B,C) -> (1,B,C)
            inp = inp.unsqueeze(0)

        # flatten time/seq dimensions if necessary
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:             # (B,S,C)
                inp = inp.reshape(-1, inp.shape[-1])
            inp = inp.t()                      # (C, N)

        # per‑sample weights √w
        if weights is not None:
            w = weights.reshape(1, -1) if weights.ndim == 1 else weights
            inp = inp * torch.sqrt(w)

        bsz = inp.shape[1]
        self.H *= self.nsamples / (self.nsamples + bsz)
        self.nsamples += bsz
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp @ inp.t()

        # periodic logging
        if self.nsamples >= self._next_log:
            print(f"[SparseGPT] collected {self.nsamples:,} samples")
            while self._next_log <= self.nsamples:
                self._next_log += PRINT_EVERY

        if DEBUG:
            self.inp1 = inp
            self.out1 = out

    # ------------------------------------------------------------------ #
    # pruning
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def fasterprune(
        self,
        sparsity: float,
        *,
        prunen: int = 0,
        prunem: int = 0,
        blocksize: int = 128,
        percdamp: float = 0.01,
        max_percdamp: float = 0.5,
        growth: float = 2.0,
    ) -> None:

        H_orig = self.H.clone()
        W_base = self.layer.weight.data.clone()

        attempt = 0
        while True:
            attempt += 1
            cur_damp = min(percdamp * (growth ** (attempt - 1)), max_percdamp)

            # fresh working copies
            H = H_orig.clone()
            W = W_base.clone()
            if isinstance(self.layer, nn.Conv2d):
                W = W.flatten(1)
            if isinstance(self.layer, transformers.Conv1D):
                W = W.t()
            W = W.float()

            try:
                # ------------------------------------------------------ #
                # prepare H + λI
                # ------------------------------------------------------ #
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                W[:, dead] = 0

                damp = cur_damp * torch.mean(torch.diag(H))
                diag = torch.arange(self.columns, device=self.dev)
                H[diag, diag] += damp

                # Cholesky + inverse
                L = torch.linalg.cholesky(H, upper=False)
                del H; torch.cuda.empty_cache()

                Hinv = torch.cholesky_inverse(L, upper=False)
                del L; torch.cuda.empty_cache()

                Hinv = torch.linalg.cholesky(Hinv, upper=True)   # U such that Uᵀ U = H⁻¹

                if hasattr(self, "quantizer") and not self.quantizer.ready():
                    self.quantizer.find_params(W, weight=True)

                tick = time.time()
                Losses = torch.zeros(self.rows, device=self.dev)
                mask = None

                for i1 in range(0, self.columns, blocksize):
                    i2 = min(i1 + blocksize, self.columns)
                    count = i2 - i1

                    W1 = W[:, i1:i2].clone()
                    Q1 = torch.zeros_like(W1)
                    Err1 = torch.zeros_like(W1)
                    Losses1 = torch.zeros_like(W1)
                    Hinv1 = Hinv[i1:i2, i1:i2]

                    if prunen == 0:
                        if mask is not None:
                            mask1 = mask[:, i1:i2]
                        else:
                            tmp = W1.pow(2) / (torch.diag(Hinv1).reshape(1, -1)).pow(2)
                            thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                            mask1 = tmp <= thresh
                    else:
                        mask1 = torch.zeros_like(W1, dtype=torch.bool)

                    for i in range(count):
                        w = W1[:, i]
                        d = Hinv1[i, i]

                        if prunen and i % prunem == 0:
                            tmp = (
                                W1[:, i : i + prunem].pow(2)
                                / (torch.diag(Hinv1)[i : i + prunem].reshape(1, -1)).pow(2)
                            )
                            mask1.scatter_(
                                1,
                                i + torch.topk(tmp, prunen, dim=1, largest=False)[1],
                                True,
                            )

                        q = w.clone()
                        q[mask1[:, i]] = 0

                        if hasattr(self, "quantizer"):
                            q = quantize(
                                q.unsqueeze(1),
                                self.quantizer.scale,
                                self.quantizer.zero,
                                self.quantizer.maxq,
                            ).flatten()

                        Q1[:, i] = q
                        Losses1[:, i] = (w - q).pow(2) / d.pow(2)

                        err1 = (w - q) / d
                        W1[:, i:] -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                        Err1[:, i] = err1

                    W[:, i1:i2] = Q1
                    Losses += Losses1.sum(1) / 2
                    W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

                torch.cuda.synchronize()
                print(
                    f"[SparseGPT] pass OK at percdamp={cur_damp:.3f} "
                    f"(time {time.time() - tick:.2f}s, err {Losses.sum().item():.4f})"
                )

                if isinstance(self.layer, transformers.Conv1D):
                    W = W.t()
                self.layer.weight.data.copy_(W.reshape_as(self.layer.weight))

                self.H = None
                torch.cuda.empty_cache()
                return

            except (RuntimeError, torch._C._LinAlgError) as err:
                if "cholesky" not in str(err).lower():
                    raise
                print(f"[SparseGPT] Cholesky failed (damp={cur_damp:.3f}); retrying…")

                if cur_damp >= max_percdamp:
                    raise RuntimeError("SparseGPT: all damping attempts failed.") from err

                # ---------- FIX: guard the deletes ----------
                for _v in ("H", "W", "Hinv"):
                    if _v in locals():
                        del locals()[_v]
                torch.cuda.empty_cache()


    @torch.no_grad()
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
