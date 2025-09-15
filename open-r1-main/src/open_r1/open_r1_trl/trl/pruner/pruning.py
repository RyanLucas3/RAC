"""
SparseGPT pruning with **budget‑aware grouping**.

We instantiate as many SparseGPT pruners in parallel as will fit inside
`memory_limit_gb` (default = 60).  Groups are processed sequentially:

  ┌─ group 1 – collect stats (hooks live) ─┐
  │  prune L1, free  │  prune L2, free …   │
  └─────────────────────────────────────────┘
  ┌─ group 2 – same pattern … ─────────────┘
"""

from __future__ import annotations
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import PreTrainedTokenizerBase
import transformers

from ..sparsegpt.sparsegpt import SparseGPT
from ..data_utils import maybe_apply_chat_template

# ──────────────────────────────────────────────────────────────────────────────
# helper utils (unchanged from upstream, kept verbatim)
# ──────────────────────────────────────────────────────────────────────────────
def _count_loader_tokens(loader: DataLoader) -> int:
    n = 0
    for batch in loader:
        n += batch["input_ids"].numel()
    return n


def _row_to_prompt(
    row: dict,
    tokenizer: PreTrainedTokenizerBase,
    prompt_column: str = "prompt",
) -> str:
    if isinstance(row.get(prompt_column), str):
        return row[prompt_column]
    if prompt_column in row:
        return maybe_apply_chat_template(row, tokenizer)["prompt"]
    if "input_ids" in row:
        return tokenizer.decode(row["input_ids"], skip_special_tokens=True)
    if "text" in row:
        return row["text"]
    raise KeyError(
        f"Cannot find '{prompt_column}', 'input_ids', or 'text' in row – "
        "please check your dataset or --dataset_prompt_column."
    )


def make_calib_loader(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    tokens: int,
    batch_size: int = 8,
    *,
    prompt_column: str = "prompt",
    weight_col: str | None = None,
) -> DataLoader:
    n_tok, prompts, weights = 0, [], []
    for row in dataset:
        p = _row_to_prompt(row, tokenizer, prompt_column)
        n_tok += len(tokenizer(p).input_ids)
        prompts.append(p)
        w = float(row.get(weight_col, 1.0)) if weight_col else 1.0
        weights.append(w)
        if n_tok >= tokens:
            break

    def _collate(batch_prompts: list[str]):
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        idx = [prompts.index(p) for p in batch_prompts]
        enc["weights"] = torch.tensor([weights[i] for i in idx], dtype=torch.float32)
        return enc

    if weight_col is None:
        return DataLoader(prompts, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(prompts, batch_size=batch_size, sampler=sampler, collate_fn=_collate)

# ──────────────────────────────────────────────────────────────────────────────
#  SparseGPT – budget‑aware grouping
# ──────────────────────────────────────────────────────────────────────────────
def _is_mlp(name: str) -> bool:
    kw = (
        "mlp",
        "ff",
        "feed_forward",
        "ffn",
        "dense_h_to_4h",
        "gate_proj",
        "down_proj",
        "up_proj",
    )
    return any(k in name.lower() for k in kw)


def _estimate_hessian_gb(layer: nn.Module) -> float:
    """Rough fp32 size of the Hessian for *layer* in **gigabytes**."""
    if isinstance(layer, nn.Conv2d):
        cols = layer.weight.data.flatten(1).shape[1]
    elif isinstance(layer, transformers.Conv1D):
        cols = layer.weight.data.t().shape[1]
    else:  # nn.Linear
        cols = layer.weight.data.shape[1]
    bytes_ = cols * cols * 4  # float32
    return bytes_ / (1024**3)  # GiB


def sparsegpt_prune(
    model: nn.Module,
    calib_loader: DataLoader,
    sparsity: float,
    *,
    prunen: int | None = None,
    prunem: int | None = None,
    device: str = "cuda",
    scope: str = "all",
    memory_limit_gb: float = 30.0,
) -> None:
    """
    Prune with SparseGPT, grouping layers so that the sum of their Hessian
    footprints never exceeds memory limit.  Everything stays in float32.
    """
    PRUNE_TYPES = (nn.Linear, nn.Conv2d, transformers.Conv1D)

    layers: list[tuple[str, nn.Module]] = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, PRUNE_TYPES)
        and m.weight.requires_grad
        and (scope == "all" or _is_mlp(n))
    ]
    print(f"[SparseGPT] total layers eligible: {len(layers)}")

    group: list[tuple[str, nn.Module]] = []
    group_mem, total_done = 0.0, 0
    label = f"{prunen}:{prunem}" if prunen and prunem else f"{sparsity*100:.1f}%"

    def _process_group(group_layers: list[tuple[str, nn.Module]], idx0: int) -> None:
        nonlocal total_done
        if not group_layers:
            return

        print(
            f"[SparseGPT] -- processing group {idx0}‑{idx0+len(group_layers)-1} "
            f"(H memory {group_mem:.2f} GB) --"
        )

        # 1) create pruners and hooks
        pruners: dict[str, SparseGPT] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(name: str):
            pr = pruners[name]

            def _hook(mod, inp, out, **kw):
                pr.add_batch(inp[0].detach(), out.detach(), weights=kw.get("weights"))
            return _hook

        for name, lyr in group_layers:
            pruners[name] = SparseGPT(lyr)
            hooks.append(lyr.register_forward_hook(_make_hook(name)))

        # 2) run calibration forward (single pass)
        with torch.inference_mode():
            for batch in calib_loader:
                tgt_dev = (
                    next(iter(model.hf_device_map.values()))
                    if hasattr(model, "hf_device_map")
                    else device
                )
                batch = {k: v.to(tgt_dev) for k, v in batch.items()}
                model(**batch)

        for h in hooks:
            h.remove()

        # 3) prune each layer sequentially inside the group
        for name, _ in group_layers:
            pr = pruners[name]
            pr.fasterprune(
                sparsity,
                prunen=(prunen or 0),
                prunem=(prunem or 0),
            )
            pr.free()
            del pruners[name]
            torch.cuda.empty_cache()
            total_done += 1
            print(f"[SparseGPT] layer {total_done}/{len(layers)} pruned (target {label})")

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # main loop – build groups under memory budget
    # ------------------------------------------------------------------ #
    for idx, (lname, layer) in enumerate(layers):
        mem = _estimate_hessian_gb(layer) * 1.15  # 15 % overhead safety
        if mem > memory_limit_gb:                 # pathological single layer
            print(
                f"[SparseGPT] WARNING: single layer '{lname}' "
                f"requires {mem:.2f} GB > budget {memory_limit_gb} GB; "
                "handling it alone."
            )
            if group:  # flush current group first
                _process_group(group, idx - len(group))
                group, group_mem = [], 0.0
            _process_group([(lname, layer)], idx)
            continue

        if group_mem + mem <= memory_limit_gb:
            group.append((lname, layer))
            group_mem += mem
        else:                                     # flush and start new group
            _process_group(group, idx - len(group))
            group, group_mem = [(lname, layer)], mem

    _process_group(group, len(layers) - len(group))

    if hasattr(model, "fuse"):
        model.fuse()

    realised = compute_sparsity(model)
    print(f"[SparseGPT] realised sparsity: {realised*100:.2f}% (budget {memory_limit_gb} GB)")


def compute_sparsity(model: nn.Module) -> float:
    """Return the fraction (0‑1) of *all* parameters that are exactly zero."""
    total, zeros = 0, 0
    with torch.no_grad():
        for p in model.parameters():
            total += p.numel()
            zeros += (p == 0).sum().item()
    return zeros / total if total > 0 else 0.0

def magnitude_prune_layerwise(
    model: torch.nn.Module,
    sparsity: float,
    device: str = "cuda",
) -> float:
    """
    Unstructured magnitude pruning applied *independently per layer*.
    """
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)"
    model.to(device).eval()

    PRUNE_TYPES = (nn.Linear, nn.Conv2d, transformers.Conv1D)

    with torch.no_grad():
        for module in model.modules():
            if not (isinstance(module, PRUNE_TYPES) and module.weight.requires_grad):
                continue

            w = module.weight.detach()
            k = int(w.numel() * sparsity)
            if k == 0:
                continue

            th = w.abs().flatten().kthvalue(k).values.item()
            w[w.abs() <= th] = 0.0

    torch.cuda.empty_cache()
    realised = compute_sparsity(model)
    print(f"[Mag-Layer] target {sparsity * 100:.1f}% → realised {realised * 100:.2f}%")
    return realised


# ─────────────────────── imports ───────────────────────
from typing import Any
import torch, torch.nn as nn
from transformers.tokenization_utils_base import BatchEncoding

# ───────────────────── helper: locate blocks ───────────
def _get_decoder_layers(model: nn.Module):
    """
    Return the transformer block stack independent of model layout.
    Works for OPT/GPT-J (model.model.decoder.layers) and
    Llama/Qwen-2/Gemma (model.model.layers).
    """
    core = getattr(model, "model", model)
    return getattr(core, "decoder", core).layers

# ───────────────────── helper: seq length ──────────────
def _batch_seq_len(batch: Any) -> int:
    """
    Token length of the first sample in *batch*.

    Accepts:
        • tuple/list            -> (input_ids, …)
        • BatchEncoding / dict  -> {'input_ids': …}
        • torch.Tensor          -> ids
        • tokenizers.Encoding   -> .ids
    """
    if isinstance(batch, (tuple, list)):
        batch = batch[0]

    if hasattr(batch, "keys") and "input_ids" in batch:
        item = batch["input_ids"]
        return item.shape[-1] if torch.is_tensor(item) else len(item[0])

    if torch.is_tensor(batch):
        return batch.shape[-1]

    if hasattr(batch, "ids"):                 # tokenizers.Encoding
        return len(batch.ids)

    raise TypeError(f"Unsupported batch type: {type(batch)}")

# ───────────────────── calibration collector ───────────

def prepare_calibration_input(model, loader, device="cuda"):
    DTYPE       = next(model.parameters()).dtype
    hidden      = model.config.hidden_size
    max_seq     = max(b["input_ids"].shape[1] for b in loader)
    nsamples    = len(loader.dataset)

    inps = torch.empty(nsamples, max_seq, hidden, dtype=DTYPE, device="cpu")
    outs = torch.empty_like(inps)
    cache = {"i": 0}

    layers = _get_decoder_layers(model)
    orig0  = layers[0]
    layers[0] = Catcher(orig0, inps, cache)

    try:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    except ValueError:
        pass

    layers[0] = orig0
    return inps, outs, None, None

# ───────────────────── WANDA pruning ╌ layer-wise ──────
def prune_wanda(model: nn.Module,
                calib_loader,
                sparsity: float,
                device: str = "cuda") -> None:
    """
    Layer-wise unstructured WANDA pruning.

    • Handles recent Qwen-2 checkpoints (≥ transformers-4.54) that require
      `position_embeddings=(cos, sin)` in every decoder-layer call.
    • Avoids Flash-Attention CUDA asserts by always supplying a valid
      attention-mask (all-ones when the original mask is None).
    """
    # ------------------------------------------------------------------ #
    # 0. setup
    # ------------------------------------------------------------------ #
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    DTYPE = next(model.parameters()).dtype

    # ------------------------------------------------------------------ #
    # 1. collect calibration activations
    # ------------------------------------------------------------------ #
    inps, outs, attn_mask, _ = prepare_calibration_input(
        model, calib_loader, device
    )
    inps, outs = inps.to(DTYPE), outs.to(DTYPE)
    nsamp, seq_len, _ = inps.shape
    layers = _get_decoder_layers(model)

    # template tensors shared across calls
    pos_ids_t = torch.arange(seq_len, dtype=torch.long,
                             device=inps.device).unsqueeze(0)

    # model component that actually exposes `rotary_emb`
    base_model = getattr(model, "model", model)

    # cache:   device → all-ones mask  (avoids re-allocation each layer)
    _full_mask_cache: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # 2. iterate over transformer blocks
    # ------------------------------------------------------------------ #
    for i, layer in enumerate(layers):
        subset = find_layers(layer)           # {name: sub-module}

        # (multi-GPU) move buffers to the shard that owns this layer
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, pos_ids = inps.to(dev), outs.to(dev), pos_ids_t.to(dev)
            if attn_mask is not None:
                attn_mask = attn_mask.to(dev)
        else:
            dev, pos_ids = inps.device, pos_ids_t

        # -------- ensure we always pass a legal attention-mask -----------
        if attn_mask is None:
            mask = _full_mask_cache.get(dev)
            if mask is None or mask.shape[1] < seq_len:
                mask = torch.ones(1, seq_len, dtype=torch.bool, device=dev)
                _full_mask_cache[dev] = mask
        else:
            mask = attn_mask
        # ----------------------------------------------------------------

        # -------- rotary position embeddings (Qwen-2) --------------------
        extra_kw = {}
        if hasattr(base_model, "rotary_emb"):
            cache_name = "_wanda_pos_emb"
            cached = getattr(base_model, cache_name, None)
            needs_new = (
                cached is None
                or cached[0].device != dev
                or cached[0].shape[1] < seq_len
            )
            if needs_new:
                dummy = torch.zeros(
                    1, seq_len, base_model.config.hidden_size,
                    dtype=DTYPE, device=dev
                )
                setattr(base_model, cache_name,
                        base_model.rotary_emb(dummy, pos_ids))
            extra_kw["position_embeddings"] = getattr(base_model, cache_name)
        # ----------------------------------------------------------------

        # 2a. attach forward hooks to accumulate row scalers
        wrappers = {n: WrappedGPT(m) for n, m in subset.items()}
        hooks = [
            subset[n].register_forward_hook(
                lambda m, inp, out, name=n:
                    wrappers[name].add_batch(inp[0].data, out.data))
            for n in wrappers
        ]

        # 2b. run each calibration slice through *this* block only
        for j in range(nsamp):
            layer(
                inps[j:j+1],
                attention_mask=mask,
                position_ids=pos_ids,
                **extra_kw
            )[0]

        for h in hooks:
            h.remove()

        # ------------------------------------------------------------------
        # 3. compute WANDA metric & apply mask
        # ------------------------------------------------------------------
        for name, sub in subset.items():
            W = sub.weight.data
            scaler = torch.sqrt(wrappers[name].scaler_row.reshape(1, -1).to(W.device))
            metric = torch.abs(W) * scaler

            # --- robust k computation -------------------------------------------------
            n_col = metric.size(1)
            k = int(round(n_col * sparsity))          # round instead of floor
            if k == 0:                                # nothing to prune → skip layer
                continue
            if k >= n_col:                            # keep at least one column
                k = n_col - 1
            # -------------------------------------------------------------------------

            idx = torch.topk(metric, k, dim=1, largest=False, sorted=False).indices
            W.scatter_(1, idx, 0)

            print(f"[WANDA] pruned layer {i} – {name} ({k}/{n_col} per row)")

        # ------------------------------------------------------------------
        # 4. propagate activations to serve next block
        # ------------------------------------------------------------------
        for j in range(nsamp):
            outs[j] = layer(
                inps[j:j+1],
                attention_mask=mask,
                position_ids=pos_ids,
                **extra_kw
            )[0].detach()
        inps, outs = outs, inps  # swap buffers

    # ------------------------------------------------------------------ #
    # 5. restore config & clean up
    # ------------------------------------------------------------------ #
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()




class Catcher(nn.Module):
    """
    Capture the hidden-state that enters the **first** decoder block during
    calibration.  We copy each sample into a pre-allocated CPU buffer `inps`
    (shape = [N, max_seq, H]), truncating or padding with zeros so every row
    fits exactly `max_seq` tokens.
    """
    def __init__(self,
                 inner: nn.Module,
                 inps_buf: torch.Tensor,
                 cache: dict):
        super().__init__()
        self.inner  = inner          # the real decoder layer
        self.inps   = inps_buf       # (N, max_seq, H) CPU buffer
        self.cache  = cache          # {'i': row_idx, …}

        # propagate attributes expected elsewhere (e.g. attention_type)
        for attr in ("attention_type",):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

    # ------------------------------------------------------------------ #
    # forward: copy -> raise ValueError to abort the full forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x, **kw):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, S, H).  We only need the first sequence in the batch.
        """
        # --- pick first sequence and move to CPU ---
        seq_cpu = x[0].detach().cpu()        # (S, H)
        seq_len, hidden = seq_cpu.shape

        # --- locate destination row in the buffer ---
        dest     = self.inps[self.cache["i"]]   # (max_seq, H)
        max_len  = dest.size(0)                 # buffer's token capacity
        dest.zero_()                            # clear previous contents

        # --- copy (truncate if too long) --------------------------------
        n = min(seq_len, max_len)
        dest[:n].copy_(seq_cpu[:n])

        # --- book-keeping ----------------------------------------------
        self.cache["i"] += 1
        self.cache["attention_mask"] = None     # masks no longer valid
        self.cache["position_ids"]   = None

        # stop the full model forward; we only wanted the hidden state
        raise ValueError

    # delegate all other attributes/methods to the wrapped layer
    def __getattr__(self, name):
        return getattr(self.inner, name)


def find_layers(module: nn.Module,
                layer_types: tuple[type[nn.Module], ...] = (nn.Linear,)) -> dict[str, nn.Module]:
    """
    Return every sub-module in *module* whose type is in *layer_types*.
    The keys are fully-qualified names (as in `named_modules()`);
    the values are the actual sub-module objects.

    Default behaviour: collect **all nn.Linear** layers.
    Extend *layer_types* if you also want nn.Conv1d, nn.Conv2d, etc.
    """
    result: dict[str, nn.Module] = {}
    for name, sub in module.named_modules():
        if isinstance(sub, layer_types):
            result[name] = sub
    return result

class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples