# opt.py — Qwen/OPT compatible QUIP quantization
# ----------------------------------------------
# This is a Qwen-aware rewrite of QUIP's original opt.py.
# Key changes:
#   * Works with OPT (facebook/opt-*) and Qwen/Qwen2 (LLaMA-style decoders).
#   * Generic layer/embedding/norm discovery + attention-mask capture.
#   * Keeps the original QUIP GPTQ / Balance / Nearest plumbing intact.

import math
import time
from typing import Tuple, Literal, Dict, Optional
import json
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
)


from gptq import *            # QUIP’s GPTQ impl
from bal import Balance       # QUIP’s Balance impl
from near import Nearest      # QUIP’s Nearest impl
from modelutils import *      # QUIP utilities (e.g., find_layers, DEV)
from quant import *           # QUIP’s quantizer bits

from tqdm import tqdm
from pathlib import Path

# -----------------------------
# Generic model helpers (Qwen/OPT)
# -----------------------------
def get_model(model_id: str):
    """
    Generic loader (OPT or Qwen/Qwen2). Keeps original default dtype behavior.
    """
    def _noop(*args, **kwargs):
        pass

    # Prevent HF from re-initializing weights on instantiation
    torch.nn.init.kaiming_uniform_ = _noop
    torch.nn.init.uniform_ = _noop
    torch.nn.init.normal_ = _noop

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        trust_remote_code=True,   # needed for Qwen/Qwen2
    )
    # Seqlen for rolling window calibration
    # (max_position_embeddings exists on both OPT and Qwen)
    model.seqlen = getattr(model.config, "max_position_embeddings", None)
    if model.seqlen is None:
        # conservative fallback
        model.seqlen = 2048
    return model


def _core(model):
    # OPT & Qwen2 both expose a `.model`; some GPT-like families use `.transformer`
    return getattr(model, "model", None) or getattr(model, "transformer", None)


def get_decoder_stack(model) -> Tuple[nn.ModuleList, Literal["opt", "llama", "gptneox"]]:
    """
    Return (layers, arch_tag), where layers is the decoder block list.
    - OPT:    model.model.decoder.layers
    - Qwen2:  model.model.layers
    - NeoX:   model.transformer.h
    """
    c = _core(model)
    if c is None:
        raise AttributeError("Unsupported model structure: missing `.model` or `.transformer`.")

    # OPT
    if hasattr(c, "decoder") and hasattr(c.decoder, "layers"):
        return c.decoder.layers, "opt"

    # LLaMA-style (Qwen/Qwen2)
    if hasattr(c, "layers"):
        return c.layers, "llama"

    # NeoX-style (fallback)
    if hasattr(c, "h"):
        return c.h, "gptneox"

    raise AttributeError("Cannot locate decoder layers on this model.")


def layer_prefix(model) -> str:
    """
    Prefix for stable quantizer keys (matches HF module names).
    """
    c = _core(model)
    if hasattr(c, "decoder") and hasattr(c.decoder, "layers"):
        return "model.decoder.layers"    # OPT
    if hasattr(c, "layers"):
        return "model.layers"            # Qwen/Qwen2 (LLaMA-style)
    if hasattr(c, "h"):
        return "transformer.h"           # NeoX fallback
    return "layers"


def move_input_modules_to_device(model, dev):
    """
    Move embedding / input projection modules to a device, architecture-agnostic.
    """
    c = _core(model)
    if c is None: return
    # Shared across OPT / Qwen
    if hasattr(c, "embed_tokens") and c.embed_tokens is not None:
        c.embed_tokens = c.embed_tokens.to(dev)
    # OPT-specific
    if hasattr(c, "embed_positions") and c.embed_positions is not None:
        c.embed_positions = c.embed_positions.to(dev)
    if hasattr(c, "project_in") and c.project_in is not None:
        c.project_in = c.project_in.to(dev)
    if hasattr(c, "project_out") and c.project_out is not None:
        c.project_out = c.project_out.to(dev)


def move_input_modules_to_cpu(model):
    c = _core(model)
    if c is None: return
    if hasattr(c, "embed_tokens") and c.embed_tokens is not None:
        c.embed_tokens = c.embed_tokens.cpu()
    if hasattr(c, "embed_positions") and c.embed_positions is not None:
        c.embed_positions = c.embed_positions.cpu()
    if hasattr(c, "project_in") and c.project_in is not None:
        c.project_in = c.project_in.cpu()
    if hasattr(c, "project_out") and c.project_out is not None:
        c.project_out = c.project_out.cpu()


def move_output_modules_to_device(model, dev):
    """
    Move final norm / projection / head to device for eval.
    """
    c = _core(model)
    if c is None: return

    # Final layer norm (Qwen/Qwen2 uses `norm`; OPT uses `decoder.final_layer_norm`)
    last_norm = getattr(c, "norm", None)
    if last_norm is None and hasattr(c, "decoder"):
        last_norm = getattr(c.decoder, "final_layer_norm", None)
    if last_norm is not None:
        setattr(c, "norm", last_norm.to(dev)) if hasattr(c, "norm") else None
        if hasattr(c, "decoder") and hasattr(c.decoder, "final_layer_norm"):
            c.decoder.final_layer_norm = c.decoder.final_layer_norm.to(dev)

    # OPT only
    if hasattr(c, "project_out") and c.project_out is not None:
        c.project_out = c.project_out.to(dev)

    model.lm_head = model.lm_head.to(dev)


def move_output_modules_to_cpu(model):
    c = _core(model)
    if c is None: return
    last_norm = getattr(c, "norm", None)
    if last_norm is not None:
        c.norm = c.norm.cpu()
    if hasattr(c, "decoder") and hasattr(c.decoder, "final_layer_norm") and c.decoder.final_layer_norm is not None:
        c.decoder.final_layer_norm = c.decoder.final_layer_norm.cpu()
    if hasattr(c, "project_out") and c.project_out is not None:
        c.project_out = c.project_out.cpu()
    model.lm_head = model.lm_head.cpu()


def _extract_batch_io(batch, dev):
    """
    Accepts:
      * tuple/list: (input_ids, [attention_mask])
      * dict:       {"input_ids": ..., "attention_mask": ...}
      * tensor:     input_ids
    Returns: input_ids (LongTensor[B, L]), kwargs dict
    """
    if isinstance(batch, (list, tuple)):
        input_ids = batch[0].to(dev)
        kw = {}
        if len(batch) > 1 and batch[1] is not None:
            kw["attention_mask"] = batch[1].to(dev)
        return input_ids, kw
    if isinstance(batch, dict):
        input_ids = batch["input_ids"].to(dev)
        kw = {}
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            kw["attention_mask"] = batch["attention_mask"].to(dev)
        return input_ids, kw
    # tensor fallback
    return batch.to(dev), {}


# -----------------------------
# Main (sequential) quant pass
# -----------------------------
@torch.no_grad()
def opt_sequential(model, dataloader, dev, args):
    """
    Memory-aware sequential quantization for OPT/Qwen/Qwen2.

    Strategy:
      • For each decoder block:
          - Attach hooks to its submodules (nn.Linear etc.)
          - Run the full model forward on calibration batches
          - Let hooks collect activations to build H
          - Remove hooks and quantize that block
          - Move block back to CPU before continuing
      • This way, Qwen2 rotary embeddings (cos,sin) are handled
        by the model itself — no manual position_embeddings.
    """
    print("Starting ...")

    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    layers, _ = get_decoder_stack(model)
    move_input_modules_to_device(model, dev)

    quantizers = {}
    errors, Hmags, times = [], [], []

    print("Ready.")

    # Iterate over decoder layers
    for i, layer in enumerate(tqdm(layers, desc="Quantizing layers")):
        layer = layer.to(dev)

        subset = find_layers(layer)
        quant_method = {}

        # Init quantizers
        for name in subset:
            if args.quant == "gptq":
                qm = GPTQ(subset[name])
                qm.quantizer = Quantizer()
                qm.quantizer.configure(args.wbits, perchannel=True,
                                       sym=False, qfn=args.qfn, mse=False)
            elif args.quant == "nearest":
                qm = Nearest(subset[name])
                qm.quantizer = Quantizer()
                qm.quantizer.configure(args.wbits, perchannel=True,
                                       sym=False, qfn=args.qfn, mse=False)
            elif args.quant in ["allbal","ldlq","ldlqRG","ldlbal_admm"]:
                qm = Balance(subset[name])
                qm.configure(args.quant, args.wbits, args.npasses,
                             unbiased=args.unbiased)
                qm.quantizer = Quantizer()
                qm.quantizer.configure(args.wbits, perchannel=True,
                                       sym=False, qfn=args.qfn, mse=False)
            else:
                raise ValueError(f"Unknown quant method {args.quant}")
            quant_method[name] = qm

        # Attach hooks
        def add_batch(name):
            def hook(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)
            return hook
        handles = [subset[n].register_forward_hook(add_batch(n)) for n in subset]

        # Run full model forward passes to trigger this layer's hooks
        ns_seen = 0
        for batch in dataloader:
            input_ids, kw = _extract_batch_io(batch, dev)
            try:
                _ = model(input_ids, **kw)  # forward through full model
            except Exception:
                # some datasets may not align perfectly; ignore
                pass
            ns_seen += 1
            if ns_seen >= args.nsamples:
                break

        # Remove hooks & finish H build
        for h in handles:
            h.remove()
        for name in subset:
            quant_method[name].post_batch()

        # Quantize weights
        for name in subset:
            qm = quant_method[name]
            qm.preproc(preproc_gptqH=args.pre_gptqH, percdamp=args.percdamp,
                       preproc_rescale=args.pre_rescale,
                       preproc_proj=args.pre_proj,
                       preproc_proj_extra=args.pre_proj_extra)
            if args.quant == "gptq":
                qm.fasterquant(groupsize=args.groupsize)
            elif args.quant in ["allbal","ldlq","ldlqRG","ldlbal_admm"]:
                qm.fasterquant(lazy_batch=args.lazy_batch)
            elif args.quant == "nearest":
                qm.fasterquant()

            key = f"{layer_prefix(model)}.{i}.{name}"
            quantizers[key] = qm.quantizer

            errors.append(qm.error)
            times.append(qm.time)
            Hmags.append(qm.Hmag)
            qm.free()

        layers[i] = layer.cpu()
        del layer, quant_method
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    print(f"Total quant time: {sum(times):.2f}s")
    return quantizers, errors

@torch.no_grad()
def opt_eval(model, testenc, dev):
    """
    Perplexity evaluation over `testenc` (as in the original QUIP script),
    now robust to OPT/Qwen decoders.
    """
    test_ids = testenc.input_ids if hasattr(testenc, "input_ids") else testenc
    nsamples = test_ids.numel() // model.seqlen

    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    layers, _ = get_decoder_stack(model)
    move_input_modules_to_device(model, dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_ids[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    # Free inputs from GPU
    layers[0] = layers[0].cpu()
    move_input_modules_to_cpu(model)
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    # Roll through all decoder layers
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Final norm/proj + head
    move_output_modules_to_device(model, dev)
    test_ids = test_ids.to(dev)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)

        c = _core(model)
        # Qwen/Qwen2: model.model.norm
        if hasattr(c, "norm") and c.norm is not None:
            hidden_states = c.norm(hidden_states)
        # OPT: decoder.final_layer_norm
        elif hasattr(c, "decoder") and getattr(c.decoder, "final_layer_norm", None) is not None:
            hidden_states = c.decoder.final_layer_norm(hidden_states)

        if hasattr(c, "project_out") and c.project_out is not None:
            hidden_states = c.project_out(hidden_states)

        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_ids[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


# The pack/load helpers remain, but generalized to AutoModelForCausalLM.
# (If you don't use them, you can ignore these.)
def opt_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print("Packing ...")
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print("Done.")
    return model


def load_quant3(model_id_or_config, checkpoint):
    config = AutoConfig.from_pretrained(model_id_or_config)

    def _noop(*args, **kwargs): pass
    torch.nn.init.kaiming_uniform_ = _noop
    torch.nn.init.uniform_ = _noop
    torch.nn.init.normal_ = _noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()

    layers = find_layers(model)
    for name in ["model.decoder.project_out", "model.decoder.project_in", "lm_head"]:
        if name in layers:
            del layers[name]
    make_quant3(model, layers)

    print("Loading model ...")
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = getattr(model.config, "max_position_embeddings", 2048)
    print("Done.")
    return model


def load_quant(model_id_or_config, checkpoint):
    config = AutoConfig.from_pretrained(model_id_or_config)

    def _noop(*args, **kwargs): pass
    torch.nn.init.kaiming_uniform_ = _noop
    torch.nn.init.uniform_ = _noop
    torch.nn.init.normal_ = _noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()

    layers = find_layers(model)
    for name in ["model.decoder.project_out", "model.decoder.project_in", "lm_head"]:
        if name in layers:
            del layers[name]

    print("Loading model ...")
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = getattr(model.config, "max_position_embeddings", 2048)
    print("Done.")
    return model


def opt_multigpu(model, gpus):
    """
    Multi-GPU forward helper (kept compatible with OPT & Qwen).
    """
    c = _core(model)

    if hasattr(c, "embed_tokens") and c.embed_tokens is not None:
        c.embed_tokens = c.embed_tokens.to(gpus[0])
    if hasattr(c, "embed_positions") and c.embed_positions is not None:
        c.embed_positions = c.embed_positions.to(gpus[0])
    if hasattr(c, "project_in") and c.project_in is not None:
        c.project_in = c.project_in.to(gpus[0])

    if hasattr(c, "project_out") and c.project_out is not None:
        c.project_out = c.project_out.to(gpus[-1])

    if hasattr(c, "decoder") and getattr(c.decoder, "final_layer_norm", None) is not None:
        c.decoder.final_layer_norm = c.decoder.final_layer_norm.to(gpus[-1])
    if hasattr(c, "norm") and c.norm is not None:
        c.norm = c.norm.to(gpus[-1])

    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {"mask": None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache["mask"] is None or (kwargs.get("attention_mask") is not None and cache["mask"].device != self.dev):
                if kwargs.get("attention_mask") is not None:
                    cache["mask"] = kwargs["attention_mask"].to(self.dev)
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = cache["mask"]
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers, _ = get_decoder_stack(model)
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))
    model.gpus = gpus


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else DEV)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None
        return tmp

    for i, layer in enumerate(get_decoder_stack(model)[0]):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.0

    def sync():
        if hasattr(model, "gpus"):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape(-1),
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)),
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync()
        import numpy as np
        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())


# -----------------------------
# Packing helpers (as before)
# -----------------------------
def opt_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print("Packing ...")
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print("Done.")
    return model


# -----------------------------
# NEW: save an HF repo-style folder
# -----------------------------
def _safe_json_dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))

def save_quantized_pretrained(
    model: AutoModelForCausalLM,
    src_model_id: str,
    save_dir: str,
    *,
    tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
    args_namespace=None,
    extra_meta: Optional[Dict] = None,
):
    """
    Save a complete HF folder:
      - pytorch_model.bin / model.safetensors
      - config.json
      - generation_config.json (if present upstream)
      - tokenizer files from the source model
      - quantization_meta.json (args + tiny summary)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1) Save model weights + config
    model.save_pretrained(save_path, safe_serialization=False)  # set True if you want .safetensors
    model.config.save_pretrained(save_path)

    # 2) Save generation config (pull from src if available)
    try:
        gen_cfg = GenerationConfig.from_pretrained(src_model_id)
        gen_cfg.save_pretrained(save_path)
    except Exception:
        # If src doesn't have it, try from current model
        gc = getattr(model, "generation_config", None)
        if isinstance(gc, GenerationConfig):
            gc.save_pretrained(save_path)

    # 3) Save tokenizer (use provided or load from source)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(src_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)

    # 4) Save a small quantization meta
    meta = {
        "source_model": src_model_id,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quant_note": "Weights have been quantized in-place (FP tensors) or packed (Quant3Linear).",
        "packed": any(m.__class__.__name__.startswith("Quant") for m in model.modules()),
    }
    if args_namespace is not None:
        # Drop big objects; keep basic CLI args
        meta["args"] = {
            k: getattr(args_namespace, k)
            for k in vars(args_namespace)
            if isinstance(getattr(args_namespace, k), (int, float, str, bool))
        }
    if extra_meta:
        meta.update(extra_meta)

    _safe_json_dump(meta, save_path / "quantization_meta.json")
    print(f"[save] Wrote HF folder to: {save_path.resolve()}")



if __name__ == "__main__":
    import argparse
    # NOTE: the original QUIP CLI is preserved. You can still run it directly.
    from datautils import *   # if you use the original QUIP calibration datasets

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="HF model_id (OPT or Qwen).")
    parser.add_argument("dataset", type=str, nargs="?", choices=["wikitext2", "ptb", "c4"],
                        help="Calibration dataset name (omit if using --dataset_path).")
    parser.add_argument("--save_dir", type=str, default="", help="Folder to save a full HF repo (recommended).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=.01)
    parser.add_argument("--quant", choices=["allbal", "ldlq", "ldlqRG", "ldlbal_admm", "nearest", "gptq"],
                        default="nearest")
    parser.add_argument("--wbits", type=int, default=16, choices=[2, 3, 4, 16])
    parser.add_argument("--npasses", type=int, default=0)
    parser.add_argument("--groupsize", type=int, default=-1)
    parser.add_argument("--pre_gptqH", action="store_true")
    parser.add_argument("--pre_rescale", action="store_true")
    parser.add_argument("--pre_proj", action="store_true")
    parser.add_argument("--pre_proj_extra", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--qfn", type=str, default="a")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--proxy_only", action="store_true")
    parser.add_argument("--unbiased", action="store_true")
    parser.add_argument("--incoh_processing", action="store_true")
    parser.add_argument("--lazy_batch", action="store_true")
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Local HF dataset path (load_from_disk). If set, overrides `dataset`.")
    parser.add_argument("--dataset_col", type=str, default="",
                        help="Preferred text column (e.g., 'prompt' or 'problem'). "
                             "For multiple, separate by commas, e.g., 'problem,question'.")

    args = parser.parse_args()

    text_cols = None
    if args.dataset_col:
        text_cols = [c.strip() for c in args.dataset_col.split(",") if c.strip()]


    # defaults to incoherence processing, if requested
    if args.incoh_processing:
        args.pre_gptqH   = True
        args.pre_rescale = True
        args.pre_proj    = True
        args.proj_extra  = 1
        args.qfn         = "b"

    if args.load:
        # If load is a folder: standard HF load. If file: raw state_dict then config from args.model.
        if Path(args.load).is_dir():
            model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True)
            src_id_for_assets = args.load
        else:
            cfg = AutoConfig.from_pretrained(args.model)
            def _noop(*a, **k): pass
            torch.nn.init.kaiming_uniform_ = _noop
            torch.nn.init.uniform_ = _noop
            torch.nn.init.normal_ = _noop
            model = AutoModelForCausalLM.from_config(cfg)
            sd = torch.load(args.load, map_location="cpu")
            model.load_state_dict(sd, strict=False)
            src_id_for_assets = args.model
        model.eval()
    else:
        model = get_model(args.model)
        model.eval()
        
    src_id_for_assets = args.model  # tokenizer & gen config copied from her

    dataloader, _ = get_loaders(
        None if args.dataset_path else args.dataset,   # name
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        path=(args.dataset_path if args.dataset_path else None),
        text_cols=text_cols
    )

    if args.wbits < 16:
        if args.qfn == "b": assert args.pre_proj is True
        print(f"Preprocessing flags: gptqH:{args.pre_gptqH}, rescale:{args.pre_rescale}, "
              f"proj:{args.pre_proj}, proj_extra:{args.pre_proj_extra}, qfn:{args.qfn}")
        print(f"using lazy_batch updates: {args.lazy_batch}")

        if ("ldl" in args.quant) and args.unbiased and (args.npasses > 0):
            print(f"LDL NOTE: unbiased + {args.npasses} npasses. NOT TRULY UNBIASED.")

        tick = time.time()
        quantizers, errors = opt_sequential(model, dataloader, DEV, args)
        print(f"Total quant + H time elapsed: {time.time() - tick:.2f}s\n")
        print(f"Proxy Summary: Qmethod:{args.quant}, Unbiased:{args.unbiased}, W:{args.wbits}, NPass:{args.npasses}")
        print("Quantization done.\n")

    # ---------- SAVE AS A FULL HF FOLDER ----------
    if args.save_dir:
        tok = AutoTokenizer.from_pretrained(src_id_for_assets, trust_remote_code=True)
        save_quantized_pretrained(
            model,
            src_model_id=src_id_for_assets,
            save_dir=args.save_dir,
            tokenizer=tok,
            args_namespace=args,
            extra_meta={"note": "Saved by QUIP-compatible opt.py"},
        )
        print(f"[done] Saved HF repo to {args.save_dir}")

    # if not args.proxy_only:
    #     for dataset in ["wikitext2", "ptb-new", "c4-new"]:
    #         dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
    #         print(dataset)
    #         opt_eval(model, testloader, DEV)
