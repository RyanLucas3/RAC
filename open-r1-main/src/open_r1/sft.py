# Copyright 2025 The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0
#
# Fine-tune any causal-LM on your corpus
# and evaluate *only* on HuggingFaceH4/MATH-500 (test set).

import logging, os, sys, pathlib
from pathlib import Path
from typing import Optional, Any

import torch, datasets, transformers
from transformers import (
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from accelerate import PartialState

# ─── local helpers ──────────────────────────────────────────────────
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from open_r1.configs          import ScriptArguments, SFTConfig
from open_r1.utils            import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks  import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from open_r1_trl              import (
    ModelConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    setup_chat_format,
)

# ════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════
class SaveWeightsEvery50Steps(TrainerCallback):
    def __init__(self, src: str, every: int = 10):
        self.every, self.src_dir = every, src
        self.save_dir = Path(f"{src.rstrip('/')}_retrained")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.every or state.global_step == 0:
            return control
        if not state.is_world_process_zero:
            return control
        kw["model"].save_pretrained(self.save_dir, safe_serialization=True)
        AutoTokenizer.from_pretrained(self.src_dir).save_pretrained(self.save_dir)
        print(f"[checkpoint] -> {self.save_dir}", flush=True)
        return control



def freeze_zero_params(model: torch.nn.Module, atol: float = 0.0):
    for p in model.parameters():
        if not p.requires_grad:
            continue
        nz = p.data.abs() > atol
        if not nz.any():
            p.requires_grad_(False)
        elif not nz.all():
            mask = nz.clone()
            p.register_hook(lambda g, m=mask: g * m.to(g))


# ════════════════════════════════════════════════════════════════════
def main(script_args, training_args, model_args):
# ════════════════════════════════════════════════════════════════════
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())

    # resume checkpoint
    last_ckpt = get_last_checkpoint(training_args.output_dir) \
        if os.path.isdir(training_args.output_dir) else None
    if last_ckpt and not training_args.resume_from_checkpoint:
        logger.info(f"Resuming from {last_ckpt}")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # ─── data / model / tokenizer ───────────────────────────────────
    corpus   = get_dataset(script_args)
    train_ds = (
        corpus[script_args.dataset_train_split]
        if isinstance(corpus, datasets.DatasetDict)
        else corpus
    )
    tok   = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    if tok.chat_template is None:
        model, tok = setup_chat_format(model, tok, format="chatml")

    freeze_zero_params(model)

    cbs = list(get_callbacks(training_args, model_args) or [])
    cbs.append(SaveWeightsEvery50Steps(model_args.model_name_or_path))

    # ─── load Math-500 test split ───────────────────────────────────
    math500_ds = (
        datasets
        .load_dataset("HuggingFaceH4/MATH-500", split="test")
        .rename_column("problem", "text")
    )

    # ─── build trainer (eval = math500 only) ────────────────────────
    trainer = SFTTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = math500_ds,       # ← sole evaluation set
        processing_class = tok,
        peft_config      = get_peft_config(model_args),
        callbacks        = cbs,
    )

    trainer.model.generation_config.eos_token_id = tok.eos_token_id

    # ─── train ──────────────────────────────────────────────────────
    if training_args.do_train:
        ckpt = training_args.resume_from_checkpoint or last_ckpt
        train_out = trainer.train(resume_from_checkpoint=ckpt)
        trainer.log_metrics("train", train_out.metrics)
        trainer.save_metrics("train", train_out.metrics)
        trainer.save_state()

        # save model & card
        trainer.model.generation_config.eos_token_id = tok.eos_token_id
        trainer.save_model(training_args.output_dir)
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(dataset_name=script_args.dataset_name, tags=["open-r1"])
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)

        # ─── evaluate on Math-500 ──────────────────────────────────
        if training_args.do_eval:
            logger.info("*** Evaluate on Math-500 ***")
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(math500_ds)
            trainer.log_metrics("eval_math500", metrics)
            trainer.save_metrics("eval_math500", metrics)

        # optional hub push
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
