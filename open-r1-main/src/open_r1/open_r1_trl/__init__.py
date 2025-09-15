# src/open_r1_trl/__init__.py
# ---------------------------------------------------------------------
# Expose the full TRL API *plus* extra helpers used by open_r1 scripts
# ---------------------------------------------------------------------
from importlib import import_module as _imp

# Core TRL namespace (dataclasses, config helpers, etc.)
_trl = _imp("open_r1_trl.trl")

# ────────────────────────────────────────────────────────────────────
# Helpers not re-exported by upstream __init__
# ────────────────────────────────────────────────────────────────────
_GRPOTrainer   = _imp("open_r1_trl.trl.trainer.grpo_trainer").GRPOTrainer
_SFTTrainer    = _imp("open_r1_trl.trl.trainer.sft_trainer").SFTTrainer
_TrlParser     = _imp("open_r1_trl.trl.scripts.utils").TrlParser
_get_peft_cfg  = _imp("open_r1_trl.trl.trainer.utils").get_peft_config

# setup_chat_format moved in some forks; grab it wherever it is
if hasattr(_trl, "setup_chat_format"):
    _setup_chat_fmt = _trl.setup_chat_format
else:
    _setup_chat_fmt = _imp("open_r1_trl.trl.trainer.utils").setup_chat_format

# ────────────────────────────────────────────────────────────────────
# Public API for `import open_r1_trl as xxx`
# ────────────────────────────────────────────────────────────────────
_EXPORTS = [
    # Dataclasses / configs that original __init__ already had
    "ScriptArguments",
    "GRPOConfig",
    "SFTConfig",
    "ModelConfig",
    "get_kbit_device_map",
    "get_quantization_config",
    # Extras we’re adding
    "GRPOTrainer",
    "SFTTrainer",
    "TrlParser",
    "get_peft_config",
    "setup_chat_format",
]

# Pull through anything upstream already provides
globals().update({n: getattr(_trl, n) for n in _EXPORTS if hasattr(_trl, n)})

# Add the symbols we imported manually
globals().update({
    "GRPOTrainer":      _GRPOTrainer,
    "SFTTrainer":       _SFTTrainer,
    "TrlParser":        _TrlParser,
    "get_peft_config":  _get_peft_cfg,
    "setup_chat_format": _setup_chat_fmt,
})

__all__ = _EXPORTS
