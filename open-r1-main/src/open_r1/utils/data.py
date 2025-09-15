from pathlib import Path
from typing import Union
import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
def _safe_load_dataset(
    name_or_path: str,
    config: str | None = None,
    split: str | None = None,
) -> Union[Dataset, DatasetDict]:
    """
    Try `load_dataset`, and if it complains that the dataset was
    saved with `save_to_disk`, fall back to `load_from_disk`.
    """
    try:
        return datasets.load_dataset(name_or_path, config, split=split)
    except ValueError as e:
        # Typical message: “You are trying to load a dataset that was saved using `save_to_disk`…”
        if "save_to_disk" in str(e) or Path(name_or_path).exists():
            ds = datasets.load_from_disk(name_or_path)
            # If the caller requested a split, honour it when possible
            if split is not None and isinstance(ds, DatasetDict):
                ds = ds[split]
            return ds
        raise  # unrelated error – propagate

def _has_all_text(row):
    """
    Return True if:
    • row["prompt"] is a plain‑text string, OR
    • row["prompt"] is a list of chat messages and every message has non‑None 'content'.

    If 'prompt' doesn't exist or is None, skip the check (return True).
    """
    prompt = row.get("prompt")
    if prompt is None:
        return True
    if isinstance(prompt, str):
        return True
    if isinstance(prompt, list):
        return all(isinstance(msg, dict) and msg.get("content") is not None for msg in prompt)
    return True

def get_dataset(args: "ScriptArguments") -> DatasetDict:
    """Load a (mixture of) dataset(s) given CLI/CFG arguments, dropping any
    rows whose messages have missing content."""
    # ─── Single dataset ────────────────────────────────────────────────────
    if args.dataset_name and not args.dataset_mixture:
        ds_config = (
            getattr(args, "dataset_config_name", None)
            or getattr(args, "dataset_config", None)
        )
        ds_split = (
            getattr(args, "dataset_split", None)
            or getattr(args, "dataset_train_split", None)
        )

        logger.info(
            f"Loading dataset: {args.dataset_name} "
            f"(config={ds_config!r}, split={ds_split!r})"
        )

        ds = _safe_load_dataset(args.dataset_name, ds_config, ds_split)
        ds = ds.filter(_has_all_text)           # ←── drop bad rows

        if isinstance(ds, Dataset):
            return DatasetDict({"train": ds})
        return ds  # already a DatasetDict

    # ─── Mixture of datasets ────────────────────────────────────────────────
    elif args.dataset_mixture:
        logger.info(
            f"Creating dataset mixture with "
            f"{len(args.dataset_mixture.datasets)} datasets"
        )
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(
                f"Loading dataset for mixture: {dataset_config.id} "
                f"(config: {dataset_config.config}, split: {dataset_config.split})"
            )

            ds = _safe_load_dataset(
                dataset_config.id,
                dataset_config.config,
                dataset_config.split,
            )
            ds = ds.filter(_has_all_text)       # ←── drop bad rows early

            # Ensure we have a plain Dataset (not Dict) after split selection
            if isinstance(ds, DatasetDict):
                ds = ds["train"]

            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)

            if dataset_config.weight is not None:
                ds = (
                    ds.shuffle(seed=seed)
                    .select(range(int(len(ds) * dataset_config.weight)))
                )
                logger.info(
                    f"Subsampled '{dataset_config.id}' to {len(ds)} rows "
                    f"(weight={dataset_config.weight})"
                )
            datasets_list.append(ds)

        if not datasets_list:
            raise ValueError("No datasets were loaded from the mixture configuration")

        combined = concatenate_datasets(datasets_list).shuffle(seed=seed)
        logger.info(f"Created dataset mixture with {len(combined)} examples")

        if args.dataset_mixture.test_split_size is not None:
            split_ds = combined.train_test_split(
                test_size=args.dataset_mixture.test_split_size, seed=seed
            )
            logger.info(
                "Split dataset into train/test with "
                f"test_size={args.dataset_mixture.test_split_size}"
            )
            return split_ds
        return DatasetDict({"train": combined})

    # ─── Invalid config ────────────────────────────────────────────────────
    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")