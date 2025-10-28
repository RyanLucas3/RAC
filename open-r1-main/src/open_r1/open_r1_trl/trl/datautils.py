import numpy as np
import torch
import os
import random
from typing import Optional, List, Sequence, Union

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only',
                           'penn_treebank',
                           split='validation')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']),
                         return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4',
        'allenai--c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train')
    valdata = load_dataset(
        'allenai/c4',
        'allenai--c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            #if trainenc.input_ids.shape[1] >= seqlen:
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc



# ----------------------------
# Local on-disk HF dataset (column-aware)
# ----------------------------
def _normalize_cols(text_cols: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if text_cols is None:
        return []
    if isinstance(text_cols, str):
        return [text_cols]
    return list(text_cols)


def _pick_text(example: dict, text_cols: Optional[Union[str, Sequence[str]]] = None) -> Optional[str]:
    """
    Priority:
      1) User-specified `text_cols` (exact matches)
      2) Common single-text fields
      3) Chat-style 'messages' (concatenate contents)
      4) 'prompt' + 'response' concatenation
    """
    # 1) explicit
    for key in _normalize_cols(text_cols):
        if key in example and isinstance(example[key], str) and example[key].strip():
            return example[key]

    # 2) common text fields
    for key in ("text", "content", "input", "inputs", "prompt", "question", "problem"):
        if key in example and isinstance(example[key], str) and example[key].strip():
            return example[key]

    # 3) chat-style
    if "messages" in example and isinstance(example["messages"], list):
        parts = []
        for m in example["messages"]:
            if isinstance(m, dict) and isinstance(m.get("content", None), str):
                parts.append(m["content"])
        if parts:
            return "\n".join(parts)

    # 4) prompt + response pair
    if "prompt" in example and "response" in example:
        p, r = example["prompt"], example["response"]
        if isinstance(p, str) and isinstance(r, str):
            return (p + ("\n" if p and r else "") + r).strip()

    return None


def get_local_hf(nsamples, seed, seqlen, model, data_path: str,
                 train_split_hint: str = "train",
                 val_split_hint: str = "validation",
                 text_cols: Optional[Union[str, Sequence[str]]] = None):
    """
    Load a HuggingFace dataset saved to disk (folder with *.arrow, dataset_info.json, state.json).
    `text_cols`: a column name (str) or list of names to pull text from; first present is used.
    """
    from datasets import load_from_disk, DatasetDict
    from transformers import AutoTokenizer

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Local dataset directory not found: {data_path}")

    ds = load_from_disk(data_path)

    if isinstance(ds, DatasetDict):
        traindata = ds[train_split_hint] if train_split_hint in ds else next(iter(ds.values()))
        valdata   = ds[val_split_hint]  if val_split_hint  in ds else None
    else:
        traindata = ds
        valdata   = None

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        enc = None
        for _try in range(1000):
            i = random.randint(0, len(traindata) - 1)
            row = traindata[i]
            txt = _pick_text(row, text_cols) if isinstance(row, dict) else None
            if not txt:
                continue
            enc = tokenizer(txt, return_tensors='pt')
            if enc.input_ids.shape[1] > seqlen:
                j0 = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
                j1 = j0 + seqlen
                inp = enc.input_ids[:, j0:j1]
                tar = inp.clone()
                tar[:, :-1] = -100
                trainloader.append((inp, tar))
                break
        else:
            # fallback: truncate if we saw at least one enc
            if enc is not None and enc.input_ids.shape[1] >= seqlen:
                inp = enc.input_ids[:, :seqlen]
                tar = inp.clone()
                tar[:, :-1] = -100
                trainloader.append((inp, tar))

    # Build validation tensor of 256 * seqlen tokens
    pool = valdata if valdata is not None else traindata
    target_len = 256 * seqlen
    acc = []
    acc_len = 0
    for k in range(min(len(pool), 5000)):
        row = pool[k]
        txt = _pick_text(row, text_cols) if isinstance(row, dict) else None
        if not txt:
            continue
        tmp = tokenizer(txt, return_tensors='pt').input_ids
        if tmp.shape[1] >= seqlen:
            s0 = 0 if tmp.shape[1] == seqlen else random.randint(0, tmp.shape[1] - seqlen)
            acc.append(tmp[:, s0:s0 + seqlen])
            acc_len += seqlen
            if acc_len >= target_len:
                break

    if acc:
        valenc_ids = torch.hstack(acc)
    else:
        # Worst-case fallback
        first_row = traindata[0] if len(traindata) > 0 else {}
        first_txt = _pick_text(first_row, text_cols) or " "
        valenc_ids = tokenizer(first_txt, return_tensors='pt').input_ids[:, :target_len]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    return trainloader, TokenizerWrapper(valenc_ids)


# ----------------------------
# Unified entry point
# ----------------------------
def get_loaders(name: Optional[str],
                nsamples: int = 128,
                seed: int = 0,
                seqlen: int = 2048,
                model: str = '',
                path: Optional[str] = None,
                text_cols: Optional[Union[str, Sequence[str]]] = None):
    """
    Pick a dataset by *name* or by local *path*. If `name` is None, `path` must be set.
      - name in {"wikitext2", "ptb", "ptb-new", "c4", "c4-new"}
      - path points to a HF 'save_to_disk' directory (with *.arrow + dataset_info.json)
      - text_cols: prefer these columns for text (e.g., "prompt" or ["problem","solution"])
    """
    if name is None and not path:
        raise ValueError("Either `name` or `path` must be provided.")

    if path:
        return get_local_hf(
            nsamples, seed, seqlen, model,
            data_path=path,
            text_cols=text_cols
        )

    # Named datasets (internet)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'ptb-new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'c4-new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)

    raise ValueError(f"Unknown dataset name: {name}")