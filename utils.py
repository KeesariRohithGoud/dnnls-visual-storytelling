"""
utils.py

Utility functions for CrossModalTemporalAttention project:
- Seeding / reproducibility
- HuggingFace dataset loading & 70/15/15 split
- Tokenization and preprocessing
- Conversion to tf.data.Dataset
"""

import random
import numpy as np
from PIL import Image
import io

import tensorflow as tf
from datasets import load_dataset
from transformers import BertTokenizerFast


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def get_tokenizer():
    """
    Use BERT-base-uncased tokenizer to match:
      - vocab_size = 30522
      - bos_token_id = 101
      - eos_token_id = 102
    as in your config.yaml.
    """
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return tokenizer


# ---------------------------------------------------------------------------
# Example preprocessing
# ---------------------------------------------------------------------------

def preprocess_example_builder(cfg, tokenizer):
    """
    Returns a function that will preprocess a single HuggingFace dataset example
    into model-ready numpy arrays.

    cfg: full config dict loaded from config.yaml
    tokenizer: BERT tokenizer
    """
    dcfg = cfg["dataset"]
    seq_len = dcfg["seq_len"]
    image_size = dcfg["image_size"]
    max_caption_len = dcfg["max_caption_len"]
    max_reason_len = dcfg["max_reason_len"]

    # configured keys (may be missing or different for some HF datasets)
    frames_key_cfg = dcfg.get("frames_key")
    captions_key_cfg = dcfg.get("captions_key")
    reason_key = dcfg.get("reason_key")

    # candidate keys to try when config lacks the expected key
    candidate_frame_keys = ["images", "frames", "frame_paths", "image_list", "image", "frames_list"]
    candidate_caption_keys = ["captions", "caption", "caption_text", "text", "story", "sentences"]

    def _resolve_key(example, cfg_key, candidates):
        # Prefer configured key if present in example
        if cfg_key and cfg_key in example:
            return cfg_key
        for k in candidates:
            if k in example:
                return k
        return None

    def _extract_frames(example):
        """Return a list-like of frames (PIL.Image or raw arrays or paths)."""
        key = _resolve_key(example, frames_key_cfg, candidate_frame_keys)
        if key is None:
            return None, None
        # Use direct indexing because Arrow record objects may not implement .get()
        try:
            val = example[key]
        except Exception:
            return key, None
        # If HF Image feature returns a dict or Image or list
        # Normalize: if single item, wrap in list
        if isinstance(val, list):
            frames = val
        else:
            frames = [val]
        return key, frames

    def _extract_captions(example):
        key = _resolve_key(example, captions_key_cfg, candidate_caption_keys)
        if key is None:
            return None, []
        try:
            val = example[key]
        except Exception:
            return key, []
        # Normalize captions into a list of strings
        if val is None:
            return key, []
        if isinstance(val, list):
            caps = val
        elif isinstance(val, str):
            caps = [val]
        elif isinstance(val, dict):
            # try nested text fields
            for nk in ("text", "caption", "captions"):
                if nk in val:
                    v = val[nk]
                    caps = v if isinstance(v, list) else [v]
                    break
            else:
                caps = [str(val)]
        else:
            caps = [str(val)]
        return key, caps

    def preprocess_example(example):
        # ----------------- Images -----------------
        frames_key_used, frames = _extract_frames(example)
        if frames is None:
            # No frames found for this example — produce a dummy zero frame list
            frames = [np.zeros((image_size, image_size, 3), dtype=np.uint8) for _ in range(seq_len)]

        # Ensure at least seq_len frames by repeating last if needed
        if len(frames) < seq_len:
            frames = frames + [frames[-1]] * (seq_len - len(frames))
        frames = frames[:seq_len]

        proc_frames = []
        for img in frames:
            # HF Image feature may return PIL.Image, numpy array, dict with 'path' or 'bytes', or a path string
            if isinstance(img, Image.Image):
                pil = img
            elif isinstance(img, dict):
                # dataset Image has {'path':..., 'bytes':...} in some cases
                if "path" in img:
                    pil = Image.open(img["path"])
                elif "bytes" in img:
                    pil = Image.open(io.BytesIO(img["bytes"]))
                else:
                    # fallback: stringify and try open
                    pil = Image.open(str(img))
            elif isinstance(img, str):
                pil = Image.open(img)
            else:
                # assume numpy array-like
                try:
                    pil = Image.fromarray(np.asarray(img))
                except Exception:
                    pil = Image.fromarray(img)

            pil = pil.convert("RGB").resize((image_size, image_size))
            arr = np.array(pil, dtype=np.float32) / 255.0  # normalize to [0,1]
            proc_frames.append(arr)
        proc_frames = np.stack(proc_frames, axis=0)  # (S, H, W, 3)

        # ----------------- Captions -----------------
        _, captions = _extract_captions(example)

        # Context captions: first seq_len captions (pad with empty if short)
        ctx_caps = list(captions[:seq_len])
        if len(ctx_caps) < seq_len:
            ctx_caps += [""] * (seq_len - len(ctx_caps))

        # Target caption: caption at index seq_len (k+1), or last if shorter
        if len(captions) > seq_len:
            target_str = captions[seq_len]
        elif len(captions) > 0:
            target_str = captions[-1]
        else:
            target_str = ""

        # Tokenize context captions
        ctx_tok = tokenizer(
            ctx_caps,
            padding="max_length",
            truncation=True,
            max_length=max_caption_len,
            add_special_tokens=True,
        )
        ctx_input_ids = np.array(ctx_tok["input_ids"], dtype=np.int32)  # (S, T)

        # Tokenize target caption
        tgt_tok = tokenizer(
            target_str,
            padding="max_length",
            truncation=True,
            max_length=max_caption_len,
            add_special_tokens=True,
        )
        tgt_input_ids = np.array(tgt_tok["input_ids"], dtype=np.int32)  # (T,)

        # ----------------- Reason text (optional) -----------------
        # reason_key may not exist on every example
        if reason_key and reason_key in example:
            try:
                reason_str = example[reason_key]
            except Exception:
                reason_str = ""
        else:
            reason_str = ""
        if reason_str is None:
            reason_str = ""
        reason_tok = tokenizer(
            reason_str,
            padding="max_length",
            truncation=True,
            max_length=max_reason_len,
            add_special_tokens=True,
        )
        reason_ids = np.array(reason_tok["input_ids"], dtype=np.int32)  # (Tr,)

        return {
            "images": proc_frames,
            "context_captions": ctx_input_ids,
            "target_caption": tgt_input_ids,
            "reason": reason_ids,
        }

    return preprocess_example


# ---------------------------------------------------------------------------
# HF → tf.data.Dataset conversion
# ---------------------------------------------------------------------------

def make_tf_dataset(hf_dataset, cfg, shuffle=False):
    """
    Convert a HuggingFace dataset (already preprocessed with preprocess_example)
    into a tf.data.Dataset suitable for Keras training.

    hf_dataset elements must have keys:
      - "images": (S, H, W, 3)
      - "context_captions": (S, T)
      - "target_caption": (T,)
      - "reason": (Tr,)
    """
    dcfg = cfg["dataset"]
    batch_size = dcfg["batch_size"]
    seq_len = dcfg["seq_len"]
    image_size = dcfg["image_size"]
    max_caption_len = dcfg["max_caption_len"]
    max_reason_len = dcfg["max_reason_len"]

    output_signature = (
        {
            "images": tf.TensorSpec(
                shape=(seq_len, image_size, image_size, 3), dtype=tf.float32
            ),
            "context_captions": tf.TensorSpec(
                shape=(seq_len, max_caption_len), dtype=tf.int32
            ),
            "target_caption": tf.TensorSpec(
                shape=(max_caption_len,), dtype=tf.int32
            ),
            "reason": tf.TensorSpec(
                shape=(max_reason_len,), dtype=tf.int32
            ),
        },
        tf.TensorSpec(shape=(max_caption_len,), dtype=tf.int32),
    )

    def generator():
        for ex in hf_dataset:
            x = {
                "images": np.array(ex["images"], dtype=np.float32),
                "context_captions": np.array(ex["context_captions"], dtype=np.int32),
                "target_caption": np.array(ex["target_caption"], dtype=np.int32),
                "reason": np.array(ex["reason"], dtype=np.int32),
            }
            y = np.array(ex["target_caption"], dtype=np.int32)
            yield x, y

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Dataset loading & 70/15/15 split
# ---------------------------------------------------------------------------

def load_and_prepare_datasets(cfg):
    """
    Load the HuggingFace dataset, split into 70% train / 15% val / 15% test,
    and apply preprocessing.

    Returns:
      train_hf, val_hf, test_hf   (preprocessed HuggingFace datasets)
    """
    dcfg = cfg["dataset"]
    seed = cfg["training"].get("seed", 42)

    hf_name = dcfg["hf_name"]
    print(f"Loading HuggingFace dataset: {hf_name}")
    raw = load_dataset(hf_name)

    # Use the base split from config (typically "train") and re-split
    base_split_name = dcfg.get("train_split", "train")
    base = raw[base_split_name]

    print("Creating 70% train, 15% val, 15% test split from base split:", base_split_name)
    # First: train vs (val+test) = 70 / 30
    train_val = base.train_test_split(test_size=0.3, seed=seed)
    # Then: val vs test = 15 / 15 out of remaining 30
    val_test = train_val["test"].train_test_split(test_size=0.5, seed=seed)

    train_ds = train_val["train"]
    val_ds = val_test["train"]
    test_ds = val_test["test"]

    print("Sizes:")
    print("  Train:", len(train_ds))
    print("  Val:  ", len(val_ds))
    print("  Test: ", len(test_ds))

    # Preprocess
    tokenizer = get_tokenizer()
    preprocess_fn = preprocess_example_builder(cfg, tokenizer)

    print("Preprocessing train set...")
    try:
        train_ds = train_ds.map(
            preprocess_fn,
            remove_columns=train_ds.column_names,
        )
    except KeyError as e:
        print("KeyError during train_ds.map(remove_columns=...):", e)
        print("Falling back to mapping without remove_columns to avoid KeyError.")
        train_ds = train_ds.map(preprocess_fn)

    print("Preprocessing validation set...")
    try:
        val_ds = val_ds.map(
            preprocess_fn,
            remove_columns=val_ds.column_names,
        )
    except KeyError as e:
        print("KeyError during val_ds.map(remove_columns=...):", e)
        print("Falling back to mapping without remove_columns to avoid KeyError.")
        val_ds = val_ds.map(preprocess_fn)

    print("Preprocessing test set...")
    try:
        test_ds = test_ds.map(
            preprocess_fn,
            remove_columns=test_ds.column_names,
        )
    except KeyError as e:
        print("KeyError during test_ds.map(remove_columns=...):", e)
        print("Falling back to mapping without remove_columns to avoid KeyError.")
        test_ds = test_ds.map(preprocess_fn)

    return train_ds, val_ds, test_ds
