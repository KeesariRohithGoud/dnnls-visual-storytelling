"""
train.py

Training script for CrossModalTemporalAttention model with:
- 70% train, 15% validation, 15% test split
- StoryReasoning dataset (daniel3303/StoryReasoning)
"""

import os
import random
import numpy as np
from PIL import Image

import tensorflow as tf
from datasets import load_dataset
from transformers import BertTokenizerFast

from model import build_compiled_model, load_config


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def get_tokenizer():
    # BERT-base-uncased vocabulary to match vocab_size, bos/eos IDs in config
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return tokenizer


def preprocess_example_builder(cfg, tokenizer):
    """
    Returns a function that will preprocess a single HF dataset example
    into model-ready numpy arrays.
    """
    dcfg = cfg["dataset"]
    seq_len = dcfg["seq_len"]
    image_size = dcfg["image_size"]
    max_caption_len = dcfg["max_caption_len"]
    max_reason_len = dcfg["max_reason_len"]

    frames_key = dcfg["frames_key"]
    captions_key = dcfg["captions_key"]
    reason_key = dcfg["reason_key"]

    def preprocess_example(example):
        # ------------- Images -------------
        frames = example[frames_key]
        # Ensure at least seq_len frames by repeating last if needed
        if len(frames) < seq_len:
            frames = frames + [frames[-1]] * (seq_len - len(frames))
        frames = frames[:seq_len]

        proc_frames = []
        for img in frames:
            # HF Image type is usually PIL.Image.Image already
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB").resize((image_size, image_size))
            arr = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]
            proc_frames.append(arr)
        proc_frames = np.stack(proc_frames, axis=0)  # (S, H, W, 3)

        # ------------- Captions -------------
        captions = example[captions_key]
        # Context captions: first seq_len captions (pad with empty if short)
        ctx_caps = list(captions[:seq_len])
        if len(ctx_caps) < seq_len:
            ctx_caps += [""] * (seq_len - len(ctx_caps))

        # Target caption: caption at index seq_len (k+1), or last if shorter
        if len(captions) > seq_len:
            target_str = captions[seq_len]
        else:
            target_str = captions[-1]

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

        # ------------- Reason text (optional) -------------
        reason_str = example.get(reason_key, "")
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
# TF Dataset conversion
# ---------------------------------------------------------------------------

def make_tf_dataset(hf_dataset, cfg, shuffle=False):
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
# Dataset loading & splitting: 70% train, 15% val, 15% test
# ---------------------------------------------------------------------------

def load_and_prepare_datasets(cfg):
    dcfg = cfg["dataset"]
    seed = cfg["training"].get("seed", 42)

    hf_name = dcfg["hf_name"]

    print(f"Loading HuggingFace dataset: {hf_name}")
    raw = load_dataset(hf_name)

    # Use the 'train' split from HF and re-split into 70/15/15
    base_split_name = dcfg.get("train_split", "train")
    base = raw[base_split_name]

    print("Creating 70% train, 15% val, 15% test split from base split:", base_split_name)
    # First: train vs (val+test) = 70 / 30
    train_val = base.train_test_split(test_size=0.3, seed=seed)
    # Then: val vs test = 15 / 15 out of the remaining 30
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
    train_ds = train_ds.map(
        preprocess_fn,
        remove_columns=train_ds.column_names,
    )

    print("Preprocessing validation set...")
    val_ds = val_ds.map(
        preprocess_fn,
        remove_columns=val_ds.column_names,
    )

    print("Preprocessing test set...")
    test_ds = test_ds.map(
        preprocess_fn,
        remove_columns=test_ds.column_names,
    )

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    # 1. Load config
    cfg = load_config("config.yaml")
    set_seed(cfg["training"].get("seed", 42))

    # 2. Load & preprocess datasets with 70/15/15 split
    train_hf, val_hf, test_hf = load_and_prepare_datasets(cfg)

    # 3. Convert to tf.data.Dataset
    train_tf = make_tf_dataset(train_hf, cfg, shuffle=True)
    val_tf = make_tf_dataset(val_hf, cfg, shuffle=False)
    test_tf = make_tf_dataset(test_hf, cfg, shuffle=False)

    # 4. Build model
    print("Building model...")
    model = build_compiled_model(config=cfg)
    model.summary()

    # 5. Callbacks
    tcfg = cfg["training"]
    save_dir = tcfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, "best_model.h5")
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=tcfg.get("save_best_only", True),
        monitor=tcfg.get("monitor_metric", "val_loss"),
        mode="min",
        save_weights_only=False,
        verbose=1,
    )

    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor=tcfg.get("monitor_metric", "val_loss"),
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    # 6. Train
    epochs = tcfg["epochs"]
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_tf,
        validation_data=val_tf,
        epochs=epochs,
        callbacks=[ckpt_cb, early_cb],
    )

    # 7. Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = model.evaluate(test_tf, return_dict=True)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
