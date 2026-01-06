"""
model.py

CrossModalTemporalAttention
TensorFlow/Keras implementation for the StoryReasoning dataset
(image sequences + captions) with cross-modal temporal attention.

- Loads hyperparameters from config.yaml
- Uses GRU for temporal encoding
- Applies a Cross-Modal Temporal Attention layer after GRU
"""

import os
import yaml
import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml"):
    """Load YAML configuration."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg





# ---------------------------------------------------------------------------
# Image Encoder
#   Input:   (B, S, H, W, 3)
#   Output:  global_feats:  (B, S, D_img)
#            spatial_feats: (B, S, P, C_spatial)
# ---------------------------------------------------------------------------

class ImageEncoder(layers.Layer):
    def __init__(self, image_feat_dim=512, image_spatial_dim=512, **kwargs):
        super().__init__(**kwargs)
        # Simple CNN backbone
        self.conv1 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.pool1 = layers.MaxPool2D(2)
        self.conv2 = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.pool2 = layers.MaxPool2D(2)
        self.conv3 = layers.Conv2D(image_spatial_dim, 3, padding="same", activation="relu")

        self.global_pool = layers.GlobalAveragePooling2D()
        self.proj_global = layers.Dense(image_feat_dim)

        self.image_spatial_dim = image_spatial_dim
        self.image_feat_dim = image_feat_dim

    def call(self, x, training=False):
        """
        x: (B, S, H, W, 3)
        """
        B = tf.shape(x)[0]
        S = tf.shape(x)[1]
        H = tf.shape(x)[2]
        W = tf.shape(x)[3]
        C = tf.shape(x)[4]

        # merge batch and sequence
        x = tf.reshape(x, (B * S, H, W, C))

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        feat_map = self.conv3(x)  # (B*S, h', w', image_spatial_dim)

        # spatial features: flatten spatial dims -> patches
        spatial_h = tf.shape(feat_map)[1]
        spatial_w = tf.shape(feat_map)[2]
        C_spatial = tf.shape(feat_map)[3]
        num_patches = spatial_h * spatial_w

        spatial_feats = tf.reshape(feat_map, (B, S, num_patches, C_spatial))

        # global features: GAP over spatial then project
        global_feats = self.global_pool(feat_map)          # (B*S, C_spatial)
        global_feats = self.proj_global(global_feats)      # (B*S, image_feat_dim)
        global_feats = tf.reshape(global_feats, (B, S, self.image_feat_dim))

        return global_feats, spatial_feats


# ---------------------------------------------------------------------------
# Text Encoder (context captions)
#   Input:  (B, S, T)
#   Output: (B, S, D_text)
# ---------------------------------------------------------------------------

class TextEncoder(layers.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_token_id=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True
        )
        self.bilstm = layers.Bidirectional(
            layers.LSTM(hidden_dim // 2, return_sequences=False)
        )
        # store hidden_dim so reshapes can use a statically-known size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id

    def build(self, input_shape):
        """
        Ensure embedding and BiLSTM sublayers are built with concrete inner dims
        so that later reshapes produce a statically-known last dimension.

        input_shape: (B, S, T)
        """
        # expected temporal length
        if len(input_shape) >= 3:
            T = input_shape[2]
        else:
            T = None

        # Build the embedding (creates embedding weights)
        # Embedding expects input shape (batch, T)
        self.embedding.build((None, T))

        # Build the BiLSTM with shape (batch, T, embed_dim)
        emb_dim = self.embedding.output_dim
        self.bilstm.build((None, T, emb_dim))

        super().build(input_shape)

    def call(self, x, training=False):
        """
        x: (B, S, T)
        """
        B = tf.shape(x)[0]
        S = tf.shape(x)[1]
        T = tf.shape(x)[2]

        # merge batch and sequence dims
        x = tf.reshape(x, (B * S, T))

        emb = self.embedding(x)  # (B*S, T, E)
        text_feat = self.bilstm(emb)  # (B*S, H_text)

        # reshape to (B, S, H_text) using the known hidden_dim so the
        # resulting symbolic tensor has a statically-known last dimension
        text_feat = tf.reshape(text_feat, (B, S, self.hidden_dim))  # (B, S, H_text)
        return text_feat


# ---------------------------------------------------------------------------
# Reason Encoder (optional)
#   Input:  (B, Tr)
#   Output: (B, D_reason)
# ---------------------------------------------------------------------------

class ReasonEncoder(layers.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_token_id=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True
        )
        self.bilstm = layers.Bidirectional(
            layers.LSTM(hidden_dim // 2, return_sequences=False)
        )
        self.pad_token_id = pad_token_id

    def build(self, input_shape):
        """
        Build embedding and BiLSTM for the reason encoder.

        input_shape: (B, Tr)
        """
        if len(input_shape) >= 2:
            Tr = input_shape[1]
        else:
            Tr = None

        self.embedding.build((None, Tr))
        emb_dim = self.embedding.output_dim
        self.bilstm.build((None, Tr, emb_dim))

        super().build(input_shape)

    def call(self, x, training=False):
        emb = self.embedding(x)  # (B, Tr, E)
        reason_feat = self.bilstm(emb)  # (B, H_reason)
        return reason_feat


# ---------------------------------------------------------------------------
# Cross-Modal Fusion via Attention (spatial image + text [+ reason])
#   image_spatial: (B, S, P, C_img)
#   text_feat:     (B, S, D_text)
#   reason_feat:   (B, D_reason) or None
#   Output:        (B, S, D_multi)
# ---------------------------------------------------------------------------

class CrossModalFusion(layers.Layer):
    def __init__(
        self,
        multimodal_dim=512,
        attn_dim=512,
        dropout=0.1,
        use_reason_in_fusion=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_reason_in_fusion = use_reason_in_fusion

        self.proj_img = layers.Dense(attn_dim)
        self.proj_txt = layers.Dense(attn_dim)
        self.attn = layers.Attention()
        self.dropout = layers.Dropout(dropout)

        self.proj_out = layers.Dense(multimodal_dim, activation="tanh")

    def call(self, image_spatial, text_feat, reason_feat=None, training=False):
        """
        image_spatial: (B, S, P, C_img)
        text_feat:     (B, S, D_text)
        reason_feat:   (B, D_reason) or None
        """
        B = tf.shape(image_spatial)[0]
        S = tf.shape(image_spatial)[1]
        P = tf.shape(image_spatial)[2]

        img_proj = self.proj_img(image_spatial)  # (B, S, P, A)
        txt_proj = self.proj_txt(text_feat)      # (B, S, A)

        A = tf.shape(img_proj)[-1]
        img_proj = tf.reshape(img_proj, (B * S, P, A))   # (B*S, P, A)
        txt_proj = tf.reshape(txt_proj, (B * S, 1, A))   # (B*S, 1, A)

        attended = self.attn([txt_proj, img_proj])  # (B*S, 1, A)
        attended = tf.reshape(attended, (B, S, A))  # (B, S, A)

        fused = tf.concat([attended, text_feat], axis=-1)  # (B, S, A + D_text)

        if self.use_reason_in_fusion and (reason_feat is not None):
            reason_expanded = tf.expand_dims(reason_feat, axis=1)   # (B, 1, D_reason)
            reason_expanded = tf.repeat(reason_expanded, repeats=S, axis=1)
            fused = tf.concat([fused, reason_expanded], axis=-1)

        fused = self.dropout(fused, training=training)
        fused = self.proj_out(fused)  # (B, S, D_multi)

        return fused


# ---------------------------------------------------------------------------
# Temporal Encoder (GRU over sequence)
#   Input:  (B, S, D_multi)
#   Output: (B, S, D_temp)   [return_sequences=True]
# ---------------------------------------------------------------------------

class TemporalEncoder(layers.Layer):
    def __init__(self, hidden_dim=512, **kwargs):
        super().__init__(**kwargs)
        # GRU instead of LSTM, with return_sequences=True
        self.gru = layers.GRU(hidden_dim, return_sequences=True)

    def call(self, x, training=False):
        """
        x: (B, S, D_multi)
        returns: (B, S, hidden_dim)
        """
        return self.gru(x)


# ---------------------------------------------------------------------------
# Cross-Modal Temporal Attention (AFTER GRU)
#   Input:  h_seq (B, S, D_temp)
#   Output: context (B, D_temp)
#   (Attention over time steps of cross-modal sequence)
# ---------------------------------------------------------------------------

class CrossModalTemporalAttentionLayer(layers.Layer):
    def __init__(self, attn_dim=512, **kwargs):
        super().__init__(**kwargs)
        # standard additive attention over time
        self.W_h = layers.Dense(attn_dim, activation="tanh")
        self.v = layers.Dense(1)

    def call(self, h_seq, training=False):
        """
        h_seq: (B, S, D_temp)
        returns:
          context: (B, D_temp)
        """
        # u_t = tanh(W_h h_t)
        u = self.W_h(h_seq)          # (B, S, attn_dim)
        scores = self.v(u)           # (B, S, 1)
        scores = tf.squeeze(scores, axis=-1)  # (B, S)

        # attention weights over time
        alpha = tf.nn.softmax(scores, axis=1)  # (B, S)
        alpha_expanded = tf.expand_dims(alpha, axis=-1)  # (B, S, 1)

        # context = sum_t alpha_t * h_t
        context = tf.reduce_sum(alpha_expanded * h_seq, axis=1)  # (B, D_temp)
        return context


# ---------------------------------------------------------------------------
# Caption Decoder
#   Input: tokens (B, T_out)
#          context (B, D_temp)
#   Output: logits (B, T_out, vocab_size)
# ---------------------------------------------------------------------------

class CaptionDecoder(layers.Layer):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True
        )
        self.lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
        self.out_dense = layers.Dense(vocab_size)
        self.pad_token_id = pad_token_id

        self.init_h = layers.Dense(hidden_dim, activation="tanh")
        self.init_c = layers.Dense(hidden_dim, activation="tanh")

    def call(self, target_tokens, context, training=False):
        emb = self.embedding(target_tokens)  # (B, T_out, E)

        h0 = self.init_h(context)  # (B, H)
        c0 = self.init_c(context)  # (B, H)

        outputs, _, _ = self.lstm(emb, initial_state=[h0, c0])
        logits = self.out_dense(outputs)  # (B, T_out, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Full CrossModalTemporalAttention Model
# ---------------------------------------------------------------------------

class CrossModalTemporalAttention(Model):
    """
    Full model:

    Inputs (dict):
      - "images":           (B, S, H, W, 3)
      - "context_captions": (B, S, T_ctx)
      - "target_caption":   (B, T_tgt)
      - (optional) "reason":(B, T_reason)
    """
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        if config is None:
            # default to loading from config.yaml
            config = load_config("config.yaml")
        self.cfg = config

        mcfg = config["model"]
        vocab_size = mcfg["vocab_size"]

        # encoders
        self.image_encoder = ImageEncoder(
            image_feat_dim=mcfg["image_feat_dim"],
            image_spatial_dim=mcfg["image_spatial_dim"],
            name="image_encoder"
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=mcfg["text_embed_dim"],
            hidden_dim=mcfg["text_hidden_dim"],
            pad_token_id=mcfg["pad_token_id"],
            name="text_encoder"
        )

        self.use_reason = mcfg.get("use_reason_in_fusion", False)
        if self.use_reason:
            self.reason_encoder = ReasonEncoder(
                vocab_size=vocab_size,
                embed_dim=mcfg["reason_embed_dim"],
                hidden_dim=mcfg["reason_hidden_dim"],
                pad_token_id=mcfg["pad_token_id"],
                name="reason_encoder"
            )
        else:
            self.reason_encoder = None

        self.cross_modal_fusion = CrossModalFusion(
            multimodal_dim=mcfg["multimodal_dim"],
            attn_dim=mcfg["cross_modal_attn_dim"],
            dropout=mcfg["cross_modal_dropout"],
            use_reason_in_fusion=self.use_reason,
            name="cross_modal_fusion"
        )

        # GRU over time
        self.temporal_encoder = TemporalEncoder(
            hidden_dim=mcfg["temporal_hidden_dim"],
            name="temporal_gru_encoder"
        )

        # Cross-modal temporal attention AFTER GRU
        self.temporal_attn = CrossModalTemporalAttentionLayer(
            attn_dim=mcfg["cross_modal_attn_dim"],
            name="cross_modal_temporal_attention"
        )

        # Caption decoder
        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embed_dim=mcfg["text_decoder_embed_dim"],
            hidden_dim=mcfg["text_decoder_hidden"],
            pad_token_id=mcfg["pad_token_id"],
            name="caption_decoder"
        )

    def call(self, inputs, training=False):
        """
        inputs: dict with keys:
          - "images":           (B, S, H, W, 3)
          - "context_captions": (B, S, T_ctx)
          - "target_caption":   (B, T_tgt)
          - (optional) "reason":(B, T_reason)
        """
        images = inputs["images"]
        context_captions = inputs["context_captions"]
        target_caption = inputs["target_caption"]
        reason = inputs.get("reason", None)

        # 1) Encode images
        img_global, img_spatial = self.image_encoder(images, training=training)

        # 2) Encode context captions
        txt_feat = self.text_encoder(context_captions, training=training)

        # 3) Encode reason (optional)
        reason_feat = None
        if self.use_reason and self.reason_encoder is not None and reason is not None:
            reason_feat = self.reason_encoder(reason, training=training)

        # 4) Cross-modal fusion (spatial image + text [+ reason])
        fused_seq = self.cross_modal_fusion(
            image_spatial=img_spatial,
            text_feat=txt_feat,
            reason_feat=reason_feat,
            training=training
        )  # (B, S, D_multi)

        # 5) Temporal GRU over fused sequence
        temporal_outputs = self.temporal_encoder(fused_seq, training=training)  # (B, S, D_temp)

        # 6) Cross-Modal Temporal Attention over GRU outputs
        story_context = self.temporal_attn(temporal_outputs, training=training)  # (B, D_temp)

        # 7) Decode target caption using context
        logits = self.decoder(target_caption, context=story_context, training=training)

        return logits


# ---------------------------------------------------------------------------
# Utility: Build a compiled Keras model using config.yaml
# ---------------------------------------------------------------------------

def build_compiled_model(config_path: str = "config.yaml", config: dict = None):
    """
    Build and compile the CrossModalTemporalAttention model.

    If `config` is None, it will be loaded from `config_path`.
    """
    if config is None:
        config = load_config(config_path)

    dcfg = config["dataset"]
    mcfg = config["model"]
    tcfg = config["training"]

    S = dcfg["seq_len"]
    T_ctx = dcfg["max_caption_len"]
    T_tgt = dcfg["max_caption_len"]
    H_img = dcfg["image_size"]
    W_img = dcfg["image_size"]

    # Inputs
    images_in = layers.Input(shape=(S, H_img, W_img, 3), name="images")
    ctx_caps_in = layers.Input(shape=(S, T_ctx), dtype="int32", name="context_captions")
    tgt_cap_in = layers.Input(shape=(T_tgt,), dtype="int32", name="target_caption")
    reason_in = layers.Input(shape=(dcfg["max_reason_len"],), dtype="int32", name="reason")

    core_model = CrossModalTemporalAttention(
        config=config,
        name=config.get("project", {}).get("name", "CrossModalTemporalAttention")
    )

    # Eager-build the core model on concrete (batch=1) tensors so all
    # sub-layers (Dense, Conv, etc.) get built with fully-defined shapes.
    # This prevents Dense layers from receiving an input dim of `None`.
    try:
        dummy_images = tf.zeros((1, S, H_img, W_img, 3), dtype=tf.float32)
        dummy_ctx = tf.zeros((1, S, T_ctx), dtype=tf.int32)
        dummy_tgt = tf.zeros((1, T_tgt), dtype=tf.int32)
        dummy_reason = tf.zeros((1, dcfg.get("max_reason_len", 0)), dtype=tf.int32)
        _ = core_model(
            {
                "images": dummy_images,
                "context_captions": dummy_ctx,
                "target_caption": dummy_tgt,
                "reason": dummy_reason,
            },
            training=False,
        )
    except Exception:
        # If eager build fails, continue — the subsequent symbolic call may still work.
        pass

    logits = core_model(
        {
            "images": images_in,
            "context_captions": ctx_caps_in,
            "target_caption": tgt_cap_in,
            "reason": reason_in,
        }
    )

    model = Model(
        inputs=[images_in, ctx_caps_in, tgt_cap_in, reason_in],
        outputs=logits,
        name=config.get("project", {}).get("name", "CrossModalTemporalAttention"),
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=tcfg["lr"])

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
    )

    return model


if __name__ == "__main__":
    # quick sanity check using config.yaml
    cfg = load_config("config.yaml")
    m = build_compiled_model(config=cfg)
    m.summary()
