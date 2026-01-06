# dnnls-visual-storytelling
Author : Keesari Rohith Goud
Cross-Modal Temporal Attention for Story Reasoning

#**Introduction and Problem Statement **
```
This project is developed as part of the Deep Neural Networks & Learning Systems (DNNLS) course and aims to improve visual storytelling models. Existing models often fail to understand
the sequence of events across multiple images. To overcome this, we enhance the storytelling system using multimodal features, temporal modeling, and attention mechanisms. The proposed
model can better understand image sequences and generate more coherent and meaningful story predictions.
```
#**Problem Definition**
```
The system integrates cross-modal fusion of image and text features with GRU-based temporal modeling and a temporal attention mechanism to generate coherent and context-aware multimodal
story continuations.
```

Evaluation Metrics
```
For evaluating storytelling performance, test loss and perplexity are used to measure prediction confidence, while BLEU-4 is applied to assess overlap between generated and reference
captions. Qualitative analysis is additionally used to examine narrative coherence and temporal consistency.
```
#**Methods**
```
We use convolutional image encoders and BiLSTM-based text encoders to extract visual and linguistic features from image–caption pairs. These features are fused using a Cross-Modal
Attention mechanism to align visual and textual information at each time step. A GRU-based temporal encoder models the narrative flow across frames, followed by a Cross-Modal Temporal
Attention layer that highlights the most informative moments in the story.Model performance is evaluated using test loss, perplexity, and BLEU score, and the dataset is split into
70/15/15 for training, validation, and testing.
```
#**Model Architecture Overview**
```
The proposed architecture incorporates a convolutional neural network (CNN)–based visual encoder and a BiLSTM-based text encoder to extract visual and linguistic representations from
image–caption pairs. These features are integrated through a Cross-Modal Attention mechanism, followed by a GRU-based temporal encoder to model narrative progression across image
sequences. A Cross-Modal Temporal Attention layer selectively emphasizes the most informative time steps, and an LSTM decoder generates coherent story continuations. The model is trained
with early stopping and evaluated using test loss, perplexity, and BLEU score to assess storytelling performance.
```
#** Reasoning aware attention**
```
The proposed mechanism enables the model to selectively focus on relevant visual features and textual context at different time steps through learned temporal relationships. This is
implemented using a Cross-Modal Temporal Attention module applied after GRU-based sequence modeling, which helps the model capture cause–effect relationships between frames and maintain
narrative coherence during story caption generation.
```
#**Code Snippet(simplified)**
```
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
```

#**Results**
```

The enhanced model produces more coherent and context-aware story predictions compared to the baseline model. The Cross-Modal Temporal Attention mechanism improves multimodal temporal
reasoning by allowing the model to focus on the most relevant frames and captions, while GRU-based sequence modeling strengthens narrative flow. These improvements result in better story
continuity and semantic alignment across frames, with consistent gains reflected in reduced test loss, stable perplexity, and qualitative improvements despite low BLEU scores due to long
and markup-heavy captions.
```
#**Quantitavtive analysis**
```
Results/loss_curve.png
```
#**Qualitative analysis**
```
Results/sample generations.png

```

#**conclusion**
```
The enhanced model produces more coherent story predictions than the baseline. Cross-Modal Temporal Attention improves temporal reasoning and multimodal alignment, resulting in better
story continuity and consistent performance gains.
```
#**Future work**
```
Scaling the model to handle longer and more complex visual stories.

Improving story generation quality using advanced attention and temporal modeling techniques.

Extending the framework to video-based inputs instea

```





