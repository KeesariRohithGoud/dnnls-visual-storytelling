# dnnls-visual-storytelling
Cross-Modal Temporal Attention for Story Reasoning

A TensorFlow implementation of a Cross-Modal Temporal Attention (CMTA) model that learns to understand short visual stories using image sequences, captions, and optional reasoning text.
The system integrates image features, caption embeddings, cross-modal fusion, temporal GRU, and a temporal attention layer to generate coherent next-step descriptions.

 1. OVERVIEW

This project focuses on story understanding from multimodal data.
Given:

A sequence of images (frames)

Corresponding captions

Optional reasoning/explanation text

the model predicts the next caption using:

🔹 Cross-Modal Feature Fusion

Spatial image features + caption features + optional reasoning text

🔹 GRU-based Temporal Modeling

Over fused features across time

🔹 Cross-Modal Temporal Attention

Attention over GRU outputs to focus on the most informative time steps

🔹 Decoder for Caption Generation

Conditioned on attention-enhanced story context

 2. Dataset

We use the HuggingFace dataset:

daniel3303/StoryReasoning


Key columns:

Column	Description
frames	List of images for each story
captions	List of captions (one per frame)
reason	Optional reasoning text

The dataset is automatically split into:

70% Train

15% Validation

15% Test

Splitting occurs inside utils.py.

 3. Model Architecture

The full model is built in model.py and includes:

 Image Encoder

CNN backbone

Spatial features

Global pooled features

🔸 Text Encoder

BERT-tokenized captions

BiLSTM (512 hidden)

🔸 Optional Reason Encoder

BiLSTM reasoning encoder

🔸 Cross-Modal Fusion

Combines:

Projected spatial image patches

Encoded captions

Optional reasoning

Attention between text ↔ image

🔸 Temporal GRU Encoder

Processes fused multimodal sequence.

🔸 Cross-Modal Temporal Attention (after GRU)

Weights time steps based on relevance.

🔸 Caption Decoder

LSTM decoder initialized from temporal context.

🛠 4. Installation
Clone the project
git clone <your_repo_url>
cd CrossModalTemporalAttention

Install dependencies
pip install -r requirements.txt

 5. Configuration

All settings are stored in:

config.yaml


This includes:

Dataset paths

Tokenization settings

Model dimensions

Training parameters

Vocabulary IDs

 6. Training

Run:

python train.py


This will:

Load config.yaml

Load HF dataset

Split 70/15/15

Preprocess images & captions

Build and compile the model

Train with checkpoints + early stopping

7. Evaluation
python train.py


Automatically evaluates on the test set after training:

Evaluating on test set...
23/23 - loss: 3.86

 8. Notebook Workflow

A full Jupyter Notebook pipeline is provided (see documentation):

Includes:

Data loading

Exploratory Data Analysis

Preprocessing

Training

Testing

Visualizing predictions (images + generated captions)

 9. Directory Structure
CrossModalTemporalAttention/
│
├── model.py              # Model definition (CMTA)
├── train.py              # Training pipeline
├── utils.py              # Utility functions + preprocessing
├── config.yaml           # All hyperparameters
├── requirements.txt      # Python environment
├── README.md             # Documentation
└── notebook.ipynb        # (Optional) Full workflow notebook

 10. Visualizing Predictions

Example:

Frame 0: A dog jumps over a log.
Frame 1: The dog runs through the forest.
Frame 2: The dog stops near a tree.

Ground-truth next caption:
"The dog looks back at its owner."

Model prediction:
"The dog turns and looks behind."

 11. Future Work

Add multi-head attention

Replace CNN with Vision Transformer

Integrate Llama or T5 decoders

Add contrastive multimodal pretraining
