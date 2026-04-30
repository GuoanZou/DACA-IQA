# DACA-IQA: Degradation-Aware CLIP for Image Quality Assessment

DACA-IQA is a modular implementation of a degradation-aware image quality assessment model that combines CLIP with Cross-Modal Mutual Adaptation (CMMA) and DA-CLIP features.

## Architecture

The model consists of three main components:

1. **CLIP with CMMA**: Vision-language model with trainable cross-modal mutual adaptation via Gram matrices
2. **DA-CLIP**: Frozen degradation-aware feature extractor providing patch-level degradation information
3. **DPIM (Dynamic Prompt Interaction Module)**: Cross-attention based module that injects DA-CLIP features into the main model

## Installation

```bash
pip install torch torchvision
pip install clip open_clip_torch
pip install pandas pillow tqdm
```

## Project Structure

```
DACA-IQA/
├── models/
│   ├── daca_iqa.py          # Main model implementation
│   ├── dpim.py              # Dynamic Prompt Interaction Module
│   └── clip_with_cmma.py    # CLIP with Cross-Modal Mutual Adaptation
├── datasets/
│   └── image_dataset.py     # Dataset loaders for various IQA benchmarks
├── losses/
│   └── mnl_loss.py          # Loss functions (KL+Rank, Ordinal, etc.)
├── utils/
│   └── data_utils.py        # Data loading utilities
└── __init__.py
```

## Usage

### Basic Usage

```python
from DACA_IQA import DACA_IQA
import torch

# Initialize model
model = DACA_IQA(
    clip_model_name="ViT-B/32",
    device='cuda:0',
    subimage_num=16,
    gram_rank=32,
    gram_alpha=0.1,
    daclip_ckpt='path/to/daclip_weights.pt',
    latent_dim=96,
    cross_attn_heads=8
)

# Forward pass
# x: [B, P, 3, 224, 224] where P is number of patches
quality_scores, probs, text_features = model(x)
```

### Training Example

```python
from DACA_IQA.losses import kl_rank_loss, ordinal_loss
from DACA_IQA.utils import set_spaq1

# Setup dataset
train_loader = set_spaq1(
    csv_file='train.xlsx',
    bs=8,
    data_set='path/to/images',
    num_workers=4,
    preprocess=preprocess,
    num_patch=16,
    test=False,
    soft_labels_csv='soft_labels.csv'
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for batch in train_loader:
    images = batch['I'].to(device)
    mos = batch['mos'].to(device)
    soft_labels = batch['soft_labels'].to(device)
    
    quality, probs, text_feats = model(images)
    
    # Combined loss
    loss_kl_rank = kl_rank_loss(quality, probs, mos, soft_labels, lambda_rank=1.0)
    loss_ordinal = ordinal_loss(text_feats, margin=0.1)
    loss = loss_kl_rank + 0.1 * loss_ordinal
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Model Components

### DACA_IQA

Main model class that integrates all components.

**Parameters:**
- `clip_ckpt`: Path to pretrained CLIP checkpoint (optional)
- `device`: Device to run on ('cuda:0' or 'cpu')
- `clip_model_name`: CLIP architecture ('ViT-B/32', 'ViT-B/16', etc.)
- `n_ctx`: Number of context tokens for prompt learning (default: 10)
- `subimage_num`: Number of patches per image (default: 16)
- `gram_rank`: Rank for Gram matrix adaptation (default: 32)
- `gram_alpha`: Initial alpha value for CMMA (default: 0.1)
- `csc`: Class-specific context for prompts (default: True)
- `daclip_ckpt`: Path to DA-CLIP checkpoint
- `latent_dim`: Latent dimension for DPIM (default: 96)
- `cross_attn_heads`: Number of attention heads in DPIM (default: 8)

### DPIM

Dynamic Prompt Interaction Module that injects degradation features via cross-attention.

**Key Features:**
- Layer-wise feature injection
- Low-dimensional latent space for efficiency
- Learnable scale factors per layer

### Loss Functions

- `kl_rank_loss`: Combined KL divergence and pairwise ranking loss
- `ordinal_loss`: Ordinal constraint on text embeddings
- `Fidelity_Loss`: Fidelity-based loss for probability distributions

## Supported Datasets

The package includes loaders for:
- SPAQ (Smartphone Photography Attribute and Quality)
- TID2013
- PIPAL
- AVA (Aesthetic Visual Analysis)
- Custom datasets with soft labels

## Citation

If you use this code in your research, please cite the relevant papers for CLIP, DA-CLIP, and the quality assessment methodology.

## License

This implementation is provided for research purposes.
