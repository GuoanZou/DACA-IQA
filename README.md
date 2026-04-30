# DACA-IQA: Distortion-Aware Cross-Modal Adaptation for Image Quality Assessment

DACA-IQA is a blind image quality assessment (BIQA) method built upon CLIP (ViT-B/32). It enhances CLIP’s ability to assess images' quality accurately via two lightweight modules:

- **Cross-Modal Mutual Modulation Adapter (CMMA)**: Low-rank bidirectional modulation between vision and text features, with only 1M extra parameters.
- **Distortion Prior Injection (DPI)**: Injects local distortion priors from a frozen DA-CLIP controller into every Transformer layer of CLIP’s visual encoder through low-rank cross-attention.

The full model achieves state‑of‑the‑art performance on multiple IQA benchmarks (CSIQ, TID2013, KADID‑10k, BID, LIVEC, KonIQ‑10k, SPAQ) while training only **2.5M parameters**.

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- open_clip_torch
- timm
- scikit‑learn
- scipy
- tqdm
- pandas

Install dependencies:
```bash
pip install torch torchvision open_clip_torch timm scikit-learn scipy tqdm pandas
📦 Pretrained Weights
DA‑CLIP controller (frozen distortion prior extractor)
Download from DA‑CLIP repository.
Place the file daclip_ViT-B-32.pt under ./weights/.

CLIP ViT‑B/32 (backbone) – automatically downloaded by open_clip if not found.

🚀 Quick Start
1. Clone repository

git clone https://github.com/ZouGuoAn/DACA-IQA.git
cd DACA-IQA
2. Prepare datasets
Organise each IQA dataset as follows (example for KonIQ‑10k):

data/
└── KonIQ-10k/
    ├── 1024x768/
    │   └── *.jpg
    └── koniq_train.csv
    └── koniq_test.csv
CSV files must contain columns image_name and mos (and optionally std for variance).
For datasets without variance, you can set a default value (e.g., std=0.5).

3. Training
Example training on KonIQ‑10k:

python train.py --dataset koniq --data_root ./data/KonIQ-10k --batch_size 16 --epochs 100 --lr 1e-3
Arguments:

--dataset : koniq, tid2013, kadid, livec, spaq, etc.

--daclip_ckpt : path to DA‑CLIP .pt file (default ./weights/daclip_ViT-B-32_mix.pt)

--pretrained_clip : use CLIP ViT‑B/32 (default)

4. Evaluation
Evaluate a trained model on test split:

python test.py --ckpt ./checkpoints/best_model.pth --dataset koniq
📊 Results (on KonIQ-10k)
Method	SRCC	PLCC	Trainable Params (M)
DACA‑IQA (ours)	0.944	0.953	2.5
