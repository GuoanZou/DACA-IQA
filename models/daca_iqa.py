"""
DACA-IQA: Degradation-Aware Cross-modal Adaption for Image Quality Assessment

Main model implementation that combines:
- CLIP with CMMA (Cross-Modal Mutual Adaptation)
- DA-CLIP image controller for degradation prior extractor
- DPIM for feature injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Optional, List
from torchvision.transforms import Normalize
import open_clip

from .clip_with_cmma import load
from .dpim import DPIM

qualitys = ['bad', 'poor', 'fair', 'good', 'excellent']


class PromptLearner(nn.Module):
    """Learnable prompt generator for quality assessment"""

    def __init__(self, classnames: List[str], clip_model, n_ctx: int = 12, csc: bool = False):
        super().__init__()
        n_cls = len(classnames)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        device = next(clip_model.parameters()).device

        if csc:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype, device=device)
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = csc
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx
        if not self.csc:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts, self.tokenized_prompts


class TextEncoder(nn.Module):
    """CLIP text encoder for quality prompts"""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        seq_len = prompts.shape[1]
        x = prompts + self.positional_embedding[:seq_len, :].type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        eot_idx = tokenized_prompts.argmax(dim=-1)
        features = x[torch.arange(x.shape[0]), eot_idx] @ self.text_projection
        return features


class DACA_IQA(nn.Module):
    """
    DACA-IQA: Degradation-Aware Cross-modal Adaption for Image Quality Assessment

    Integrates DA-CLIP patch tokens via DPIM for degradation-aware quality prediction.
    """

    def __init__(self,
                 clip_ckpt: Optional[str] = None,
                 device: str = 'cuda:0',
                 clip_model_name: str = "ViT-B/32",
                 n_ctx: int = 10,
                 subimage_num: int = 16,
                 gram_rank: int = 32,
                 gram_alpha: float = 0.1,
                 csc: bool = True,
                 daclip_ckpt: str = None,
                 latent_dim: int = 96,
                 cross_attn_heads: int = 8):
        super().__init__()
        self.device = device
        self.num_patch = subimage_num

        # Main model: CLIP with CMMA
        self.model = load(
            clip_model_name,
            device=device,
            gram_rank=gram_rank,
            gram_alpha=gram_alpha
        )
        self.model = self.model.float()
        if clip_ckpt is not None:
            ckpt = torch.load(clip_ckpt, map_location=device)
            self.model.load_state_dict(ckpt, strict=False)

        # DA-CLIP degradation prior extractor (frozen)
        if daclip_ckpt is not None:
            self.daclip, _ = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=daclip_ckpt)
            self.daclip = self.daclip.to(device).float()
            for param in self.daclip.parameters():
                param.requires_grad = False

            self.degradations = [
                'motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy',
                'raindrop', 'rainy', 'shadowed', 'snowy', 'uncompleted'
            ]
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.degradation_texts = tokenizer(self.degradations).to(device)
        else:
            self.daclip = None

        # DPIM
        vision_layers = self.model.visual.transformer.layers
        embed_dim = 768
        self.dpim = DPIM(
            num_layers=vision_layers,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            num_heads=cross_attn_heads
        )

        # Prompt learner(CoOp)
        self.prompt_learner = PromptLearner(
            classnames=qualitys,
            clip_model=self.model,
            n_ctx=n_ctx,
            csc=csc
        )
        self.text_encoder = TextEncoder(self.model)

        # Image preprocessing
        self.normalize = Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )

        if self.daclip is not None:
            self.daclip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
            self.daclip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

        self._setup_trainable_parameters()

    def _setup_trainable_parameters(self):
        """Freeze backbone, unfreeze CMMA adapters, prompt_learner and DPIM"""
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze CMMA adapters
        cmma_keys = ("S_vis_attn", "S_text_attn", "S_vis_mlp", "S_text_mlp", "alpha")
        for name, p in self.model.named_parameters():
            if any(k in name for k in cmma_keys):
                p.requires_grad = True

        for p in self.prompt_learner.parameters():
            p.requires_grad = True

        for p in self.dpim.parameters():
            p.requires_grad = True

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}, Trainable: {trainable:,}, Ratio: {trainable/total*100:.2f}%")

    def forward(self, x):
        """
        Args:
            x: [B, P, 3, 224, 224] where P = subimage_num
        Returns:
            quality: [B] predicted quality scores
            probs: [B, 5] quality level probabilities
            text_feats: [5, D] text features
        """
        B, P, C, H, W = x.shape
        x_flat = x.view(B * P, C, H, W)

        # Extract DA-CLIP patch tokens
        if self.daclip is not None:
            da_img = (x_flat - self.daclip_mean) / self.daclip_std
            with torch.no_grad():
                _, hiddens = self.daclip.visual_control(da_img, output_hiddens=True)
                hidden_feature = hiddens[-1]
                hidden_feature = hidden_feature.permute(1, 0, 2)
                patch_tokens = hidden_feature[:, 1:, :]
        else:
            patch_tokens = None

        # Main model vision encoder with DPIM injection
        visual = self.model.visual
        x = visual.conv1(x_flat)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                    dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        # Layer-wise processing with DPIM
        for i, block in enumerate(visual.transformer.resblocks):
            if patch_tokens is not None and i < self.dpim.num_layers:
                x_ = x.permute(1, 0, 2)
                x_ = self.dpim(i, x_, patch_tokens)
                x = x_.permute(1, 0, 2)

            x = block(x)

        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        img_feats = x[:, 0, :]
        if visual.proj is not None:
            img_feats = img_feats @ visual.proj
        img_feats = F.normalize(img_feats, dim=-1)

        # Text features
        prompts, tokenized = self.prompt_learner()
        text_feats = self.text_encoder(prompts, tokenized)
        text_feats = F.normalize(text_feats, dim=-1)

        # Quality prediction
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (img_feats @ text_feats.t())
        logits = logits.view(B, P, -1).mean(dim=1)
        probs = F.softmax(logits, dim=1)

        weights = torch.tensor([1, 2, 3, 4, 5], device=self.device, dtype=probs.dtype)
        quality = (probs * weights).sum(dim=1)
        return quality, probs, text_feats
