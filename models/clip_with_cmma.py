"""
CLIP with CMMA (Cross-Modal Mutual Modulation)

CLIP model with trainable cross-modal mutual adaptation via Gram matrices.
Each ResidualAttentionBlock has independent trainable alpha parameters.
"""

import hashlib
import os
import urllib
import warnings
from tqdm import tqdm
from typing import Tuple, Union, List
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model downloaded but SHA256 mismatch")

    return download_target


def available_models() -> List[str]:
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit: bool = False, download_root: str = None,
         gram_rank: int = 32, gram_alpha: float = 0.1):
    """
    Load a CLIP model.
    gram_alpha: 仅用于 alpha 参数的初始化值，后续每个 block 的 alpha 独立且可训练
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as state dict")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict(),
                             gram_rank=gram_rank, gram_alpha=gram_alpha).to(device)
        if str(device) == "cpu":
            model.float()
        return model

    # --- JIT path (unchanged) ---
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    return model


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    - alpha: nn.Parameter（每个 block 独立可训练）
    - 初始化时用 init_val 初始化 alpha 参数
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 layer_idx: int = 0, modal: str = 'text',
                 S_vis_attn: nn.Parameter = None, S_text_attn: nn.Parameter = None,
                 S_vis_mlp: nn.Parameter = None, S_text_mlp: nn.Parameter = None,
                 alpha_init_val: float = 0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

        # Gram adapter parameters
        self.S_vis_attn = S_vis_attn
        self.S_text_attn = S_text_attn
        self.S_vis_mlp = S_vis_mlp
        self.S_text_mlp = S_text_mlp

        self.alpha_attn = nn.Parameter(
            torch.tensor(np.random.randn() * 0.1, dtype=torch.float32)
        )
        self.alpha_mlp = nn.Parameter(
            torch.tensor(np.random.randn() * 0.1, dtype=torch.float32)
        )

        self.modal = modal
        self.d_model = d_model

    def _apply_gram_adapter(self, x: torch.Tensor, S_self: nn.Parameter, S_cross: nn.Parameter, alpha: nn.Parameter):
        """
        Gram 适配器:
        T = S_self @ (S_cross.T @ S_cross) @ S_self.T
        x = x + alpha * (x @ T)
        """
        if S_self is None or S_cross is None:
            return x

        gram = S_cross.T @ S_cross           # (r, r)
        T = S_self @ gram @ S_self.T          # (D, D)
        out = torch.matmul(x, T)

        return x + alpha * out

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # attention path
        x_ln1 = self.ln_1(x)
        if self.modal == 'text':
            x_ln1 = self._apply_gram_adapter(x_ln1, self.S_text_attn, self.S_vis_attn, self.alpha_attn)
        else:
            x_ln1 = self._apply_gram_adapter(x_ln1, self.S_vis_attn, self.S_text_attn, self.alpha_attn)

        x = x + self.attention(x_ln1)

        # mlp path
        x_ln2 = self.ln_2(x)
        if self.modal == 'text':
            x_ln2 = self._apply_gram_adapter(x_ln2, self.S_text_mlp, self.S_vis_mlp, self.alpha_mlp)
        else:
            x_ln2 = self._apply_gram_adapter(x_ln2, self.S_vis_mlp, self.S_text_mlp, self.alpha_mlp)
        x = x + self.mlp(x_ln2)
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 S_vis_attn_list: List[nn.Parameter] = None, S_text_attn_list: List[nn.Parameter] = None,
                 S_vis_mlp_list: List[nn.Parameter] = None, S_text_mlp_list: List[nn.Parameter] = None,
                 alpha_init_val: float = 0.1, modal: str = 'text'):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(
                width, heads, attn_mask,
                layer_idx=i, modal=modal,
                S_vis_attn=S_vis_attn_list[i] if S_vis_attn_list is not None else None,
                S_text_attn=S_text_attn_list[i] if S_text_attn_list is not None else None,
                S_vis_mlp=S_vis_mlp_list[i] if S_vis_mlp_list is not None else None,
                S_text_mlp=S_text_mlp_list[i] if S_text_mlp_list is not None else None,
                alpha_init_val=alpha_init_val   # 初始化值，每个 block 会叠加少量随机扰动
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 S_vis_attn_list: List[nn.Parameter] = None, S_text_attn_list: List[nn.Parameter] = None,
                 S_vis_mlp_list: List[nn.Parameter] = None, S_text_mlp_list: List[nn.Parameter] = None,
                 alpha_init_val: float = 0.1):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads,
            S_vis_attn_list=S_vis_attn_list,
            S_text_attn_list=S_text_attn_list,
            S_vis_mlp_list=S_vis_mlp_list,
            S_text_mlp_list=S_text_mlp_list,
            alpha_init_val=alpha_init_val,
            modal='vision'
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_token=True, pos_embedding=False):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                      dtype=x.dtype, device=x.device), x], dim=1)

        if pos_embedding:
            positional_embedding_resize = F.interpolate(
                self.positional_embedding.unsqueeze(0).unsqueeze(0),
                size=(x.size(1), x.size(2)), mode='bicubic').squeeze(0).squeeze(0)
            x = x + positional_embedding_resize.to(x.dtype)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        token = self.ln_post(x[:, 1:, :])
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
            token = token @ self.proj

        if return_token:
            return x, token
        else:
            return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 gram_rank: int = 8,
                 gram_alpha: float = 0.1   # 仅用于 alpha 参数的初始化值
                 ):
        super().__init__()

        self.context_length = context_length
        print(f"[CLIP] vision_width={vision_width}, transformer_width={transformer_width}, "
              f"vision_layers={vision_layers}, transformer_layers={transformer_layers}, "
              f"gram_rank={gram_rank}, gram_alpha(init)={gram_alpha}")

        num_layers = min(
            vision_layers if isinstance(vision_layers, int) else len(vision_layers),
            transformer_layers
        )
        if num_layers != vision_layers or num_layers != transformer_layers:
            warnings.warn(f"Vision ({vision_layers}) and text ({transformer_layers}) layers differ. "
                          f"Using only first {num_layers} for Gram adapter.")
        self.num_layers = num_layers
        self.gram_rank = gram_rank

        # Gram adapter 参数 (与原版一致)
        self.S_vis_attn = nn.ParameterList([
            nn.Parameter(torch.randn(vision_width, gram_rank) * 0.02)
            for _ in range(num_layers)
        ])
        self.S_text_attn = nn.ParameterList([
            nn.Parameter(torch.randn(transformer_width, gram_rank) * 0.02)
            for _ in range(num_layers)
        ])
        self.S_vis_mlp = nn.ParameterList([
            nn.Parameter(torch.randn(vision_width, gram_rank) * 0.02)
            for _ in range(num_layers)
        ])
        self.S_text_mlp = nn.ParameterList([
            nn.Parameter(torch.randn(transformer_width, gram_rank) * 0.02)
            for _ in range(num_layers)
        ])

        # Vision encoder
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers, output_dim=embed_dim, heads=vision_heads,
                input_resolution=image_resolution, width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                S_vis_attn_list=self.S_vis_attn,
                S_text_attn_list=self.S_text_attn,
                S_vis_mlp_list=self.S_vis_mlp,
                S_text_mlp_list=self.S_text_mlp,
                alpha_init_val=gram_alpha
            )

        # Text transformer
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            S_vis_attn_list=self.S_vis_attn,
            S_text_attn_list=self.S_text_attn,
            S_vis_mlp_list=self.S_vis_mlp,
            S_text_mlp_list=self.S_text_mlp,
            alpha_init_val=gram_alpha,
            modal='text'
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        if isinstance(self.visual, VisionTransformer):
            image_features, image_tokens = self.visual(image.type(self.dtype))
            return image_features, image_tokens
        else:
            image_features = self.visual(image.type(self.dtype))
            return image_features, None

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text, pos_embedding=False, text_features=None):
        image_features, _ = self.encode_image(image)
        if text_features is None:
            text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# ---------------------------------------------------------------------------
# 以下未改动部分（与 clip_gram_mslr.py 保持一致）
# ---------------------------------------------------------------------------

class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, return_token=False, pos_embedding=False):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if return_token:
            x, tokens = self.attnpool(x, return_token, pos_embedding)
            return x, tokens
        else:
            x = self.attnpool(x, return_token, pos_embedding)
            return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, return_token=False, pos_embedding=False):
        n, c, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        if pos_embedding:
            positional_embedding_resize = F.interpolate(
                self.positional_embedding.unsqueeze(0).unsqueeze(0),
                size=(x.size(0), x.size(2)), mode='bicubic').squeeze(0).squeeze(0)
            x = x + positional_embedding_resize[:, None, :].to(x.dtype)

        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
            out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True, training=self.training, need_weights=False
        )
        if return_token:
            return x[0], x[1:]
        else:
            return x[0]


def convert_weights(model: nn.Module):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, gram_rank: int = 8, gram_alpha: float = 0.1):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        gram_rank=gram_rank, gram_alpha=gram_alpha
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
