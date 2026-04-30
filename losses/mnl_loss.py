import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import itertools

EPS = 1e-2
esp = 1e-8

class Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss, self).__init__()

    def forward(self, p, g):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))

        return torch.mean(loss)

class Fidelity_Loss_distortion(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss_distortion, self).__init__()

    def forward(self, p, g):
        loss = 0
        for i in range(p.size(1)):
            p_i = p[:, i]
            g_i = g[:, i]
            g_i = g_i.view(-1, 1)
            p_i = p_i.view(-1, 1)
            loss_i = torch.sqrt(p_i * g_i + esp)
            loss = loss + loss_i
        loss = 1 - loss
        #loss = loss / p.size(1)
        return torch.mean(loss)


class Multi_Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Multi_Fidelity_Loss, self).__init__()

    def forward(self, p, g):

        loss = 0
        for i in range(p.size(1)):
            p_i = p[:, i]
            g_i = g[:, i]
            g_i = g_i.view(-1, 1)
            p_i = p_i.view(-1, 1)
            loss_i = 1 - (torch.sqrt(p_i * g_i + esp) + torch.sqrt((1 - p_i) * (1 - g_i) + esp))
            loss = loss + loss_i
        loss = loss / p.size(1)

        return torch.mean(loss)

eps = 1e-12


def loss_m(y_pred, y):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1  #
    preds = y_pred-(y_pred + 10).t()
    gts = y.t() - y
    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]
    return torch.sum(F.relu(preds * torch.sign(gts))) / preds.size(0)
    #return torch.sum(F.relu((y_pred-(y_pred + 10).t()) * torch.sign((y.t()-y)))) / y_pred.size(0) / (y_pred.size(0)-1)


def loss_m2(y_pred, y, gstd):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1  #
    preds = y_pred-y_pred.t()
    gts = y - y.t()
    g_var = gstd * gstd + gstd.t() * gstd.t() + eps

    #signed = torch.sign(gts)

    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]
    g_var = g_var[triu_indices[0], triu_indices[1]]
    #signed = signed[triu_indices[0], triu_indices[1]]

    constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
    g = 0.5 * (1 + torch.erf(gts / torch.sqrt(g_var)))
    p = 0.5 * (1 + torch.erf(preds / constant))

    g = g.view(-1, 1)
    p = p.view(-1, 1)

    loss = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    return loss


def loss_m3(y_pred, y):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1  #
    y_pred = y_pred.unsqueeze(1)
    y = y.unsqueeze(1)
    preds = y_pred-y_pred.t()
    gts = y - y.t()

    #signed = torch.sign(gts)

    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]
    g = 0.5 * (torch.sign(gts) + 1)

    constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
    p = 0.5 * (1 + torch.erf(preds / constant))

    g = g.view(-1, 1)
    p = p.view(-1, 1)

    loss = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    return loss

def loss_m4(y_pred_all, per_num, y_all):
    """prediction monotonicity related loss"""
    loss = 0
    pos_idx = 0
    for task_num in per_num:
        y_pred = y_pred_all[pos_idx:pos_idx+task_num]
        y = y_all[pos_idx:pos_idx+task_num]
        pos_idx = pos_idx + task_num

        #assert y_pred.size(0) > 1  #
        if y_pred.size(0) == 0:
            continue
        y_pred = y_pred.unsqueeze(1)
        y = y.unsqueeze(1)

        preds = y_pred - y_pred.t()
        gts = y - y.t()

        # signed = torch.sign(gts)

        triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
        preds = preds[triu_indices[0], triu_indices[1]]
        gts = gts[triu_indices[0], triu_indices[1]]
        g = 0.5 * (torch.sign(gts) + 1)

        constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
        p = 0.5 * (1 + torch.erf(preds / constant))

        g = g.view(-1, 1)
        p = p.view(-1, 1)

        loss += torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    loss = loss / len(per_num)

    return loss


def kl_rank_loss(y_pred, probs_pred, y, soft_labels, lambda_rank=1.0):
    """
    组合损失：KL散度 + 成对排序损失 (内部实现，不调用外部函数)
    Args:
        y_pred:     预测分数, shape [B] 或 [B,1]
        probs_pred: 预测的等级概率分布, shape [B, 5]
        y:          真实MOS分数, shape [B]
        soft_labels: 真实软标签（高斯离散化）, shape [B, 5]
        lambda_rank: 排序损失权重，默认为1.0
    Returns:
        总损失（标量）
    """
    # ---------- 1. 确保分数为一维 ----------
    if y_pred.dim() > 1:
        y_pred = y_pred.reshape(-1)
    if y.dim() > 1:
        y = y.reshape(-1)
    y = y.detach().float()
    N = y_pred.shape[0]

    # ---------- 2. 分数标准化（保留原逻辑）----------
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred_norm = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y_norm = (y - m) / (sigma + 1e-8)

    # ---------- 3. 成对排序损失（内嵌实现）----------
    if N >= 2:
        # 差值矩阵
        pred_diff = y_pred_norm[:, None] - y_pred_norm[None, :]  # [N, N]
        gt_diff   = y_norm[:, None]   - y_norm[None, :]
        # 符号函数（忽略对角线）
        sign_gt = torch.sign(gt_diff)
        # 核心公式: max(0, -sign(gt_diff) * (pred_diff - gt_diff))
        indicat = -sign_gt * (pred_diff - gt_diff)
        loss_ij = torch.clamp(indicat, min=0.0)
        # 忽略对角线
        eye = torch.eye(N, device=y_pred.device)
        loss_ij = loss_ij * (1 - eye)
        # 平均到每个有序对
        rank_loss = loss_ij.sum() / (N * (N - 1))
    else:
        rank_loss = torch.tensor(0.0, device=y_pred.device)

    # ---------- 4. KL 散度损失 ----------
    eps = 1e-8
    kl_div = F.kl_div(
        torch.log(probs_pred + eps),
        soft_labels,
        reduction='batchmean'
    )

    # ---------- 5. 总损失 ----------
    total_loss = kl_div + lambda_rank * rank_loss
    return total_loss   

def ordinal_loss(text_feats, margin=0.1):
    """
    text_feats: [5, D]  L2-normalized text embeddings for the five quality levels
    margin: margin δ
    """
    # 计算余弦相似度矩阵 (已归一化，点积即为余弦相似度)
    sim = torch.mm(text_feats, text_feats.t())  # [5, 5]
    
    loss = 0.0
    cnt = 0
    for i in range(4):          # i = 0,1,2,3
        for j in range(i+2, 5): # j = i+2, i+3, i+4
            # 相邻相似度: sim[i, i+1]
            # 非相邻相似度: sim[i, j]
            loss += torch.clamp(sim[i, i+1] - sim[i, j] + margin, min=0.0)
            cnt += 1
    return loss / cnt