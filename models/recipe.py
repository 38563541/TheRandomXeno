"""
models/recipe.py — 1126 訓練配方選項中，作用在模型層級（CNN_TRX／CNN_TRXWithRelation）的部分。

所有東西都只在選項打開時才建立，而且由模型 __init__ 在既有模組（含 setup_adapters）之後呼叫，
所以預設設定下 state_dict 不會多出鍵、既有層的隨機初始化順序不變。

logit_scale（預設 None = 不縮放，logits 就是匹配頭輸出的 −距離）
    logit_scale_learnable=False：logits = logit_scale × (−距離)，常數。
    logit_scale_learnable=True ：參數存 log(scale)（log 參數化），前向取 exp；初始值 = logit_scale。
"""
import math

import torch
import torch.nn as nn


def setup_logit_scale(model, args):
    """在模型 __init__ 的最後面呼叫（setup_adapters 之後）。"""
    scale = getattr(args, "logit_scale", None)
    learnable = bool(getattr(args, "logit_scale_learnable", False))
    model._logit_scale_const = None
    if scale is None:
        if learnable:
            raise ValueError("logit_scale_learnable=true 需要同時設定 logit_scale（作為初始值）")
        return
    scale = float(scale)
    if not scale > 0:
        raise ValueError(f"logit_scale 必須 > 0，收到 {scale}")
    if learnable:
        model.log_logit_scale = nn.Parameter(torch.tensor(math.log(scale)))
        print(f"[INFO] logit_scale 可學（log 參數化），初始 {scale}", flush=True)
    else:
        model._logit_scale_const = scale
        print(f"[INFO] logit_scale 固定 {scale}", flush=True)


def logit_scale_enabled(model):
    return getattr(model, "_logit_scale_const", None) is not None or hasattr(model, "log_logit_scale")


def apply_logit_scale(model, logits):
    """只在 logit_scale_enabled(model) 時呼叫：回傳 scale × logits。"""
    if hasattr(model, "log_logit_scale"):
        return torch.exp(model.log_logit_scale) * logits
    return model._logit_scale_const * logits


def logit_scale_value(model):
    """給 log 用：目前的 scale；沒開回傳 None。"""
    if hasattr(model, "log_logit_scale"):
        return float(torch.exp(model.log_logit_scale.detach()))
    return getattr(model, "_logit_scale_const", None)


# ---------------------------------------------------------------------------
# aux_cls_weight（預設 0 = 不建立）：train 類別的輔助分類頭
# ---------------------------------------------------------------------------
def setup_aux_head(model, train_class_ids, feat_dim, device):
    """由 Learner 在 VideoDataset 建好之後、收 trainable 參數／建 optimizer 之前呼叫。
    對照表只含 train split 的類別（全域類別編號 → 0..K−1），存成模型上的 buffer aux_class_ids
    （checkpoint 會一起存）與 dict aux_class_map。分類頭 = nn.Linear(feat_dim, K)。"""
    ids = sorted(int(c) for c in train_class_ids)
    model.register_buffer("aux_class_ids", torch.tensor(ids, dtype=torch.long, device=device))
    model.aux_class_map = {c: i for i, c in enumerate(ids)}
    model.aux_head = nn.Linear(feat_dim, len(ids)).to(device)
    print(f"[INFO] aux_cls_weight>0：輔助分類頭 Linear({feat_dim}, {len(ids)})，train 類別 {len(ids)} 類", flush=True)
    return len(ids)


def aux_enabled(model):
    return getattr(model, "aux_head", None) is not None


def aux_targets(model, global_ids, device):
    """全域類別編號 → 0..K−1。遇到對照表外的類別直接報錯（代表 val／test 的影片混進訓練）。"""
    out = []
    for g in global_ids:
        g = int(g)
        if g not in model.aux_class_map:
            raise RuntimeError(f"aux_cls：類別 {g} 不在 train 類別對照表（{len(model.aux_class_map)} 類）裡，"
                               f"val／test 的影片混進訓練了？")
        out.append(model.aux_class_map[g])
    return torch.tensor(out, dtype=torch.long, device=device)


def aux_cls_loss(model, model_dict, support_labels, real_target_labels, batch_class_list, device):
    """support 的全域類別 = batch_class_list[support_labels]；query 的全域類別 = real_target_labels。"""
    feats = torch.cat([model_dict["support_video_feats"], model_dict["query_video_feats"]], dim=0)
    bcl = batch_class_list.detach().cpu()
    sup_global = [bcl[int(l)] for l in support_labels.detach().cpu()]
    qry_global = list(real_target_labels.detach().cpu())
    tgt = aux_targets(model, sup_global + qry_global, device)
    logits = model.aux_head(feats.to(model.aux_head.weight.dtype))
    return torch.nn.functional.cross_entropy(logits, tgt)
