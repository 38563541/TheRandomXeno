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

