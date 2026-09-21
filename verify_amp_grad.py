"""
verify_amp_grad.py — 0922 階段 4c

⚠️ 這個腳本不判定通過或不通過。AMP 本來就不會跟 fp32 位元相同，這裡只是
量化偏差大小，讓之後設計準確率驗證時有一個基準可以比對。

GPU、RN50、train 模式、BN momentum 設 0、dropout 設 0（排除跟 AMP 無關的
隨機性來源）、同一份初始權重、同一個 batch，--grad_ckpt --ckpt_prefix 14
--ckpt_segments 7：
  - fp32 跑兩次 → 底線（GPU 本身的不確定性，非確定性 cuDNN 演算法選擇等）
  - bf16 跑一次
  - fp16 跑一次

對三組參數報 ||g - g_fp32|| / ||g_fp32|| 與 cosine similarity：
  backbone 第一層 conv、backbone 最後一個 block、匹配頭第一層 Linear
另外報 loss 的差。
"""
import sys, os, types, math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from models.stage2_model import CNN_TRXWithRelation

device = "cuda:0"
WAY, SHOT, QPC, SEQ = 5, 1, 5, 8


def make_args(amp="off"):
    return types.SimpleNamespace(
        trans_linear_in_dim=2048, trans_linear_out_dim=1152,
        way=WAY, shot=SHOT, query_per_class=QPC, trans_dropout=0.1,
        seq_len=SEQ, img_size=224, method="resnet50", num_gpus=1,
        temp_set=[2], grad_ckpt=True, ckpt_segments=7, ckpt_prefix=14,
        ckpt_policy="full", freeze_backbone=False, matching="bidirectional",
        set_aggregation="pool", tau=0.1, relation_level="tuple",
        use_intra_relation=True, use_inter_relation=True,
        inter_style="true_hyrsm", intra_depth=1, amp=amp,
    )


def zero_stochastic(model):
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.momentum = 0.0
        elif isinstance(mod, nn.Dropout):
            mod.p = 0.0
        elif isinstance(mod, nn.MultiheadAttention):
            mod.dropout = 0.0


def build_model(amp):
    m = CNN_TRXWithRelation(make_args(amp)).to(device).train()
    zero_stochastic(m)
    return m


def pick_probe_names(model):
    resnet_names = [n for n, _ in model.named_parameters() if n.startswith("resnet.")]
    block_idxs = sorted({int(n.split(".")[1]) for n in resnet_names if n.split(".")[1].isdigit()})
    first_idx, last_idx = block_idxs[0], block_idxs[-1]
    conv1_name = next(n for n in resnet_names if n.startswith(f"resnet.{first_idx}.") and n.endswith("weight"))
    last_block_names = [n for n in resnet_names if n.startswith(f"resnet.{last_idx}.")]
    last_block_conv_name = next((n for n in last_block_names if "conv" in n and n.endswith("weight")), last_block_names[0])
    head_name = next(n for n, _ in model.named_parameters() if "embed_linear.weight" in n)
    return conv1_name, last_block_conv_name, head_name


def run_once(model, ctx, tgt, lbl):
    model.zero_grad()
    _amp = getattr(model.args, "amp", "off")
    if _amp == "off":
        out = model(ctx, lbl, tgt)["logits"]
        loss = out.sum()
        loss.backward()
    else:
        dtype = torch.bfloat16 if _amp == "bf16" else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            out = model(ctx, lbl, tgt)["logits"]
            loss = out.sum()
        loss.backward()
    grads = {name: p.grad.detach().clone().float()
             for name, p in model.named_parameters() if p.grad is not None}
    return loss.item(), grads


def rel_err_and_cos(g_a, g_b):
    diff_norm = (g_a - g_b).norm().item()
    a_norm = g_a.norm().item()
    rel = diff_norm / a_norm if a_norm > 0 else float("nan")
    cos = torch.nn.functional.cosine_similarity(g_a.flatten(), g_b.flatten(), dim=0).item()
    return rel, cos


print("建立模型（fp32 基準模型，其餘從它複製權重）...")
m_fp32a = build_model("off")

conv1_name, last_block_name, head_name = pick_probe_names(m_fp32a)
print(f"探測參數：backbone 第一層={conv1_name}  backbone 最後一個block={last_block_name}  head 第一層={head_name}")

m_fp32b = build_model("off")
m_bf16  = build_model("bf16")
m_fp16  = build_model("fp16")
for m in (m_fp32b, m_bf16, m_fp16):
    m.load_state_dict(m_fp32a.state_dict())

torch.manual_seed(99)
ctx = torch.randn(WAY * SHOT * SEQ, 3, 224, 224, device=device)
tgt = torch.randn(WAY * QPC  * SEQ, 3, 224, 224, device=device)
lbl = torch.arange(WAY, device=device)

loss_fp32a, g_fp32a = run_once(m_fp32a, ctx, tgt, lbl)
loss_fp32b, g_fp32b = run_once(m_fp32b, ctx, tgt, lbl)
loss_bf16,  g_bf16  = run_once(m_bf16,  ctx, tgt, lbl)
loss_fp16,  g_fp16  = run_once(m_fp16,  ctx, tgt, lbl)

print(f"\nloss: fp32_a={loss_fp32a:.6f}  fp32_b={loss_fp32b:.6f}  bf16={loss_bf16:.6f}  fp16={loss_fp16:.6f}")
print(f"|loss_fp32b - loss_fp32a| = {abs(loss_fp32b-loss_fp32a):.6e}  (GPU 底線)")
print(f"|loss_bf16  - loss_fp32a| = {abs(loss_bf16 -loss_fp32a):.6e}")
print(f"|loss_fp16  - loss_fp32a| = {abs(loss_fp16 -loss_fp32a):.6e}")

probe_names = {"backbone第一層conv": conv1_name, "backbone最後一個block": last_block_name, "head第一層Linear": head_name}

print("\n=== 梯度偏差（相對 fp32_a）===")
for label, pname in probe_names.items():
    print(f"\n[{label}] ({pname})")
    for tag, gdict in [("fp32_b(底線)", g_fp32b), ("bf16", g_bf16), ("fp16", g_fp16)]:
        if pname not in gdict:
            print(f"  {tag}: 該參數沒有梯度（可能在未涵蓋段外或其他原因）")
            continue
        rel, cos = rel_err_and_cos(g_fp32a[pname], gdict[pname])
        print(f"  {tag}: ||g-g_fp32||/||g_fp32|| = {rel:.4e}   cosine similarity = {cos:.8f}")

print("\n（描述性結果，不是驗收——AMP 跟 fp32 本來就不會位元相同）")
