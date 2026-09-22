"""
verify_amp_grad_v2.py — 0922b 階段 1：重做梯度偏差量測，修正 0922 4c 的兩個問題。

問題 1（已確認）：0922 的 4c 用 `loss = out.sum()`（logits 原始總和，可正可負，
不是分類 loss），不是 run.py 訓練時實際用的 loss。這裡改用 utils.loss()——
run.py 的 train_task() 實際呼叫的同一個函式：
    task_loss = self.loss(target_logits, target_labels, self.device) / tasks_per_batch

問題 2（已確認）：0922 4c 的「fp32 跑兩次」底線在這個受控單步設定下是
0.000000e+00，沒有尺度可比。真正該比的底線是「換一個 episode，梯度本來就
會變多少」——SGD 本身的雜訊，不是 GPU 決定性問題。

四組比較（GPU、RN50、--grad_ckpt --ckpt_prefix 14 --ckpt_segments 7、
train 模式、BN momentum=0、dropout=0、同一組初始權重）：
    fp32(ep1) vs fp32(ep1) 重跑   — GPU 底線
    fp32(ep1) vs fp32(ep2)        — 換 episode 的梯度差異（SGD 雜訊尺度）
    fp16(ep1) vs fp32(ep1)        — AMP 偏差
    bf16(ep1) vs fp32(ep1)        — 參考

探測點同 4c：resnet.0.weight、resnet.7.0.conv1.weight、
transformers.0.matching.embed_linear.weight。
"""
import sys, os, types
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from models.stage2_model import CNN_TRXWithRelation
from utils import loss as real_loss_fn

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
        tasks_per_batch=16,
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


def make_episode(seed):
    g = torch.Generator(device=device).manual_seed(seed)
    ctx = torch.randn(WAY * SHOT * SEQ, 3, 224, 224, device=device, generator=g)
    tgt = torch.randn(WAY * QPC  * SEQ, 3, 224, 224, device=device, generator=g)
    lbl = torch.arange(WAY, device=device)
    target_labels = torch.randint(0, WAY, (WAY * QPC,), device=device, generator=g)
    return ctx, tgt, lbl, target_labels


def run_once(model, ctx, tgt, lbl, target_labels):
    model.zero_grad()
    _amp = getattr(model.args, "amp", "off")
    if _amp == "off":
        out = model(ctx, lbl, tgt)["logits"]
        l = real_loss_fn(out, target_labels, device) / model.args.tasks_per_batch
        l.backward()
    else:
        dtype = torch.bfloat16 if _amp == "bf16" else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            out = model(ctx, lbl, tgt)["logits"]
            l = real_loss_fn(out, target_labels, device) / model.args.tasks_per_batch
        l.backward()
    grads = {name: p.grad.detach().clone().float()
             for name, p in model.named_parameters() if p.grad is not None}
    return l.item(), grads


def rel_err_and_cos(g_a, g_b):
    diff_norm = (g_a - g_b).norm().item()
    a_norm = g_a.norm().item()
    rel = diff_norm / a_norm if a_norm > 0 else float("nan")
    cos = torch.nn.functional.cosine_similarity(g_a.flatten(), g_b.flatten(), dim=0).item()
    return rel, cos


print("建立模型（fp32 基準模型，bf16/fp16 從它複製權重）...")
m_fp32 = build_model("off")
conv1_name, last_block_name, head_name = pick_probe_names(m_fp32)
print(f"探測參數：backbone第一層={conv1_name}  backbone最後一個block={last_block_name}  head第一層={head_name}")

m_bf16 = build_model("bf16")
m_fp16 = build_model("fp16")
m_bf16.load_state_dict(m_fp32.state_dict())
m_fp16.load_state_dict(m_fp32.state_dict())

ep1 = make_episode(seed=99)
ep2 = make_episode(seed=100)  # 不同的 episode（不同輸入+不同 target_labels）

loss_fp32_ep1_r1, g_fp32_ep1_r1 = run_once(m_fp32, *ep1)
loss_fp32_ep1_r2, g_fp32_ep1_r2 = run_once(m_fp32, *ep1)   # 同一個 episode 重跑
loss_fp32_ep2,    g_fp32_ep2    = run_once(m_fp32, *ep2)   # 換一個 episode
loss_bf16_ep1,    g_bf16_ep1    = run_once(m_bf16, *ep1)
loss_fp16_ep1,    g_fp16_ep1    = run_once(m_fp16, *ep1)

print(f"\nloss（真正的分類 loss，utils.loss()）：")
print(f"  fp32(ep1)重跑1 = {loss_fp32_ep1_r1:.6f}")
print(f"  fp32(ep1)重跑2 = {loss_fp32_ep1_r2:.6f}")
print(f"  fp32(ep2)      = {loss_fp32_ep2:.6f}")
print(f"  bf16(ep1)      = {loss_bf16_ep1:.6f}")
print(f"  fp16(ep1)      = {loss_fp16_ep1:.6f}")
print(f"\n|loss_fp32(ep1)重跑1 - 重跑2| = {abs(loss_fp32_ep1_r1-loss_fp32_ep1_r2):.6e}  (GPU 底線)")
print(f"|loss_fp32(ep1) - loss_fp32(ep2)| = {abs(loss_fp32_ep1_r1-loss_fp32_ep2):.6e}  (換episode)")
print(f"|loss_bf16(ep1) - loss_fp32(ep1)| = {abs(loss_bf16_ep1-loss_fp32_ep1_r1):.6e}  (AMP bf16偏差)")
print(f"|loss_fp16(ep1) - loss_fp32(ep1)| = {abs(loss_fp16_ep1-loss_fp32_ep1_r1):.6e}  (AMP fp16偏差)")

probe_names = {"backbone第一層conv": conv1_name, "backbone最後一個block": last_block_name, "head第一層Linear": head_name}

comparisons = [
    ("fp32(ep1)重跑1 vs fp32(ep1)重跑2  [GPU底線]", g_fp32_ep1_r1, g_fp32_ep1_r2),
    ("fp32(ep1) vs fp32(ep2)  [換episode/SGD雜訊尺度]", g_fp32_ep1_r1, g_fp32_ep2),
    ("fp16(ep1) vs fp32(ep1)  [AMP偏差]", g_fp32_ep1_r1, g_fp16_ep1),
    ("bf16(ep1) vs fp32(ep1)  [參考]", g_fp32_ep1_r1, g_bf16_ep1),
]

print("\n=== 梯度偏差 ===")
for label, pname in probe_names.items():
    print(f"\n[{label}] ({pname})")
    for tag, g_a, g_b in comparisons:
        if pname not in g_a or pname not in g_b:
            print(f"  {tag}: 該參數沒有梯度")
            continue
        rel, cos = rel_err_and_cos(g_a[pname], g_b[pname])
        print(f"  {tag}:")
        print(f"    ||g_a-g_b||/||g_a|| = {rel:.4e}   cosine similarity = {cos:.8f}")

print("\n（描述性結果，不是驗收——重點是把 AMP 偏差跟「換 episode 的雜訊尺度」放在同一把尺上比）")
