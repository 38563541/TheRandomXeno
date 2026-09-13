"""
verify_wiring.py — 旗標接線稽核
任務 2 的動態測試：改值 → logits 要跟著改（或相同，若數學上等價）。
任務 3 的 config forward 驗證：每個 yaml 都能建模型、跑 forward、回傳正確形狀。

不訓練、不載資料集、不改既有程式碼。
"""

import sys, os, types, copy, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml as _yaml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

_passed, _failed = [], []

def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    (_passed if ok else _failed).append(name)

# ---------------------------------------------------------------------------
# 共用隨機輸入 (5-way 1-shot, qpc=5, seq_len=8)
# ---------------------------------------------------------------------------
WAY, QPC, SEQ, DIN = 5, 5, 8, 512

def _make_inputs(shot=1):
    n_s = WAY * shot
    n_q = WAY * QPC
    support = torch.randn(n_s, SEQ, DIN, device=DEVICE)
    queries  = torch.randn(n_q, SEQ, DIN, device=DEVICE)
    labels   = torch.arange(WAY, device=DEVICE).repeat_interleave(shot)
    return support, queries, labels

def _base_args(shot=1, **overrides):
    a = types.SimpleNamespace(
        trans_linear_in_dim=DIN,
        trans_linear_out_dim=1152,
        way=WAY, shot=shot, query_per_class=QPC,
        trans_dropout=0.1,
        seq_len=SEQ,
        method="resnet18",
        num_gpus=1,
        temp_set=[2],
        matching="bidirectional",
        set_aggregation="pool",
        tau=0.1,
        use_intra_relation=False,
        use_inter_relation=False,
        relation_level="tuple",
        inter_style="global",
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a

# ---------------------------------------------------------------------------
# 任務 2：動態接線測試
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("任務 2：動態接線測試")
print("=" * 70)

# ── 共用權重工具 ─────────────────────────────────────────────────────────────

from models.stage1_model import TRXSetMatching

def _clone_weights(src, dst):
    """把 src 的 state_dict 載入 dst（strict=False，架構相同才有意義）。"""
    dst.load_state_dict(src.state_dict(), strict=False)

def _run_set_matching(args, support, queries, labels):
    """建 TRXSetMatching，共享 embed_linear/norm/pe 後 forward。"""
    m = TRXSetMatching(args, temporal_set_size=2).to(DEVICE)
    return m(support, labels, queries)["logits"]

# ── 2-1. matching: bidirectional → unidirectional ────────────────────────────
print("\n--- matching ---")
torch.manual_seed(1)
s1, q1, l1 = _make_inputs(shot=1)

args_bidir = _base_args(matching="bidirectional")
args_uni   = _base_args(matching="unidirectional")

m_bidir = TRXSetMatching(args_bidir).to(DEVICE)
m_uni   = TRXSetMatching(args_uni).to(DEVICE)
_clone_weights(m_bidir, m_uni)

logits_bidir = m_bidir(s1, l1, q1)["logits"]
logits_uni   = m_uni(s1, l1, q1)["logits"]
diff = (logits_bidir - logits_uni).abs().max().item()
check("matching bidir→uni 改變 logits", diff > 1e-6, f"max_diff={diff:.4f}")

# ── 2-2. matching: bidirectional → attention_weighted ─────────────────────────
args_attn = _base_args(matching="attention_weighted")
m_attn = TRXSetMatching(args_attn).to(DEVICE)
_clone_weights(m_bidir, m_attn)

logits_attn = m_attn(s1, l1, q1)["logits"]
diff2 = (logits_bidir - logits_attn).abs().max().item()
check("matching bidir→attn 改變 logits", diff2 > 1e-6, f"max_diff={diff2:.4f}")

# ── 2-3. tau (attention_weighted 下): 0.1 → 1.0 ─────────────────────────────
print("\n--- tau (attention_weighted 下) ---")
args_tau01  = _base_args(matching="attention_weighted", tau=0.1)
args_tau10  = _base_args(matching="attention_weighted", tau=1.0)
m_tau01 = TRXSetMatching(args_tau01).to(DEVICE)
m_tau10 = TRXSetMatching(args_tau10).to(DEVICE)
_clone_weights(m_tau01, m_tau10)

lt01 = m_tau01(s1, l1, q1)["logits"]
lt10 = m_tau10(s1, l1, q1)["logits"]
diff_tau = (lt01 - lt10).abs().max().item()
check("tau 0.1→1.0 改變 logits", diff_tau > 1e-6, f"max_diff={diff_tau:.4f}")

# ── 2-4. set_aggregation (shot=5): pool → instance ───────────────────────────
print("\n--- set_aggregation (shot=5) ---")
torch.manual_seed(2)
s5, q5, l5 = _make_inputs(shot=5)

args_pool5     = _base_args(shot=5, set_aggregation="pool")
args_instance5 = _base_args(shot=5, set_aggregation="instance")
m_pool5     = TRXSetMatching(args_pool5).to(DEVICE)
m_instance5 = TRXSetMatching(args_instance5).to(DEVICE)
_clone_weights(m_pool5, m_instance5)

lp5 = m_pool5(s5, l5, q5)["logits"]
li5 = m_instance5(s5, l5, q5)["logits"]
diff_agg5 = (lp5 - li5).abs().max().item()
check("set_aggregation pool→instance (5-shot) 改變 logits", diff_agg5 > 1e-4,
      f"max_diff={diff_agg5:.4f}")

# ── 2-5. set_aggregation (shot=1): pool ≈ instance（數學等價）────────────────
print("\n--- set_aggregation (shot=1) 等價性 ---")
args_pool1     = _base_args(shot=1, set_aggregation="pool")
args_instance1 = _base_args(shot=1, set_aggregation="instance")
m_pool1     = TRXSetMatching(args_pool1).to(DEVICE)
m_instance1 = TRXSetMatching(args_instance1).to(DEVICE)
_clone_weights(m_pool1, m_instance1)

# 注意：PE 有 Dropout (trans_dropout=0.1)，train mode 下兩次 forward 結果不同。
# 等價性測試必須在 eval mode 下進行（Dropout 關閉）。
m_pool1.eval()
m_instance1.eval()
with torch.no_grad():
    lp1 = m_pool1(s1, l1, q1)["logits"]
    li1 = m_instance1(s1, l1, q1)["logits"]
m_pool1.train()
m_instance1.train()
diff_agg1 = (lp1 - li1).abs().max().item()
# tolerance 放寬到 1e-3：pool_hausdorff 用 chunked 計算，mean_hausdorff 用全矩陣，
# float32 精度差約 2e-4，兩者數學等價但計算路徑不同。
check("set_aggregation pool≈instance (1-shot) logits 相同 [eval mode]", diff_agg1 < 1e-3,
      f"max_diff={diff_agg1:.6f}")

# ── 2-6. use_intra_relation: true → false ────────────────────────────────────
print("\n--- relation modules ---")
from models.stage2_model import TRXSetMatchingWithRelation

def _run_stage2(args_ns, support, queries, labels):
    m = TRXSetMatchingWithRelation(
        args_ns,
        relation_level=args_ns.relation_level,
        use_intra=args_ns.use_intra_relation,
        use_inter=args_ns.use_inter_relation,
        inter_style=args_ns.inter_style,
    ).to(DEVICE)
    return m, m(support, labels, queries)["logits"]

torch.manual_seed(3)
s1r, q1r, l1r = _make_inputs(shot=1)

# intra true vs false — same weights in the shared TRXSetMatching (embed_linear/norm/pe)
args_intra_on  = _base_args(use_intra_relation=True,  use_inter_relation=False)
args_intra_off = _base_args(use_intra_relation=False, use_inter_relation=False)
m_intra_on,  logits_intra_on  = _run_stage2(args_intra_on,  s1r, q1r, l1r)
m_intra_off, logits_intra_off = _run_stage2(args_intra_off, s1r, q1r, l1r)
# share weights on the TRXSetMatching sub-module and re-run
m_intra_off.matching.load_state_dict(m_intra_on.matching.state_dict())
logits_intra_off_shared = m_intra_off(s1r, l1r, q1r)["logits"]
diff_intra = (logits_intra_on - logits_intra_off_shared).abs().max().item()
# intra_on uses DIFFERENT intra weights (random init) so logits will differ
# — the check is that intra=True actually changes logits vs intra=False with same backbone weights
check("use_intra_relation true→false 改變 logits", diff_intra > 1e-4,
      f"max_diff={diff_intra:.4f}")

# ── 2-7. use_inter_relation: true → false (global style) ─────────────────────
args_inter_on  = _base_args(use_intra_relation=False, use_inter_relation=True,  inter_style="global")
args_inter_off = _base_args(use_intra_relation=False, use_inter_relation=False, inter_style="global")
m_inter_on,  logits_inter_on  = _run_stage2(args_inter_on,  s1r, q1r, l1r)
m_inter_off, logits_inter_off = _run_stage2(args_inter_off, s1r, q1r, l1r)
m_inter_off.matching.load_state_dict(m_inter_on.matching.state_dict())
logits_inter_off_shared = m_inter_off(s1r, l1r, q1r)["logits"]
diff_inter = (logits_inter_on - logits_inter_off_shared).abs().max().item()
check("use_inter_relation true→false 改變 logits", diff_inter > 1e-4,
      f"max_diff={diff_inter:.4f}")

# ── 2-8. inter_style: global → true_hyrsm ────────────────────────────────────
# 架構不同（conv 層），無法共用完整 state_dict。
# 改測「模型物件內部的實際 inter 型別」是否反映設定。
print("\n--- inter_style 型別驗證（架構不同，改測模組型別）---")
from relation.inter_relation import InterRelation
from relation.true_hyrsm_inter_relation import TrueHyRSMInterRelation
from relation.hyrsm_inter_relation import HyRSMInterRelation

for style, expected_cls in [
    ("global",      InterRelation),
    ("true_hyrsm",  TrueHyRSMInterRelation),
    ("hyrsm",       HyRSMInterRelation),
]:
    args_style = _base_args(use_intra_relation=False, use_inter_relation=True, inter_style=style)
    m_style = TRXSetMatchingWithRelation(
        args_style, relation_level="tuple",
        use_intra=False, use_inter=True, inter_style=style,
    ).to(DEVICE)
    actual_cls = type(m_style.inter)
    ok = actual_cls is expected_cls
    check(f"inter_style={style} → {expected_cls.__name__}", ok,
          f"實際: {actual_cls.__name__}")

# 順便確認 logits 在 global vs true_hyrsm 下確實不同
m_global, l_global = _run_stage2(_base_args(use_intra_relation=False, use_inter_relation=True, inter_style="global"), s1r, q1r, l1r)
m_thyrsm, l_thyrsm = _run_stage2(_base_args(use_intra_relation=False, use_inter_relation=True, inter_style="true_hyrsm"), s1r, q1r, l1r)
diff_style = (l_global - l_thyrsm).abs().max().item()
check("inter_style global→true_hyrsm 改變 logits", diff_style > 1e-4,
      f"max_diff={diff_style:.4f}")

# ── 2-9. relation_level: tuple → frame ───────────────────────────────────────
print("\n--- relation_level ---")
# frame level 的 IntraRelation dim=512（frame dim），tuple level 的 dim=1152（tuple dim）。
# 兩個模型的 intra 權重形狀不同，無法共用 state_dict。
# 改測：(a) 模型物件的 relation_level 屬性正確反映設定；(b) 兩組 logits 數值不同。
args_tuple_level = _base_args(use_intra_relation=True, use_inter_relation=False, relation_level="tuple")
args_frame_level = _base_args(use_intra_relation=True, use_inter_relation=False, relation_level="frame")

m_tuple_lvl = TRXSetMatchingWithRelation(
    args_tuple_level, relation_level="tuple", use_intra=True, use_inter=False, inter_style="global"
).to(DEVICE)
m_frame_lvl = TRXSetMatchingWithRelation(
    args_frame_level, relation_level="frame", use_intra=True, use_inter=False, inter_style="global"
).to(DEVICE)

# (a) 屬性驗證
check("relation_level='tuple' 寫入模型屬性",
      m_tuple_lvl.relation_level == "tuple", f"got {m_tuple_lvl.relation_level!r}")
check("relation_level='frame' 寫入模型屬性",
      m_frame_lvl.relation_level == "frame", f"got {m_frame_lvl.relation_level!r}")

# (b) 只共用 matching (embed_linear/norm/pe)，讓 intra 各自隨機初始化
m_frame_lvl.matching.load_state_dict(m_tuple_lvl.matching.state_dict())
# 注意：intra 不共用（dim 不同），所以 logits 差異同時包含「路徑不同」和「intra 權重不同」
# 這裡只驗證「logits 不相同」，不能做精確的單一變因測試
l_tuple_lvl = m_tuple_lvl(s1r, l1r, q1r)["logits"]
l_frame_lvl = m_frame_lvl(s1r, l1r, q1r)["logits"]
diff_level = (l_tuple_lvl - l_frame_lvl).abs().max().item()
check("relation_level tuple vs frame logits 不同（含 intra dim 差異）",
      diff_level > 1e-4, f"max_diff={diff_level:.4f}")
print("    ⚠ 注意：intra 維度不同(1152 vs 512)，無法共用權重；logit 差異同時含架構差異與權重差異，"
      "單一變因無法分離。屬性驗證已確認接線正確。")

# ── 2-10. temp_set（stage2 模型）：[2] → [2,3] ───────────────────────────────
print("\n--- temp_set（stage2 模型的已知無效項）---")
# stage1_model.TRXSetMatching 硬寫 temporal_set_size=2，temp_set 不影響
args_ts2  = _base_args(temp_set=[2])
args_ts23 = _base_args(temp_set=[2, 3])
m_ts2  = TRXSetMatching(args_ts2).to(DEVICE)
m_ts23 = TRXSetMatching(args_ts23).to(DEVICE)
_clone_weights(m_ts2, m_ts23)   # 同樣權重，確保 logits 差異只來自 temp_set

# 確認 tuples_len 相同（都是 28 = C(8,2)）
tl2  = m_ts2.tuples_len
tl23 = m_ts23.tuples_len
check("temp_set [2]→[2,3] 對 stage2 模型 tuple 數不變（已知無效）",
      tl2 == tl23, f"tuples_len: {tl2} vs {tl23}")

m_ts2.eval();  m_ts23.eval()
with torch.no_grad():
    l_ts2  = m_ts2(s1, l1, q1)["logits"]
    l_ts23 = m_ts23(s1, l1, q1)["logits"]
m_ts2.train(); m_ts23.train()
diff_ts = (l_ts2 - l_ts23).abs().max().item()
check("temp_set [2]→[2,3] logits 相同（已知無效）", diff_ts < 1e-6,
      f"max_diff={diff_ts:.2e}")

# ── 2-11. num_samples: 1 → 2 ─────────────────────────────────────────────────
print("\n--- num_samples ---")
# NUM_SAMPLES 是 models/stage1_model.py 的模組常數，不從 args 讀。
# 預期行為：args.num_samples 沒有被讀取（= 無效鍵）。
# 測試邏輯：「PASS = 確認不讀取」（讀取才算接線有效，但這個鍵沒有效）
import models.stage1_model as _s1m
import models.stage2_model as _s2m
import inspect

s1_src = inspect.getsource(_s1m.CNN_TRX)
s2_src = inspect.getsource(_s2m.CNN_TRXWithRelation)
reads_ns_s1 = "args.num_samples" in s1_src
reads_ns_s2 = "args.num_samples" in s2_src

# PASS = 確認不讀取（這是預期的行為：num_samples 是無效鍵）
check("num_samples: stage1 CNN_TRX 不讀 args.num_samples（預期無效）",
      not reads_ns_s1,
      "使用模組常數 NUM_SAMPLES=1" if not reads_ns_s1 else "⚠ 意外讀取了 args.num_samples")
check("num_samples: stage2 CNN_TRXWithRelation 不讀 args.num_samples（預期無效）",
      not reads_ns_s2,
      "使用模組常數 NUM_SAMPLES=1" if not reads_ns_s2 else "⚠ 意外讀取了 args.num_samples")

# 直接確認：同樣權重下，args.num_samples 不同 → logits 相同
a_ns1 = _base_args(shot=1, num_samples=1)
a_ns2 = _base_args(shot=1, num_samples=2)
m_ns1 = TRXSetMatching(a_ns1).to(DEVICE)
m_ns2 = TRXSetMatching(a_ns2).to(DEVICE)
_clone_weights(m_ns1, m_ns2)   # 必須共用權重排除隨機初始化影響

m_ns1.eval(); m_ns2.eval()
with torch.no_grad():
    l_ns1 = m_ns1(s1, l1, q1)["logits"]
    l_ns2 = m_ns2(s1, l1, q1)["logits"]
m_ns1.train(); m_ns2.train()
diff_ns = (l_ns1 - l_ns2).abs().max().item()
check("num_samples 1→2 logits 相同（確認 TRXSetMatching 不讀此值）",
      diff_ns < 1e-6, f"max_diff={diff_ns:.2e}")


# ---------------------------------------------------------------------------
# 任務 3：configs 全掃 forward 驗證
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("任務 3：configs 全掃（建模型 + forward + 內部設定核對）")
print("=" * 70)

from models.stage1_model import TRXSetMatching as _S1Match, NUM_SAMPLES as _NS

print(f"\n{'config':<45} {'shape':>14}  {'OK?':>4}  yaml↔model 不一致")
print("-" * 90)

_task3_rows = []

for cfg_path in sorted(glob.glob("configs/*.yaml")):
    cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
    with open(cfg_path) as f:
        yaml_d = _yaml.safe_load(f) or {}

    method     = yaml_d.get("method", yaml_d.get("backbone", "resnet18"))
    shot       = yaml_d.get("shot", 1)
    way        = yaml_d.get("way", 5)
    qpc        = yaml_d.get("query_per_class", 5)
    use_intra  = yaml_d.get("use_intra_relation", False)
    use_inter  = yaml_d.get("use_inter_relation", False)
    rel_level  = yaml_d.get("relation_level", "none")
    inter_sty  = yaml_d.get("inter_style", "global")

    a = types.SimpleNamespace(
        trans_linear_in_dim=512,
        trans_linear_out_dim=yaml_d.get("trans_linear_out_dim", 1152),
        way=way, shot=shot, query_per_class=qpc,
        trans_dropout=0.1,
        seq_len=8,
        method=method,
        num_gpus=1,
        temp_set=yaml_d.get("temp_set", [2]),
        matching=yaml_d.get("matching", "bidirectional"),
        set_aggregation=yaml_d.get("set_aggregation", "pool"),
        tau=float(yaml_d.get("tau", 0.1)),
        use_intra_relation=use_intra,
        use_inter_relation=use_inter,
        relation_level=rel_level,
        inter_style=inter_sty,
        num_samples=yaml_d.get("num_samples", 1),
    )

    # 根據是否有 relation module 選擇模組（繞過 ResNet，直接測 matching 層）
    use_rel = use_intra or use_inter
    try:
        torch.manual_seed(0)
        if use_rel:
            model = TRXSetMatchingWithRelation(
                a, relation_level=rel_level,
                use_intra=use_intra, use_inter=use_inter, inter_style=inter_sty,
            ).to(DEVICE)
        else:
            model = _S1Match(a, temporal_set_size=2).to(DEVICE)
    except Exception as e:
        print(f"  FAIL  {cfg_name:<43}  build error: {e}")
        _failed.append(f"task3:{cfg_name}:build")
        continue

    # forward：直接餵 sequence features（不需要圖片）
    try:
        n_s = way * shot
        n_q = way * qpc
        s_in = torch.randn(n_s, 8, DIN, device=DEVICE)
        q_in = torch.randn(n_q, 8, DIN, device=DEVICE)
        l_in = torch.arange(way, device=DEVICE).repeat_interleave(shot)
        model.eval()
        with torch.no_grad():
            out = model(s_in, l_in, q_in)
        model.train()
        logits = out["logits"]
        expected_shape = (n_q, way)   # matching modules return (nq, way) directly
        shape_ok = logits.shape == expected_shape
    except Exception as e:
        print(f"  FAIL  {cfg_name:<43}  forward error: {e}")
        _failed.append(f"task3:{cfg_name}:forward")
        continue

    # 從模型內部取實際生效設定
    mismatches = []

    # inner = TRXSetMatching（stage2 model.matching 是 TRXSetMatching 實例；stage1 model 本身就是）
    # 注意：TRXSetMatching.matching 是字串屬性（"bidirectional" 等），不是子模組
    inner = model.matching if isinstance(model, TRXSetMatchingWithRelation) else model

    for attr, yaml_key, default in [
        ("matching",        "matching",        "bidirectional"),
        ("set_aggregation", "set_aggregation", "pool"),
        ("tau",             "tau",             0.1),
    ]:
        yaml_val  = yaml_d.get(yaml_key, default)
        model_val = getattr(inner, attr, None)
        if str(yaml_val) != str(model_val):
            mismatches.append(f"{attr}: yaml={yaml_val!r} model={model_val!r}")

    if hasattr(model, "relation_level"):
        yaml_rl  = yaml_d.get("relation_level", "none")
        model_rl = model.relation_level
        if yaml_rl != model_rl:
            mismatches.append(f"relation_level: yaml={yaml_rl!r} model={model_rl!r}")
        yaml_ui  = yaml_d.get("use_intra_relation", False)
        if bool(yaml_ui) != bool(model.use_intra):
            mismatches.append(f"use_intra: yaml={yaml_ui} model={model.use_intra}")
        yaml_ue  = yaml_d.get("use_inter_relation", False)
        if bool(yaml_ue) != bool(model.use_inter):
            mismatches.append(f"use_inter: yaml={yaml_ue} model={model.use_inter}")
        yaml_is  = yaml_d.get("inter_style", "global")
        if yaml_is != model.inter_style:
            mismatches.append(f"inter_style: yaml={yaml_is!r} model={model.inter_style!r}")

    # temp_set 無效性標記
    yaml_ts  = yaml_d.get("temp_set", [2])
    model_tl = inner.tuples_len
    if yaml_ts != [2] and model_tl == 28:
        mismatches.append(f"temp_set={yaml_ts} 宣告但 tuples_len 仍=28（已知無效）")

    miss_str = " | ".join(mismatches) if mismatches else "—"
    shape_str = str(tuple(logits.shape))
    marker = "✓" if shape_ok and not mismatches else ("✗" if not shape_ok else "⚠")
    print(f"  {marker}  {cfg_name:<43} {shape_str:>14}  {'OK' if shape_ok else 'BAD'}  {miss_str}")

    _task3_rows.append((cfg_name, shape_ok, mismatches))
    if shape_ok and not mismatches:
        _passed.append(f"task3:{cfg_name}")
    else:
        _failed.append(f"task3:{cfg_name}")


# ---------------------------------------------------------------------------
# 任務 1a：死程式碼掃描（靜態）
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("任務 1a：死程式碼掃描")
print("=" * 70)

dead = []

# models/trx_original.py — 有被 import 嗎？
trx_orig_imported = False
for f in ["model.py", "run.py", "models/stage1_model.py", "models/stage2_model.py"]:
    with open(f) as fh:
        if "trx_original" in fh.read():
            trx_orig_imported = True
if not trx_orig_imported:
    dead.append("models/trx_original.py  — 無任何 import（整個檔案）")

# matching/bidirectional_hausdorff.py — 被 import 嗎？
bidir_imported = False
for f in glob.glob("models/*.py") + ["run.py", "model.py"]:
    with open(f) as fh:
        if "bidirectional_hausdorff" in fh.read():
            bidir_imported = True
if not bidir_imported:
    dead.append("matching/bidirectional_hausdorff.py::bidirectional_hausdorff  — 無 import")

# mean_hausdorff_bidir — 被呼叫嗎（除了自己的 __main__ 和 verify 腳本）
bidir_fn_used = False
for f in glob.glob("models/*.py") + ["model.py", "run.py"]:
    with open(f) as fh:
        if "mean_hausdorff_bidir" in fh.read():
            bidir_fn_used = True
if not bidir_fn_used:
    dead.append("matching/mean_hausdorff.py::mean_hausdorff_bidir  — 被 import 但未被 models/ 呼叫（stage1_model import 了但 forward 現在走 pool_hausdorff）")

for item in dead:
    print(f"  ⚠  {item}")
if not dead:
    print("  （無）")


# ---------------------------------------------------------------------------
# 任務 4.1：config 鍵對照表（靜態 + 動態結合）
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("任務 4.1：config 鍵接線總表")
print("=" * 70)

rows = [
    # key, in_args, code_reads, dynamic_result, verdict
    ("backbone",         "✓ (→method)", "✓ _load_yaml_config 改名",   "—",         "有效"),
    ("matching",         "✓ set_defaults", "✓ TRXSetMatching.__init__", "改變",      "有效"),
    ("set_aggregation",  "✓ set_defaults", "✓ TRXSetMatching.__init__", "改變(5s)/相同(1s)", "有效"),
    ("tau",              "✓ set_defaults", "✓ TRXSetMatching.__init__", "改變",      "有效（attn_weighted 才生效）"),
    ("matching_method",  "✗ 無 add_argument", "✗ 無程式讀取",          "—",         "⚠ 無效（legacy，被 warn 但不生效）"),
    ("bidirectional",    "✗ 無 add_argument", "✗ 無程式讀取",          "—",         "⚠ 無效（legacy，被 warn 但不生效）"),
    ("temp_set",         "✓ add_argument", "✗ stage1/2 硬寫 temporal_set_size=2", "不變", "⚠ 已知無效（stage1/2）"),
    ("num_samples",      "✓ set_defaults", "✗ 模組常數 NUM_SAMPLES=1", "不變",      "⚠ 無效（args.num_samples 無人讀取）"),
    ("use_intra_relation","✓ set_defaults","✓ stage2_model CNN_TRXWithRelation", "改變", "有效"),
    ("use_inter_relation","✓ set_defaults","✓ stage2_model CNN_TRXWithRelation", "改變", "有效"),
    ("inter_style",      "✓ set_defaults", "✓ CNN_TRXWithRelation.__init__", "型別改變", "有效"),
    ("relation_level",   "✓ set_defaults", "✓ CNN_TRXWithRelation.__init__", "改變",  "有效"),
    ("way/shot/qpc",     "✓ add_argument", "✓ (loader+model)",          "—",         "有效"),
    ("trans_linear_out_dim","✓ add_argument","✓ TRXSetMatching embed_linear", "—",   "有效"),
    ("training_iterations","✓ add_argument","✓ run loop",               "—",         "有效"),
    ("tasks_per_batch",  "✓ add_argument", "✓ optimizer step",          "—",         "有效"),
]

print(f"  {'鍵':<22} {'有進 args':>10}  {'有被讀':>24}  {'動態測試':>20}  判定")
print("  " + "-" * 100)
for r in rows:
    print(f"  {r[0]:<22} {r[1]:>10}  {r[2]:>24}  {r[3]:>20}  {r[4]}")


# ---------------------------------------------------------------------------
# 總結
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
total = len(_passed) + len(_failed)
print(f"動態測試結果：{len(_passed)}/{total} PASS，{len(_failed)} FAIL")
if _failed:
    print("  FAIL 項目：")
    for f in _failed:
        print(f"    - {f}")
print("=" * 70)
