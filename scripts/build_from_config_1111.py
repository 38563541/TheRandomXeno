"""1111：在指定的程式樹上，用該樹自己的 run.py 參數解析建模型（CPU），輸出 state_dict 鍵名／形狀，
可選擇做一次固定 seed 的前向並把 logits 存檔。用法：
  python build_from_config_1111.py <tree> <config> <out.pt> [--forward] [--load ckpt]
每棵樹各跑一個 subprocess，避免同一個 process 重複 import 不同版本的模組。"""
import os, sys, tempfile
tree, cfg, out = sys.argv[1], sys.argv[2], sys.argv[3]
do_fwd = "--forward" in sys.argv
ckpt = sys.argv[sys.argv.index("--load") + 1] if "--load" in sys.argv else None
os.chdir(tree)
sys.path.insert(0, tree)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import run
tmp = tempfile.mkdtemp(prefix="b1111_")
sys.argv = ["run.py", "--config", cfg, "--dataset", "hmdb", "--split", "3", "-c", tmp]
args = run.Learner.parse_command_line(None)
from model import CNN_TRX
torch.manual_seed(0)
m = CNN_TRX(args).eval()
res = {"keys": {k: tuple(v.shape) for k, v in m.state_dict().items()},
       "inter_style": getattr(m.transformers[0], "inter_style", None),
       "module_file": sys.modules["model"].__file__}
if ckpt:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    r = m.load_state_dict(sd, strict=True)
    res["load"] = f"strict=True OK missing={list(r.missing_keys)} unexpected={list(r.unexpected_keys)}"
if do_fwd:
    torch.manual_seed(1)
    n_s, n_q, L = args.way * args.shot, args.way * 1, args.seq_len
    sup = torch.rand(n_s * L, 3, 224, 224)
    qry = torch.rand(n_q * L, 3, 224, 224)
    lbl = torch.arange(args.way).repeat_interleave(args.shot)
    with torch.no_grad():
        res["logits"] = m(sup, lbl, qry)["logits"]
torch.save(res, out)
print("OK", cfg, "inter_style=", res["inter_style"], "n_keys=", len(res["keys"]),
      "logits=", tuple(res["logits"].shape) if do_fwd else None, res.get("load", ""))
