import torch
import numpy as np
import argparse
import os
import pickle
import csv
import yaml
import re
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion
from model import CNN_TRX
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import video_reader
import random
import datetime
import time


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def _load_yaml_config(path):
    """
    Load a YAML config file and return a flat dict suitable for
    parser.set_defaults(**config_dict).

    Alias handling:
        backbone -> method   (YAML uses 'backbone'; argparse uses 'method')

    All YAML keys not recognised by argparse are passed through silently
    so that stage2 keys (use_intra_relation, relation_level, …) are stored
    on args without needing argparse declarations at this stage.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # backbone is the human-readable alias; argparse uses --method
    if "backbone" in cfg and "method" not in cfg:
        cfg["method"] = cfg.pop("backbone")
    elif "backbone" in cfg:
        cfg.pop("backbone")   # method already specified; drop alias

    return cfg


# ---------------------------------------------------------------------------
# CSV result logger
# ---------------------------------------------------------------------------

_CSV_PATH = os.path.join(os.path.dirname(__file__), "experiments", "results", "results.csv")
_CSV_COLUMNS = [
    "timestamp", "config_file", "dataset", "split",
    "backbone", "temp_set", "matching", "set_aggregation", "tau",
    "relation_level", "use_intra_relation", "use_inter_relation", "inter_style",
    "decouple_gate", "decouple_mode", "intra_depth",
    "way", "shot", "query_per_class",
    "iteration", "mean_accuracy", "confidence_interval",
]

def _log_result_csv(args, iteration, mean_accuracy, confidence_interval):
    """
    Append one row to experiments/results/results.csv.
    Creates the file with a header row if it does not yet exist.

    Args:
        args               : parsed argparse namespace (provides dataset, split, etc.)
        iteration          : int — training iteration at which test was run
        mean_accuracy      : float — mean test accuracy (%)
        confidence_interval: float — 95% confidence interval (±%)
    """
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)
    write_header = not os.path.exists(_CSV_PATH)

    row = {
        "timestamp":           datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_file":         getattr(args, "config_file", "none"),
        "dataset":             args.dataset,
        "split":               args.split,
        "backbone":            getattr(args, "method", ""),
        "temp_set":            str(getattr(args, "temp_set", "")),
        "matching":            getattr(args, "matching", ""),
        "set_aggregation":     getattr(args, "set_aggregation", ""),
        "tau":                 getattr(args, "tau", ""),
        "relation_level":      getattr(args, "relation_level", ""),
        "use_intra_relation":  getattr(args, "use_intra_relation", ""),
        "use_inter_relation":  getattr(args, "use_inter_relation", ""),
        "inter_style":         getattr(args, "inter_style", ""),
        "decouple_gate":       getattr(args, "decouple_gate", ""),
        "decouple_mode":       getattr(args, "decouple_mode", ""),
        "intra_depth":         getattr(args, "intra_depth", 1),
        "way":                 args.way,
        "shot":                args.shot,
        "query_per_class":     getattr(args, "query_per_class", ""),
        "iteration":           iteration,
        "mean_accuracy":       round(float(mean_accuracy), 4),
        "confidence_interval": round(float(confidence_interval), 4),
    }

    with open(_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)





def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        # ------------------------------------------------------------------
        # Global seed (all four sources).  worker_init_fn propagates to
        # DataLoader workers so each worker gets a deterministic but distinct
        # seed derived from the global seed and the worker id.
        # ------------------------------------------------------------------
        if self.args.seed is not None:
            _s = self.args.seed
            random.seed(_s)
            np.random.seed(_s)
            torch.manual_seed(_s)
            torch.cuda.manual_seed_all(_s)
            print(f"[INFO] seed={_s} (random, numpy, torch, cuda)", flush=True)

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        self.writer = SummaryWriter()
        
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()

        self.vd = video_reader.VideoDataset(self.args)

        # worker_init_fn: give each worker a deterministic but distinct seed.
        # Only active when --seed is set; otherwise same behaviour as before.
        def _worker_init_fn(worker_id):
            base = self.args.seed if self.args.seed is not None else 0
            # Phase 0.3: multiply by 1000 to prevent seed collision across seeds.
            # base+worker_id caused seeds 42/43 workers 0-9 to overlap 9 of 10.
            seed_w = base * 1000 + worker_id
            random.seed(seed_w)
            np.random.seed(seed_w)
            torch.manual_seed(seed_w)

        _wifn = _worker_init_fn if self.args.seed is not None else None
        self.video_loader = torch.utils.data.DataLoader(
            self.vd, batch_size=1, num_workers=self.args.num_workers,
            worker_init_fn=_wifn)
        self.test_loader  = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=0)
        
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        
        # Phase 2.3: only collect parameters that require gradients.
        # When freeze_backbone=True this excludes backbone params (saves optimizer state).
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[INFO] optimizer 收了 {sum(p.numel() for p in trainable)/1e6:.2f} M 個可訓練參數",
              flush=True)
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(trainable, lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(trainable, lr=self.args.learning_rate)
        self.test_accuracies = TestAccuracies(self.test_set)
        
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def _log_mem_line(self, iteration, tag):
        """階段7 soak：印一行顯存快照。win_alloc/win_res 是自上次
        reset_peak_memory_stats() 以來的峰值（呼叫方決定要不要 reset，
        這裡不 reset）；cur_res 是呼叫當下的瞬時 reserved（絕對不 reset，
        看 C 類階梯用）；retries 是 allocator 重試次數，process 啟動以來
        累積，看碎片化用。回傳 (win_alloc, win_res) 供呼叫方更新全程峰值。"""
        _st = torch.cuda.memory_stats()
        _wa = torch.cuda.max_memory_allocated() / 1024**3
        _wr = torch.cuda.max_memory_reserved()  / 1024**3
        _cur = torch.cuda.memory_reserved() / 1024**3
        _retry = _st.get("num_alloc_retries", 0)
        print_and_log(self.logfile,
            "[mem]{} iter {:>6} win_alloc {:.3f} win_res {:.3f} cur_res {:.3f} retries {}".format(
                tag, iteration, _wa, _wr, _cur, _retry))
        return _wa, _wr

    def init_model(self):
        model = CNN_TRX(self.args)
        model = model.to(self.device)
        if self.args.num_gpus > 1:
            model.distribute_model()

        # Phase 2.4: freeze_backbone + grad_ckpt は無意味で遅くなるだけ → 無効化
        if getattr(self.args, "freeze_backbone", False) and getattr(self.args, "grad_ckpt", False):
            print("[WARN] freeze_backbone と grad_ckpt は同時に使えません；grad_ckpt を無効化します",
                  flush=True)
            self.args.grad_ckpt = False

        # ── BN momentum correction for grad_ckpt ─────────────────────────────
        # checkpoint_sequential re-runs forward for backward; BN in training
        # mode would update running stats twice per batch, doubling momentum
        # from 0.1 to ~0.19.  Correct to m = 1-sqrt(1-0.1) ≈ 0.0513 so that
        # one grad-ckpt cycle equals one normal cycle in expectation.
        #
        # bn_fix_scope="all" (舊行為): 套到所有 BN，包含 checkpoint_sequential
        # 尾段（不被重算）的 BN — 對那些層是錯的修正，因為它們每個 iteration
        # 只更新一次。bn_fix_scope="covered" (預設): 只套到真正被
        # checkpoint_sequential 重算的 chain[:n_ckpt] 段。
        #
        # ⚠️ 這裡「一次 iteration 更新幾次」跟 forward() 裡 support/query 各跑一次
        # backbone 是兩件不同的事，兩者疊在一起算才是每個 BN 真正的更新次數：
        #   - 未開 grad_ckpt: 每個 BN 每 iteration 更新 2 次（support 1 次 + query 1 次），
        #     這是 PyTorch resnet 的常態，不是 bug，momentum=0.1 是對這個常態算的。
        #   - 開 grad_ckpt 後: chain[:n_ckpt] 段的 BN 因為 checkpoint 重算，
        #     在 support 那次forward 內部就變成 2 次，query 那次forward 內部又 2 次，
        #     合計每 iteration 4 次；chain[n_ckpt:] 尾段不受影響，維持 2 次。
        #     這裡的修正只處理「checkpoint 重算造成的加倍」，不是「support/query
        #     造成的加倍」——後者從沒被修正過，也不需要被修正。
        #
        # ⚠️ 驗證這件事別用 register_forward_hook 數次數：non-reentrant
        # checkpoint 的 backward 重算不會觸發 forward hook（PyTorch 2.5.1 驗證過），
        # 但底層運算（含 BN running_mean 的更新副作用）確實有重跑。用 hook 計數
        # 量出來的「涵蓋段 vs 未涵蓋段呼叫次數一樣」是假的，不代表沒有加倍。
        # 若要驗證，直接比較 running_mean 位移，不要透過 hook：
        #   manual 2x forward（手動呼叫兩次 BN）        running_mean = [-0.0055, 0.0081, 0.0024, -0.0112]
        #   manual 1x forward（只呼叫一次）              running_mean = [-0.0029, 0.0042, 0.0013, -0.0059]
        #   checkpoint(fn, x) 一次 + backward()          running_mean = [-0.0055, 0.0081, 0.0024, -0.0112]
        # checkpoint 版本的數值跟「手動兩次」完全相同，證明重算確實讓底層計算多跑
        # 了一次，即使 hook 只回報呼叫一次。細節見 0918_階段4_等價性與BN修正.md。
        if getattr(self.args, "grad_ckpt", False):
            import math
            m = 1 - math.sqrt(1 - 0.1)
            scope = getattr(self.args, "bn_fix_scope", "covered")
            n_bn_total = sum(1 for mod in model.modules()
                              if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm))
            if scope == "all":
                n_bn = 0
                for mod in model.modules():
                    if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                        mod.momentum = m
                        n_bn += 1
                print(f"[INFO] grad_ckpt on (bn_fix_scope=all): adjusted momentum of "
                      f"{n_bn} BN layers to {m:.5f}", flush=True)
            else:
                chain = model._ckpt_chain()
                n = len(chain)
                _ckpt_prefix = getattr(self.args, "ckpt_prefix", None)
                if _ckpt_prefix:
                    # 0921：--ckpt_prefix 給了 N，涵蓋範圍就是 chain[:N] 本身，
                    # 不再用 checkpoint_sequential 的 segment_size*(segments-1) 公式。
                    n_ckpt = min(_ckpt_prefix, n)
                else:
                    segs = min(getattr(self.args, "ckpt_segments", 8) or 8, n)
                    segment_size = n // segs
                    n_ckpt = segment_size * (segs - 1)
                covered_ids = {id(sub) for blk in chain[:n_ckpt] for sub in blk.modules()
                               if isinstance(sub, torch.nn.modules.batchnorm._BatchNorm)}
                n_bn = 0
                for mod in model.modules():
                    if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and id(mod) in covered_ids:
                        mod.momentum = m
                        n_bn += 1
                print(f"[INFO] grad_ckpt on (bn_fix_scope=covered): adjusted momentum of "
                      f"{n_bn} / {n_bn_total} BN layers to {m:.5f} (chain[:{n_ckpt}] of {n})",
                      flush=True)

        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        return train_set, validation_set, test_set


    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        # ------------------------------------------------------------------
        # Config file (YAML).  CLI args override YAML values when both given.
        # ------------------------------------------------------------------
        parser.add_argument(
            "--config",
            default=None,
            metavar="PATH",
            help="Path to a YAML config file (e.g. configs/stage1_hausdorff.yaml). "
                 "Values in the file set argparse defaults; explicit CLI flags override them.",
        )

        # Pre-parse to extract --config before the full parse so we can call
        # set_defaults() with YAML values before argparse reads argv properly.
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument("--config", default=None)
        pre_args, _ = pre_parser.parse_known_args()

        yaml_cfg = {}
        if pre_args.config is not None:
            yaml_cfg = _load_yaml_config(pre_args.config)
        # NOTE: parser.set_defaults(**yaml_cfg) is called AFTER all add_argument()
        # calls below, so that YAML values correctly override argparse defaults.

        parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf"], default="ssv2", help="Dataset to use.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        
        parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=100020, help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--test_iters', nargs='+', type=int, help='iterations to test at. Default is for ssv2 otam split.', default=[75000])
        parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=1000, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=10, help="Num dataloader workers.")
        parser.add_argument("--method", choices=["resnet18", "resnet34", "resnet50"], default="resnet50", help="method")
        parser.add_argument("--trans_linear_out_dim", type=int, default=1152, help="Transformer linear_out_dim")
        parser.add_argument("--trans_linear_in_dim", type=int, default=-1, help="Transformer linear_in_dim (default: auto from backbone: 512 for resnet18/34, 2048 for resnet50)")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
        parser.add_argument("--save_freq", type=int, default=5000, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument('--temp_set', nargs='+', type=int, help='cardinalities e.g. 2,3 is pairs and triples', default=[2,3])
        parser.add_argument("--scratch", type=str, default=os.path.expanduser("~/trx_data"),
                    help="Root dir containing video_datasets/{data,splits} and checkpoints")

        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")
        parser.add_argument("--split", type=int, default=7, help="Dataset split.")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        parser.add_argument('--seed', type=int, default=None,
                            help='Global random seed for reproducibility '
                                 '(random, numpy, torch, cuda). None = no seeding.')
        parser.add_argument('--grad_ckpt', action='store_true', default=False,
                            help='Gradient checkpointing on the ResNet backbone. '
                                 'Uses use_reentrant=False; ReLU inplace is disabled in __init__ '
                                 'and BN momentum is corrected in init_model().')
        parser.add_argument('--ckpt_segments', type=int, default=8,
                            help='checkpoint_sequential 段數（攤平後的 block 數上限）。'
                                 '8 是 Colab 實測在不同切法之間的最佳點：RN50 從 16.15 GB 降到 '
                                 '11.01 GB，時間只多 3%%（這個 3%% 講的是切法之間的差，不是開/關 '
                                 'ckpt 的差）。開 ckpt 本身的純 GPU 時間代價：本機 3080 Ti 用 '
                                 '--profile_time 量過 RN34（唯一有裸跑對照組的 backbone），扣掉 '
                                 'data wait 後，backbone+head+backward+step 合計裸跑 377.0 ms、'
                                 'flat8 444.5 ms，多 17.9%%（含 data wait 的 wall clock 差是 16.1%%，'
                                 '兩者一致）。RN50 沒有裸跑對照組（裸跑就 OOM），無法直接量出 '
                                 'RN50 開 ckpt 的時間代價，只能引用 RN34 的量測結果。'
                                 '0 = 不限制（等同 len(chain)）。'
                                 '若同時給了 --ckpt_prefix，這個 flag 改為「前綴內要切幾段」，'
                                 '不再是 checkpoint_sequential 的段數。')
        parser.add_argument('--ckpt_prefix', type=int, default=None,
                            help='0921 新增：恰好 checkpoint chain[:N] 這 N 個 module，其餘'
                                 '（chain[N:]）直接跑，不 checkpoint。跟 --ckpt_segments 合用時，'
                                 'segments 決定 N 個 module 內部要切幾組（純顯存旋鈕，不影響時間'
                                 '——時間代價只看 N）。不給這個 flag 時行為與舊的 '
                                 'checkpoint_sequential(chain, segments, x) 路徑完全相同，'
                                 '預設 None 不啟用。')
        parser.add_argument('--freeze_backbone', action='store_true', default=False,
                            help='Freeze the ResNet backbone: no grads, BN in eval mode. '
                                 'Used for the frozen second track.')
        parser.add_argument('--profile_memory', action='store_true', default=False,
                            help='每 2*tasks_per_batch 個 iteration 記錄顯存峰值並重置統計')
        parser.add_argument('--bn_fix_scope', choices=['all', 'covered'], default='covered',
                            help='grad_ckpt 開啟時 BN momentum 修正的套用範圍。'
                                 '"all"＝套到所有 BN（舊行為，對 checkpoint_sequential 尾段是錯的）；'
                                 '"covered"＝只套到真正被重算的 chain[:n_ckpt] 段（預設，正確行為）。')
        parser.add_argument('--loss_csv', default=None,
                            help='指定路徑時，逐 iteration 把 (iteration, loss) 寫入這個 csv。'
                                 '預設 None＝不寫，行為與現在完全相同。')
        parser.add_argument('--profile_time', action='store_true', default=False,
                            help='把每個 iteration 拆成 data wait / backbone fwd（support+query 合計）'
                                 '/ head fwd / backward+step 四段，用 cuda.Event 量 GPU 段落、'
                                 'perf_counter 量 dataloader 等待，跑完整個 run 後印中位數報告。')

        # decouple_gate / decouple_mode — SupportDecoupleRelation args.
        # Use mutually exclusive group for gate so YAML "decouple_gate: false"
        # works correctly.  str2bool does not exist in this repo → do NOT use it.
        _dg = parser.add_mutually_exclusive_group()
        _dg.add_argument("--decouple_gate",    dest="decouple_gate",
                         action="store_true",  help="Enable cosine gate in SupportDecoupleRelation.")
        _dg.add_argument("--no_decouple_gate", dest="decouple_gate",
                         action="store_false", help="Disable cosine gate.")
        parser.set_defaults(decouple_gate=True)
        parser.add_argument("--decouple_mode", type=str, default="remove",
                            choices=["remove", "inject"],
                            help="Decouple mode: 'remove' subtracts other-class signal, "
                                 "'inject' adds it (ablation only).")
        parser.add_argument("--intra_depth", type=int, default=1,
                            help="number of stacked IntraRelation blocks")

        # ------------------------------------------------------------------
        # YAML config key validation — runs after all add_argument() calls
        # so that `known` reflects every declared argument.
        # ------------------------------------------------------------------
        if yaml_cfg:
            known = {a.dest for a in parser._actions}

            # Stage 2 keys intentionally passed through without add_argument.
            # ⚠️  Every new stage2 flag must be listed here (e.g. decouple_gate,
            #     decouple_mode) so it is not mis-classified as unknown.
            # Note: 'backbone' is NOT listed here — _load_yaml_config already
            #       consumes it (aliasing backbone→method and popping the key)
            #       before this guard runs, so it will never appear in yaml_cfg.
            passthrough = {
                "matching", "set_aggregation", "tau",
                "use_intra_relation", "use_inter_relation",
                "relation_level", "inter_style",
            }

            # Keys confirmed to be read by no model code.
            # They appear in all 17 configs but are harmless.
            # Kept as deprecated (not unknown) so the label is accurate,
            # and flagged every run as a reminder to clean them up.
            deprecated = {"num_samples", "matching_method", "bidirectional"}

            unknown = set(yaml_cfg) - known - passthrough - deprecated
            if unknown:
                print(f"[WARN] config 有 {len(unknown)} 個未知的鍵，不會生效: "
                      f"{sorted(unknown)}", flush=True)
            hit = deprecated & set(yaml_cfg)
            if hit:
                print(f"[WARN] config 有廢棄鍵 {sorted(hit)}"
                      f"（程式碼不讀取，對模型無作用）", flush=True)

        # Apply YAML config defaults AFTER all add_argument() calls so that
        # YAML values override the argparse-level defaults (not the other way round).
        if yaml_cfg:
            parser.set_defaults(**yaml_cfg)

        args = parser.parse_args()

        # Store config file path on args for use by STEP 5 CSV logger.
        args.config_file = args.config if args.config is not None else "none"

        #if args.scratch == "bc":
        #    args.scratch = "/mnt/storage/home/tp8961/scratch"
        #elif args.scratch == "bp":
        #    args.num_gpus = 4
        #    # this is low becuase of RAM constraints for the data loader
        #    args.num_workers = 3
        #    args.scratch = "/work/tp8961"
        
        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)
        # ===== 自動加時間戳子目錄，或 resume 時找最新子目錄 =====
        if args.resume_from_checkpoint:
            # Find the most-recently-modified subdirectory to resume from.
            base = args.checkpoint_dir
            if not os.path.isdir(base):
                print(f"Can't resume: checkpoint_dir does not exist: {base}")
                exit(1)
            subdirs = sorted(
                [d for d in os.listdir(base)
                 if os.path.isdir(os.path.join(base, d))],
                reverse=True,
            )
            if not subdirs:
                print(f"Can't resume: no subdirectories found in {base}")
                exit(1)
            args.checkpoint_dir = os.path.join(base, subdirs[0])
            print(f"[resume] Resolved checkpoint directory: {args.checkpoint_dir}", flush=True)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.checkpoint_dir = os.path.join(
                args.checkpoint_dir,
                f"{args.dataset}_split{args.split}_{timestamp}"
            )
        # ======================================================
        if (args.method == "resnet50") or (args.method == "resnet34"):
            args.img_size = 224
        if args.trans_linear_in_dim == -1:  # not explicitly provided — auto-set from backbone
            if args.method == "resnet50":
                args.trans_linear_in_dim = 2048
            else:
                args.trans_linear_in_dim = 512
        
        if args.dataset == "ssv2":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/somethingsomethingv2TrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/somethingsomethingv2_256x256q5_7l8.zip")
        elif args.dataset == "kinetics":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/kineticsTrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/kinetics_256q5_1.zip")
        elif args.dataset == "ucf":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/ucfTrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/UCF-101_320.zip")
        elif args.dataset == "hmdb":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/hmdb_ARN")
            args.path = os.path.join(args.scratch, "video_datasets/data/hmdb51_256q5.zip")

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:

                # ------------------------------------------------------------------
                # Test-only mode: load checkpoint, evaluate once, log to CSV, exit.
                # ------------------------------------------------------------------
                if self.args.test_model_path is not None:
                    ckpt_path = self.args.test_model_path
                    checkpoint = torch.load(ckpt_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    m = re.search(r'(\d+)', os.path.basename(ckpt_path))
                    iteration = int(m.group(1)) if m else 0
                    print(f"[test-only] Loaded {ckpt_path}  iteration={iteration}", flush=True)
                    accuracy_dict = self.test(session)
                    if getattr(self.args, "profile_memory", False):
                        print_and_log(self.logfile, "[mem] EVAL PEAK  alloc {:.3f} GB  reserved {:.3f} GB".format(
                            torch.cuda.max_memory_allocated() / 1024**3,
                            torch.cuda.max_memory_reserved()  / 1024**3))
                    print(accuracy_dict)
                    self.test_accuracies.print(self.logfile, accuracy_dict)
                    _item = self.args.dataset
                    if _item in accuracy_dict:
                        if getattr(self.args, "profile_memory", False):
                            print_and_log(self.logfile, "[mem] profiling run — 跳過 results.csv 寫入")
                        else:
                            _log_result_csv(
                                self.args,
                                iteration=iteration,
                                mean_accuracy=accuracy_dict[_item]["accuracy"],
                                confidence_interval=accuracy_dict[_item]["confidence"],
                            )
                    self.logfile.close()
                    return
                # ------------------------------------------------------------------

                train_accuracies = []
                losses = []
                total_iterations = self.args.training_iterations

                _loss_csv_f = None
                if getattr(self.args, "loss_csv", None):
                    _loss_csv_f = open(self.args.loss_csv, "w", buffering=1)
                    _loss_csv_f.write("iteration,loss\n")

                _prof_time = getattr(self.args, "profile_time", False)
                _prof_rows = [] if _prof_time else None

                iteration = self.start_iteration
                _loader_iter = iter(self.video_loader)
                while True:
                    # 逐行對應原本 `for task_dict in self.video_loader:` 的行為：
                    # 先取一筆（不管要不要用），再判斷是否該停——flag 關閉時跟原本
                    # 逐位元組相同，只是把隱式的 for-iterator 換成顯式 next() 好包計時。
                    if _prof_time:
                        _iter_t0 = time.perf_counter()
                    try:
                        task_dict = next(_loader_iter)
                    except StopIteration:
                        break
                    if iteration >= total_iterations:
                        break
                    if _prof_time:
                        _data_ms = (time.perf_counter() - _iter_t0) * 1000
                    iteration += 1
                    torch.set_grad_enabled(True)

                    task_loss, task_accuracy = self.train_task(task_dict)
                    if _loss_csv_f is not None:
                        _loss_csv_f.write(f"{iteration},{task_loss.item():.10f}\n")
                    train_accuracies.append(task_accuracy)
                    losses.append(task_loss)

                    # optimize
                    _step_ms = 0.0
                    if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                        if _prof_time:
                            _es0 = torch.cuda.Event(enable_timing=True)
                            _es1 = torch.cuda.Event(enable_timing=True)
                            _es0.record()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            _es1.record()
                            torch.cuda.synchronize()
                            _step_ms = _es0.elapsed_time(_es1)
                        else:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    self.scheduler.step()

                    if _prof_time:
                        _iter_total_ms = (time.perf_counter() - _iter_t0) * 1000
                        _prof_rows.append(dict(
                            iteration=iteration,
                            data_ms=_data_ms,
                            h2d_ms=getattr(self, "_prof_h2d_ms", float("nan")),
                            backbone_ms=getattr(self.model, "_prof_backbone_ms", float("nan")),
                            head_ms=getattr(self.model, "_prof_head_ms", float("nan")),
                            backward_step_ms=getattr(self, "_prof_backward_ms", float("nan")) + _step_ms,
                            total_ms=_iter_total_ms,
                        ))

                    if getattr(self.args, "profile_memory", False):
                        _win = 100  # soak (階段7)：每 100 iteration 取樣一次
                        if (iteration + 1) % _win == 0:
                            _wa, _wr = self._log_mem_line(iteration + 1, "")
                            self._mem_peak_alloc = max(getattr(self, "_mem_peak_alloc", 0.0), _wa)
                            self._mem_peak_res   = max(getattr(self, "_mem_peak_res",   0.0), _wr)
                            torch.cuda.reset_peak_memory_stats()

                    if (iteration + 1) % self.args.print_freq == 0:
                        # print training stats
                        print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                      .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                              torch.Tensor(train_accuracies).mean().item()))
                        train_accuracies = []
                        losses = []

                    if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                        self.save_checkpoint(iteration + 1)


                    if ((iteration + 1) in self.args.test_iters) and (iteration + 1) != total_iterations:
                        if getattr(self.args, "profile_memory", False):
                            self._log_mem_line(iteration + 1, "[pre-eval]")
                        accuracy_dict = self.test(session)
                        if getattr(self.args, "profile_memory", False):
                            self._log_mem_line(iteration + 1, "[post-eval]")
                        print(accuracy_dict)
                        self.test_accuracies.print(self.logfile, accuracy_dict)
                        # --- CSV logging (STEP 5) ---
                        _item = self.args.dataset
                        if _item in accuracy_dict:
                            if getattr(self.args, "profile_memory", False):
                                print_and_log(self.logfile, "[mem] profiling run — 跳過 results.csv 寫入")
                            else:
                                _log_result_csv(
                                    self.args,
                                    iteration=iteration + 1,
                                    mean_accuracy=accuracy_dict[_item]["accuracy"],
                                    confidence_interval=accuracy_dict[_item]["confidence"],
                                )

                if getattr(self.args, "profile_memory", False):
                    print_and_log(self.logfile, "[mem] RUN PEAK  alloc {:.3f} GB  reserved {:.3f} GB".format(
                        getattr(self, "_mem_peak_alloc", 0.0), getattr(self, "_mem_peak_res", 0.0)))

                if _prof_time and _prof_rows:
                    def _median(key):
                        return float(np.median([r[key] for r in _prof_rows]))
                    data_med     = _median("data_ms")
                    h2d_med      = _median("h2d_ms")
                    backbone_med = _median("backbone_ms")
                    head_med     = _median("head_ms")
                    bs_med       = _median("backward_step_ms")
                    total_med    = _median("total_ms")
                    seg_sum      = data_med + h2d_med + backbone_med + head_med + bs_med
                    err_pct      = abs(seg_sum - total_med) / total_med * 100 if total_med else float("nan")
                    print_and_log(self.logfile,
                        "[time] n={}  data={:.3f}ms  H2D={:.3f}ms  backbone(支+查)={:.3f}ms  head={:.3f}ms  "
                        "backward+step={:.3f}ms  | 五段加總={:.3f}ms  total(wall)={:.3f}ms  誤差={:.2f}%".format(
                            len(_prof_rows), data_med, h2d_med, backbone_med, head_med, bs_med,
                            seg_sum, total_med, err_pct))

                # save the final model
                torch.save(self.model.state_dict(), self.checkpoint_path_final)

                if _loss_csv_f is not None:
                    _loss_csv_f.close()

        self.logfile.close()

    def train_task(self, task_dict):
        if getattr(self.args, "profile_time", False):
            _eh0 = torch.cuda.Event(enable_timing=True)
            _eh1 = torch.cuda.Event(enable_timing=True)
            _eh0.record()
            context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)
            _eh1.record()
            torch.cuda.synchronize()
            self._prof_h2d_ms = _eh0.elapsed_time(_eh1)
        else:
            context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)

        model_dict = self.model(context_images, context_labels, target_images)
        target_logits = model_dict['logits']

        task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        if getattr(self.args, "profile_time", False):
            _eb0 = torch.cuda.Event(enable_timing=True)
            _eb1 = torch.cuda.Event(enable_timing=True)
            _eb0.record()
            task_loss.backward(retain_graph=False)
            _eb1.record()
            torch.cuda.synchronize()
            self._prof_backward_ms = _eb0.elapsed_time(_eb1)
        else:
            task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def test(self, session):
        self.model.eval()
        with torch.no_grad():

                self.vd.train = False
                accuracy_dict ={}
                accuracies = []
                iteration = 0
                item = self.args.dataset
                for task_dict in self.test_loader:
                    if iteration >= self.args.num_test_tasks:
                        break
                    iteration += 1

                    context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)
                    model_dict = self.model(context_images, context_labels, target_images)
                    target_logits = model_dict['logits']
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
                self.vd.train = True
        self.model.train()
        
        return accuracy_dict


    def prepare_task(self, task_dict, images_to_device = True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list  

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]


    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
        print_and_log(self.logfile, f"Resuming from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.start_iteration = checkpoint['iteration']
        print_and_log(self.logfile, f"Loaded checkpoint at iteration {self.start_iteration}")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    main()
