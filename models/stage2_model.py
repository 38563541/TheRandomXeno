"""
Stage 2 model: TRXSetMatching + optional IntraRelation / InterRelation.

Two ablation options controlled by 'relation_level' in the YAML config:

    Option A (relation_level="frame"):
        ResNet → [IntraRelation] → [InterRelation] → Tuple Sampling → Hausdorff → Logits
        Relation enriches per-frame representations BEFORE tuples are formed.

    Option B (relation_level="tuple"):
        ResNet → Tuple Sampling → [IntraRelation] → [InterRelation] → Hausdorff → Logits
        Relation enriches per-tuple representations AFTER tuples are formed.
        This is the primary Stage 2 contribution.

Both intra and inter can be independently toggled via:
    use_intra_relation: true/false
    use_inter_relation: true/false

The distance function is dispatched from the config, exactly as in Stage 1:
    matching:        unidirectional | bidirectional | attention_weighted
    set_aggregation: pool | instance
    tau:             float (attention_weighted only)
Defaults (bidirectional + pool) reproduce the previously hard-coded behaviour
bit-for-bit, so all results obtained before this dispatch was added remain valid.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from utils import split_first_dim_linear
from matching.mean_hausdorff import pool_hausdorff, instance_class_distance
from models.stage1_model import TRXSetMatching, NUM_SAMPLES
from relation.intra_relation import IntraRelation
from relation.inter_relation import InterRelation
from relation.hyrsm_inter_relation import HyRSMInterRelation
from relation.true_hyrsm_inter_relation import TrueHyRSMInterRelation
from relation.support_decouple_relation import SupportDecoupleRelation


class TRXSetMatchingWithRelation(nn.Module):
    """
    Stage 2: TRXSetMatching with optional relation modules.

    Args:
        args           : argparse namespace
        relation_level : "frame" (Option A) or "tuple" (Option B)
        use_intra      : bool — apply IntraRelation
        use_inter      : bool — apply InterRelation
    """

    def __init__(self, args, relation_level="tuple", use_intra=True, use_inter=True,
                 inter_style="global"):
        super().__init__()
        assert relation_level in ("frame", "tuple"), \
            f"relation_level must be 'frame' or 'tuple', got {relation_level!r}"
        assert inter_style in ("global", "hyrsm", "true_hyrsm", "decouple"), \
            f"inter_style must be 'global', 'hyrsm', 'true_hyrsm', or 'decouple', got {inter_style!r}"
        # Phase 0.2: frame 分支只支援 global/hyrsm；decouple/true_hyrsm 的 API 與 frame 不相容
        assert not (relation_level == "frame" and inter_style in ("decouple", "true_hyrsm")), \
            f"inter_style={inter_style!r} 只支援 relation_level='tuple'"

        self.args           = args
        self.relation_level = relation_level
        self.use_intra      = use_intra
        self.use_inter      = use_inter
        self.inter_style    = inter_style

        # Reuse Stage 1 tuple embedding + distance computation
        self.matching = TRXSetMatching(args, temporal_set_size=2)

        # Relation dim depends on where it is applied
        if relation_level == "frame":
            rel_dim = args.trans_linear_in_dim   # 512 for ResNet18
        else:
            rel_dim = args.trans_linear_out_dim  # 1152

        num_heads = 8

        depth = getattr(args, "intra_depth", 1)
        if use_intra:
            if depth == 1:
                self.intra = IntraRelation(rel_dim, num_heads)  # 保留舊鍵名（向後相容）
            else:
                self.intra = nn.Sequential(*[IntraRelation(rel_dim, num_heads)
                                             for _ in range(depth)])
        else:
            self.intra = nn.Identity()
        if use_inter:
            if inter_style == "decouple":
                self.inter = SupportDecoupleRelation(
                    rel_dim, num_heads,
                    gate=getattr(args, "decouple_gate", True),
                    mode=getattr(args, "decouple_mode", "remove"),
                )
            elif inter_style == "true_hyrsm":
                self.inter = TrueHyRSMInterRelation(rel_dim, num_heads)
            elif inter_style == "hyrsm":
                self.inter = HyRSMInterRelation(rel_dim, num_heads)
            else:
                self.inter = InterRelation(rel_dim, num_heads)
        else:
            self.inter = None

        print(
            f"[INFO] TRXSetMatchingWithRelation: relation_level={relation_level}, "
            f"use_intra={use_intra}, use_inter={use_inter}, intra_depth={depth}, "
            f"inter_style={inter_style}, rel_dim={rel_dim}"
        )

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)
        return torch.reshape(torch.nonzero(class_mask, as_tuple=False), (-1,))

    def _class_distance(self, q_set, class_s):
        """
        Query-set to class-support-set distance, dispatched from the config.

        Reads matching / set_aggregation / tau off self.matching (the shared
        TRXSetMatching instance), so Stage 1 and Stage 2 always agree on what
        the distance function is.

        Defaults are matching="bidirectional" and set_aggregation="pool", which
        is numerically identical to the previously hard-coded
        mean_hausdorff_bidir() — so existing results are unaffected.

        Args:
            q_set   : (nq, T, d)
            class_s : (k_shot, T, d)   support tuples for ONE class
        Returns:
            distances: (nq,)
        """
        cfg = self.matching   # TRXSetMatching holds the parsed distance config
        if cfg.set_aggregation == "pool":
            return pool_hausdorff(
                q_set, class_s,
                mode=cfg.matching, tau=cfg.tau, chunk_s=cfg.support_chunk,
            )
        return instance_class_distance(
            q_set, class_s, mode=cfg.matching, tau=cfg.tau,
        )

    def forward(self, support_set, support_labels, queries):
        """
        Args:
            support_set    : [n_support, seq_len, d_in]
            support_labels : [n_support,]
            queries        : [n_queries, seq_len, d_in]

        Returns:
            dict with 'logits': [n_queries, way]
        """
        device          = queries.device
        unique_labels   = torch.unique(support_labels)
        s_set_per_query = None   # set to [nq, ns, T, d] only for true_hyrsm

        if self.relation_level == "frame":
            # ── Option A: relation on frame embeddings ──────────────────────
            # IntraRelation: enrich each video's frame sequence independently
            queries     = self.intra(queries)      # [nq, seq_len, d_in]
            support_set = self.intra(support_set)  # [ns, seq_len, d_in]

            # InterRelation: cross-attend across all queries and all support
            if self.inter is not None:
                queries, support_set = self.inter(queries, support_set)

            # Build tuple sets from enhanced frame features
            q_set = self.matching._make_pair_set(queries)      # [nq, T, d_out]
            s_set = self.matching._make_pair_set(support_set)  # [ns, T, d_out]

        else:
            # ── Option B: relation on tuple embeddings ───────────────────────
            # Build tuple sets first (no relation yet)
            q_set = self.matching._make_pair_set(queries)      # [nq, T, d_out]
            s_set = self.matching._make_pair_set(support_set)  # [ns, T, d_out]

            # IntraRelation: enrich each video's tuple sequence
            q_set = self.intra(q_set)  # [nq, T, d_out]
            s_set = self.intra(s_set)  # [ns, T, d_out]

            # Support decouple: class-conditioned support rewriting.
            # Runs ONCE per episode, BEFORE the class loop.
            # query is deliberately NOT modified so all 5 class distances
            # remain in a common scale (design principle).
            if self.inter is not None and self.inter_style == "decouple":
                s_set = self.inter(s_set, support_labels)

            # Global InterRelation: cross-attend across all queries and all support
            if self.inter is not None and self.inter_style == "global":
                q_set, s_set = self.inter(q_set, s_set)

            # True HyRSM: global-pool → self-attn → expand-concat-conv
            # Returns enhanced_query [nq, T, d] and enhanced_support [nq, ns, T, d]
            if self.inter is not None and self.inter_style == "true_hyrsm":
                q_set, s_set_per_query = self.inter(q_set, s_set)
                # s_set_per_query: [nq, ns, T, d] — support differs per query

        # ── Per-class Hausdorff distances ────────────────────────────────────
        n_queries    = q_set.shape[0]
        all_distances = torch.zeros(
            n_queries, self.args.way, device=device, dtype=q_set.dtype
        )

        for c in unique_labels:
            idx     = self._extract_class_indices(support_labels, c)

            if s_set_per_query is not None:
                # true_hyrsm: support is per-query → loop over queries
                class_s_pq = s_set_per_query[:, idx, :, :]   # [nq, k, T, d]
                for qi in range(n_queries):
                    d = self._class_distance(
                        q_set[qi].unsqueeze(0),   # [1, T, d]
                        class_s_pq[qi],            # [k, T, d]
                    )
                    all_distances[qi, c.long()] = -d
            else:
                class_s = torch.index_select(s_set, 0, idx)  # [k_shot, T, d_out]

                # HyRSM (old class-specific): enrich query inside loop
                if self.inter is not None and self.inter_style == "hyrsm":
                    q_for_dist = self.inter(q_set, class_s)
                else:
                    q_for_dist = q_set

                dist = self._class_distance(q_for_dist, class_s)   # [nq,]
                all_distances[:, c.long()] = -dist

        return {"logits": all_distances}


class CNN_TRXWithRelation(nn.Module):
    """
    ResNet backbone + TRXSetMatchingWithRelation (Stage 2).

    Reads relation config from args:
        args.relation_level      : "frame" or "tuple"
        args.use_intra_relation  : bool
        args.use_inter_relation  : bool
    """

    def __init__(self, args):
        super().__init__()
        self.train()
        self.args = args

        relation_level = getattr(args, "relation_level",     "tuple")
        use_intra      = getattr(args, "use_intra_relation", True)
        use_inter      = getattr(args, "use_inter_relation", True)
        inter_style    = getattr(args, "inter_style",        "global")

        # Backbone (last avg-pool removed)
        if args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # D: set inplace=False on all ReLUs so use_reentrant=False works correctly
        _n_relu = 0
        for m in self.resnet.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False
                _n_relu += 1
        if _n_relu:
            print(f"[INFO] set inplace=False on {_n_relu} ReLU modules in resnet")

        # Phase 2.1: freeze backbone (BN kept in eval via train() override)
        if getattr(args, "freeze_backbone", False):
            for p in self.resnet.parameters():
                p.requires_grad_(False)
            self.resnet.eval()
            print("[INFO] backbone FROZEN: requires_grad=False, BN in eval mode", flush=True)

        self.transformers = nn.ModuleList([
            TRXSetMatchingWithRelation(
                args,
                relation_level=relation_level,
                use_intra=use_intra,
                use_inter=use_inter,
                inter_style=inter_style,
            )
        ])

    def _ckpt_chain(self):
        """攤平 layer1-4 成個別 residual block 的視圖，不重新註冊 → state_dict 鍵名不變。"""
        out = []
        for m in self.resnet:
            if isinstance(m, torch.nn.Sequential):
                out.extend(list(m))
            else:
                out.append(m)
        return out

    @staticmethod
    def _split_even(n, k):
        """把 n 個元素平均分成 k 組的 (start, end) 邊界，餘數分配給前面的組。"""
        base, rem = divmod(n, k)
        bounds = []
        idx = 0
        for i in range(k):
            size = base + (1 if i < rem else 0)
            bounds.append((idx, idx + size))
            idx += size
        return bounds

    @staticmethod
    def _sac_policy_fn(ctx, op, *args, **kwargs):
        """0922 selective checkpointing 的 policy：conv 的輸出一定存，其他
        （BN/ReLU 等）偏好重算。只有 aten.convolution 回傳 MUST_SAVE。"""
        from torch.utils.checkpoint import CheckpointPolicy
        if op == torch.ops.aten.convolution.default:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def _ckpt_forward_prefix(self, x, N, segs, policy="full"):
        """0921 新增：恰好 checkpoint chain[:N] 這 N 個 module，其餘直接跑。

        不用 checkpoint_sequential(chain[:N], segs, x) —— 那個函式的最後一段
        本來就不 checkpoint（已在 0918 查證過），用它包前綴會讓實際涵蓋
        < N，N 的意義會模糊掉。這裡手動把 chain[:N] 分成 segs 組，每組包成
        nn.Sequential 個別呼叫 checkpoint()，segs 只影響要存幾個段邊界
        （顯存），不影響涵蓋範圍（N 決定，也就是時間代價）。

        0922 新增 policy="save_conv"：selective activation checkpointing，
        conv 輸出存下來、BN/ReLU 等便宜運算重算，透過 context_fn 傳
        policy_fn 給 checkpoint()。policy="full"（預設）是原本的行為，
        整組全部重算，不傳 context_fn。
        """
        from torch.utils.checkpoint import checkpoint

        chain = self._ckpt_chain()
        N = min(N, len(chain))
        prefix, suffix = chain[:N], chain[N:]

        context_fn = None
        if policy == "save_conv":
            from torch.utils.checkpoint import create_selective_checkpoint_contexts
            import functools
            context_fn = functools.partial(create_selective_checkpoint_contexts,
                                            self._sac_policy_fn)

        if N > 0:
            segs = max(1, min(segs, N))
            for start, end in self._split_even(N, segs):
                group = torch.nn.Sequential(*prefix[start:end])
                if context_fn is not None:
                    x = checkpoint(group, x, use_reentrant=False, context_fn=context_fn)
                else:
                    x = checkpoint(group, x, use_reentrant=False)

        for m in suffix:
            x = m(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        # 凍結時 backbone 永遠保持 eval（BN running stats 不更新）
        # guard: __init__ 在 self.resnet 建立前就呼叫 self.train()，必須防衛
        if getattr(self, "resnet", None) is not None and \
                getattr(getattr(self, "args", None), "freeze_backbone", False):
            self.resnet.eval()
        return self

    def forward(self, context_images, context_labels, target_images):
        _prof = self.training and getattr(self.args, "profile_time", False)
        if _prof:
            _e0 = torch.cuda.Event(enable_timing=True)
            _e1 = torch.cuda.Event(enable_timing=True)
            _e2 = torch.cuda.Event(enable_timing=True)
            _e0.record()

        _ckpt_prefix = getattr(self.args, "ckpt_prefix", None)
        if self.training and getattr(self.args, "grad_ckpt", False) and _ckpt_prefix:
            # 0921：手動前綴 checkpoint，涵蓋範圍 = chain[:N]，段數只影響顯存。
            # 0922：ckpt_policy="save_conv" 時走 selective checkpointing，
            # branch 旗標分辨 full／save_conv。
            _policy = getattr(self.args, "ckpt_policy", "full") or "full"
            self._last_ckpt_branch = "prefix_save_conv" if _policy == "save_conv" else "prefix_full"
            segs = getattr(self.args, "ckpt_segments", 8) or 8
            context_features = self._ckpt_forward_prefix(
                context_images, _ckpt_prefix, segs, _policy).squeeze()
            target_features  = self._ckpt_forward_prefix(
                target_images,  _ckpt_prefix, segs, _policy).squeeze()
        elif self.training and getattr(self.args, "grad_ckpt", False):
            self._last_ckpt_branch = "checkpoint_sequential"
            from torch.utils.checkpoint import checkpoint_sequential
            # use_reentrant=False: inplace ReLUs are patched to inplace=False
            # in __init__, so non-reentrant checkpointing is safe.
            # _ckpt_chain() 攤平 layer1-4，不重新註冊，state_dict 鍵名不變。
            chain = self._ckpt_chain()
            segs  = min(getattr(self.args, "ckpt_segments", 8) or 8, len(chain))
            context_features = checkpoint_sequential(
                chain, segs, context_images, use_reentrant=False).squeeze()
            target_features  = checkpoint_sequential(
                chain, segs, target_images,  use_reentrant=False).squeeze()
        else:
            self._last_ckpt_branch = "none"
            context_features = self.resnet(context_images).squeeze()
            target_features  = self.resnet(target_images).squeeze()

        if _prof:
            _e1.record()

        # 0922 顯存拆分：support+query 兩次 backbone 都跑完、進匹配頭之前。
        # 用 memory_allocated()（不是 max_memory_allocated()），不需要 sync。
        if self.training and getattr(self.args, "profile_memory", False):
            self._mem_bb = torch.cuda.memory_allocated()

        dim = int(context_features.shape[1])
        context_features = context_features.reshape(-1, self.args.seq_len, dim)
        target_features  = target_features.reshape(-1, self.args.seq_len, dim)

        all_logits = [
            t(context_features, context_labels, target_features)['logits']
            for t in self.transformers
        ]
        all_logits    = torch.stack(all_logits, dim=-1)
        sample_logits = torch.mean(all_logits, dim=-1)

        if _prof:
            _e2.record()
            torch.cuda.synchronize()
            # backbone: support + query 兩次 resnet forward（或 checkpoint 重算）合計；
            # head: reshape（可忽略）+ transformer 們的 forward。
            self._prof_backbone_ms = _e0.elapsed_time(_e1)
            self._prof_head_ms     = _e1.elapsed_time(_e2)

        return {
            'logits': split_first_dim_linear(
                sample_logits, [NUM_SAMPLES, target_features.shape[0]]
            )
        }

    def distribute_model(self):
        if self.args.num_gpus > 1:
            self.resnet = torch.nn.DataParallel(
                self.resnet,
                device_ids=list(range(self.args.num_gpus))
            )
            self.transformers.cuda(0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import types

    args = types.SimpleNamespace(
        trans_linear_in_dim=512,
        trans_linear_out_dim=1152,
        way=5, shot=1, query_per_class=5,
        trans_dropout=0.1,
        seq_len=8,
        img_size=84,
        method="resnet18",
        num_gpus=1,
        temp_set=[2],
        use_intra_relation=True,
        use_inter_relation=True,
        relation_level="tuple",
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for level in ("frame", "tuple"):
        args.relation_level = level
        model = CNN_TRXWithRelation(args).to(device)

        support = torch.rand(args.way * args.shot * args.seq_len, 3, 84, 84).to(device)
        target  = torch.rand(args.way * args.query_per_class * args.seq_len, 3, 84, 84).to(device)
        labels  = torch.arange(args.way).to(device)

        out = model(support, labels, target)
        expected = (NUM_SAMPLES, args.way * args.query_per_class, args.way)
        assert out['logits'].shape == expected, \
            f"Option {'A' if level == 'frame' else 'B'}: expected {expected}, got {out['logits'].shape}"
        print(f"Option {'A' if level == 'frame' else 'B'} ({level} level): logits shape {out['logits'].shape} ✓")

    print("All smoke tests passed.")
