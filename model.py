<<<<<<< Updated upstream
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
import math
from itertools import combinations 

from torch.autograd import Variable

import torchvision.models as models

NUM_SAMPLES=1

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)
'''
class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        
        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way).cuda()

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0,2,1,3)
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2,-1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict



    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
'''        
class TemporalCrossTransformerHausdorffPairs(nn.Module):
    """
    Pairs-only (temporal_set_size=2) version that REPLACES TRX attention/prototype with
    bidirectional Mean-Hausdorff set matching over tuple embeddings.

    Key idea:
      - Keep query tuples as a set Q (size T= C(seq_len,2))
      - Keep class support tuples as a set S (size k_bs*T)
      - Distance(Q,S) = 0.5*( mean_q min_s d(q,s)  +  mean_s min_q d(s,q) )
      - Logit = -Distance
    """
    

    def __init__(self, args, temporal_set_size=2):
        
        super().__init__()
        assert temporal_set_size == 2, "This class is for pairs-only. Use temporal_set_size=2."
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        # Map concatenated pair features (2 * d_in) -> d_out (shared for K/V in original; here we just need an embedding)
        self.embed_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)
        self.norm = nn.LayerNorm(self.args.trans_linear_out_dim)

        # Precompute all pair indices as a buffer: shape (tuples_len, 2)
        frame_idxs = list(range(self.args.seq_len))
        pair_list = list(combinations(frame_idxs, temporal_set_size))  # list of tuples length 2
        tuples = torch.tensor(pair_list, dtype=torch.long)            # (T, 2)
        self.register_buffer("tuples", tuples, persistent=False)
        self.tuples_len = tuples.shape[0]
        
        # Chunk size for support tuples when computing min-dist (time <-> memory trade-off)
        # Smaller => lower VRAM, slower. Start with 256; reduce if still tight.
        self.support_chunk = 64
        print(f"[INFO] HausdorffPairs tuples_len={self.tuples_len}, support_chunk={self.support_chunk}")


    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def _make_pair_set(self, x):
        """
        x: (N, seq_len, d_in)
        return: (N, T, d_out) where T=C(seq_len,2)
        """
        device = x.device
        tuples = self.tuples.to(device)  # (T, 2)

        # Add PE
        x = self.pe(x)  # (N, seq_len, d_in)

        # Build pair features: concatenate features at two time indices
        # We'll loop over T (28 when seq_len=8), which is cheap and avoids big advanced indexing buffers.
        pair_feats = []
        for p in tuples:  # p: (2,)
            # index_select along time dim (-2)
            xp = torch.index_select(x, dim=-2, index=p)      # (N, 2, d_in)
            xp = xp.reshape(x.shape[0], -1)                  # (N, 2*d_in)
            pair_feats.append(xp)
        pair_feats = torch.stack(pair_feats, dim=1)          # (N, T, 2*d_in)

        # Embed + norm
        z = self.embed_linear(pair_feats)                    # (N, T, d_out)
        z = self.norm(z)
        return z

    def _mean_hausdorff_bidir(self, q_set, s_set, chunk_s=256):
        """
        q_set: (nq, Tq, d)
        s_set: (ns, Ts, d) OR flattened (NsTot, d) after reshape

        We'll flatten s_set to (S, d) where S = ns*Ts.
        Then compute:
          d_q2s = mean over q-tuples of min over s-tuples ||q - s||^2
          d_s2q = mean over s-tuples of min over q-tuples ||s - q||^2
        Return: 0.5*(d_q2s + d_s2q), shape (nq,)
        """

        nq, Tq, d = q_set.shape
        s_flat = s_set.reshape(-1, d)  # (S, d)
        S = s_flat.shape[0]
        device = q_set.device
        dtype = q_set.dtype

        # ---------- Q -> S ----------
        # min_dist_q: (nq, Tq)
        min_dist_q = torch.full((nq, Tq), float("inf"), device=device, dtype=dtype)

        # Precompute q norms (nq, Tq, 1)
        q_norm = (q_set ** 2).sum(dim=-1, keepdim=True)  # (nq, Tq, 1)

        for start in range(0, S, chunk_s):
            end = min(start + chunk_s, S)
            s_chunk = s_flat[start:end]                                # (c, d)
            s_norm = (s_chunk ** 2).sum(dim=-1).view(1, 1, -1)         # (1,1,c)

            # dist^2 = ||q||^2 + ||s||^2 - 2 q·s
            # q_set: (nq,Tq,d), s_chunk.T: (d,c) => (nq,Tq,c)
            qs = torch.matmul(q_set, s_chunk.t())                      # (nq,Tq,c)
            dist2 = q_norm + s_norm - 2.0 * qs                         # (nq,Tq,c)

            # update min over s
            min_dist_q = torch.minimum(min_dist_q, dist2.min(dim=-1).values)

        d_q2s = min_dist_q.mean(dim=-1)  # (nq,)

        # ---------- S -> Q ----------
        # For each s tuple, min over all q tuples (across Tq) for each query
        # We want mean over s tuples of min_q dist(s,q). For each query, compute:
        #   min over q tuples: (S,) then mean -> (nq,)
        min_dist_s = torch.full((nq, S), float("inf"), device=device, dtype=dtype)

        # Flatten q to (nq, Tq, d), we already have; flatten q tuples dimension for matmul convenience
        q_flat = q_set.reshape(nq * Tq, d)                              # (nq*Tq, d)
        q_flat_t = q_flat.t()                                           # (d, nq*Tq)
        q_flat_norm = (q_flat ** 2).sum(dim=-1).view(1, -1)             # (1, nq*Tq)

        for start in range(0, S, chunk_s):
            end = min(start + chunk_s, S)
            s_chunk = s_flat[start:end]                                 # (c, d)
            s_norm = (s_chunk ** 2).sum(dim=-1).view(-1, 1)             # (c,1)

            # dist^2 between each s in chunk and all q tuples (nq*Tq)
            # (c, d) @ (d, nq*Tq) => (c, nq*Tq)
            sq = torch.matmul(s_chunk, q_flat_t)                        # (c, nq*Tq)
            dist2 = s_norm + q_flat_norm - 2.0 * sq                     # (c, nq*Tq)

            # reshape to (c, nq, Tq) then min over Tq => (c, nq)
            dist2 = dist2.view(end - start, nq, Tq)
            min_over_qtuple = dist2.min(dim=-1).values                  # (c, nq)

            # store into min_dist_s: we want (nq, S). transpose:
            min_dist_s[:, start:end] = min_over_qtuple.t()              # (nq, c)

        d_s2q = min_dist_s.mean(dim=-1)  # (nq,)

        return 0.5 * (d_q2s + d_s2q)

    def forward(self, support_set, support_labels, queries):
        """
        support_set: (n_support, seq_len, d_in)
        queries:     (n_queries, seq_len, d_in)
        support_labels: (n_support,) values in {0..way-1}
        """
        device = queries.device
        unique_labels = torch.unique(support_labels)

        # Build pair sets (tuple embeddings)
        q_set = self._make_pair_set(queries)        # (nq, T, d_out)
        s_set = self._make_pair_set(support_set)    # (ns, T, d_out)

        n_queries = q_set.shape[0]
        all_distances_tensor = torch.zeros(n_queries, self.args.way, device=device, dtype=q_set.dtype)

        for c in unique_labels:
            idx = self._extract_class_indices(support_labels, c)
            class_s = torch.index_select(s_set, 0, idx)  # (k_bs, T, d_out)

            # Bidirectional mean-Hausdorff distance (squared L2)
            dist = self._mean_hausdorff_bidir(q_set, class_s, chunk_s=self.support_chunk)  # (nq,)

            # logits = -distance
            all_distances_tensor[:, c.long()] = -dist

        return {"logits": all_distances_tensor}

class CNN_TRX(nn.Module):
    """
    Standard Resnet connected to a Temporal Cross Transformer.
    
    """
    def __init__(self, args):
        super(CNN_TRX, self).__init__()

        self.train()
        self.args = args

        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -1
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])

        #self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set])
        print("[INFO] Using TemporalCrossTransformerHausdorffPairs (pairs-only).")
 
        self.transformers = nn.ModuleList([TemporalCrossTransformerHausdorffPairs(args, temporal_set_size=2)])


    def forward(self, context_images, context_labels, target_images):

        context_features = self.resnet(context_images).squeeze()
        target_features = self.resnet(target_images).squeeze()

        dim = int(context_features.shape[1])

        context_features = context_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        all_logits = [t(context_features, context_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_features.shape[0]])}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)



if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 1
            self.query_per_class = 5
            self.trans_dropout = 0.1
            self.seq_len = 8 
            self.img_size = 84
            self.method = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2]
    args = ArgsObject()
    torch.manual_seed(0)
    
    device = 'cuda:0'
    model = CNN_TRX(args).to(device)
    
    support_imgs = torch.rand(args.way * args.shot * args.seq_len,3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len ,3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0,1,2,3,4]).to(device)

    print("Support images input shape: {}".format(support_imgs.shape))
    print("Target images input shape: {}".format(target_imgs.shape))
    print("Support labels input shape: {}".format(support_imgs.shape))

    out = model(support_imgs, support_labels, target_imgs)

    print("TRX returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(out['logits'].shape))





=======
"""
model.py — thin model factory, backward compatible with run.py.

run.py does:
    from model import CNN_TRX
    model = CNN_TRX(args)

CNN_TRX() inspects args and returns the right model:

    Stage 1 (default): use_intra_relation=false AND use_inter_relation=false
        → CNN_TRX from models/stage1_model.py (no relation modules)

    Stage 2: use_intra_relation=true OR use_inter_relation=true
        → CNN_TRXWithRelation from models/stage2_model.py

Stage is selected automatically from the YAML config — no code change needed
when switching between stage1_*.yaml and stage2_*.yaml configs.

All implementation lives in:
    models/stage1_model.py       — TRXSetMatching (active Stage 1)
    models/stage2_model.py       — TRXSetMatchingWithRelation (Stage 2)
    models/trx_original.py       — original prototype TRX (reference baseline)
    matching/mean_hausdorff.py
    matching/bidirectional_hausdorff.py
    matching/attention_weighted_hausdorff.py
    relation/intra_relation.py
    relation/inter_relation.py
"""

from models.stage1_model import CNN_TRX as _Stage1, NUM_SAMPLES  # noqa: F401


def CNN_TRX(args):
    """
    Model factory. Returns Stage 1 or Stage 2 model based on args.

    Stage 2 is activated when either use_intra_relation or
    use_inter_relation is True in the config.
    """
    use_relation = (
        getattr(args, "use_intra_relation", False)
        or getattr(args, "use_inter_relation", False)
    )
    if use_relation:
        from models.stage2_model import CNN_TRXWithRelation
        return CNN_TRXWithRelation(args)
    return _Stage1(args)


__all__ = ["CNN_TRX", "NUM_SAMPLES"]
>>>>>>> Stashed changes
