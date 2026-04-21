"""
Minimal reproduction of the crash at test evaluation in run.py.
Exactly mirrors run.py: test() is called while inside the outer training for-loop.
"""
import torch
import numpy as np
import os, sys, types, traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import video_reader
from model import CNN_TRX

args = types.SimpleNamespace(
    dataset='hmdb', split=3, seq_len=8, img_size=224,
    way=5, shot=5, query_per_class=5, query_per_class_test=1,
    trans_linear_in_dim=512, trans_linear_out_dim=1152,
    trans_dropout=0.1, temp_set=[2], num_gpus=1, num_samples=1,
    method='resnet18', num_workers=10, debug_loader=False,
    use_intra_relation=True, use_inter_relation=True, relation_level='frame',
    traintestlist='/home/ccwu/Documents/work/trx/trx_data/video_datasets/splits/hmdb_ARN',
    path='/home/ccwu/Documents/work/trx/trx_data/video_datasets/data/hmdb51_256q5.zip',
    zip=True, scratch='/home/ccwu/Documents/work/trx/trx_data',
    num_test_tasks=50,  # small number so it completes quickly
    training_iterations=5,
    print_freq=1, save_freq=9999, test_iters=[5],
    learning_rate=0.001, tasks_per_batch=4, sch=[1000000],
    opt='sgd',
)

device = torch.device('cuda')

print("Loading model...", flush=True)
model = CNN_TRX(args).to(device)

ckpt_path = 'checkpoints/stage2_optA_hmdb3_5shot/hmdb_split3_20260420_203803/checkpoint25000.pt'
ckpt = torch.load(ckpt_path, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print("Checkpoint loaded.", flush=True)

vd = video_reader.VideoDataset(args)
loader      = torch.utils.data.DataLoader(vd, batch_size=1, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(vd, batch_size=1, num_workers=0)  # 0 workers: run in main process

def prepare_task(task_dict):
    ctx_img = task_dict['support_set'][0].to(device)
    tgt_img = task_dict['target_set'][0].to(device)
    ctx_lbl = task_dict['support_labels'][0].type(torch.LongTensor).to(device)
    tgt_lbl = task_dict['target_labels'][0].type(torch.LongTensor).to(device)
    return ctx_img, tgt_img, ctx_lbl, tgt_lbl

def test():
    print("  [test] model.eval()", flush=True)
    model.eval()
    with torch.no_grad():
        print("  [test] switching loader to test mode", flush=True)
        vd.train = False
        accuracies = []
        print("  [test] starting test iteration loop", flush=True)
        print(f"  [test] VRAM before loop: {torch.cuda.memory_allocated()/1e6:.0f}MB alloc / {torch.cuda.memory_reserved()/1e6:.0f}MB reserved", flush=True)
        test_iter = iter(test_loader)
        print("  [test] iterator created", flush=True)
        for i in range(args.num_test_tasks):
            print(f"  [test] fetching batch {i}...", flush=True)
            task_dict = next(test_iter)
            print(f"  [test] batch {i} fetched, moving to device...", flush=True)
            ctx_img, tgt_img, ctx_lbl, tgt_lbl = prepare_task(task_dict)
            print(f"  [test] ctx={ctx_img.shape} tgt={tgt_img.shape} — running model...", flush=True)
            print(f"  [test] VRAM: {torch.cuda.memory_allocated()/1e6:.0f}MB alloc / {torch.cuda.memory_reserved()/1e6:.0f}MB reserved", flush=True)
            out = model(ctx_img, ctx_lbl, tgt_img)
            print(f"  [test] model done", flush=True)
            logits = out['logits'][0]
            preds = logits.argmax(dim=-1)
            acc = (preds == tgt_lbl).float().mean().item()
            accuracies.append(acc)
            print(f"  [test] task {i}: acc={acc:.3f}", flush=True)
        vd.train = True
    model.train()
    mean_acc = np.mean(accuracies) * 100
    print(f"  [test] DONE. mean_acc={mean_acc:.2f}%", flush=True)
    return mean_acc

print("Starting TF session + training loop...", flush=True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

try:
    with tf.compat.v1.Session(config=config) as session:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

        iteration = 0
        # ---- This exactly mirrors run.py: test() called INSIDE the outer for-loop ----
        for task_dict in loader:
            if iteration >= args.training_iterations:
                break
            iteration += 1

            ctx_img, tgt_img, ctx_lbl, tgt_lbl = prepare_task(task_dict)
            out = model(ctx_img, ctx_lbl, tgt_img)
            loss = torch.nn.functional.cross_entropy(out['logits'][0], tgt_lbl)
            loss.backward()
            if iteration % args.tasks_per_batch == 0:
                optimizer.step()
                optimizer.zero_grad()

            print(f"Train iter {iteration}/{args.training_iterations}  loss={loss.item():.4f}", flush=True)

            if iteration in args.test_iters:
                print(f"\n=== Triggering test at iteration {iteration} ===", flush=True)
                acc = test()
                print(f"=== Test complete: {acc:.2f}% ===\n", flush=True)

        print("Training loop done.", flush=True)

except Exception as e:
    print(f"\n!!! EXCEPTION: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("Script finished cleanly.", flush=True)
