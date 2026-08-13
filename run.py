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
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.args.num_workers)
        self.test_loader  = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=0)
        
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        self.test_accuracies = TestAccuracies(self.test_set)
        
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        model = CNN_TRX(self.args)
        model = model.to(self.device) 
        if self.args.num_gpus > 1:
            model.distribute_model()
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
                    print(accuracy_dict)
                    self.test_accuracies.print(self.logfile, accuracy_dict)
                    _item = self.args.dataset
                    if _item in accuracy_dict:
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

                iteration = self.start_iteration
                for task_dict in self.video_loader:
                    if iteration >= total_iterations:
                        break
                    iteration += 1
                    torch.set_grad_enabled(True)

                    task_loss, task_accuracy = self.train_task(task_dict)
                    train_accuracies.append(task_accuracy)
                    losses.append(task_loss)

                    # optimize
                    if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    self.scheduler.step()
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
                        accuracy_dict = self.test(session)
                        print(accuracy_dict)
                        self.test_accuracies.print(self.logfile, accuracy_dict)
                        # --- CSV logging (STEP 5) ---
                        _item = self.args.dataset
                        if _item in accuracy_dict:
                            _log_result_csv(
                                self.args,
                                iteration=iteration + 1,
                                mean_accuracy=accuracy_dict[_item]["accuracy"],
                                confidence_interval=accuracy_dict[_item]["confidence"],
                            )

                # save the final model
                torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.logfile.close()

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)

        model_dict = self.model(context_images, context_labels, target_images)
        target_logits = model_dict['logits']

        task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

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
