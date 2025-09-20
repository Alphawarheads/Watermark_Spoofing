import argparse
import sys
import os
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import yaml
from data_utils_SSL import (
    genSpoof_list,
    Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2021_eval,
    Dataset_in_the_wild_eval,
    genSpoof_list_multi,
    Dataset_MultiTrain,
)
from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm
from torchvision import transforms
import gc


def dedup_score_file(path: str, key_col: int = 0, sep: str = None):
    """
    De-duplicate a score file line-by-line: use column `key_col` (default 0: utt_id)
    as the key and keep only the first occurrence of each key while preserving the
    original order. If `sep` is None, split on any whitespace.
    """
    import tempfile, os

    seen = set()
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".dedup.", dir=dir_name)
    os.close(fd)
    kept, dropped = 0, 0
    with open(path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            parts = line.rstrip("\n")
            key = parts.split(sep)[key_col]  # default: split on whitespace
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            fout.write(line)
            kept += 1
    os.replace(tmp_path, path)
    return kept, dropped


# ----------------------- Distributed helpers -----------------------
def dist_is_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if dist_is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist_is_initialized() else 1


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def setup_distributed():
    """
    Initialize torch.distributed from torchrun environment variables.
    No new CLI args are introduced.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"  # Use 'nccl' for ROCm as well
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, local_rank
    else:
        return False, 0


def cleanup_distributed():
    if dist_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ----------------------- Training / Eval -----------------------
def evaluate_accuracy(dev_loader, model, device):
    # Logic unchanged; simply disable the progress bar on non-rank0
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    pbar = tqdm(dev_loader, disable=(get_rank() != 0))
    with torch.no_grad():
        for batch_x, batch_y in pbar:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)
            batch_out = model(batch_x)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)

    # Distributed aggregation
    if dist_is_initialized():
        t = torch.tensor([val_loss, num_total, num_correct], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        val_loss, num_total, num_correct = t.tolist()

    val_loss = val_loss / max(num_total, 1.0)
    acc = 100.0 * (num_correct / max(num_total, 1.0))
    return val_loss, acc


def produce_evaluation_file(dataset, model, device, save_path, allowed_ids=None):
    """
    Distributed-safe writing: each rank writes to `save_path.rank{r}`, and rank 0
    merges them into `save_path`. After merging, rank 0 performs a de-duplication
    over the final file by utt_id (keeping the first occurrence).
    """
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False) if dist_is_initialized() else None
    data_loader = DataLoader(
        dataset, batch_size=8, shuffle=False, drop_last=False, sampler=sampler, num_workers=32, pin_memory=True
    )

    model.eval()
    rank = get_rank()
    world_size = get_world_size()
    tmp_path = save_path if world_size == 1 else f"{save_path}.rank{rank}"

    with open(tmp_path, "w") as fh:
        pbar = tqdm(data_loader, disable=(rank != 0))
        with torch.no_grad():
            for batch_x, utt_id in pbar:
                fname_list, score_list = [], []
                batch_x = batch_x.to(device, non_blocking=True)
                torch.set_printoptions(threshold=10_000)
                batch_out = model(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
                fname_list.extend(utt_id)
                score_list.extend(batch_score.tolist())
                for f, cm in zip(fname_list, score_list):
                    fh.write(f"{f} {cm}\n")

    # Merge shards
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            with open(save_path, "w") as out:
                for r in range(world_size):
                    part = f"{save_path}.rank{r}"
                    with open(part, "r") as fh:
                        out.write(fh.read())
            # Cleanup shards
            for r in range(world_size):
                try:
                    os.remove(f"{save_path}.rank{r}")
                except OSError:
                    pass
        dist.barrier()

    # Rank 0 only: final de-dup + logging
    if rank == 0:
        try:
            kept, dropped = dedup_score_file(save_path, key_col=0, sep=None)
            print0(f"[dedup] {save_path}: kept={kept}, dropped={dropped}")
        except Exception as e:
            print0(f"[dedup] {save_path}: skipped due to error: {e}")
        print0(f"Scores saved to {save_path}")


def train_epoch(train_loader, model, lr, optim, device):
    """
    Keep your interface and loss logic; only add distributed friendliness and a
    minor speed-up (non_blocking). Returns: running_loss_avg (this process),
    num_total (this process).
    """
    running_loss = 0.0
    num_total = 0.0
    model.train()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    pbar = tqdm(train_loader, disable=(get_rank() != 0))
    for batch_x, batch_y in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)

        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)

        running_loss += (batch_loss.item() * batch_size)

        optim.zero_grad(set_to_none=True)  # Equivalent and more efficient; behavior unchanged
        batch_loss.backward()
        optim.step()

    running_loss_avg = running_loss / max(num_total, 1.0)
    return running_loss_avg, num_total


# ===================== main =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof2021 baseline system")
    # Dataset
    parser.add_argument(
        "--database_path",
        type=str,
        default="/public/home/qinxy/AudioData/Antispoofing/ASVspoof2019/LA/",
        help="Change this to user's full directory address of LA database (ASVspoof2019 for training & development (used as validation), ASVspoof2021 DF for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.",
    )
    parser.add_argument(
        "--protocols_path",
        type=str,
        default="/public/home/qinxy/AudioData/Antispoofing/ASVspoof2019/LA/",
        help="Change to the path of the user's DF database protocols directory",
    )

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--loss", type=str, default="weighted_CCE")
    # Model
    parser.add_argument("--seed", type=int, default=1234, help="random seed (default: 1234)")
    parser.add_argument("--model_path", type=str, default=None, help="Model checkpoint")
    parser.add_argument("--comment", type=str, default=None, help="Comment to describe the saved model")
    # Auxiliary arguments
    parser.add_argument("--track", type=str, default="DF", choices=["LA", "In-the-Wild", "DF"], help="LA/PA/DF")
    parser.add_argument("--eval_output", type=str, default=None, help="Path to save the evaluation result")
    parser.add_argument("--eval", action="store_true", default=False, help="eval mode")
    parser.add_argument("--is_eval", action="store_true", default=False, help="eval database")
    parser.add_argument("--eval_part", type=int, default=0)
    # Backend options
    parser.add_argument(
        "--cudnn-deterministic-toggle",
        action="store_false",
        default=True,
        help="use cudnn-deterministic? (default true)",
    )
    parser.add_argument(
        "--cudnn-benchmark-toggle",
        action="store_true",
        default=False,
        help="use cudnn-benchmark? (default false)",
    )

    # ============================= Rawboost data augmentation ============================= #
    parser.add_argument("--algo", type=int, default=3, help="Rawboost algos...")
    parser.add_argument("--nBands", type=int, default=5)
    parser.add_argument("--minF", type=int, default=20)
    parser.add_argument("--maxF", type=int, default=8000)
    parser.add_argument("--minBW", type=int, default=100)
    parser.add_argument("--maxBW", type=int, default=1000)
    parser.add_argument("--minCoeff", type=int, default=10)
    parser.add_argument("--maxCoeff", type=int, default=100)
    parser.add_argument("--minG", type=int, default=0)
    parser.add_argument("--maxG", type=int, default=0)
    parser.add_argument("--minBiasLinNonLin", type=int, default=5)
    parser.add_argument("--maxBiasLinNonLin", type=int, default=20)
    parser.add_argument("--N_f", type=int, default=5)
    parser.add_argument("--P", type=int, default=10)
    parser.add_argument("--g_sd", type=int, default=2)
    parser.add_argument("--SNRmin", type=int, default=10)
    parser.add_argument("--SNRmax", type=int, default=40)
    # ============================= Rawboost data augmentation ============================= #

    if not os.path.exists("models"):
        os.mkdir("models")
    args = parser.parse_args()

    # --- Print args (sorted by key); only rank 0 prints ---
    # We haven't initialized distributed yet; print raw args once (single-process phase)
    print("==== Parsed Args ====")
    for k in sorted(vars(args).keys()):
        print(f"{k}: {getattr(args, k)}")
    print("=====================")

    # cuDNN flags according to your args
    torch.backends.cudnn.deterministic = args.cudnn_deterministic_toggle
    torch.backends.cudnn.benchmark = args.cudnn_benchmark_toggle

    # Reproducibility
    set_random_seed(args.seed, args)

    # Setup distributed
    use_dist, local_rank = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    track = args.track
    model_tag = "model"
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_save_path = os.path.join("models", model_tag)
    if rank == 0 and (not os.path.exists(model_save_path)):
        os.mkdir(model_save_path)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    print0("Device:", device, "| World size:", world_size, "| Rank:", rank)

    # Model
    model = Model(args, device).to(device)
    nb_params = sum(p.numel() for p in model.parameters())
    print0("nb_params:", nb_params)

    # Wrap with DDP if needed (necessary change: find_unused_parameters=True to support unused branches)
    if use_dist:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )
    else:
        # Single machine single GPU/CPU: do not wrap with DataParallel (keep stable)
        pass

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Optional load (compatible with DP/DDP checkpoints)
    if args.model_path:
        state = torch.load(args.model_path, map_location=device)
        if any(k.startswith("module.") for k in state.keys()):
            from collections import OrderedDict

            new_state = OrderedDict()
            for k, v in state.items():
                new_state[k.replace("module.", "")] = v
            state = new_state
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(state)
        print0("Model loaded : {}".format(args.model_path))

    # ====================== EVAL MODES ======================
    if args.track == "In-the-Wild":
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print0("no. of eval trials", len(file_eval))
        eval_set = Dataset_in_the_wild_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        if args.eval_output is None:
            raise ValueError("--eval_output must be set in eval mode.")
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        cleanup_distributed()
        sys.exit(0)

    prefix_2021 = "ASVspoof2021.{}".format(track)
    if args.eval:
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print0("no. of eval trials", len(file_eval))
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        if args.eval_output is None:
            raise ValueError("--eval_output must be set in eval mode.")
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        cleanup_distributed()
        sys.exit(0)

    # ====================== TRAIN ======================
    # Example (kept commented): original single-set training
    # d_label_trn, file_train = genSpoof_list(
    #     dir_meta=os.path.join(args.protocols_path + 'ASVspoof2021_DF_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),
    #     is_train=True, is_eval=False
    # )
    # print0('no. of training trials', len(file_train))
    # train_set = Dataset_ASVspoof2019_train(
    #     args, list_IDs=file_train, labels=d_label_trn,
    #     base_dir=os.path.join(args.database_path + 'ASVspoof2019_LA_train/flac/'),
    #     algo=args.algo
    # )

    # 1) Assemble the training sets you want to mix
    datasets = [
        # {
        #     "protocol": os.path.join('/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/protocols/original/ASVspoof2019.LA.cm.train.trn.txt'),
        #     "base_dir": "/public/home/qinxy/AudioData/Antispoofing/ASVspoof2019/LA/ASVspoof2019_LA_train/flac/",
        # },
        # {
        #     "protocol": "/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/trainData/asvspoof2019_50_watermarked_train.txt",
        #     "base_dir": "/public/home/qinxy/AudioData/Antispoofing/ASVspoof2019/LA/ASVspoof2019_LA_train/half_A/",
        # },
        # {
        #     "protocol": "/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/trainData/odss_systems.txt",
        #     "base_dir": "/public/home/qinxy/AudioData/Antispoofing/odss/",
        # },
        # {
        #     "protocol": "/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/trainData/timit_systems.txt",
        #     "base_dir": "/public/home/qinxy/AudioData/Antispoofing/TIMIT-TTS/AUG/",
        # },
        # {
        #     "protocol": "/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/trainData/asv5_train_systems.txt",
        #     "base_dir": "/public/home/qinxy/AudioData/Antispoofing/ASVspoof5/flac_T/",
        # },
        # {
        #     "protocol": "/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/trainData/for_systems.txt",
        #     "base_dir": "/public/home/qinxy/AudioData/Antispoofing/FoR/for-norm/for-norm/",
        # },
        # ... add as many as you need
    ]

    # 2) Build the mixed list
    d_label_trn, file_train, d_wm, d_base = genSpoof_list_multi(datasets)
    print0("no. of training trials", len(file_train))

    # 3) Dataset (returns (wav, cls_label, wm_label))
    train_set = Dataset_MultiTrain(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir_map=d_base,
        wm_labels=d_wm,
        algo=args.algo,
    )

    # 4) The rest of the DataLoader / training loop remains unchanged

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if use_dist else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=32,  # Multithreaded acceleration: keep your value
        shuffle=(not use_dist),  # In distributed mode, shuffling is handled by the sampler
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler,
    )

    del train_set, d_label_trn

    # Your original dev set (commented out) can be restored if needed; this script does not change it
    # d_label_dev, file_dev = ...
    # dev_set = ...
    # dev_loader = ...

    num_epochs = args.num_epochs
    writer = SummaryWriter("logs/{}".format(model_tag)) if get_rank() == 0 else None

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(num_epochs):
        if use_dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss_local, num_total_local = train_epoch(train_loader, model, args.lr, optimizer, device)

        # Global sample-weighted aggregation: sum(loss*count), sum(count) → global average
        loss_sum = torch.tensor([running_loss_local * num_total_local], dtype=torch.float64, device=device)
        cnt_sum = torch.tensor([num_total_local], dtype=torch.float64, device=device)
        if dist_is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_sum, op=dist.ReduceOp.SUM)
        running_loss_global = (loss_sum.item() / max(cnt_sum.item(), 1.0))

        stop_tensor = torch.zeros(1, device=device)
        if get_rank() == 0:
            # Keep your early-stopping logic unchanged (based on training loss)
            if running_loss_global < best_val_loss:
                best_val_loss = running_loss_global
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if writer is not None:
                writer.add_scalar("loss", running_loss_global, epoch)
            print0("\nEpoch {} - train_loss(avg): {:.6f}".format(epoch, running_loss_global))

            # Rank 0 only: save checkpoint
            target_model = model.module if hasattr(model, "module") else model
            torch.save(target_model.state_dict(), os.path.join(model_save_path, f"epoch_{epoch}.pth"))

            if patience_counter >= 1:
                print0("Early stopping triggered, best model is epoch: {}".format(epoch - 1))
                stop_tensor.fill_(1.0)

        # Broadcast whether to stop early
        if dist_is_initialized():
            dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() > 0.5:
            break

    if writer is not None:
        writer.close()

    # Training summary: print the best epoch (according to your definition)
    if get_rank() == 0:
        if best_epoch >= 0:
            print0(f"\nTraining finished. Best model: epoch_{best_epoch}.pth  (train_loss={best_val_loss:.6f})")
        else:
            print0("\nTraining finished. No best epoch recorded.")

    cleanup_distributed()
