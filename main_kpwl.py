#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import copy
import hashlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data_utils_SSL_kpwl import (
    genSpoof_list,                                   # Return (labels, files, wm_labels)
    Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2021_eval,
    Dataset_in_the_wild_eval
)
from model_kpwl import Model  # Return (task_logprob, domain_logit, h); this script only uses task_logprob
from core_scripts.startup_config import set_random_seed


# ====================== Utils ======================
def dedup_score_file(path: str, key_col: int = 0, sep: str = None):
    import tempfile
    seen = set()
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path)+".dedup.", dir=dir_name)
    os.close(fd)
    kept, dropped = 0, 0
    with open(path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            parts = line.rstrip("\n")
            key = parts.split(sep)[key_col]
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            fout.write(line)
            kept += 1
    os.replace(tmp_path, path)
    return kept, dropped

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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
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


# ====================== Optional: disk cache (no value changes) ======================
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, cache_dir: str, dataset_signature: str):
        self.base = base_ds
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.prefix = hashlib.sha1(dataset_signature.encode('utf-8')).hexdigest()[:16]

    def __len__(self):
        return len(self.base)

    def _key(self, idx: int) -> str:
        return os.path.join(self.cache_dir, f"{self.prefix}_{idx:08d}.pt")

    def __getitem__(self, idx):
        path = self._key(idx)
        if os.path.exists(path):
            return torch.load(path, map_location='cpu')
        data = self.base[idx]
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save(data, tmp)
        try:
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
        return data


# ====================== Eval (task head only) ======================
def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.NLLLoss(weight=weight)
    pbar = tqdm(dev_loader, disable=(get_rank()!=0))
    with torch.inference_mode():
        for batch in pbar:
            if len(batch) == 2:
                batch_x, batch_y = batch
            else:
                batch_x, batch_y = batch[0], batch[1]
            B = batch_x.size(0)
            num_total += B
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)
            out = model(batch_x)  # do not use adversarial branch
            task_logprob = out[0] if isinstance(out, (tuple, list)) else out
            _, batch_pred = task_logprob.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            batch_loss = criterion(task_logprob, batch_y)
            val_loss += (batch_loss.item() * B)

    if dist_is_initialized():
        t = torch.tensor([val_loss, num_total, num_correct], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        val_loss, num_total, num_correct = t.tolist()

    val_loss = val_loss / max(num_total, 1.0)
    acc = 100.0 * (num_correct / max(num_total, 1.0))
    return val_loss, acc

def produce_evaluation_file(dataset, model, device, save_path, allowed_ids=None):
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False) if dist_is_initialized() else None
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False,
                             sampler=sampler, num_workers=32, pin_memory=True)

    model.eval()
    rank = get_rank()
    world_size = get_world_size()
    tmp_path = save_path if world_size == 1 else f"{save_path}.rank{rank}"

    with open(tmp_path, 'w') as fh:
        pbar = tqdm(data_loader, disable=(rank!=0))
        with torch.inference_mode():
            for batch in pbar:
                if len(batch) == 2:
                    batch_x, utt_id = batch
                else:
                    batch_x, utt_id = batch[0], batch[-1]
                batch_x = batch_x.to(device, non_blocking=True)
                out = model(batch_x)
                task_logprob = out[0] if isinstance(out, (tuple, list)) else out
                score = task_logprob[:, 1].data.cpu().numpy().ravel().tolist()
                for f, cm in zip(utt_id, score):
                    fh.write(f'{f} {cm}\n')

    if world_size > 1:
        dist.barrier()
        if rank == 0:
            with open(save_path, 'w') as out:
                for r in range(world_size):
                    part = f"{save_path}.rank{r}"
                    with open(part, 'r') as fh:
                        out.write(fh.read())
            for r in range(world_size):
                try:
                    os.remove(f"{save_path}.rank{r}")
                except OSError:
                    pass
        dist.barrier()

    if rank == 0:
        kept, dropped = dedup_score_file(save_path)
        print0(f"[dedup] {save_path}: kept={kept}, dropped={dropped}")
        print0(f'Scores saved to {save_path}')


# ====================== Training: distillation & L2-SP ======================
@torch.no_grad()
def teacher_logits(teacher_model, x, device):
    out = teacher_model(x.to(device, non_blocking=True))  # log-prob
    return out[0] if isinstance(out, (tuple, list)) else out

def l2sp_loss(model, anchor, exclude_prefix=()):
    loss = 0.0
    tgt = model.module if hasattr(model, "module") else model
    for n, p in tgt.named_parameters():
        if (not p.requires_grad):
            continue
        if any(n.startswith(px) for px in exclude_prefix):
            continue
        a = anchor.get(n, None)
        if a is not None:
            loss = loss + F.mse_loss(p, a, reduction="mean")
    return loss


# ====================== Data builder ======================
def build_train_loader(*, meta_path: str, base_dir: str, args, use_dist: bool):
    # Watermark training set: use GRL version genSpoof_list (returns wm_labels, not used here)
    d_label_trn, file_train, d_wm = genSpoof_list(
        dir_meta=meta_path, is_train=True, is_eval=False
    )

    if get_rank() == 0:
        print(f"[build_train_loader] meta={meta_path}")
        print(f"[build_train_loader] base={base_dir}")
        print(f"[build_train_loader] #trials={len(file_train)}")

    base_ds = Dataset_ASVspoof2019_train(
        args, list_IDs=file_train, labels=d_label_trn,
        base_dir=base_dir, algo=args.algo, wm_labels=d_wm
    )

    if args.cache_dir:
        sig = f"{meta_path}|{base_dir}|{len(file_train)}"
        base_ds = CachedDataset(base_ds, args.cache_dir, sig)
        if get_rank() == 0:
            print(f"[cache] enabled at {args.cache_dir}")

    train_sampler = DistributedSampler(base_ds, shuffle=True, drop_last=True) if use_dist else None
    train_loader = DataLoader(
        base_ds, batch_size=args.batch_size, num_workers=32,
        shuffle=(not use_dist), drop_last=True, pin_memory=True, sampler=train_sampler
    )
    return base_ds, train_sampler, train_loader, len(file_train)


# ====================== main ======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='稳健微调：水印域上短训练（任务CE + KD蒸馏 + L2-SP），不使用GRL/不改模型')

    # Data Path
    parser.add_argument('--wm_meta', type=str,
        default='/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/database/trainData/asvspoof2019_50_watermarked_train.txt',
        help='水印训练 protocol')
    parser.add_argument('--wm_base', type=str,
        default='/public/home/qinxy/AudioData/Antispoofing/ASVspoof2019/LA/ASVspoof2019_LA_train/half_A/',
        help='水印训练 base_dir')

    # Evaluation root (optional)
    parser.add_argument('--database_path', type=str, default='/public/home/qinxy/AudioData/Antispoofing/ASVspoof2019/LA/',help="the flac folder with the audio")
    parser.add_argument('--protocols_path', type=str, default='/public/home/qinxy/AudioData/Antispoofing/ASVspoof2019/LA/',help="the actual txt file")
    parser.add_argument('--eval_output', type=str, default=None, help='evaluation score output')
    parser.add_argument('--eval_track', type=str, default='DF', choices=['LA', 'In-the-Wild', 'DF'])
    parser.add_argument('--eval', action='store_true', default=False, help='eval mode')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=2, help='Recommend 1–2 epochs')
    parser.add_argument('--lr', type=float, default=3e-7, help='Conservatively small learning rate to avoid drift')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Distillation & L2-SP
    parser.add_argument('--beta_kd', type=float, default=0.3, help='Weight of KD distillation loss')
    parser.add_argument('--mu_l2sp', type=float, default=1e-4, help='Coefficient of L2-SP regularization')
    parser.add_argument('--freeze_fc3', type=lambda s: s.lower()!='false', default=True,
                        help='Whether to freeze the classifier head fc3 (strongly recommended True)')
    parser.add_argument('--freeze_ssl', type=lambda s: s.lower()!='false', default=True,
                        help='Whether to freeze the SSL frontend (recommended True; set False if adapting to watermark)')


    # Model loading
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_path', type=str, required=True, help='Trained model weights (used to initialize the student and teacher)')
    parser.add_argument('--comment', type=str, default=None)

    # 其它
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', default=True)
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', default=False)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--torch_threads', type=int, default=0)
    parser.add_argument('--algo', type=int, default=3, help='Rawboost algos...')
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    print("you are running main_ft_wm_kd_l2sp.py (No-GRL Fine-tune on Watermark) | keep accuracy safe")
    print("==== Parsed Args ====")
    for k in sorted(vars(args).keys()):
        print(f"{k}: {getattr(args, k)}")
    print("=====================")

    # Reproducibility
    torch.backends.cudnn.deterministic = args.cudnn_deterministic_toggle
    torch.backends.cudnn.benchmark = args.cudnn_benchmark_toggle
    if args.torch_threads and args.torch_threads > 0:
        try:
            torch.set_num_threads(args.torch_threads)
            print0(f"[threads] torch.set_num_threads({args.torch_threads})")
        except Exception as e:
            print0(f"[threads] set_num_threads failed: {e}")
    set_random_seed(args.seed, args)

    # Distributed
    use_dist, local_rank = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    # Device
    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    print0('Device:', device, '| World size:', world_size, '| Rank:', rank)

    # Model
    model = Model(args, device).to(device)
    if use_dist:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True
        )

    # Load trained weights (student init = your current main model)
    state = torch.load(args.model_path, map_location=device)
    if any(k.startswith('module.') for k in state.keys()):
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        state = new_state
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state, strict=False)
    print0(f'Model loaded from: {args.model_path}')
    if args.eval:
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print0('no. of eval trials', len(file_eval))
        if args.eval_track == 'In-the-Wild':
            eval_set = Dataset_in_the_wild_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        else:
            eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))

        if args.eval_output is None:
            print0("[eval] --eval_output not set")
            
        else:
            produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)
    # Freezing strategy (stable)
    if args.freeze_fc3 and hasattr(target, 'fc3'):
        for p in target.fc3.parameters():
            p.requires_grad_(False)
        print0("[freeze] fc3 frozen")

    if args.freeze_ssl and hasattr(target, 'ssl_model'):
        # Completely freeze SSL frontend
        if hasattr(target.ssl_model, 'set_freeze'):
            target.ssl_model.set_freeze(True)
        else:
            for p in target.ssl_model.parameters():
                p.requires_grad_(False)
        print0("[freeze] SSL backbone frozen")

    # Teacher model (frozen)
    # teacher = copy.deepcopy(target).to(device).eval()
    # target is the currently trained model (may be wrapped by DDP); args/device already available
    base = target.module if hasattr(target, "module") else target

    # 1) Create the same type of model
    teacher = Model(args, device).to(device)

    # 2) Copy weights (strict match)
    teacher.load_state_dict(base.state_dict(), strict=True)

    # 3) Freeze + eval (ensure teacher only does forward; no updates)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad_(False)
    print0("[teacher] snapshot created (frozen)")

    # L2-SP anchor: build based on “currently trainable parameters”
    anchor = {n: p.detach().clone() for n, p in target.named_parameters() if p.requires_grad}
    nb_trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    print0(f"[trainable] params: {nb_trainable/1e6:.2f}M")

    # Optimizer (only train parameters with requires_grad=True; domain head not distinguished/used here)
    trainable_params = [p for p in target.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # ========= Build training set (watermark only) =========
    train_set, train_sampler, train_loader, _ = build_train_loader(
        meta_path=args.wm_meta, base_dir=args.wm_base, args=args, use_dist=use_dist
    )
    if get_rank() == 0:
        print('no. of training trials (WM)', len(train_loader.dataset))

    # Logging
    model_tag = 'model_ft_wm'
    if args.comment:
        model_tag += f'_{args.comment}'
    model_save_path = os.path.join('models', model_tag)
    if rank == 0 and (not os.path.exists(model_save_path)):
        os.mkdir(model_save_path)
    writer = SummaryWriter('logs/{}'.format(model_tag)) if get_rank() == 0 else None

    # Training
    num_epochs = args.num_epochs
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    task_crit = nn.NLLLoss(weight=weight)

    global_step = 0
    for epoch in range(num_epochs):
        if use_dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        target.train()
        pbar = tqdm(train_loader, disable=(get_rank()!=0))
        running = {'task':0.0, 'kd':0.0, 'l2sp':0.0, 'total':0.0, 'n':0}

        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                batch_x, batch_y = batch[0], batch[1]
            else:
                raise RuntimeError("Dataset must return at least (x, y)")

            B = batch_x.size(0)
            running['n'] += B

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.view(-1).long().to(device, non_blocking=True)

            # Student forward (no adversarial)
            task_logprob, *_ = model(batch_x)
            L_task = task_crit(task_logprob, batch_y)

            # Teacher distillation (symmetric KL)
            with torch.no_grad():
                logp_teacher = teacher_logits(teacher, batch_x, device)
            KD = F.kl_div(logp_teacher, task_logprob.exp(), reduction='batchmean') + \
                 F.kl_div(task_logprob, logp_teacher.exp(), reduction='batchmean')

            # L2-SP (anchor = trainable parameters at the start of fine-tuning)
            L_anchor = l2sp_loss(model, anchor)

            L = L_task + args.beta_kd * KD + args.mu_l2sp * L_anchor

            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

            running['task']  += float(L_task.item()) * B
            running['kd']    += float(KD.item())      * B
            running['l2sp']  += float(L_anchor.item())* B
            running['total'] += float(L.item())       * B
            global_step += 1

            if get_rank()==0:
                pbar.set_description(f"L:{L.item():.4f}|T:{L_task.item():.4f}|KD:{KD.item():.4f}|A:{L_anchor.item():.4f}")

        # epoch stats
        n = max(1, running['n'])
        avg_task  = running['task']/n
        avg_kd    = running['kd']/n
        avg_l2sp  = running['l2sp']/n
        avg_total = running['total']/n

        if writer is not None and get_rank()==0:
            writer.add_scalar('train/loss_task',  avg_task,  epoch)
            writer.add_scalar('train/loss_kd',    avg_kd,    epoch)
            writer.add_scalar('train/loss_l2sp',  avg_l2sp,  epoch)
            writer.add_scalar('train/loss_total', avg_total, epoch)

        if get_rank()==0:
            print0(f"\nEpoch {epoch} | total:{avg_total:.6f} | task:{avg_task:.6f} | kd:{avg_kd:.6f} | l2sp:{avg_l2sp:.6f}")

        # Save
        if get_rank()==0:
            tgt_to_save = model.module if hasattr(model, "module") else model
            torch.save(tgt_to_save.state_dict(), os.path.join(model_save_path, f'epoch_{epoch}.pth'))

    if writer is not None:
        writer.close()

    # ========== Optional: evaluate after training ==========
    

    cleanup_distributed()
