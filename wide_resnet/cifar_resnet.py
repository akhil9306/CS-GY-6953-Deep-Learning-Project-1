"""CIFAR-10 ResNet Architecture Challenge (<5M trainable params, trained from scratch).

One self-contained file: model, data pipeline, training, distillation, inference, and a
multi-GPU launcher used when this file runs as a Kaggle kernel.

Usage (local / cluster):
    python cifar_resnet.py --name s0 --arch wrn28_s --val_split 5000 --epochs 200
    python cifar_resnet.py --name t0 --arch wrn28_10 --epochs 200
    python cifar_resnet.py --name kd --arch wrn28_s --teacher_ckpt runs/t0/final.pt --kd_alpha 0.9

As a Kaggle kernel (no CLI args): runs every job in PLAN, one queue per GPU.
"""
import argparse
import copy
import csv
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PLAN = None  # filled in by make_kernel.py when staging a Kaggle kernel

MAX_PARAMS = 5_000_000


# ----------------------------------------------------------------------------- model
class PreActBlock(nn.Module):
    """Pre-activation residual block: y = S(x) + F(x), F = conv(relu(bn(conv(relu(bn(x))))))."""

    def __init__(self, cin, cout, stride, drop):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.shortcut = None
        if stride != 1 or cin != cout:  # learned projection skip
            self.shortcut = nn.Conv2d(cin, cout, 1, stride, 0, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        sc = self.shortcut(o) if self.shortcut is not None else x
        o = self.conv1(o)
        o = self.conv2(self.drop(F.relu(self.bn2(o))))
        return o + sc


class WideResNet(nn.Module):
    def __init__(self, depth, widths, num_classes=10, drop=0.0, stem=16):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4"
        n = (depth - 4) // 6
        self.stem = nn.Conv2d(3, stem, 3, 1, 1, bias=False)
        layers, cin = [], stem
        for i, w in enumerate(widths):
            for j in range(n):
                layers.append(PreActBlock(cin, w, 2 if (i > 0 and j == 0) else 1, drop))
                cin = w
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        o = self.blocks(self.stem(x))
        o = F.relu(self.bn(o))
        o = F.adaptive_avg_pool2d(o, 1).flatten(1)
        return self.fc(o)


class ShakeDropFn(torch.autograd.Function):
    """ShakeDrop (Yamada et al. 2019): scale the residual branch by b + a - b*a forward and b + c - b*c backward,
    with b ~ Bernoulli(p), a ~ U(-1, 1), c ~ U(0, 1) drawn per sample."""

    @staticmethod
    def forward(ctx, x, p):
        gate = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(p)
        alpha = torch.empty_like(gate).uniform_(-1, 1)
        ctx.save_for_backward(gate)
        return x * (gate + alpha - gate * alpha)

    @staticmethod
    def backward(ctx, grad):
        (gate,) = ctx.saved_tensors
        beta = torch.empty_like(gate).uniform_(0, 1)
        return grad * (gate + beta - gate * beta), None


class PyramidBlock(nn.Module):
    """PyramidNet basic block: BN-conv-BN-ReLU-conv-BN, plus a parameter-free shortcut
    (avg-pool when downsampling, zero-padded channels): y = S(x) + ShakeDrop(F(x))."""

    def __init__(self, cin, cout, stride, p_shake):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.stride, self.pad, self.p_shake = stride, cout - cin, p_shake

    def forward(self, x):
        o = self.conv1(self.bn1(x))
        o = self.bn3(self.conv2(F.relu(self.bn2(o))))
        if self.p_shake < 1.0:
            o = ShakeDropFn.apply(o, self.p_shake) if self.training else o * self.p_shake
        sc = F.avg_pool2d(x, 2) if self.stride == 2 else x
        if self.pad:
            sc = F.pad(sc, (0, 0, 0, 0, 0, self.pad))
        return o + sc


class PyramidNet(nn.Module):
    """PyramidNet-<depth> (Han et al. 2017): width grows linearly by alpha over all blocks, from 16 to 16+alpha.
    shake_pL < 1 enables ShakeDrop with survival p decaying linearly from 1 to shake_pL."""

    def __init__(self, depth, alpha, num_classes=10, shake_pL=1.0, drop=0.0):
        super().__init__()
        assert (depth - 2) % 6 == 0, "PyramidNet (basic) depth must be 6n+2"
        n = (depth - 2) // 6
        total = 3 * n
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16))
        layers, cin, width = [], 16, 16.0
        for i in range(total):
            width += alpha / total
            stride = 2 if i in (n, 2 * n) else 1
            p = 1.0 - (i + 1) / total * (1.0 - shake_pL)
            layers.append(PyramidBlock(cin, int(round(width)), stride, p))
            cin = int(round(width))
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        o = self.blocks(self.stem(x))
        o = F.relu(self.bn(o))
        o = F.adaptive_avg_pool2d(o, 1).flatten(1)
        return self.fc(o)


ARCHS = {
    # submission candidates (<5M)
    "wrn28_s": dict(depth=28, widths=(64, 128, 224)),  # WRN-28-4 with a slimmer last stage: 4.84M params
    "wrn22_4": dict(depth=22, widths=(64, 128, 256)),
    "wrn16_4": dict(depth=16, widths=(64, 128, 256)),  # tiny, for smoke tests
    "pyr110_sd": dict(kind="pyramid", depth=110, alpha=96, shake_pL=0.5),  # PyramidNet-110 + ShakeDrop
    "pyr20_sd": dict(kind="pyramid", depth=20, alpha=48, shake_pL=0.5),  # tiny, for smoke tests
    # teacher only (never submitted)
    "wrn28_10": dict(depth=28, widths=(160, 320, 640)),
}


def model_from_cfg(cfg):
    """cfg as stored in checkpoints; checkpoints from rounds 1-2 have no 'kind' and are Wide-ResNets."""
    cfg = dict(cfg)
    kind = cfg.pop("kind", "wrn")
    return PyramidNet(**cfg) if kind == "pyramid" else WideResNet(**cfg)


def build_model(arch, drop=0.0):
    return model_from_cfg(dict(**ARCHS[arch], drop=drop))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- data
def find_data_root(hint=None):
    cands = [hint] if hint else []
    cands += [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train_labels.csv", recursive=True)]
    cands += [os.path.join(os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0])), "data"), "data"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "train_labels.csv")):
            return c
    raise FileNotFoundError("could not find train_labels.csv; pass --data_dir")


def load_images(folder, filenames):
    return np.stack([np.array(Image.open(os.path.join(folder, f)).convert("RGB")) for f in filenames])


def load_data(root, cache_dir):
    cache = os.path.join(cache_dir, "cifar_cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["ids"]
    with open(os.path.join(root, "train_labels.csv")) as f:
        rows = list(csv.DictReader(f))
    xtr = load_images(os.path.join(root, "train_images"), [r["filename"] for r in rows])
    ytr = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    with open(os.path.join(root, "test.csv")) as f:
        trows = list(csv.DictReader(f))
    xte = load_images(os.path.join(root, "test_images"), [r["filename"] for r in trows])
    ids = np.array([int(r["id"]) for r in trows], dtype=np.int64)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache, xtr=xtr, ytr=ytr, xte=xte, ids=ids)
    return xtr, ytr, xte, ids


def stratified_split(y, n_val, seed=0):
    """Fixed per-class holdout so every run validates on the same images."""
    rng = np.random.RandomState(seed)
    per_class = n_val // 10
    val = np.concatenate([rng.permutation(np.where(y == c)[0])[:per_class] for c in range(10)])
    train = np.setdiff1d(np.arange(len(y)), val)
    return train, np.sort(val)


class Cutout:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):  # img: normalized CHW tensor
        h, w = img.shape[1:]
        cy, cx = np.random.randint(h), np.random.randint(w)
        y0, y1 = max(0, cy - self.size // 2), min(h, cy + self.size // 2)
        x0, x1 = max(0, cx - self.size // 2), min(w, cx + self.size // 2)
        img[:, y0:y1, x0:x1] = 0.0
        return img


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y, self.t = x, y, transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.t(Image.fromarray(self.x[i])), int(self.y[i])


def make_transforms(mean, std, aug, cutout):
    import torchvision.transforms as T
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    train = [T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()]
    if aug == "ta":
        train.append(T.TrivialAugmentWide())
    elif aug == "aa":
        train.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
    train += norm
    if cutout > 0:
        train.append(Cutout(cutout))
    return T.Compose(train), T.Compose(norm)


# ----------------------------------------------------------------------------- training utils
class EMA:
    """Exponential moving average of all weights and BN buffers."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.steps = 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


@torch.no_grad()
def predict_logits(model, x_uint8, mean, std, device, tta=True, bs=500):
    """Logits for uint8 NHWC images; tta=True averages with the horizontal flip."""
    model.eval()
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(x_uint8), bs):
        xb = torch.from_numpy(x_uint8[i:i + bs]).to(device).permute(0, 3, 1, 2).float().div_(255)
        xb = ((xb - mean_t) / std_t).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            lo = model(xb).float()
            if tta:
                lo = (lo + model(torch.flip(xb, dims=[3])).float()) / 2
        out.append(lo.cpu())
    return torch.cat(out).numpy()


def kd_loss(s_logits, t_logits, y, alpha, T, smoothing):
    hard = F.cross_entropy(s_logits, y, label_smoothing=smoothing)
    soft = F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1), reduction="batchmean") * T * T
    return alpha * soft + (1 - alpha) * hard


def write_submission(path, ids, preds):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, preds):
            w.writerow([int(i), int(p)])


# ----------------------------------------------------------------------------- train
def train(args):
    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "log.txt"), "a")

    def log(msg):
        line = f"[{args.name} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"args: {json.dumps(vars(args))}")
    root = find_data_root(args.data_dir)
    xtr, ytr, xte, ids = load_data(root, args.cache_dir)
    if args.prepare_only:
        log(f"cached {len(xtr)} train / {len(xte)} test images")
        return
    if args.limit:
        xtr, ytr = xtr[:args.limit], ytr[:args.limit]
    if args.val_split:
        tr_idx, va_idx = stratified_split(ytr, args.val_split)
        xva, yva = xtr[va_idx], ytr[va_idx]
        xtr, ytr = xtr[tr_idx], ytr[tr_idx]
    else:
        xva = yva = None
    # normalization stats come from the training images only
    mean = tuple((xtr.reshape(-1, 3).mean(0) / 255).tolist())
    std = tuple((xtr.reshape(-1, 3).std(0) / 255).tolist())
    log(f"data root={root} train={len(xtr)} val={0 if xva is None else len(xva)} test={len(xte)}")

    model = build_model(args.arch, args.drop)
    n_params = count_params(model)
    log(f"arch={args.arch} trainable params={n_params:,} ({'OK' if n_params <= MAX_PARAMS else 'OVER 5M - teacher only'})")
    model.to(device, memory_format=torch.channels_last)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    teacher = None
    if args.teacher_ckpt:
        ck = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher = model_from_cfg(ck["cfg"])
        teacher.load_state_dict(ck["ema"] if args.teacher_weights == "ema" and ck.get("ema") else ck["model"])
        teacher.to(device, memory_format=torch.channels_last).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if xva is not None:
            t_acc = (predict_logits(teacher, xva, mean, std, device).argmax(1) == yva).mean()
            log(f"teacher {args.teacher_ckpt} val acc (flip-TTA) = {t_acc:.4f}")

    tf_train, _ = make_transforms(mean, std, args.aug, args.cutout)
    loader = torch.utils.data.DataLoader(
        ArrayDataset(xtr, ytr, tf_train), batch_size=args.bs, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 and args.no_bn_wd else decay).append(p)
    lr = args.lr * args.bs / 128
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                          lr=lr, momentum=0.9, nesterov=True)
    steps_per_epoch = len(loader)
    sched_state = {"total": args.epochs * steps_per_epoch, "warm": args.warmup * steps_per_epoch}

    def lr_lambda(step):  # linear warmup then cosine to 0; total can shrink after timing epoch 2
        if step < sched_state["warm"]:
            return (step + 1) / sched_state["warm"]
        prog = (step - sched_state["warm"]) / max(1, sched_state["total"] - sched_state["warm"])
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    epochs = args.epochs
    history = []
    epoch = 0
    last_path = os.path.join(out_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        if ema is not None:
            ema.model.load_state_dict(st["ema"])
            ema.steps = st["ema_steps"]
        sched_state.update(st["sched_state"])
        epoch, epochs, history = st["epoch"], st["epochs"], st["history"]
        log(f"resumed from {last_path} at epoch {epoch}/{epochs}")

    def compute_loss(xb, yb, t_logits):
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            lo = model(xb).float()
        if t_logits is not None:
            return lo, kd_loss(lo, t_logits, yb, args.kd_alpha, args.kd_T, args.smoothing)
        return lo, F.cross_entropy(lo, yb, label_smoothing=args.smoothing)

    # SAM (Foret et al. 2021): after the normal backward, move the weights by e = rho * g / ||g|| (the locally
    # "worst" direction), recompute the gradient there, move back, and let the optimizer apply that gradient.
    # e is independent of the fp16 loss scale (it cancels in g / ||g||). BN running stats are frozen during
    # the 2nd pass so they are only updated once per step.
    sam_params = [p for p in model.parameters() if p.requires_grad]
    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def sam_perturb_and_regrad(xb, yb, t_logits):
        ps = [p for p in sam_params if p.grad is not None]
        gnorm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in ps]))).item()
        if not (math.isfinite(gnorm) and gnorm > 0):
            return  # overflowed fp16 step: keep the grads, GradScaler skips this step anyway
        with torch.no_grad():
            orig = [p.detach().clone() for p in ps]
            torch._foreach_add_(ps, [p.grad for p in ps], alpha=args.sam_rho / gnorm)
        opt.zero_grad(set_to_none=True)
        moms = [m.momentum for m in bn_layers]
        for m in bn_layers:
            m.momentum = 0.0
        _, loss2 = compute_loss(xb, yb, t_logits)
        scaler.scale(loss2).backward()
        for m, mom in zip(bn_layers, moms):
            m.momentum = mom
        with torch.no_grad():
            torch._foreach_copy_(ps, orig)

    while epoch < epochs:
        model.train()
        t0 = time.time()
        tot_loss, correct, seen = 0.0, 0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            t_logits = None
            if teacher is not None:
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    t_logits = teacher(xb).float()
            logits, loss = compute_loss(xb, yb, t_logits)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.sam_rho > 0:
                sam_perturb_and_regrad(xb, yb, t_logits)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            tot_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
        ep_time = time.time() - t0
        epoch += 1

        # after epoch 2 (epoch 1 includes cudnn autotuning), shrink the schedule to fit the time budget
        if epoch == 2 and args.time_budget_h > 0:
            remaining = args.time_budget_h * 3600 - (time.time() - t_start) - 300
            fit = epoch + int(remaining / (ep_time * 1.05))
            if fit < epochs:
                log(f"time budget {args.time_budget_h}h: epoch takes {ep_time:.0f}s -> reducing epochs {epochs} -> {fit}")
                epochs = fit
                sched_state["total"] = epochs * steps_per_epoch

        rec = {"epoch": epoch, "loss": tot_loss / seen, "train_acc": correct / seen,
               "lr": opt.param_groups[0]["lr"], "time": ep_time}
        if xva is not None and (epoch % args.eval_every == 0 or epoch > epochs - 10 or epoch == epochs):
            rec["val_acc"] = float((predict_logits(model, xva, mean, std, device, tta=False).argmax(1) == yva).mean())
            if ema is not None:
                rec["val_acc_ema"] = float((predict_logits(ema.model, xva, mean, std, device, tta=False).argmax(1) == yva).mean())
        history.append(rec)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()) + f" /{epochs}")
        if args.ckpt_every and (epoch % args.ckpt_every == 0 or epoch == epochs):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "ema": ema.model.state_dict() if ema else None,
                        "ema_steps": ema.steps if ema else 0, "sched_state": sched_state,
                        "epoch": epoch, "epochs": epochs, "history": history}, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)

    # ------------------------------------------------------------------ final evaluation + test predictions
    cfg = dict(**ARCHS[args.arch], drop=args.drop)
    torch.save({"cfg": cfg, "model": model.state_dict(), "ema": ema.model.state_dict() if ema else None,
                "mean": mean, "std": std, "params": n_params, "args": vars(args)}, os.path.join(out_dir, "final.pt"))
    summary = {"name": args.name, "arch": args.arch, "params": n_params, "epochs_run": epochs,
               "train_size": len(xtr), "hours": (time.time() - t_start) / 3600}
    variants = {"raw": model}
    if ema is not None:
        variants["ema"] = ema.model
    for tag, m in variants.items():
        if xva is not None:
            summary[f"val_acc_{tag}"] = float((predict_logits(m, xva, mean, std, device, tta=False).argmax(1) == yva).mean())
            summary[f"val_acc_{tag}_tta"] = float((predict_logits(m, xva, mean, std, device, tta=True).argmax(1) == yva).mean())
        test_logits = predict_logits(m, xte, mean, std, device, tta=True)
        np.save(os.path.join(out_dir, f"test_logits_{tag}.npy"), test_logits)
        if n_params <= MAX_PARAMS:
            write_submission(os.path.join(out_dir, f"submission_{args.name}_{tag}.csv"), ids, test_logits.argmax(1))
    json.dump({"summary": summary, "history": history}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    log(f"SUMMARY {json.dumps(summary)}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--arch", default="wrn28_s", choices=list(ARCHS))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--time_budget_h", type=float, default=0, help="shrink epochs to fit this many hours (0=off)")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="lr for batch 128; scaled linearly with --bs")
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--no_bn_wd", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--aug", default="ta", choices=["none", "ta", "aa"])
    p.add_argument("--cutout", type=int, default=16)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--val_split", type=int, default=0, help="hold out this many train images (stratified)")
    p.add_argument("--teacher_ckpt", default="")
    p.add_argument("--teacher_weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--kd_alpha", type=float, default=0.9)
    p.add_argument("--kd_T", type=float, default=4.0)
    p.add_argument("--sam_rho", type=float, default=0.0, help="SAM neighbourhood radius (0 = plain SGD; 0.05 typical)")
    p.add_argument("--resume", action="store_true", help="continue from <out_dir>/<name>/last.pt if present")
    p.add_argument("--ckpt_every", type=int, default=10, help="save last.pt every N epochs (0 = never)")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prepare_only", action="store_true", help="only decode PNGs into the npz cache")
    p.add_argument("--limit", type=int, default=0, help="debug: use only the first N train images")
    p.add_argument("--data_dir", default="")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--out_dir", default="runs")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- kaggle launcher
def launch(plan):
    """plan: {"gpu0": [[args...], ...], "gpu1": [...]}; jobs on one GPU run sequentially."""
    work = os.environ.get("KAGGLE_WORK", "/kaggle/working")
    find_data_root()  # fail fast
    os.makedirs(f"{work}/cache", exist_ok=True)
    # keep a copy of the exact code next to the outputs and run every job from it
    me = globals().get("__file__") or sys.argv[0]
    script = f"{work}/cifar_resnet_run.py"
    with open(me, encoding="utf-8") as fi, open(script, "w", encoding="utf-8") as fo:
        fo.write(fi.read())
    subprocess.run([sys.executable, script, "--name", "prepare", "--prepare_only",
                    "--cache_dir", f"{work}/cache", "--out_dir", f"{work}/prepare"], check=True)

    def resolve(job):  # "TEACHER:<run>" -> path of that run's checkpoint (this kernel or an attached one)
        out = []
        for a in job:
            if isinstance(a, str) and a.startswith("TEACHER:"):
                run = a.split(":", 1)[1]
                hits = glob.glob(f"{work}/runs/{run}/final.pt") + \
                    glob.glob(f"/kaggle/input/**/{run}/final.pt", recursive=True)
                if not hits:
                    raise FileNotFoundError(f"teacher checkpoint for run {run} not found")
                a = hits[0]
            out.append(str(a))
        return out

    def worker(gpu, jobs):
        for job in jobs:
            cmd = [sys.executable, script] + resolve(job) + ["--cache_dir", f"{work}/cache", "--out_dir", f"{work}/runs"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            print(f"[gpu{gpu}] start: {' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd, env=env)
            print(f"[gpu{gpu}] exit code {r.returncode}", flush=True)

    n_gpu = max(1, torch.cuda.device_count())
    threads = []
    for i, key in enumerate(sorted(plan)):
        th = threading.Thread(target=worker, args=(i % n_gpu, plan[key]))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    for path in sorted(glob.glob(f"{work}/runs/*/metrics.json")):
        print(json.dumps(json.load(open(path))["summary"]), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1 and PLAN is not None:
        launch(PLAN)
    else:
        train(parse_args())
