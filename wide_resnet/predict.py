"""Write a submission CSV from a trained checkpoint with multi-view test-time augmentation.

    python predict.py outputs/r1/runs/s50/final.pt submissions/submission_r1_s50_tta10.csv

Views: center + 2px shifts (left/right/up/down), each with and without horizontal flip
(10 views, softmax-averaged). Chosen on the 5k held-out split with s45: 97.62% vs 97.46% for flip-only.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from cifar_resnet import model_from_cfg, count_params, find_data_root, load_data, stratified_split, write_submission

SHIFTS = [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)]


@torch.no_grad()
def predict_tta(model, x_uint8, mean, std, device, bs=250):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    probs = []
    for i in range(0, len(x_uint8), bs):
        xb = torch.from_numpy(x_uint8[i:i + bs]).to(device).permute(0, 3, 1, 2).float().div_(255)
        xb = F.pad((xb - mean_t) / std_t, (2, 2, 2, 2), mode="reflect")
        p = 0
        for dx, dy in SHIFTS:
            v = xb[:, :, 2 + dy:34 + dy, 2 + dx:34 + dx]
            p = p + F.softmax(model(v), 1) + F.softmax(model(torch.flip(v, dims=[3])), 1)
        probs.append((p / (2 * len(SHIFTS))).cpu())
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_csv")
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
    ap.add_argument("--val", action="store_true", help="also report accuracy on the 5k held-out split")
    ap.add_argument("--cache_dir", default="cache")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    model = model_from_cfg(ck["cfg"])
    model.load_state_dict(ck["ema"] if args.weights == "ema" and ck.get("ema") else ck["model"])
    model.to(device).eval()
    n = count_params(model)
    assert n <= 5_000_000, f"{n:,} params exceeds the 5M limit"
    print(f"{args.ckpt} ({args.weights}): {n:,} trainable params")

    xtr, ytr, xte, ids = load_data(find_data_root(), args.cache_dir)
    if args.val:
        _, va = stratified_split(ytr, 5000)
        acc = (predict_tta(model, xtr[va], ck["mean"], ck["std"], device).argmax(1) == ytr[va]).mean()
        print(f"held-out acc (10-view TTA): {acc:.4f}  [only meaningful for models trained with --val_split 5000]")
    probs = predict_tta(model, xte, ck["mean"], ck["std"], device)
    np.save(args.out_csv.replace(".csv", "_probs.npy"), probs)
    write_submission(args.out_csv, ids, probs.argmax(1))
    print(f"wrote {args.out_csv}: {len(ids)} rows, class counts {np.bincount(probs.argmax(1), minlength=10).tolist()}")


if __name__ == "__main__":
    main()
