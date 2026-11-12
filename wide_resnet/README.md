# Wide-ResNet student with knowledge distillation (and PyramidNet)

This folder holds the code for the **Wide-ResNet + distillation** and **PyramidNet** approaches in the report (*Earlier approaches*, Table 5 and the distillation figure). It is self-contained and separate from the code of the final SLA + SAM model.

## Results

All validation numbers use a stratified 45,000 / 5,000 split (seed 0). Kaggle scores are for the version retrained on all 50,000 images (PyramidNet: 45,000).

| Model | Params | Validation (single view) | Validation (TTA) | Kaggle public |
|---|---|---|---|---|
| WRN-28 student (`s45` / `s50`) | 4,839,578 | 97.16% | 97.62% | 0.892 |
| WRN-28-10 teacher (`t45`, not eligible) | 36,479,194 | 97.88% | 98.02% | – |
| WRN-28 + KD from WRN-28-10 (`kd45` / `kd50`) | 4,839,578 | 97.60% | 97.84% | 0.892 |
| PyramidNet-110 + ShakeDrop + KD (`pk45`) | 4,804,512 | – | 97.18% | 0.883 |

TTA is 10 views (original plus four 2-pixel shifts, each also flipped) for the students and the horizontal flip for the teacher.

## Files

| File | What it is |
|---|---|
| `cifar_resnet.py` | Everything for training: models (pre-activation Wide-ResNet, PyramidNet + ShakeDrop), data loading and the fixed split, augmentation (crop, flip, TrivialAugment, Cutout), label smoothing, EMA, knowledge distillation with a live teacher, optional SAM (`--sam_rho`), checkpoint/resume, flip-TTA predictions |
| `predict.py` | Writes a submission CSV with 10-view TTA; `--val` also reports held-out accuracy; asserts ≤ 5M parameters |
| `make_kernel.py` | Packages runs as Kaggle GPU kernels (how `s45`, `s50`, `t45`, `kd45` and `kd50` were trained on Kaggle T4 GPUs) |
| `notebooks/pyramidnet_colab.ipynb` | Colab notebook for the PyramidNet student (`pk45` / `pk50`) on an A100 |
| `notebooks/sam_colab.ipynb` | Colab notebook for the WRN-28 + KD recipe with SAM (`kdsam45` / `kdsam50`) |
| `results/*_metrics.json` | Summary and per-epoch history (loss, train accuracy, held-out accuracy) of each run |
| `submissions/` | The two submitted CSVs (both 0.892 public) |
| `plot_kd_curves.py`, `figures/wrn_kd_curves.pdf` | The report figure and the script that draws it from `results/` |

## Reproduce

Put the competition data (folder with `train_labels.csv`, `train_images/`, `test.csv`, `test_images/`) in `data/`. Requires PyTorch, torchvision, NumPy and Pillow, plus a GPU.

```bash
# student alone: validation run (45k/5k) and final run (all 50k)
python cifar_resnet.py --name s45 --arch wrn28_s --epochs 250 --val_split 5000
python cifar_resnet.py --name s50 --arch wrn28_s --epochs 250

# teacher (36.5M, never submitted)
python cifar_resnet.py --name t45 --arch wrn28_10 --epochs 200 --val_split 5000

# distilled student (KD loss 0.9 * KL at T = 4 + 0.1 * CE)
python cifar_resnet.py --name kd45 --arch wrn28_s --epochs 250 --val_split 5000 --teacher_ckpt runs/t45/final.pt
python cifar_resnet.py --name kd50 --arch wrn28_s --epochs 250 --teacher_ckpt runs/t45/final.pt

# PyramidNet-110 + ShakeDrop student
python cifar_resnet.py --name pk45 --arch pyr110_sd --epochs 300 --wd 1e-4 --val_split 5000 --teacher_ckpt runs/t45/final.pt

# 10-view TTA submission (+ held-out accuracy for a 45k run)
python predict.py runs/kd45/final.pt kd45.csv --val
python predict.py runs/kd50/final.pt submission_kd50_tta10.csv
```

Each run writes `runs/<name>/final.pt`, `metrics.json` and flip-TTA submission CSVs. Add `--resume` to continue from `runs/<name>/last.pt` (saved every 10 epochs) after an interruption.

Model checkpoints and the dataset are not included because of their size.
