# Wide ResNet trials: baseline → distillation → ensemble → SLA + SAM (final model)

This folder holds the notebooks for our main line of experiments. The final model is in
`DL_Project1_SLA_SAM_e600_cutrot.ipynb`: a wide 3-stage ResNet with **4,808,440 parameters**, trained for 600 epochs with
Self-supervised Label Augmentation (SLA), Sharpness-Aware Minimization (SAM), AutoAugment and Cutout.

| Final model | Value |
|---|---|
| Trainable parameters | **4,808,440** (under the 5M limit) |
| Validation accuracy, 8-view inference | **98.04%** (97.74% single view) |
| Kaggle public leaderboard | **0.906** |
| Kaggle private leaderboard | **0.878** |

## Notebooks, in the order we ran them

| # | Notebook | What it does | Val. accuracy | Kaggle public |
|---|---|---|---|---|
| 1 | `DL_Project1_ResNet_CIFAR10.ipynb` | Baseline: wide 3-stage ResNet (C = 80/160/320, B = 2/3/2, 4,798,810 params), 200 epochs, AutoAugment + Cutout, label smoothing, EMA, flip TTA | 96.52% (96.64% TTA) | 0.876 |
| 2 | `DL_Project1_KD.ipynb` | Knowledge distillation: trains the same student without a teacher, a 52.8M-parameter teacher (C = 192/384/768, B = 4/4/4), and the student distilled from the teacher (T = 4, α = 0.9) | student 96.60% → distilled 97.02%, teacher 97.38% | 0.880 / 0.879 (distilled) / 0.880 (teacher) |
| 3 | `DL_Project1_Ensemble.ipynb` | 4 ResNets on 4 disjoint 45k/5k folds; 3-of-4 majority vote, soft vote and SAMME boosting-weighted vote; BatchNorm recalibration on the test images | 96.87% (pooled, flip TTA) | 0.885 vote / **0.886** soft / 0.884 boost / 0.876 BN recal. |
| 4 | `DL_Project1_SLA_SAM.ipynb` | SLA (4 rotations, 40-way joint labels, 8-view inference) + SAM (ρ = 0.05), 200 epochs, 4,808,440 params | 97.20% (97.62% 8 views) | 0.901 |
| 5 | `DL_Project1_SLA_SAM_e600_cutrot.ipynb` | **Final model.** Same as 4, but 600 epochs and Cutout applied separately to each rotated copy | 97.74% (**98.04%** 8 views) | **0.906** (private 0.878) |

All validation numbers are on the same stratified 5,000-image hold-out (500 per class, seed 42); the ensemble models are
each validated on their own fold. The ensemble (4 × 4.8M parameters) and the distillation teacher (52.8M) exceed the 5M
limit and were only used as experiments.

## Training logs and graphs

`training_n_outputs/` has one folder per notebook with the plots it produced and the per-epoch training logs:

| Folder | Graphs | Training logs (per epoch) |
|---|---|---|
| `DL_Project1_ResNet_CIFAR10/` | loss/accuracy curves, confusion matrix | `wresnet_80-160-320_2-3-2_gpu_history.csv` |
| `DL_Project1_KD/` | curves of all three models, per-class accuracy | `student_baseline_history.csv`, `teacher_history.csv`, `student_kd_history.csv` |
| `DL_Project1_Ensemble/` | curves of the 4 models | `model0_history.csv` … `model3_history.csv` |
| `DL_Project1_SLA_SAM/` | raw vs. EMA validation loss and accuracy | `sla4_sam0.05_e200_history.csv` |
| `DL_Project1_SLA_SAM_e600_cutrot/` | raw vs. EMA validation loss and accuracy | `sla4_sam0.05_e600_cutrot_history.csv` |

The history files have one row per epoch with the learning rate, training loss and accuracy, and the validation loss and
accuracy of both the raw and the EMA weights (for the SLA notebooks the training accuracy is on the 40 joint
class × rotation labels).

## How to run

1. Put the competition `data.zip` (with `train_labels.csv`, `train_images/`, `test.csv`, `test_images/`) in the working
   directory set in each notebook's **Config** cell (`WORK_DIR` / `data_root`).
2. Run all cells. Notebooks 2–5 decode the images once and cache them as `cifar_kaggle_uint8.npz`; notebook 1 extracts
   `data.zip` into `data_root` and loads the PNGs directly.
3. Training is resumable: if the session stops, run all cells again and it continues from the last checkpoint.
   Finished runs are skipped and their saved checkpoint is loaded.
4. Each notebook prints the **number of trainable parameters** and the **final validation accuracy**, plots the training
   curves and writes the Kaggle submission CSV to its output folder.

The final notebook writes `submission_sla4_sam0.05_e600_cutrot.csv` (8-view inference, the one we submitted) and
`submission_sla4_sam0.05_e600_cutrot_single.csv` (single view). Setting `use_full_train = True` in its Config cell trains
the same recipe on all 50,000 images without a validation split.

**Requirements:** a CUDA GPU, PyTorch 2.x with torchvision (uses `torchvision.transforms.v2`), NumPy, pandas, Matplotlib
and Pillow. `torch.compile` is used where available and switched off automatically on Windows.

Model checkpoints and the dataset are not included because of their size.

## Main references

- He et al. 2016, *Deep Residual Learning for Image Recognition*; Zagoruyko & Komodakis 2016, *Wide Residual Networks*
- Lee, Hwang & Shin 2020, *Self-supervised Label Augmentation via Input Transformations*
- Foret et al. 2021, *Sharpness-Aware Minimization for Efficiently Improving Generalization*
- Cubuk et al. 2019, *AutoAugment*; DeVries & Taylor 2017, *Cutout*
- Hinton, Vinyals & Dean 2015, *Distilling the Knowledge in a Neural Network*
- Model code structure adapted from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
