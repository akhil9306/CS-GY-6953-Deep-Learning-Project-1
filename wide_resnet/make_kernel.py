"""Stage a Kaggle kernel folder: kernels/<name>/{run.py, kernel-metadata.json}.

run.py is cifar_resnet.py with PLAN filled in. Push with (from inside the folder):
    cd kernels/<name> && kaggle kernels push -p .
"""
import json
import os
import sys

USER = "nikhilkindi"
COMP = "cifar-10-res-net-architecture-challenge-dl"
HERE = os.path.dirname(os.path.abspath(__file__))

S = ["--arch", "wrn28_s", "--epochs", "250", "--workers", "2"]
T = ["--arch", "wrn28_10", "--epochs", "200", "--workers", "2"]
# PyramidNet-110 (alpha 96, 4.80M) + ShakeDrop; wd 1e-4 as in the PyramidNet/ShakeDrop papers
P = ["--arch", "pyr110_sd", "--epochs", "300", "--wd", "1e-4", "--workers", "2"]

PLANS = {
    # round 1: validate the student recipe, train a full-data student, and train the teacher
    "cifar-r1": dict(sources=[], plan={
        "gpu0": [["--name", "s45", "--val_split", "5000", "--time_budget_h", "4.5"] + S,
                 ["--name", "s50", "--time_budget_h", "5.0"] + S],
        "gpu1": [["--name", "t45", "--val_split", "5000", "--time_budget_h", "10.5"] + T],
    }),
    # round 2: distill the teacher into the <5M student (45k run measures the gain, 50k run is the final model)
    "cifar-r2": dict(sources=[f"{USER}/cifar-r1"], plan={
        "gpu0": [["--name", "kd45", "--val_split", "5000", "--teacher_ckpt", "TEACHER:t45", "--time_budget_h", "10.5"] + S],
        "gpu1": [["--name", "kd50", "--teacher_ckpt", "TEACHER:t45", "--time_budget_h", "10.5"] + S],
    }),
    # round 3: PyramidNet + ShakeDrop student, taught by the same teacher
    "cifar-r3": dict(sources=[f"{USER}/cifar-r1"], plan={
        "gpu0": [["--name", "pk45", "--val_split", "5000", "--teacher_ckpt", "TEACHER:t45", "--time_budget_h", "11.0"] + P],
        "gpu1": [["--name", "pk50", "--teacher_ckpt", "TEACHER:t45", "--time_budget_h", "11.0"] + P],
    }),
}


def stage(name):
    cfg = PLANS[name]
    out = os.path.join(HERE, "kernels", name)
    os.makedirs(out, exist_ok=True)
    src = open(os.path.join(HERE, "cifar_resnet.py"), encoding="utf-8").read()
    assert "\nPLAN = None" in src
    src = src.replace("\nPLAN = None", "\nPLAN = " + json.dumps(cfg["plan"]), 1)
    open(os.path.join(out, "run.py"), "w", encoding="utf-8").write(src)
    meta = {
        "id": f"{USER}/{name}", "title": name, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "dataset_sources": [], "competition_sources": [COMP],
        "kernel_sources": cfg["sources"], "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    print("staged", out)


if __name__ == "__main__":
    for n in sys.argv[1:] or PLANS:
        stage(n)
