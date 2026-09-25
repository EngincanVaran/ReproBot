"""Ablate the four settings Tang 2013 never states, to find which one kills training.

Imports the model, loss and data loaders from the ACTUAL generated train.py, so this
tests the real reproduction rather than a reimplementation of it. Every value the paper
DOES state is held fixed: lr 0.1 linearly decayed, PCA to 70 dims, two 512-unit hidden
layers, minibatch 200, Gaussian input noise sigma 1.0 linearly decayed, top-layer L2
cost 0.001. Only the four guesses the Coder had to make are varied, one at a time.

Each config uses the paper's 400-epoch schedule but stops at EPOCHS, so learning rate
and noise follow the real run's early trajectory - the phase in which it collapsed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn

GEN = "/workspace/paper/train.py"
DATA = "/workspace/data"
SCHEDULE_EPOCHS = 400  # the paper's horizon: lr and noise decay against this
EPOCHS = 30            # the real run had collapsed by epoch 20
LR, BATCH, NOISE0, L2_TOP = 0.1, 200, 1.0, 0.001
SEED = 42

spec = importlib.util.spec_from_file_location("gen", GEN)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

CONFIGS = {
    # name:           (momentum, C,   init,         whiten)
    "A_as_generated": (0.9, 1.0, "pytorch_default", False),
    "B_momentum_0.5": (0.5, 1.0, "pytorch_default", False),
    "C_momentum_0.0": (0.0, 1.0, "pytorch_default", False),
    "D_C_0.1":        (0.9, 0.1, "pytorch_default", False),
    "E_pca_whitened": (0.9, 1.0, "pytorch_default", True),
    "F_init_N0.01":   (0.9, 1.0, "gauss_0.01",      False),
}


def dead_fraction(model: nn.Module, probe: torch.Tensor) -> tuple[float, float]:
    """Fraction of hidden units that output exactly 0 for EVERY probe sample."""
    with torch.no_grad():
        h1 = torch.relu(model.fc1(probe))
        h2 = torch.relu(model.fc2(h1))
    d1 = (h1.max(dim=0).values == 0).float().mean().item()
    d2 = (h2.max(dim=0).values == 0).float().mean().item()
    return d1, d2


def run(args: tuple) -> dict:
    name, (momentum, C, init, whiten), Xtr, Ytr, Ttr, Xte, Yte = args
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = gen.DLSVMNet(in_dim=Xtr.shape[1], hidden_dim=512, num_classes=10)
    if init == "gauss_0.01":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum, weight_decay=0.0)

    xtr, ttr, ytr = torch.from_numpy(Xtr), torch.from_numpy(Ttr), torch.from_numpy(Ytr)
    xte, yte = torch.from_numpy(Xte), torch.from_numpy(Yte)
    probe = xtr[:2000]
    n = xtr.shape[0]
    g = torch.Generator().manual_seed(SEED)
    history = []
    t0 = time.time()

    for epoch in range(EPOCHS):
        frac = epoch / SCHEDULE_EPOCHS
        for pg in opt.param_groups:
            pg["lr"] = LR * (1 - frac)
        noise = NOISE0 * (1 - frac)
        perm = torch.randperm(n, generator=g)
        loss_sum = correct = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            x = xtr[idx] + torch.randn(len(idx), xtr.shape[1], generator=g) * noise
            opt.zero_grad()
            s = model(x)
            loss, _ = gen.l2_svm_loss(s, ttr[idx], C, L2_TOP, model.top.linear.weight)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += (s.argmax(1) == ytr[idx]).sum().item()
        d1, d2 = dead_fraction(model, probe)
        with torch.no_grad():
            test_err = 100 * (1 - (model(xte).argmax(1) == yte).float().mean().item())
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / n, 4),
            "train_err": round(100 * (1 - correct / n), 2),
            "test_err_clean": round(test_err, 2),
            "dead_fc1": round(d1, 3),
            "dead_fc2": round(d2, 3),
        })
    return {"name": name, "seconds": round(time.time() - t0, 1), "history": history}


def main() -> None:
    Xtr_raw, Ytr = gen.load_mnist_flat(DATA, train=True)
    Xte_raw, Yte = gen.load_mnist_flat(DATA, train=False)
    from sklearn.decomposition import PCA

    feats = {}
    for whiten in (False, True):
        p = PCA(n_components=70, whiten=whiten, random_state=SEED)
        tr = p.fit_transform(Xtr_raw).astype(np.float32)
        te = p.transform(Xte_raw).astype(np.float32)
        feats[whiten] = (tr, te)
        sd = tr.std(axis=0)
        print(f"PCA whiten={whiten}: per-feature std  PC1={sd[0]:.3f}  "
              f"PC10={sd[9]:.3f}  PC70={sd[69]:.3f}  mean||x||^2={float((tr**2).sum(1).mean()):.1f}",
              flush=True)

    Ttr = gen.make_onehot_pm1(Ytr, 10)
    jobs = [(name, cfg, feats[cfg[3]][0], Ytr, Ttr, feats[cfg[3]][1], Yte)
            for name, cfg in CONFIGS.items()]
    with Pool(len(jobs)) as pool:
        results = pool.map(run, jobs)
    json.dump(results, open("/workspace/out/ablation.json", "w"), indent=1)
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
