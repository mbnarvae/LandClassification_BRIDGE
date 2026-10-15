"""Plain-PyTorch segmentation training loop (kept small and hackable; Lightning optional)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from bridge_lc.evaluation.metrics import confusion_matrix, macro_f1, mean_iou


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.05
    batch_size: int = 8
    ignore_index: int = 0
    time_budget_s: float | None = None  # stop early when exceeded (autoresearch)
    device: str = "auto"


def _device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, num_classes: int, device, ignore_index: int = 0) -> dict:
    import torch

    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            logits = getattr(out, "output", out)
            pred = logits.argmax(1).cpu().numpy()
            cm += confusion_matrix(y.numpy(), pred, num_classes - 1, ignore_index)
    return {"macro_f1": macro_f1(cm, ignore_index), "miou": mean_iou(cm, ignore_index), "cm": cm}


def train(model, train_ds, val_ds, num_classes: int, cfg: TrainConfig, class_weights=None) -> dict:
    """Train ``model``; ``num_classes`` includes nodata (0). Returns best val metrics."""
    import torch
    from torch.utils.data import DataLoader

    device = _device(cfg.device)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=cfg.ignore_index)
    tl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=cfg.batch_size)

    t0, best = time.time(), {"macro_f1": -1.0}
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in tl:
            out = model(x.to(device))
            loss = loss_fn(getattr(out, "output", out), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        metrics = evaluate(model, vl, num_classes, device, cfg.ignore_index)
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}  val_macro_f1 {metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best["macro_f1"]:
            best = metrics | {"epoch": epoch}
        if cfg.time_budget_s and time.time() - t0 > cfg.time_budget_s:
            print("time budget reached")
            break
    return best
