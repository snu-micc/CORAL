import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .data import MolPool, PairDataset, make_loader
from .model import CORAL, save_model
from .predict import predict, regression_metrics

METRIC_KEYS = ["mae", "rmse", "r2", "spearman"]

HIDDEN = 128
LAYERS = 6
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PAIRS_PER_EPOCH = 20000
HARD_RATIO = 0.7
MIN_DELTA = 0.3
WINDOW = 1.0
TEMPERATURE = 1.0
PATIENCE = 30
MIN_IMPROVEMENT = 1e-4


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def effective_temperature(base, min_delta):
    return math.sqrt(base ** 2 + 2.0 * min_delta ** 2)


def evaluate_pool(model, pool, reference_pool, device, amp):
    estimates, _ = predict(model, pool, reference_pool, device, amp=amp, progress=False)
    return regression_metrics(pool.values, estimates)


def train(args):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    train_pool = MolPool(args.train)
    valid_pool = MolPool(args.valid) if args.valid else None

    if not train_pool.labeled:
        raise ValueError(f"{args.train} has molecules without a value")

    sample = train_pool.graph(0)
    model = CORAL(
        node_in=int(sample.x.size(1)),
        edge_in=int(sample.edge_attr.size(1)),
        global_in=int(sample.g.size(1)),
        hidden=HIDDEN,
        layers=LAYERS,
        dropout=DROPOUT,
    ).to(device)

    temperature = effective_temperature(TEMPERATURE, MIN_DELTA)
    dataset = PairDataset(
        train_pool,
        n_pairs=PAIRS_PER_EPOCH,
        temperature=temperature,
        hard_ratio=HARD_RATIO,
        min_delta=MIN_DELTA,
        window=WINDOW,
    )
    loader = make_loader(dataset, args.batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    fields = ["epoch", "train_loss"]
    if valid_pool is not None:
        fields += [f"val_{k}" for k in METRIC_KEYS]
    fields += ["lr", "seconds"]

    log_path = os.path.join(args.output, "log.csv")
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(fields)

    print(f"device={device}  train={len(train_pool)}  "
          f"valid={len(valid_pool) if valid_pool else 0}")
    print(f"temperature={temperature:.3f}  params="
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best = float("inf")
    patience = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        running = 0.0

        for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            target = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch["a"], batch["b"], device).squeeze(-1)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())

        scheduler.step()
        train_loss = running / max(len(loader), 1)

        row = {"epoch": epoch, "train_loss": round(train_loss, 6),
               "lr": optimizer.param_groups[0]["lr"],
               "seconds": round(time.time() - started, 1)}
        line = f"epoch {epoch:>4}  loss {train_loss:.4f}"
        monitor = train_loss

        if valid_pool is not None:
            metrics = evaluate_pool(model, valid_pool, train_pool, device, use_amp)
            for k in METRIC_KEYS:
                row[f"val_{k}"] = round(metrics[k], 4)
            monitor = metrics["rmse"]
            line += (f" | val mae {metrics['mae']:.3f} rmse {metrics['rmse']:.3f} "
                     f"r2 {metrics['r2']:.3f} rho {metrics['spearman']:.3f}")

        print(line)

        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([row.get(k, "") for k in fields])

        if monitor < best - MIN_IMPROVEMENT:
            best = monitor
            patience = 0
            save_model(model, os.path.join(args.output, "best.pt"),
                       {"epoch": epoch, "score": best})
            print(f"{'':>10}new best ({best:.4f}) saved")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"early stop at epoch {epoch}")
                break

    save_model(model, os.path.join(args.output, "last.pt"), {"epoch": epoch})
    print(f"\ndone. best={best:.4f}")
    print(f"checkpoints: {os.path.join(args.output, 'best.pt')}")
    print(f"log:         {log_path}")
