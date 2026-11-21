import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from torch_geometric.data import Batch
from tqdm.auto import tqdm

TOLERANCE_RATIO = 0.15


def neighbor_count(n):
    for limit, k in ((100, 1), (300, 2), (500, 3), (1000, 5), (3000, 7),
                     (5000, 9), (7000, 10), (10000, 12)):
        if n <= limit:
            return k
    return 15


@torch.inference_mode()
def score_graphs(model, graphs, device, batch_size=512, amp=True, desc=None):
    model.eval()
    scores = np.empty(len(graphs), dtype=np.float32)
    use_amp = bool(amp) and torch.device(device).type == "cuda"
    for start in tqdm(range(0, len(graphs), batch_size), desc=desc, leave=False,
                      disable=desc is None):
        end = min(start + batch_size, len(graphs))
        batch = Batch.from_data_list(graphs[start:end]).to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
            out = model.score(batch, device)
        scores[start:end] = out.detach().squeeze(-1).float().cpu().numpy()
    return scores


def anchor(query_scores, ref_scores, ref_values):
    order = np.argsort(ref_scores.astype(np.float32))
    sorted_scores = ref_scores.astype(np.float32)[order]
    sorted_values = np.asarray(ref_values, dtype=np.float64)[order]

    tolerance = (float(np.std(sorted_scores)) + 1e-6) * TOLERANCE_RATIO
    k = neighbor_count(len(sorted_scores))

    ranks = np.clip(np.searchsorted(sorted_scores, query_scores, side="left"),
                    0, len(sorted_scores) - 1)

    out = np.empty(len(query_scores), dtype=np.float64)
    for i, (score, rank) in enumerate(zip(query_scores, ranks)):
        lo = max(0, rank - k)
        hi = min(len(sorted_scores), rank + k + 1)
        window_scores = sorted_scores[lo:hi]
        window_values = sorted_values[lo:hi]
        mask = np.abs(window_scores - score) <= tolerance
        out[i] = float(np.mean(window_values[mask])) if mask.any() else float(sorted_values[rank])
    return out


def predict(model, query_pool, reference_pool, device, batch_size=512, amp=True,
            progress=True):
    ref_scores = score_graphs(model, reference_pool.graphs, device, batch_size, amp,
                              desc="reference" if progress else None)
    query_scores = score_graphs(model, query_pool.graphs, device, batch_size, amp,
                                desc="query" if progress else None)
    values = anchor(query_scores, ref_scores, reference_pool.values)
    return values, query_scores


def regression_metrics(true, pred):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    residual = pred - true
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) + 1e-12
    metrics = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - ss_res / ss_tot),
    }
    if len(true) > 1:
        metrics["spearman"] = float(spearmanr(true, pred).correlation)
        metrics["kendall"] = float(kendalltau(true, pred).correlation)
    else:
        metrics["spearman"] = 0.0
        metrics["kendall"] = 0.0
    return metrics
