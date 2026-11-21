import argparse
import os
import tempfile

import pandas as pd
import torch

from coral.data import MolPool, build_dataset, read_table, save_dataset
from coral.model import load_model
from coral.predict import predict, regression_metrics

METRICS = ["mae", "rmse", "r2", "spearman"]


def name_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_pool(path, workers):
    if path.endswith(".npz"):
        return MolPool(path), None
    smiles, values = read_table(path)
    records, skipped = build_dataset(smiles, values, workers=workers)
    if skipped:
        print(f"{name_of(path)}: skipped {len(skipped)} molecules")
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    save_dataset(records, tmp.name)
    return MolPool(tmp.name), tmp.name


def main():
    parser = argparse.ArgumentParser(description="Predict with a trained CORAL model")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-r", "--reference", required=True)
    parser.add_argument("-i", "--input", required=True, nargs="+")
    parser.add_argument("-o", "--output", default="predictions.csv")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.model, device)

    reference, ref_tmp = load_pool(args.reference, workers=4)
    if not reference.labeled:
        raise ValueError("every molecule in the reference file needs a value")

    print(f"device={device}  reference={len(reference)}")

    temporary = [ref_tmp]
    summary = []
    stem, ext = os.path.splitext(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    for path in args.input:
        pool, tmp = load_pool(path, workers=4)
        temporary.append(tmp)

        values, scores = predict(model, pool, reference, device)

        table = pd.DataFrame({"smiles": pool.smiles, "score": scores, "prediction": values})
        if pool.labeled:
            table["value"] = pool.values
            table["error"] = table["prediction"] - table["value"]
            summary.append({"dataset": name_of(path), "n": len(pool),
                            **regression_metrics(pool.values, values)})

        dest = args.output if len(args.input) == 1 else f"{stem}_{name_of(path)}{ext or '.csv'}"
        table.to_csv(dest, index=False)
        print(f"saved {dest}")

    if summary:
        print()
        print(f"{'dataset':<20}{'n':>6}" + "".join(f"{k:>11}" for k in METRICS))
        for row in summary:
            print(f"{row['dataset']:<20}{row['n']:>6}" +
                  "".join(f"{row[k]:>11.4f}" for k in METRICS))

    for path in temporary:
        if path and os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    main()
