import argparse
import os
import sys

from coral.data import build_dataset, read_table, save_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Turn a CSV of SMILES into a CORAL dataset")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    smiles, values = read_table(args.input)
    print(f"read {len(smiles)} molecules from {args.input}")

    records, skipped = build_dataset(smiles, values, workers=args.workers)
    if not records:
        print("no molecule could be featurized", file=sys.stderr)
        raise SystemExit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    save_dataset(records, args.output)

    print(f"featurized {len(records)} molecules")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:3]}{' ...' if len(skipped) > 3 else ''}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
