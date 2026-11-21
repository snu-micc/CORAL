import argparse

from coral.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train a CORAL model")
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", default="")
    parser.add_argument("-o", "--output", default="runs/coral")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
