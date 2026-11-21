# CORAL: Comparative Reference-Anchored Learning for Molecular Property Prediction

CORAL predicts molecular properties by comparing molecules instead of predicting absolute values.

Training uses pairwise comparisons (is A larger than B?). At inference a query is scored against a
**reference pool** of molecules with known values, so swapping the pool changes predictions without
retraining.

Built for aqueous pKa, but works for any continuous property.

## Contents

- [Environment](#environment)
- [Data](#data)
- [Quick start](#quick-start)
- [Models](#models)
- [Full Pipeline](#full-pipeline)

## Environment

Create a conda environment named `coral` with Python 3.9:

```bash
conda create -n coral python=3.9 -y
conda activate coral
conda install -c conda-forge rdkit -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install -r requirements.txt
```

A GPU is recommended but everything runs on CPU.

## Data

Experimental pKa data was collected from

- https://github.com/dptech-corp/Uni-pKa
- https://github.com/Shualdon/QupKake

Two curated sets are in `data/`:

| file | molecules | role |
|---|---|---|
| `acid.csv` | 7,080 | acidic pKa; reference pool for the released acidic model |
| `base.csv` | 8,447 | basic pKa; reference pool for the released basic model |

Both have two columns, `smiles` and `value`. The `value` column may be omitted when you only
want predictions.

```csv
smiles,value
CC(=O)O,4.76
c1ccccc1O,9.95
```

### Reference pool

Inference needs a **reference pool**: a CSV of molecules whose values you know. It turns the
model's relative score into an absolute value, and it is the main thing you control at inference
time.

## Quick start

Predict with a pretrained model. Featurize the reference pool once, then reuse it.

```bash
python preprocess.py -i data/acid.csv -o data/acid.npz --workers 8

python predict.py -m models/coral_pka_acid.pt \
                  -r data/acid.npz \
                  -i my_molecules.csv \
                  -o predictions.csv
```

`-r` and `-i` take a `.csv` or a `.npz` made by `preprocess.py`. Pass several files to `-i` and
each gets its own output file.


## Full Pipeline

### Step 0. Featurize

Converts SMILES into molecular graphs and caches them in a `.npz`. Atom features include MMFF94
partial charges, so this is the slowest stage; the cache means you pay for it once.

```bash
python preprocess.py -i train.csv -o train.npz --workers 8
python preprocess.py -i valid.csv -o valid.npz --workers 8
python preprocess.py -i test.csv  -o test.npz  --workers 8
```

| option | default |
|---|---|
| `-i, --input` | required |
| `-o, --output` | required |
| `--workers` | `1` |

### Step 1. Train

```bash
python train.py --train train.npz --valid valid.npz -o runs/mymodel
```

Each epoch prints the training loss and, when `--valid` is given, the reference-anchored
MAE / RMSE / R² and Spearman correlation on the validation set. The same numbers go to
`runs/mymodel/log.csv`.

```
device=cuda  train=296  valid=79
temperature=1.086  params=498,050
epoch    1  loss 0.6555 | val mae 1.409 rmse 2.035 r2 0.595 rho 0.777
          new best (2.0346) saved
epoch    2  loss 0.6043 | val mae 1.456 rmse 2.143 r2 0.551 rho 0.780
```

`best.pt` is the epoch with the lowest validation RMSE; `last.pt` is written once when training
ends.

| option | default |
|---|---|
| `--train` | required |
| `--valid` | none |
| `-o, --output` | `runs/coral` |
| `--epochs` | `200` |
| `--batch-size` | `32` |
| `--lr` | `5e-4` |
| `--seed` | `2026` |
| `--device` | auto |

Model size, pair sampling and early stopping are fixed at the values used in the paper and live at
the top of `coral/trainer.py`.

### Step 2. Predict

```bash
python predict.py -m runs/mymodel/best.pt \
                  -r data/acid.npz \
                  -i test_a.npz test_b.npz \
                  -o runs/pred.csv
```

When the input has a `value` column, metrics are printed and written next to the predictions.

```
dataset                  n        mae       rmse         r2   spearman
test_a                  40     0.5826     0.7871     0.8991     0.9207
test_b                 112     0.8051     1.1289     0.8041     0.8781
```

Output columns are `smiles`, `score` (position on the learned axis), `prediction`, and `value` and
`error` when ground truth is available.

| option | default |
|---|---|
| `-m, --model` | required |
| `-r, --reference` | required |
| `-i, --input` | required, one or more files |
| `-o, --output` | `predictions.csv` |
| `--device` | auto |

### Step 3. Use from Python

```python
import torch
from coral import load_model, MolPool, predict, regression_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("models/coral_pka_acid.pt", device)

reference = MolPool("data/acid.npz")
query = MolPool("query.npz")

values, scores = predict(model, query, reference, device)
print(regression_metrics(query.values, values))
```


