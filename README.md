#Task A : NYC yellow taxi fare regression

This repository is a small end-to-end example: download **NYC TLC** trip data in **Parquet** format, optionally merge monthly files per fleet type, and train a **PyTorch** feedforward network to predict **yellow taxi** `fare_amount` from trip features—with preprocessing (imputation, scaling, feature selection, optional PCA), **dropout**, **weight decay**, **learning-rate decay**, and **early stopping**.

Trip data is published by the [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). TLC files are large; downloaded data is **gitignored** and should stay local.

## What’s included

- **`scripts/download_nyc_tlc_parquet.py`** — Downloads monthly Parquet files from TLC’s CloudFront URLs into `dataset_nyc_parquet/{yellow,green,fhv,fhvhv}/`. Probes each URL and skips months that do not exist (FHV and high-volume FHV coverage varies by year; yellow and green are available for the full catalog).
- **`scripts/combine_nyc_tlc_parquet.py`** — Concatenates all monthly Parquets **within each type** into `dataset_nyc_parquet/combined/{type}_all.parquet`. Yellow, green, FHV, and HVFHV have **different schemas**; this is a vertical stack per type, not a single merged table across fleets.
- **`docs/CONCEPTS.md`** — Definitions and examples for the data pipeline, preprocessing, neural-network building blocks (activations, dropout, batch norm), training (optimizers, weight decay, LR decay, early stopping), metrics, and **K-fold cross-validation**.
- **`notebooks/yellow_taxi_fare_ann_pytorch.ipynb`** — Same pipeline as above, plus **experiments**: shallow vs deep vs deep+dropout (ReLU), then ReLU vs Tanh vs Leaky ReLU vs Sigmoid (deep+dropout), tables and bar charts, and **example predictions** from the validation-champion model.
- **`notebooks/yellow_taxi_fare_cv_pytorch.ipynb`** — **K-fold cross-validation** of the deep + dropout + **AdamW weight decay (L2)** model: refit sklearn preprocessing **inside each fold**, report mean ± std of out-of-fold validation error, and an informal comparison to a single train/validation/test split.
- **`notebooks/taxi_fare_ann_individual_lab.ipynb`** — **Individual lab / coursework report** structure (problem, **peer-reviewed related work**, dataset exploration, **ANN-only** methods, experiments comparing two MLPs, results, discussion, appendix checklist) aligned to a typical “neural networks for a real-world task” brief.

## Requirements

- Python 3.10+ recommended  
- See **`requirements.txt`** (`pandas`, `pyarrow`, `requests`, `torch`, `scikit-learn`, plotting and Jupyter packages).

## Setup

```bash
git clone https://github.com/adityajoshi16439238/Task-A.git
cd Task A
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Download TLC Parquet data

Default year range in the script is **2024–2025** (adjust to save disk and time). Example:

```bash
# All four types (missing months are skipped automatically)
python scripts/download_nyc_tlc_parquet.py

# Wider range — can require very large disk and time
python scripts/download_nyc_tlc_parquet.py --start-year 2009 --end-year 2026 --quiet-missing

# Yellow taxis only
python scripts/download_nyc_tlc_parquet.py --types yellow
```

Files land under **`dataset_nyc_parquet/`** (ignored by Git). Re-runs skip re-download when local file size matches the server’s `Content-Length` unless you pass **`--no-skip-existing`**.

## Combine monthly files (optional)

```bash
python scripts/combine_nyc_tlc_parquet.py
```

Outputs: `dataset_nyc_parquet/combined/yellow_all.parquet`, etc., for types that had at least one monthly file. Combining many years can use a lot of **RAM**; narrow the download range if needed.

## Run the notebooks

```bash
jupyter notebook notebooks/yellow_taxi_fare_ann_pytorch.ipynb
jupyter notebook notebooks/yellow_taxi_fare_cv_pytorch.ipynb
```

Each notebook looks for **`dataset_nyc_parquet/combined/yellow_all.parquet`**, or otherwise **`dataset_nyc_parquet/yellow/`**. Use `MAX_ROWS` to cap rows for faster runs. The CV notebook trains **K** models (default 5); reduce `N_SPLITS` or `MAX_ROWS` if it is too slow.

## Project layout

```
Task A/
├── README.md
├── requirements.txt
├── docs/
│   └── CONCEPTS.md
├── scripts/
│   ├── download_nyc_tlc_parquet.py
│   └── combine_nyc_tlc_parquet.py
├── notebooks/
│   ├── yellow_taxi_fare_ann_pytorch.ipynb
│   ├── yellow_taxi_fare_cv_pytorch.ipynb
│   └── taxi_fare_ann_individual_lab.ipynb
└── dataset_nyc_parquet/     # created locally; not committed
```

## License

See **`LICENSE`** in the repository root.
