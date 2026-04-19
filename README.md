#Task A : NYC yellow taxi fare regression

This repository is a small end-to-end example: download **NYC TLC** trip data in **Parquet** format, optionally merge monthly files per fleet type, and train a **PyTorch** feedforward network to predict **yellow taxi** `fare_amount` from trip features—with preprocessing (imputation, scaling, feature selection, optional PCA), **dropout**, **weight decay**, **learning-rate decay**, and **early stopping**.

Trip data is published by the [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The Dataset is Yellow Taxi Trip Records for January 2021. File name is yellow_tripdata_2021-01.parquet.


## What’s included

- **`notebooks/yellow_taxi_fare_ann_pytorch.ipynb`**experiments**: shallow vs deep vs deep+dropout (ReLU), then ReLU vs Tanh vs Leaky ReLU vs Sigmoid (deep+dropout), tables and bar charts, and **example predictions** from the validation-champion model.
- **`notebooks/yellow_taxi_fare_cv_pytorch.ipynb`** — **K-fold cross-validation** of the deep + dropout + **AdamW weight decay (L2)** model: refit sklearn preprocessing **inside each fold**, report mean ± std of out-of-fold validation error, and an informal comparison to a single train/validation/test split.

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

Files land under **`dataset_nyc_parquet/`** yellow_tripdata_2021-01.parquet`**.


## Run the notebooks

```bash
jupyter notebook notebooks/yellow_taxi_fare_ann_pytorch.ipynb
jupyter notebook notebooks/yellow_taxi_fare_cv_pytorch.ipynb
```

 Use `MAX_ROWS` to cap rows for faster runs. The CV notebook trains **K** models (default 5); reduce `N_SPLITS` or `MAX_ROWS` if it is too slow.

## Project layout

```
Task A/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── yellow_taxi_fare_ann_pytorch.ipynb
│   └── yellow_taxi_fare_cv_pytorch.ipynb
└── dataset_nyc_parquet/
    └── yellow_tripdata_2021-01.parquet

```

## License

See **`LICENSE`** in the repository root.
