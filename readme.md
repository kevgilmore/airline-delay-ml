# Flight Delay Prediction

A supervised ML pipeline that predicts flight delays using UK CAA punctuality statistics.

## Project Structure

```
├── data/raw/              # Raw CAA punctuality CSVs
├── models/                # Saved plots and model artifacts
├── notebooks/             # Jupyter notebooks (exploration only)
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline_model.ipynb
├── src/
│   ├── load_data.py       # Load and combine raw CSVs
│   ├── preprocess.py      # Clean data, define binary target
│   ├── features.py        # Feature selection and encoding
│   ├── train.py           # Model training
│   ├── evaluate.py        # Metrics and plots
│   └── pipeline.py        # End-to-end orchestration
├── requirements.txt
└── readme.md
```

## Problem

Each row in the CAA dataset represents a route-airline-month combination with aggregated punctuality stats (delay bucket percentages, average delay, cancellations).

The binary target **`is_delayed`** is 1 when more than 50% of flights on a route were over 15 minutes late.

## Baseline Model

- **Algorithm**: Random Forest (200 trees, max depth 10, balanced class weights)
- **Features**: airport, destination country, airline, arrival/departure flag, flight counts, cancellation rate, prior-year average delay
- **Split**: 80/20 stratified train/test

### Results

| Metric   | Value |
|----------|-------|
| ROC-AUC  | 0.82  |
| Accuracy | 81%   |
| Recall (Delayed) | 61% |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python -m src.pipeline
```

Individual modules can also be run standalone:

```bash
python -m src.load_data       # inspect raw data
python -m src.preprocess      # check cleaning and target distribution
python -m src.features        # view feature matrix shape
python -m src.evaluate        # train + full evaluation with plots
```
