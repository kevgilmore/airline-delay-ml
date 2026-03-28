# Flight Delay Prediction

Predicts flight delay probability using UK CAA punctuality data and a Random Forest classifier.

**[Try the live demo →](#)** *(update with your GitHub Pages URL)*

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m src.pipeline        # train and evaluate
python -m src.save_model      # save model artifacts
python app.py                 # run local web app
python -m src.build_static    # build static site for GitHub Pages
```

## How it works

The CAA dataset has one row per route-airline-month with delay bucket percentages. A route is labelled "delayed" when >50% of its flights were >15 minutes late.

The baseline model (ROC-AUC 0.82) uses airport, destination country, airline, direction, flight volume, and cancellation rate as features.
