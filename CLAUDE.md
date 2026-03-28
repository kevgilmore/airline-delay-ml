# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate virtual environment (required before all commands)
source .venv/bin/activate

# Run full pipeline (load → preprocess → featurise → train → evaluate)
python -m src.pipeline

# Run individual stages
python -m src.load_data       # inspect raw data schema
python -m src.preprocess      # check cleaning + target distribution
python -m src.features        # view feature matrix shape
python -m src.evaluate        # train + metrics + save plots to models/
python -m src.save_model      # save model.joblib + model_metadata.joblib to models/

# Web app (requires saved model — run save_model first)
python app.py                 # Flask dev server on http://127.0.0.1:5000
python -m src.build_static    # build static site to docs/ for GitHub Pages

# Install dependencies
pip install -r requirements.txt
```

There is no test suite, linter, or build system configured.

## Architecture

Functional ML pipeline where each stage is a separate module in `src/`. Every module has a `__main__` block so it can run standalone. Data flows linearly:

```
load_data → preprocess → features → train → evaluate
```

`pipeline.py` orchestrates the full sequence.

**Key design decisions:**
- Functions over classes throughout — no OOP abstractions
- One-hot encoding happens in `features.py` via pandas `get_dummies`, not sklearn transformers
- Binary target `is_delayed` is defined in `preprocess.py`: 1 when >50% of flights on a route were >15 min late (threshold constant: `DELAY_THRESHOLD`)
- Feature lists (`CATEGORICAL_FEATURES`, `NUMERIC_FEATURES`) are module-level constants in `features.py`
- Class imbalance (~5.5% positive) handled via `class_weight="balanced"` in Random Forest
- Plots save to `models/` directory alongside any future model artifacts
- Notebooks are for exploration only — core logic lives in `src/`
- `app.py` (repo root) is a Flask frontend that loads saved model artifacts from `models/`. It reconstructs one-hot encoded feature vectors from form inputs by zero-initialising the full column schema and setting the appropriate `{feature}_{value}` column to 1
- `save_model.py` persists both the model and a metadata dict (column schema, dropdown values, numeric defaults) so the Flask app never needs to import the training pipeline
- `build_static.py` pre-computes all predictions and bakes them into a single `docs/index.html` — no backend needed for GitHub Pages deployment
- GitHub Actions workflow (`.github/workflows/deploy.yml`) trains the model, builds the static site, and deploys to Pages on push to main

## Data

Raw UK CAA punctuality CSVs in `data/raw/`. Each row is a route-airline-month aggregate (not per-flight). Columns include delay bucket percentages, average delay, cancellation stats, and prior-year comparisons. New monthly CSVs can be dropped into `data/raw/` and will be auto-loaded.
