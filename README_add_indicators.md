Usage: invert observed `B_score` into five Tanaka indicators

Run from repository root:

```bash
python3 scripts/add_indicators.py \
  --input raw_data/selected_scores.csv \
  --output raw_data/selected_scores_with_indicators.csv \
  --lambda_reg 0.01 \
  --maxiter 500 \
  --seed 42
```

Dependencies: install from `requirements.txt` or use Poetry:

```bash
python3 -m pip install -r requirements.txt
```

Using Poetry (recommended):

```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -
# Install dependencies
poetry install
# Run the script inside the virtual environment
poetry run python3 scripts/add_indicators.py \
  --input raw_data/selected_scores.csv \
  --output raw_data/selected_scores_with_indicators.csv \
  --lambda_reg 0.01 \
  --maxiter 500 \
  --seed 42
```

Notes:
- `x_prior` is random per-sample (uniform in (0.01,0.99)).
- The script writes columns: `H,S,A,B_indicator,C,B_recalc`.