# Cafe Sales Data Cleaning

A Python pipeline that cleans the "dirty cafe sales" dataset and produces two outputs: a readable cleaned CSV and a scaled, encoded, model-ready feature matrix. It is written as clean, vectorized pandas so it can be reused on new batches of data.

## Quick start

```bash
python -m pip install pandas numpy
python src/cafe_cleaner.py
```

This cleans `uncleaned/dirty_cafe_sales.csv` and writes the results to `cleaned/` and `model_ready/`. Requires Python 3.10 or later.

## Data source

The raw data is *Cafe Sales - Dirty Data for Cleaning Training* by **Ahmed Mohamed**, published on Kaggle:
https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training/data

`uncleaned/dirty_cafe_sales.csv` is that file, unmodified. Everything in `cleaned/` and `model_ready/` is derived from it by this pipeline.

## Project structure

```
root/
├── uncleaned/        raw input CSVs
│   └── dirty_cafe_sales.csv
├── src/
│   ├── cafe_cleaner.py       pipeline + command-line interface
│   ├── artifacts.json        learned values reused on later runs
│   └── cleaning_report.json  summary of the latest run
├── cleaned/          cleaned_<name>.csv
└── model_ready/      model_ready_<name>.csv
```

`cafe_cleaner.py` finds the project root from its own location (the parent of `src/`), so it must stay inside `src/`. You can run it from any directory.

## Usage

```bash
python src/cafe_cleaner.py                           # cleans uncleaned/dirty_cafe_sales.csv
python src/cafe_cleaner.py uncleaned/new_batch.csv   # cleans a different file
python src/cafe_cleaner.py --refit                   # relearn artifacts from the input
```

**Fit vs. transform mode.** If `src/artifacts.json` does not exist, the script learns the item price map, fallback statistics, scaler stats, and category lists from the input and saves them (FIT mode). If it exists, the script reuses it (TRANSFORM mode), so new batches are cleaned, scaled, and encoded exactly like the original. Use `--refit` to ignore the saved file and relearn.

Other options: `--scaling minmax` for min-max instead of z-score scaling, and `--drop-iqr-outliers` to also drop rows flagged by the IQR check.

**What gets overwritten.** Output names come from the input name with `dirty_` removed, so `dirty_cafe_sales.csv` writes `cleaned/cleaned_cafe_sales.csv` and `model_ready/model_ready_cafe_sales.csv`, replacing any existing files with those names. A different input name writes new files and leaves the old ones alone. `src/cleaning_report.json` is rewritten on every run.

## How the pipeline cleans

**Routine steps**

- **Standardize.** Blanks, `ERROR`, and `UNKNOWN` become real missing values; IDs must match `TXN_` plus seven digits.
- **Cast types.** Numbers are cast, text is trimmed and title-cased, and dates use explicit formats only, so day and month are never silently swapped.
- **Deduplicate.** Invalid IDs, exact duplicate rows, and duplicate transaction IDs are removed, keeping the most complete record.
- **Scale and encode.** Numerics are z-scored (or min-max scaled) and categoricals one-hot encoded from the lists in `artifacts.json`. In transform mode, a category not seen during fit (say a new payment type like `Crypto`) is mapped to `Unknown` and a warning is logged.

**Imputation**

The key fact in this data: `Total Spent = Quantity × Price Per Unit`, and every item has exactly one price. So most gaps can be recovered exactly instead of guessed.

- Missing price: taken from the item.
- Missing quantity: total ÷ price.
- Missing total: quantity × price.
- Missing item: recovered from the price only when that price belongs to one item (1.0 Cookie, 1.5 Tea, 2.0 Coffee, 5.0 Salad). Prices of 3.0 (Cake or Juice) and 4.0 (Sandwich or Smoothie) are ambiguous, so those items stay `Unknown`.
- Rows that lost two or more numeric fields fall back to the quantity mode and the per-item median price, then the total is recomputed.
- Payment method and location become an explicit `Unknown` class. Their known values are split almost evenly, so filling with the most common value would invent a majority.

**Dates**

- Missing or invalid dates take the date from the row above (the next valid date if the first row is missing), and `date_missing` marks every filled row.
- The file is not sorted by date, so filled dates are effectively arbitrary. Treat `month` and `day_of_week` as unreliable for those rows.

**Outliers**

- Rows where quantity × price ≠ total, or with non-positive values, are dropped.
- A per-item IQR check (3 × IQR fences) flags extreme totals but does not drop them unless asked. It runs per item because a global IQR would wrongly flag legitimate $25 salad orders.

## Results on the included dataset

| | |
|---|---|
| Rows in / out | 10,000 / 9,997 |
| Dropped | 3 rows failing the quantity × price = total check |
| Duplicates found | 0 |
| Items left as `Unknown` | 477 (ambiguous price, or price also missing) |
| IQR outliers flagged | 0 |

After cleaning, `Total = Quantity × Price` holds on every row.

Missing values before cleaning and how each was handled:

| Column | Missing | Handling |
|---|---|---|
| item | 969 | Recovered from price when unique; otherwise `Unknown` |
| quantity | 479 | Total ÷ price; 23 left for the fallback |
| price_per_unit | 533 | From the item, or total ÷ quantity; 6 left for the fallback |
| total_spent | 502 | Quantity × price; 23 left for the fallback |
| payment_method | 3,178 | `Unknown` |
| location | 3,961 | `Unknown` |
| transaction_date | 460 | Date from the row above |

## Outputs

**`cleaned/cleaned_cafe_sales.csv`** (19 columns) is the readable, auditable version. It keeps the original columns with real values and text categories, with dates as `DD/MM/YYYY`. It adds `*_imputed` flags (item, quantity, price, total, payment method, location), `date_missing`, `month`, `day_of_week`, `rule_violation`, and `iqr_outlier`.

**`model_ready/model_ready_cafe_sales.csv`** (23 columns) is the same rows as numbers only: z-scored `quantity`, `price_per_unit`, and `total_spent`; `month`, `day_of_week`, and `date_missing`; and one-hot columns for item (9), payment method (4), and location (3), each with an `Unknown` column. It has no date column and no imputed flags. `transaction_id` is included only for traceability, so drop it before training. If you predict `total_spent`, also drop the quantity and price columns, since total is exactly their product.
