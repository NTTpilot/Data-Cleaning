"""
cafe_cleaner.py - reusable, vectorized cleaning pipeline for the dirty cafe sales data.

Two modes, same code path:
  FIT       (no artifacts passed): learns price map, scaler stats and category lists
            from the data, saves them to artifacts.json.
  TRANSFORM (artifacts passed):    reuses those learned values on new data, so
            future batches are cleaned and scaled consistently with the first one.

Project layout (ROOT = parent folder of src/):
  ROOT/uncleaned/    raw input csvs
  ROOT/src/          this script, artifacts.json, cleaning_report.json
  ROOT/cleaned/      cleaned_<name>.csv
  ROOT/model_ready/  model_ready_<name>.csv

CLI (run from anywhere):
  python src/cafe_cleaner.py                          # cleans uncleaned/dirty_cafe_sales.csv
  python src/cafe_cleaner.py uncleaned/new_batch.csv  # reuses src/artifacts.json if it exists
  python src/cafe_cleaner.py --refit                  # relearn artifacts from this input
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("cafe_cleaner")


# --------------------------------- #
# Config
# --------------------------------- #
@dataclass
class Config:
    sentinels: tuple = ("ERROR", "UNKNOWN", "N/A", "NA", "NAN", "NULL", "NONE", "")
    id_pattern: str = r"^TXN_\d{7}$"
    id_col: str = "transaction_id"
    date_col: str = "transaction_date"
    date_in_formats: tuple = ("%Y-%m-%d", "%d/%m/%Y")   # accepted input formats, tried in order
    date_out_format: str = "%d/%m/%Y"                   # day/month/year in cleaned output
    numeric_cols: tuple = ("quantity", "price_per_unit", "total_spent")
    categorical_cols: tuple = ("item", "payment_method", "location")
    unknown_label: str = "Unknown"
    price_purity: float = 0.95      # min share of rows agreeing on an item's price
    iqr_k: float = 3.0              # fence multiplier for per-item IQR outlier flag
    drop_rule_violations: bool = True    # quantity*price != total, non-positive values
    drop_iqr_outliers: bool = False      # flag only by default (see README note below)
    scaling: str = "standard"            # "standard" (z-score) or "minmax"


# --------------------------------- #
# 1. load + missing/corrupted value standardisation
# --------------------------------- #
def to_snake(name: str) -> str:
    return re.sub(r"\W+", "_", name.strip().lower()).strip("_")


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [to_snake(c) for c in df.columns]
    return df


def standardize_missing(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Strip whitespace; turn ERROR/UNKNOWN/blank/etc. into real NaN."""
    df = df.copy()
    sentinels = set(cfg.sentinels)
    for c in df.columns:
        s = df[c].str.strip()
        df[c] = s.mask(s.str.upper().isin(sentinels))
    return df


# --------------------------------- #
# 2. Type casting + string normalisation
# --------------------------------- #
def cast_types(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df[cfg.id_col] = df[cfg.id_col].str.upper()
    df.loc[~df[cfg.id_col].fillna("").str.match(cfg.id_pattern), cfg.id_col] = np.nan
    for c in cfg.numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    raw_dates = df[cfg.date_col]
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for fmt in cfg.date_in_formats:  # explicit formats only, no guessing (avoids d/m vs m/d mix-ups)
        parsed = parsed.fillna(pd.to_datetime(raw_dates, format=fmt, errors="coerce"))
    df[cfg.date_col] = parsed
    for c in cfg.categorical_cols:
        df[c] = df[c].str.title()
    return df


# --------------------------------- #
# 3. Duplicates
# --------------------------------- #
def drop_duplicates(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop rows with no valid ID, exact duplicates, then duplicate IDs
    (keeping the most complete record per ID)."""
    df = df.dropna(subset=[cfg.id_col]).drop_duplicates()
    completeness = df.isna().sum(axis=1)
    df = (
        df.assign(_n_missing=completeness)
        .sort_values("_n_missing", kind="stable")
        .drop_duplicates(subset=cfg.id_col, keep="first")
        .sort_index()
        .drop(columns="_n_missing")
    )
    return df


# --------------------------------- #
# 4. Imputation (relationship based first, distribution-based fallback)
# --------------------------------- #
def learn_price_map(df: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """item -> unit price, only for items whose price is (near) constant."""
    valid = df.dropna(subset=["item", "price_per_unit"])
    mode = valid.groupby("item")["price_per_unit"].agg(lambda s: s.mode().iloc[0])
    agree = valid.assign(_m=valid["item"].map(mode)).assign(
        _hit=lambda d: d["price_per_unit"] == d["_m"]
    ).groupby("item")["_hit"].mean()
    return {k: float(v) for k, v in mode[agree >= cfg.price_purity].items()}


def invert_unique(price_map: dict[str, float]) -> dict[float, str]:
    """price -> item, only where exactly one item has that price."""
    inv: dict[float, list[str]] = {}
    for item, price in price_map.items():
        inv.setdefault(price, []).append(item)
    return {p: items[0] for p, items in inv.items() if len(items) == 1}


def impute_numeric_relational(df: pd.DataFrame, price_map: dict[str, float]) -> pd.DataFrame:
    """Total = Quantity x Price, so if any two of the three are known, the third can be inferred."""
    df = df.copy()
    before = df[["item", "quantity", "price_per_unit", "total_spent"]].notna()
    q, p, t = df["quantity"], df["price_per_unit"], df["total_spent"]

    p = p.fillna(df["item"].map(price_map))
    for _ in range(2):  # second pass picks up values unlocked by the first
        t = t.fillna(q * p)
        q_cand = (t / p).replace([np.inf, -np.inf], np.nan)
        q_ok = q_cand.where(np.isclose(q_cand, q_cand.round()) & (q_cand > 0))
        q = q.fillna(q_ok.round())
        p_cand = (t / q).replace([np.inf, -np.inf], np.nan)
        p = p.fillna(p_cand.where(p_cand > 0))
    df["quantity"], df["price_per_unit"], df["total_spent"] = q, p, t

    # item from price where that price belongs to exactly one item
    df["item"] = df["item"].fillna(df["price_per_unit"].map(invert_unique(price_map)))

    after = df[["item", "quantity", "price_per_unit", "total_spent"]].notna()
    for c in before.columns:
        df[f"{c}_imputed"] = after[c] & ~before[c]
    return df


def impute_numeric_fallback(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Distribution-based fallback for rows where 2+ of the 3 numerics were lost:
    quantity -> mode, price -> per-item median (else global median),
    total -> quantity x price."""
    df = df.copy()
    miss_q, miss_p = df["quantity"].isna(), df["price_per_unit"].isna()
    df["quantity"] = df["quantity"].fillna(stats["quantity_mode"])
    item_med = df["item"].map(stats["price_median_by_item"])
    df["price_per_unit"] = df["price_per_unit"].fillna(item_med).fillna(stats["price_median"])
    miss_t = df["total_spent"].isna()
    df["total_spent"] = df["total_spent"].fillna(df["quantity"] * df["price_per_unit"])
    df["quantity_imputed"] |= miss_q
    df["price_per_unit_imputed"] |= miss_p
    df["total_spent_imputed"] |= miss_t
    return df


def impute_categorical(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Categoricals ( mainly payment/location) are ~uniform with 15-30% missing, so
    mode-filling would fabricate a majority. Instead, we fill with 'Unknown' and flag the imputed rows."""
    df = df.copy()
    for c in cfg.categorical_cols:
        if f"{c}_imputed" not in df:
            df[f"{c}_imputed"] = False
        was_missing = df[c].isna()
        df[c] = df[c].fillna(cfg.unknown_label)
        df[f"{c}_imputed"] |= was_missing
    return df


def map_unseen_categories(df: pd.DataFrame, art: dict, cfg: Config) -> pd.DataFrame:
    """Transform mode: any category not seen during fit (e.g. 'Crypto') becomes
    'Unknown', flagged in `*_imputed`, with a logged warning."""
    df = df.copy()
    for c in cfg.categorical_cols:
        known = set(art["categories"][c])
        unseen = ~df[c].isin(known)
        if unseen.any():
            counts = df.loc[unseen, c].value_counts().to_dict()
            log.warning("Unseen %s values mapped to '%s': %s", c, cfg.unknown_label, counts)
            df.loc[unseen, c] = cfg.unknown_label
            df[f"{c}_imputed"] |= unseen
    return df


def handle_dates(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Missing/invalid dates take the date from the row above (forward fill);
    leading gaps at the top of the file take the next valid date (back fill).
    Filled rows are flagged in `date_missing`."""
    df = df.copy()
    d = df[cfg.date_col]
    df["date_missing"] = d.isna()
    d = d.ffill().bfill()
    df[cfg.date_col] = d
    df["month"] = d.dt.month.astype("Int64")
    df["day_of_week"] = d.dt.dayofweek.astype("Int64")
    return df


# --------------------------------------------------------------------------- #
# 5. Outliers
# --------------------------------------------------------------------------- #
def flag_outliers(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Two checks:
    - rule_violation: non-positive values or quantity x price != total (corrupted)
    - iqr_outlier:    total outside k x IQR fences *within each item*
                      (a global IQR would wrongly flag legit 5 x $5 salads)."""
    df = df.copy()
    q, p, t = df["quantity"], df["price_per_unit"], df["total_spent"]
    df["rule_violation"] = (q <= 0) | (p <= 0) | (t <= 0) | ~np.isclose(q * p, t)

    g = df.groupby("item")["total_spent"]
    q1, q3 = g.transform("quantile", 0.25), g.transform("quantile", 0.75)
    iqr = q3 - q1
    df["iqr_outlier"] = (iqr > 0) & ((t < q1 - cfg.iqr_k * iqr) | (t > q3 + cfg.iqr_k * iqr))
    return df


def drop_outliers(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    if cfg.drop_rule_violations:
        mask |= df["rule_violation"]
    if cfg.drop_iqr_outliers:
        mask |= df["iqr_outlier"]
    return df.loc[~mask]


# --------------------------------------------------------------------------- #
# 6. Scaling + model matrix
# --------------------------------------------------------------------------- #
def fit_scaler(df: pd.DataFrame, cols) -> dict:
    return {
        c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=0) or 1.0),
            "min": float(df[c].min()), "max": float(df[c].max())}
        for c in cols
    }


def apply_scaler(df: pd.DataFrame, scaler: dict, method: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c, s in scaler.items():
        if method == "minmax":
            rng = (s["max"] - s["min"]) or 1.0
            out[f"{c}_scaled"] = (df[c] - s["min"]) / rng
        else:
            out[f"{c}_scaled"] = (df[c] - s["mean"]) / s["std"]
    return out


def build_model_matrix(df: pd.DataFrame, art: dict, cfg: Config) -> pd.DataFrame:
    parts = [df[[cfg.id_col]], apply_scaler(df, art["scaler"], cfg.scaling)]
    parts.append(df[["month", "day_of_week", "date_missing"]].astype("float").assign(
        date_missing=df["date_missing"].astype(int)))
    for c in cfg.categorical_cols:
        cat = pd.Categorical(df[c], categories=art["categories"][c])
        parts.append(pd.get_dummies(cat, prefix=c, dtype=int).set_index(df.index))
    return pd.concat(parts, axis=1)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(raw: pd.DataFrame, cfg: Config | None = None, artifacts: dict | None = None):
    cfg = cfg or Config()
    report: dict = {"rows_in": len(raw)}
    fit = artifacts is None

    df = standardize_missing(raw, cfg)
    report["missing_after_standardize"] = df.isna().sum().to_dict()
    df = cast_types(df, cfg)

    n = len(df)
    df = drop_duplicates(df, cfg)
    report["duplicates_or_bad_id_dropped"] = n - len(df)

    if fit:
        artifacts = {"price_map": learn_price_map(df, cfg)}
    price_map = artifacts["price_map"]

    df = impute_numeric_relational(df, price_map)
    report["still_missing_after_relational"] = {
        c: int(df[c].isna().sum()) for c in cfg.numeric_cols}

    if fit:
        artifacts["fallback_stats"] = {
            "quantity_mode": float(df["quantity"].mode().iloc[0]),
            "price_median": float(df["price_per_unit"].median()),
            "price_median_by_item": df.groupby("item")["price_per_unit"].median().to_dict(),
        }
    df = impute_numeric_fallback(df, artifacts["fallback_stats"])
    df = impute_categorical(df, cfg)
    if not fit:
        df = map_unseen_categories(df, artifacts, cfg)
    df = handle_dates(df, cfg)

    df = flag_outliers(df, cfg)
    report["rule_violations"] = int(df["rule_violation"].sum())
    report["iqr_outliers_flagged"] = int(df["iqr_outlier"].sum())
    df = drop_outliers(df, cfg)

    if fit:
        artifacts["scaler"] = fit_scaler(df, cfg.numeric_cols)
        artifacts["categories"] = {
            c: sorted(set(df[c].unique()) | {cfg.unknown_label}) for c in cfg.categorical_cols}

    model_df = build_model_matrix(df, artifacts, cfg)
    report["rows_out"] = len(df)
    report["imputed_cells"] = {
        c: int(df[c].sum()) for c in df.columns if c.endswith("_imputed")}
    return df.reset_index(drop=True), model_df.reset_index(drop=True), artifacts, report


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
UNCLEANED_DIR = ROOT / "uncleaned"
CLEANED_DIR = ROOT / "cleaned"
MODEL_DIR = ROOT / "model_ready"
DEFAULT_INPUT = UNCLEANED_DIR / "dirty_cafe_sales.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean cafe sales data.")
    ap.add_argument("input", nargs="?", default=str(DEFAULT_INPUT),
                    help=f"raw csv (default: {DEFAULT_INPUT})")
    ap.add_argument("--artifacts", default=str(SRC_DIR / "artifacts.json"),
                    help="artifacts file: reused if it exists, created if not")
    ap.add_argument("--refit", action="store_true",
                    help="ignore existing artifacts and relearn them from this input")
    ap.add_argument("--scaling", choices=["standard", "minmax"], default="standard")
    ap.add_argument("--drop-iqr-outliers", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config(scaling=args.scaling, drop_iqr_outliers=args.drop_iqr_outliers)

    in_path, art_path = Path(args.input), Path(args.artifacts)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    artifacts = None
    if art_path.exists() and not args.refit:
        artifacts = json.loads(art_path.read_text())
        log.info("TRANSFORM mode: reusing %s", art_path)
    else:
        log.info("FIT mode: learning artifacts from %s", in_path.name)

    clean, model, artifacts, report = run_pipeline(load_raw(in_path), cfg, artifacts)

    name = in_path.stem.replace("dirty_", "")
    for d in (SRC_DIR, CLEANED_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEANED_DIR / f"cleaned_{name}.csv", index=False, date_format=cfg.date_out_format)
    model.to_csv(MODEL_DIR / f"model_ready_{name}.csv", index=False)
    art_path.write_text(json.dumps(artifacts, indent=2))
    (SRC_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2, default=int))
    log.info(json.dumps(report, indent=2, default=int))
    log.info("Wrote:\n  %s\n  %s", CLEANED_DIR / f"cleaned_{name}.csv", MODEL_DIR / f"model_ready_{name}.csv")


if __name__ == "__main__":
    main()
