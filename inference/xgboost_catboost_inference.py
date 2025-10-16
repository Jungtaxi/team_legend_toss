"""
XGBoost + CatBoost Ensemble Inference
- Load trained models from outputs/{run_name}/ directory
- Perform inference on test data with 5-fold ensemble
- Generate submission files (xgb, catboost, weighted)
"""

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from pathlib import Path
import logging
import argparse
from typing import Tuple, List

# ==========================================================
# Constants (must match training)
# ==========================================================
ID_COLUMN = "ID"
TARGET_COLUMN = "clicked"

CATEGORICAL_FEATURES = ["gender", "inventory_id", "hour", "age_group", "day_of_week",
                        "time_segment", "seq_length_bin_str", "age_hour"]
EXCLUDED_FEATURES = ["seq"]
BINARY_FEATURES = [
    "l_feat_1", "l_feat_2", "l_feat_8", "l_feat_13",
    "l_feat_16", "l_feat_19", "l_feat_21", "l_feat_22", "l_feat_24"
]

# ==========================================================
# Feature Engineering (identical to training)
# ==========================================================

def create_seq_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Sequence features using Polars expressions"""
    return lf.with_columns([
        (pl.col("seq").str.count_matches(",") + 1).alias("seq_length"),
        (pl.col("seq").str.count_matches(",") + 1).clip(upper_bound=5000).alias("seq_length_clipped"),
        (pl.col("seq").str.count_matches(",") + 1).log1p().alias("seq_length_log"),
        pl.when(pl.col("seq").str.count_matches(",") + 1 <= 100).then(0)
          .when(pl.col("seq").str.count_matches(",") + 1 <= 200).then(1)
          .when(pl.col("seq").str.count_matches(",") + 1 <= 500).then(2)
          .when(pl.col("seq").str.count_matches(",") + 1 <= 1000).then(3)
          .otherwise(4)
          .alias("seq_length_bin"),
        ((pl.col("seq").str.count_matches(",") + 1) <= 100).cast(pl.Int8).alias("seq_very_short"),
        (pl.col("seq").str.len_chars() / ((pl.col("seq").str.count_matches(",") + 1) * 3 + 1)).alias("seq_diversity_proxy"),
        pl.col("seq").str.contains(",74,").cast(pl.Int8).alias("seq_has_74"),
        pl.col("seq").str.contains(",101,").cast(pl.Int8).alias("seq_has_101"),
        pl.col("seq").str.contains(",479,").cast(pl.Int8).alias("seq_has_479"),
        pl.col("seq").str.contains(",57,").cast(pl.Int8).alias("seq_has_57"),
        pl.col("seq").str.contains(",408,").cast(pl.Int8).alias("seq_has_408"),
    ]).with_columns([
        pl.col("seq_length_bin").cast(pl.Utf8).alias("seq_length_bin_str")
    ])


def create_history_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """History features with noise handling"""
    schema = lf.collect_schema()
    cols = schema.names()
    exprs = []

    if "history_a_1" in cols:
        exprs.extend([
            pl.col("history_a_1").clip(upper_bound=10.0).alias("history_a_1_clipped"),
            pl.col("history_a_1").fill_null(0).log1p().alias("history_a_1_log"),
            pl.when(pl.col("history_a_1") <= 0.05).then(0)
              .when(pl.col("history_a_1") <= 0.1).then(1)
              .when(pl.col("history_a_1") <= 0.2).then(2)
              .when(pl.col("history_a_1") <= 1.0).then(3)
              .otherwise(4)
              .alias("history_a_1_bin"),
            (pl.col("history_a_1") > 0.2).cast(pl.Int8).alias("history_a_1_high"),
            (pl.col("history_a_1") > 1.0).cast(pl.Int8).alias("history_a_1_very_high"),
        ])

    if "history_a_4" in cols:
        exprs.extend([
            pl.col("history_a_4").clip(lower_bound=-5000.0).alias("history_a_4_clipped"),
            (pl.col("history_a_4") > -200).cast(pl.Int8).alias("history_a_4_near_zero"),
            pl.when(pl.col("history_a_4") <= -1000).then(0)
              .when(pl.col("history_a_4") <= -500).then(1)
              .when(pl.col("history_a_4") <= -200).then(2)
              .otherwise(3)
              .alias("history_a_4_bin"),
        ])

    history_cols = [f"history_a_{i}" for i in range(1, 8)]
    existing_history = [col for col in history_cols if col in cols]

    if existing_history:
        exprs.extend([
            pl.mean_horizontal(existing_history).alias("history_a_mean"),
            pl.concat_list(existing_history).list.std().alias("history_a_std"),
            pl.max_horizontal(existing_history).alias("history_a_max"),
        ])

    return lf.with_columns(exprs) if exprs else lf


def create_time_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Time-based features"""
    schema = lf.collect_schema()
    cols = schema.names()
    exprs = []

    if "hour" in cols:
        hour_col = pl.col("hour").cast(pl.Int32)
        exprs.extend([
            pl.when(hour_col.is_in([2, 3, 4])).then(pl.lit("dawn"))
              .when(hour_col.is_in([7, 8, 9, 10, 11])).then(pl.lit("morning"))
              .when(hour_col.is_in([12, 13, 14, 15, 16, 17])).then(pl.lit("afternoon"))
              .when(hour_col.is_in([18, 19, 20])).then(pl.lit("evening"))
              .when(hour_col.is_in([21, 22, 23])).then(pl.lit("night"))
              .when(hour_col.is_in([0, 1])).then(pl.lit("midnight"))
              .when(hour_col.is_in([5, 6])).then(pl.lit("early"))
              .otherwise(pl.lit("other"))
              .alias("time_segment"),
            hour_col.is_in([2, 3, 4]).cast(pl.Int8).alias("is_dawn"),
            hour_col.is_in([7, 8, 9, 10]).cast(pl.Int8).alias("is_morning_rush"),
        ])

    if "day_of_week" in cols:
        dow_col = pl.col("day_of_week").cast(pl.Int32)
        exprs.extend([
            (dow_col == 2).cast(pl.Int8).alias("is_tuesday"),
            dow_col.is_in([6, 7]).cast(pl.Int8).alias("is_weekend"),
        ])

    return lf.with_columns(exprs) if exprs else lf


def create_interaction_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Interaction features"""
    schema = lf.collect_schema()
    cols = schema.names()
    exprs = []

    if "history_a_1" in cols and "seq_length" in cols:
        exprs.extend([
            (pl.col("history_a_1") * pl.col("seq_length").log1p()).alias("h1_x_seqlen"),
            (pl.col("history_a_1") * pl.col("seq_length").log1p() + 1).log1p().alias("h1_x_seqlen_log"),
        ])

    if "age_group" in cols and "hour" in cols:
        exprs.append(
            (pl.col("age_group").cast(pl.Utf8) + pl.lit("_") + pl.col("hour").cast(pl.Utf8)).alias("age_hour")
        )

    return lf.with_columns(exprs) if exprs else lf


def create_feat_group_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Feature group aggregations"""
    schema = lf.collect_schema()
    all_cols = schema.names()
    exprs = []

    for prefix in ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']:
        cols = [col for col in all_cols if col.startswith(f'{prefix}_')]
        if cols:
            exprs.extend([
                pl.mean_horizontal(cols).alias(f'{prefix}_mean'),
                pl.max_horizontal(cols).alias(f'{prefix}_max'),
                pl.min_horizontal(cols).alias(f'{prefix}_min'),
                pl.concat_list(cols).list.std().alias(f'{prefix}_std'),
            ])

    feat_a_cols = [col for col in all_cols if col.startswith('feat_a_')]
    if feat_a_cols:
        exprs.extend([
            pl.sum_horizontal([pl.col(c) != 0 for c in feat_a_cols]).alias("feat_a_nonzero_count"),
            (pl.sum_horizontal([pl.col(c) != 0 for c in feat_a_cols]) == 0).cast(pl.Int8).alias("feat_a_is_sparse"),
        ])

    return lf.with_columns(exprs) if exprs else lf


def engineer_all_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Apply all feature engineering"""
    lf = create_seq_features_polars(lf)
    lf = create_history_features_polars(lf)
    lf = create_time_features_polars(lf)
    lf = create_interaction_features_polars(lf)
    lf = create_feat_group_features_polars(lf)
    return lf


# ==========================================================
# Data Preparation
# ==========================================================

def prepare_for_xgboost(
    lf: pl.LazyFrame,
    streaming: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """Prepare data for XGBoost inference"""
    lf = engineer_all_features_polars(lf)

    exclude_cols = set(EXCLUDED_FEATURES) | {ID_COLUMN, TARGET_COLUMN}

    engine = "streaming" if streaming else None
    df_pl = lf.collect(engine=engine)
    df = df_pl.to_pandas()

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    cat_cols = [c for c in feature_cols if str(X[c].dtype) in ("object", "bool", "category")]

    categorical_feature_indices = []
    for i, col in enumerate(feature_cols):
        if col in cat_cols:
            X[col] = X[col].astype("category")
            X[col] = (
                X[col]
                .cat.add_categories(["__MISSING__"])
                .fillna("__MISSING__")
                .cat.codes
                .astype("int32")
            )
            categorical_feature_indices.append(i)
        else:
            if not np.issubdtype(X[col].dtype, np.number):
                X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].astype("float32")

    return X, feature_cols, categorical_feature_indices


def prepare_for_catboost(
    lf: pl.LazyFrame,
    streaming: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare data for CatBoost inference"""
    lf = engineer_all_features_polars(lf)

    exclude_cols = set(EXCLUDED_FEATURES) | {ID_COLUMN, TARGET_COLUMN}

    engine = "streaming" if streaming else None
    df_pl = lf.collect(engine=engine)
    df = df_pl.to_pandas()

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    cat_features = [col for col in X.columns if col in CATEGORICAL_FEATURES]
    for col in cat_features:
        X[col] = X[col].astype(str).fillna("__MISSING__")

    for col in BINARY_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(0).astype(int)

    return X, cat_features


def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """Weighted LogLoss"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    mask0, mask1 = (y_true == 0), (y_true == 1)
    ll0 = -np.mean(np.log(1 - y_pred[mask0])) if mask0.sum() else 0
    ll1 = -np.mean(np.log(y_pred[mask1])) if mask1.sum() else 0
    return 0.5 * ll0 + 0.5 * ll1


# ==========================================================
# Main Inference
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="XGBoost + CatBoost Ensemble Inference")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to model directory (e.g., outputs/polars-ooc-stack-phase2-noise-20250117-143022)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to data directory containing test.parquet")
    parser.add_argument("--output-dir", type=str, default="./submissions",
                        help="Path to output directory for submission files")

    args = parser.parse_args()

    log = logging.getLogger("xgb_cat_inference")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    log.info("\n" + "=" * 80)
    log.info("XGBoost + CatBoost Ensemble Inference")
    log.info("=" * 80)

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\nModel directory: {model_dir}")
    log.info(f"Data directory: {data_dir}")
    log.info(f"Output directory: {output_dir}")

    # Find model files
    xgb_models = sorted(model_dir.glob("xgb_fold*.json"))
    cat_models = sorted(model_dir.glob("catboost_fold*.cbm"))

    n_folds = len(xgb_models)
    log.info(f"\nFound {n_folds} XGBoost models and {len(cat_models)} CatBoost models")

    if n_folds == 0 or len(cat_models) == 0:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    # Load test data
    log.info("\n1. Loading and preprocessing test data")
    test_lf = pl.scan_parquet(f"{data_dir}/test.parquet")
    test_ids = test_lf.select([ID_COLUMN]).collect(engine="streaming").to_pandas()[ID_COLUMN]
    log.info(f"   Test samples: {len(test_ids):,}")

    # Prepare data for XGBoost
    log.info("\n2. Preparing data for XGBoost")
    X_xgb, feat_xgb, xgb_cat_idx = prepare_for_xgboost(test_lf, streaming=True)
    log.info(f"   XGBoost features: {X_xgb.shape}")

    # Prepare data for CatBoost
    log.info("\n3. Preparing data for CatBoost")
    X_cat, cat_feats = prepare_for_catboost(test_lf, streaming=True)
    log.info(f"   CatBoost features: {X_cat.shape}")
    log.info(f"   Categorical features: {len(cat_feats)}")

    # XGBoost inference
    log.info(f"\n4. XGBoost inference ({n_folds}-fold ensemble)")
    test_xgb_preds = []

    X_xgb_np = X_xgb.to_numpy()
    dtest = xgb.DMatrix(X_xgb_np)

    for i, model_path in enumerate(xgb_models, 1):
        log.info(f"   Loading fold {i}: {model_path.name}")
        model = xgb.Booster()
        model.load_model(str(model_path))

        preds = model.predict(dtest)
        test_xgb_preds.append(preds)
        log.info(f"   Fold {i} predictions: Mean={preds.mean():.4f}, Std={preds.std():.4f}")

    # CatBoost inference
    log.info(f"\n5. CatBoost inference ({len(cat_models)}-fold ensemble)")
    test_cat_preds = []

    test_pool = Pool(X_cat, cat_features=cat_feats)

    for i, model_path in enumerate(cat_models, 1):
        log.info(f"   Loading fold {i}: {model_path.name}")
        model = CatBoostClassifier()
        model.load_model(str(model_path))

        preds = model.predict_proba(test_pool)[:, 1]
        test_cat_preds.append(preds)
        log.info(f"   Fold {i} predictions: Mean={preds.mean():.4f}, Std={preds.std():.4f}")

    # Ensemble
    log.info("\n6. Creating ensemble predictions")

    # Median aggregation (robust against outliers)
    test_xgb_final = np.median(test_xgb_preds, axis=0)
    test_cat_final = np.median(test_cat_preds, axis=0)

    log.info(f"   XGBoost ensemble: Mean={test_xgb_final.mean():.4f}, Std={test_xgb_final.std():.4f}")
    log.info(f"   CatBoost ensemble: Mean={test_cat_final.mean():.4f}, Std={test_cat_final.std():.4f}")

    # Weighted ensemble (equal weights if no training scores available)
    # You can adjust weights based on training performance
    w_xgb = 0.5
    w_cat = 0.5
    test_weighted = w_xgb * test_xgb_final + w_cat * test_cat_final

    log.info(f"   Weighted ensemble (XGB={w_xgb:.2f}, CAT={w_cat:.2f})")
    log.info(f"   Mean={test_weighted.mean():.4f}, Std={test_weighted.std():.4f}")

    # Save submissions
    log.info("\n7. Generating submission files")

    submissions = {
        "xgboost.csv": test_xgb_final,
        "catboost.csv": test_cat_final,
        "xgb_cat_weighted.csv": test_weighted
    }

    for filename, preds in submissions.items():
        sub = pd.DataFrame({ID_COLUMN: test_ids, TARGET_COLUMN: preds})
        path = output_dir / filename
        sub.to_csv(path, index=False)
        log.info(f"   Saved: {path}")

    log.info("\n" + "=" * 80)
    log.info("Inference completed!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
