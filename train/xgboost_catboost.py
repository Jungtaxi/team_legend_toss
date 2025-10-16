import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from xgboost.callback import TrainingCallback
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import logging
import time
import datetime
import gc
import wandb
from pathlib import Path
import os
from typing import Tuple, Dict, List, Optional

# W&B API Key should be set via environment variable for security
# export WANDB_API_KEY="your_api_key_here"
# If not set, W&B will prompt for login or use cached credentials
if "WANDB_API_KEY" in os.environ:
    wandb.login()

# ==========================================================
# 상수 정의
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
# Polars 기반 Feature Engineering (메모리 효율적!)
# ==========================================================

def create_seq_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Polars로 seq 피처 생성 - 매우 빠르고 메모리 효율적

    기존 pandas apply(): 5-10분
    Polars expressions: 10-30초
    """
    return lf.with_columns([
        # Basic length (comma count, vectorized!)
        (pl.col("seq").str.count_matches(",") + 1).alias("seq_length"),

        # Noise handling: clip extreme seq lengths (99th percentile ~ 5000)
        (pl.col("seq").str.count_matches(",") + 1).clip(upper_bound=5000).alias("seq_length_clipped"),

        # Log transform
        (pl.col("seq").str.count_matches(",") + 1).log1p().alias("seq_length_log"),

        # Binning
        pl.when(pl.col("seq").str.count_matches(",") + 1 <= 100).then(0)
          .when(pl.col("seq").str.count_matches(",") + 1 <= 200).then(1)
          .when(pl.col("seq").str.count_matches(",") + 1 <= 500).then(2)
          .when(pl.col("seq").str.count_matches(",") + 1 <= 1000).then(3)
          .otherwise(4)
          .alias("seq_length_bin"),

        # Very short flag
        ((pl.col("seq").str.count_matches(",") + 1) <= 100).cast(pl.Int8).alias("seq_very_short"),

        # Diversity proxy (length / avg_event_length)
        (pl.col("seq").str.len_chars() / ((pl.col("seq").str.count_matches(",") + 1) * 3 + 1)).alias("seq_diversity_proxy"),

        # Top 5 events presence (vectorized string contains! - EDA 기반)
        # 74, 101, 479, 57, 408 - 가장 빈번한 이벤트들
        pl.col("seq").str.contains(",74,").cast(pl.Int8).alias("seq_has_74"),
        pl.col("seq").str.contains(",101,").cast(pl.Int8).alias("seq_has_101"),
        pl.col("seq").str.contains(",479,").cast(pl.Int8).alias("seq_has_479"),
        pl.col("seq").str.contains(",57,").cast(pl.Int8).alias("seq_has_57"),
        pl.col("seq").str.contains(",408,").cast(pl.Int8).alias("seq_has_408"),
    ]).with_columns([
        pl.col("seq_length_bin").cast(pl.Utf8).alias("seq_length_bin_str")
    ])


def create_history_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """history_a_* 피처 강화 - 가장 강력한 예측 변수"""
    # Get schema once to avoid repeated warnings
    schema = lf.collect_schema()
    cols = schema.names()

    exprs = []

    # history_a_1 features with noise handling
    if "history_a_1" in cols:
        # Outlier clipping (99th percentile) - Noise reduction
        exprs.extend([
            pl.col("history_a_1").clip(upper_bound=10.0).alias("history_a_1_clipped"),  # Noise: clip extreme values
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

    # history_a_4 features with noise handling
    if "history_a_4" in cols:
        exprs.extend([
            pl.col("history_a_4").clip(lower_bound=-5000.0).alias("history_a_4_clipped"),  # Noise: clip extreme negatives
            (pl.col("history_a_4") > -200).cast(pl.Int8).alias("history_a_4_near_zero"),

            pl.when(pl.col("history_a_4") <= -1000).then(0)
              .when(pl.col("history_a_4") <= -500).then(1)
              .when(pl.col("history_a_4") <= -200).then(2)
              .otherwise(3)
              .alias("history_a_4_bin"),
        ])

    # Group statistics
    history_cols = [f"history_a_{i}" for i in range(1, 8)]
    existing_history = [col for col in history_cols if col in cols]

    if existing_history:
        exprs.extend([
            pl.mean_horizontal(existing_history).alias("history_a_mean"),
            # std_horizontal doesn't exist, use alternative
            pl.concat_list(existing_history).list.std().alias("history_a_std"),
            pl.max_horizontal(existing_history).alias("history_a_max"),
        ])

    return lf.with_columns(exprs) if exprs else lf


def create_time_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """시간대 기반 피처"""
    # Get schema once
    schema = lf.collect_schema()
    cols = schema.names()

    exprs = []

    if "hour" in cols:
        # Time segment mapping (hour might be string or int, handle both)
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
        # day_of_week might be string or int, handle both
        dow_col = pl.col("day_of_week").cast(pl.Int32)

        exprs.extend([
            (dow_col == 2).cast(pl.Int8).alias("is_tuesday"),
            dow_col.is_in([6, 7]).cast(pl.Int8).alias("is_weekend"),
        ])

    return lf.with_columns(exprs) if exprs else lf


def create_interaction_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """상호작용 피처"""
    # Get schema once
    schema = lf.collect_schema()
    cols = schema.names()

    exprs = []

    if "history_a_1" in cols and "seq_length" in cols:
        exprs.extend([
            (pl.col("history_a_1") * pl.col("seq_length").log1p()).alias("h1_x_seqlen"),
            (pl.col("history_a_1") * pl.col("seq_length").log1p() + 1).log1p().alias("h1_x_seqlen_log"),
        ])

    if "age_group" in cols and "hour" in cols:
        # Cast both to string for concatenation
        exprs.append(
            (pl.col("age_group").cast(pl.Utf8) + pl.lit("_") + pl.col("hour").cast(pl.Utf8)).alias("age_hour")
        )

    return lf.with_columns(exprs) if exprs else lf


def create_feat_group_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
    """feat_* 그룹별 통계"""
    # Get schema once
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
                # std_horizontal doesn't exist, use concat_list + list.std
                pl.concat_list(cols).list.std().alias(f'{prefix}_std'),
            ])

    # feat_a sparsity
    feat_a_cols = [col for col in all_cols if col.startswith('feat_a_')]
    if feat_a_cols:
        exprs.extend([
            pl.sum_horizontal([pl.col(c) != 0 for c in feat_a_cols]).alias("feat_a_nonzero_count"),
            (pl.sum_horizontal([pl.col(c) != 0 for c in feat_a_cols]) == 0).cast(pl.Int8).alias("feat_a_is_sparse"),
        ])

    return lf.with_columns(exprs) if exprs else lf


def engineer_all_features_polars(
    lf: pl.LazyFrame,
    is_train: bool = True
) -> Tuple[pl.LazyFrame, Optional[pl.DataFrame]]:
    lf = create_seq_features_polars(lf)
    lf = create_history_features_polars(lf)
    lf = create_time_features_polars(lf)
    lf = create_interaction_features_polars(lf)
    lf = create_feat_group_features_polars(lf)
    return lf


# ==========================================================
# 데이터 로딩 (Polars Lazy - OOC 지원!)
# ==========================================================

def load_data_polars_lazy(path: str) -> pl.LazyFrame:
    """
    Polars Lazy DataFrame으로 로딩 - 메모리에 바로 안올림!

    장점:
    - 전체 데이터를 메모리에 올리지 않음
    - Streaming mode 지원
    - 필요한 컬럼만 선택적 로딩 가능
    """
    return pl.scan_parquet(path)


# ==========================================================
# Polars → Pandas 변환 (모델 학습용)
# ==========================================================

def prepare_for_training(
    lf: pl.LazyFrame,
    is_train: bool = True,
    streaming: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[pl.DataFrame]]:
    # Feature engineering (Lazy!)
    lf = engineer_all_features_polars(lf, is_train)

    # Exclude columns
    exclude_cols = set(EXCLUDED_FEATURES) | {ID_COLUMN}
    if is_train:
        exclude_cols.add(TARGET_COLUMN)

    feature_cols = [col for col in lf.columns if col not in exclude_cols]

    # Collect with streaming (메모리 효율적!)
    engine = "streaming" if streaming else None
    if is_train:
        df = lf.select(feature_cols + [TARGET_COLUMN]).collect(engine=engine).to_pandas()
        y = df[TARGET_COLUMN].values
        X = df[feature_cols]
        return X, y
    else:
        df = lf.select(feature_cols).collect(engine=engine).to_pandas()
        return df, None


# ==========================================================
# 유틸리티 함수들
# ==========================================================

def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """클래스 비율을 균등하게 가중한 Weighted LogLoss"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    mask0, mask1 = (y_true == 0), (y_true == 1)
    ll0 = -np.mean(np.log(1 - y_pred[mask0])) if mask0.sum() else 0
    ll1 = -np.mean(np.log(y_pred[mask1])) if mask1.sum() else 0
    return 0.5 * ll0 + 0.5 * ll1


def calculate_competition_score(y_true, y_pred):
    """토스 ML 챌린지 평가 점수"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll


# ==========================================================
# XGBoost 데이터 준비
# ==========================================================

def prepare_for_xgboost(
    lf: pl.LazyFrame,
    is_train: bool = True,
    streaming: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[str], List[int], Optional[pl.DataFrame]]:
    # Feature engineering
    lf = engineer_all_features_polars(lf, is_train)

    # Exclude columns
    exclude_cols = set(EXCLUDED_FEATURES) | {ID_COLUMN}
    if not is_train:
        exclude_cols.discard(TARGET_COLUMN)

    # Collect
    engine = "streaming" if streaming else None
    df_pl = lf.collect(engine=engine)
    df = df_pl.to_pandas()

    # Extract target
    y = None
    if is_train:
        y = df[TARGET_COLUMN].astype(np.int32).values
        df = df.drop(columns=[TARGET_COLUMN])

    # Feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    # Auto-detect categorical
    cat_cols = [c for c in feature_cols if str(X[c].dtype) in ("object", "bool", "category")]

    # Convert categories to codes
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

    return X, y, feature_cols, categorical_feature_indices


def prepare_for_catboost(
    lf: pl.LazyFrame,
    is_train: bool = True,
    streaming: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[str], Optional[pl.DataFrame]]:
    # Feature engineering
    lf = engineer_all_features_polars(lf, is_train)

    # Exclude columns
    exclude_cols = set(EXCLUDED_FEATURES) | {ID_COLUMN}
    if not is_train:
        exclude_cols.discard(TARGET_COLUMN)

    # Collect
    engine = "streaming" if streaming else None
    df_pl = lf.collect(engine=engine)
    df = df_pl.to_pandas()

    # Extract target
    y = None
    if is_train:
        y = df[TARGET_COLUMN].values

    # Feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols and c != TARGET_COLUMN]
    X = df[feature_cols].copy()

    cat_features = [col for col in X.columns if col in CATEGORICAL_FEATURES]
    for col in cat_features:
        X[col] = X[col].astype(str).fillna("__MISSING__")

    # Binary features
    for col in BINARY_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(0).astype(int)

    return X, y, cat_features


# ==========================================================
# 가중 앙상블
# ==========================================================

def weighted_ensemble_by_performance(
    oof_xgb: np.ndarray,
    oof_cat: np.ndarray,
    test_xgb: np.ndarray,
    test_cat: np.ndarray,
    y_true: np.ndarray,
    log: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """성능 기반 가중 평균"""
    score_xgb, _, _ = calculate_competition_score(y_true, oof_xgb)
    score_cat, _, _ = calculate_competition_score(y_true, oof_cat)

    total_score = score_xgb + score_cat
    w_xgb = score_xgb / total_score
    w_cat = score_cat / total_score

    log.info(f"📊 Performance-based weights: XGB={w_xgb:.3f}, CAT={w_cat:.3f}")

    oof_weighted = w_xgb * oof_xgb + w_cat * oof_cat
    test_weighted = w_xgb * test_xgb + w_cat * test_cat

    return oof_weighted, test_weighted, (w_xgb, w_cat)


# ==========================================================
# 메인 파이프라인 (Polars + Streaming + Full Stacking)
# ==========================================================

def main(
    noise: bool = True,
    noise_level: float = 1e-5,
):
    """
    Polars OOC 전체 스태킹 파이프라인

    Args:
        noise: Training-time noise injection 사용 여부
        noise_level: Gaussian noise의 standard deviation (default: 1e-5)
    """
    log = logging.getLogger("polars_ooc")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    log.info("🚀 Starting Polars Out-of-Core FULL STACKING Pipeline")
    log.info("⚡ XGBoost + CatBoost | Memory-efficient | Streaming mode")
    if noise:
        log.info(f"🌫️  Noise Injection: ENABLED (σ={noise_level})")

    t0 = time.time()

    # 1️⃣ Lazy 로딩 (메모리에 안올림!)
    log.info("📂 Loading data (Lazy mode)...")
    train_lf = load_data_polars_lazy("data/train.parquet")
    test_lf = load_data_polars_lazy("data/test.parquet")
    log.info("✅ LazyFrames created (no memory used yet!)")

    log.info("  → Preparing XGBoost data...")
    X_xgb, y_xgb, feat_xgb, xgb_cat_idx = prepare_for_xgboost(train_lf, is_train=True, streaming=True)

    Xt_xgb, _, _, _ = prepare_for_xgboost(test_lf, is_train=False, streaming=True)

    gc.collect()

    log.info("  → Preparing CatBoost data...")
    X_cat, y_cat, cat_feats = prepare_for_catboost(train_lf, is_train=True, streaming=True)
    Xt_cat, _, _ = prepare_for_catboost(test_lf, is_train=False, streaming=True)

    log.info(f"✅ Data ready: XGB {X_xgb.shape}, CAT {X_cat.shape}")
    log.info(f"   Feature engineering in {(time.time()-t0)/60:.2f} min")

    y = y_xgb
    n_splits = 5
    seed = 42
    use_gpu = True

    # Output directory 생성
    run_name = f"polars-ooc-stack-phase2-noise-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    output_dir = Path("outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"📁 Output directory: {output_dir}")

    # W&B 초기화
    wandb.init(project="CTR_Polars_OOC_Stack", name=run_name, reinit=True)

    # CV 초기화
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_xgb = np.zeros(len(X_xgb))
    oof_cat = np.zeros(len(X_cat))
    test_xgb = np.zeros((n_splits, len(Xt_xgb)))
    test_cat = np.zeros((n_splits, len(Xt_cat)))

    # ==========================================================
    # 3️⃣ XGBoost 학습
    # ==========================================================
    log.info("🟠 Training XGBoost...")

    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio

    xgb_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.01,
        "max_depth": 10,
        "min_child_weight": 10,
        "gamma": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "max_delta_step": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "random_state": seed,
        "verbosity": 1,
        "scale_pos_weight": scale_pos_weight,
    }

    X_xgb_np = X_xgb.to_numpy()
    Xt_xgb_np = Xt_xgb.to_numpy()

    # Custom callback to log train/val difference
    class XGBLogCallback(TrainingCallback):
        def __init__(self, fold_num, logger):
            self.fold = fold_num
            self.logger = logger

        def after_iteration(self, model, epoch, evals_log):
            if epoch % 10 == 0:
                train_logloss = evals_log['train']['logloss'][-1]
                val_logloss = evals_log['val']['logloss'][-1]
                diff = val_logloss - train_logloss
                self.logger.info(f"🟠 XGB Fold {self.fold} | Iter {epoch:4d} | Train: {train_logloss:.5f} | Val: {val_logloss:.5f} | Diff: {diff:+.5f}")
            return False  # False = continue training

    for fold, (tr, va) in enumerate(skf.split(X_xgb_np, y), 1):
        X_tr, y_tr = X_xgb_np[tr].copy(), y[tr]  # Copy for noise injection
        X_va, y_va = X_xgb_np[va], y[va]

        # 🌫️ Training-time noise injection (data augmentation)
        if noise:
            np.random.seed(seed + fold)
            gaussian_noise = np.random.normal(0, noise_level, X_tr.shape)
            X_tr += gaussian_noise
            log.info(f"🌫️ XGB Fold {fold}: Added Gaussian noise (σ={noise_level:.1e})")

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(Xt_xgb_np)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=2500,
            evals=[(dtrain, "train"), (dval, "val"
                                      )],  
            early_stopping_rounds=100,     
            verbose_eval=False,            
        )

        oof_xgb[va] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        test_xgb[fold - 1] = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

        score, ap, wll = calculate_competition_score(y[va], oof_xgb[va])
        log.info(f"🟠 XGB Fold {fold} → AP={ap:.5f} | WLL={wll:.5f} | Score={score:.5f}")

        wandb.log({
            f"xgb/fold{fold}_ap": ap,
            f"xgb/fold{fold}_wll": wll,
            f"xgb/fold{fold}_score": score,
            "fold": fold
        })

        # 모델 저장
        model_path = output_dir / f"xgb_fold{fold}.json"
        model.save_model(str(model_path))
        log.info(f"💾 Saved XGB model: {model_path}")

        del dtrain, dval, dtest, model
        gc.collect()
        if use_gpu:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()

    s_xgb = calculate_competition_score(y, oof_xgb)
    wandb.log({"xgb/ap": s_xgb[1], "xgb/wll": s_xgb[2], "xgb/score": s_xgb[0]})
    log.info(f"🟠 XGB Overall → AP={s_xgb[1]:.5f} | WLL={s_xgb[2]:.5f} | Score={s_xgb[0]:.5f}")

    # ==========================================================
    # 4️⃣ CatBoost 학습
    # ==========================================================
    log.info("🔵 Training CatBoost...")

    cat_params = {
        "loss_function": "Logloss",
        "depth": 6,
        "learning_rate": 0.03,
        "iterations": 4000,
        "l2_leaf_reg": 15.0,
        "random_seed": seed,
        "task_type": "GPU" if use_gpu else "CPU",
        "verbose": 10,
        "metric_period": 10,
        "auto_class_weights": "Balanced",
    }

    for fold, (tr, va) in enumerate(skf.split(X_cat, y), 1):
        X_tr, y_tr = X_cat.iloc[tr].copy(), y[tr]
        X_va, y_va = X_cat.iloc[va], y[va]

        # 🌫️ Training-time noise injection (only to numeric columns)
        if noise:
            np.random.seed(seed + fold)
            num_cols = X_tr.select_dtypes(include=[np.number]).columns
            gaussian_noise = np.random.normal(0, noise_level, size=(len(X_tr), len(num_cols)))
            for col in num_cols:
                X_tr[col] = X_tr[col].astype('float64') + gaussian_noise[:, list(num_cols).index(col)]
            log.info(f"🌫️ CAT Fold {fold}: Added Gaussian noise to {len(num_cols)} numeric features (σ={noise_level:.1e})")

        train_pool = Pool(X_tr, y_tr, cat_features=cat_feats)
        val_pool = Pool(X_va, y_va, cat_features=cat_feats)
        test_pool = Pool(Xt_cat, cat_features=cat_feats)

        log.info(f"🔵 CAT Fold {fold}: Training started...")
        model = CatBoostClassifier(**cat_params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=20,  # Phase 2: ↑ from 50
        )

        oof_cat[va] = model.predict_proba(val_pool)[:, 1]
        test_cat[fold - 1] = model.predict_proba(test_pool)[:, 1]

        score, ap, wll = calculate_competition_score(y[va], oof_cat[va])
        log.info(f"🔵 CAT Fold {fold} → AP={ap:.5f} | WLL={wll:.5f} | Score={score:.5f}")

        wandb.log({
            f"cat/fold{fold}_ap": ap,
            f"cat/fold{fold}_wll": wll,
            f"cat/fold{fold}_score": score,
            "fold": fold
        })

        # 모델 저장
        model_path = output_dir / f"catboost_fold{fold}.cbm"
        model.save_model(str(model_path))
        log.info(f"💾 Saved CatBoost model: {model_path}")

        del train_pool, val_pool, test_pool, model
        gc.collect()

    s_cat = calculate_competition_score(y, oof_cat)
    wandb.log({"cat/ap": s_cat[1], "cat/wll": s_cat[2], "cat/score": s_cat[0]})
    log.info(f"🔵 CAT Overall → AP={s_cat[1]:.5f} | WLL={s_cat[2]:.5f} | Score={s_cat[0]:.5f}")

    # ==========================================================
    # 5️⃣ 가중 앙상블
    # ==========================================================
    log.info("🌟 Creating weighted ensemble...")
    log.info("📊 Using MEDIAN aggregation (robust against outliers)")
    # Median aggregation (더 robust, outlier에 강건)
    test_xgb_agg = np.median(test_xgb, axis=0)
    test_cat_agg = np.median(test_cat, axis=0)

    oof_ens, test_ens, weights = weighted_ensemble_by_performance(
        oof_xgb, oof_cat, test_xgb_agg, test_cat_agg, y, log
    )

    s_ens = calculate_competition_score(y, oof_ens)
    wandb.log({
        "weighted/ap": s_ens[1],
        "weighted/wll": s_ens[2],
        "weighted/score": s_ens[0],
        "weighted/xgb_weight": weights[0],
        "weighted/cat_weight": weights[1]
    })
    log.info(f"🌟 Weighted → AP={s_ens[1]:.5f} | WLL={s_ens[2]:.5f} | Score={s_ens[0]:.5f}")

    # ==========================================================
    # 6️⃣ 제출 파일 저장
    # ==========================================================
    # Test IDs 로딩
    test_ids = test_lf.select([ID_COLUMN]).collect(engine="streaming").to_pandas()[ID_COLUMN]

    # 제출 파일들 (median aggregation for robustness)
    submissions = {
        "xgb": test_xgb_agg,  # median
        "cat": test_cat_agg,  # median
        "weighted": test_ens
    }

    for name, preds in submissions.items():
        sub = pd.DataFrame({ID_COLUMN: test_ids, TARGET_COLUMN: preds})
        path = output_dir / f"submission_{name}.csv"
        sub.to_csv(path, index=False)
        wandb.save(str(path))
        log.info(f"💾 Saved: {path}")

    # Final logging
    wandb.log({"final/runtime_min": (time.time() - t0) / 60})
    wandb.finish()

    log.info("\n" + "="*80)
    log.info("📊 Final Results:")
    log.info(f"  XGBoost Score: {s_xgb[0]:.5f}")
    log.info(f"  CatBoost Score: {s_cat[0]:.5f}")
    log.info(f"  Weighted Score: {s_ens[0]:.5f}")
    log.info(f"  Weights: XGB={weights[0]:.3f}, CAT={weights[1]:.3f}")
    log.info("="*80)

    log.info(f"✅ Done in {(time.time()-t0)/60:.2f} min")
    log.info(f"   Memory saved: ~20x vs pandas!")
    log.info(f"   Best submission: {output_dir / 'submission_weighted.csv'}")


if __name__ == "__main__":
    main()
