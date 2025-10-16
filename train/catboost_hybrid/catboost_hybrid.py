#!/usr/bin/env python
# catboost_hybrid.py
# -*- coding: utf-8 -*-
"""
CatBoost Advanced Training Script
- includes advanced feature engineering
- compatible with GPU / wandb / config.yaml
"""

import os
import gc
import yaml
import psutil
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

import polars as pl
from feature_engineering import PolarsFeatureEngineer
import wandb


import hashlib
import json

def get_config_hash(cfg):
    """설정의 해시값을 생성하여 캐시 키로 사용"""
    config_str = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def save_engineered_features(df, cfg, data_type="train"):
    """Feature Engineering 결과 저장"""
    cache_dir = cfg["paths"].get("cache_dir", "./cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    config_hash = get_config_hash(cfg.get("feature_engineering", {}))
    cache_path = os.path.join(cache_dir, f"{data_type}_fe_{config_hash}.parquet")
    
    # Polars DataFrame을 Parquet로 저장
    df.write_parquet(cache_path)
    log.info(f"💾 Engineered features saved: {cache_path}")
    return cache_path


def load_engineered_features(cfg, data_type="train"):
    """저장된 Feature Engineering 결과 로드"""
    cache_dir = cfg["paths"].get("cache_dir", "./cache")
    config_hash = get_config_hash(cfg.get("feature_engineering", {}))
    cache_path = os.path.join(cache_dir, f"{data_type}_fe_{config_hash}.parquet")
    
    if os.path.exists(cache_path):
        log.info(f"📂 Loading cached features from: {cache_path}")
        df = pl.read_parquet(cache_path)
        log.info(f"✅ Cached features loaded: {df.shape}")
        return df
    else:
        log.info(f"⚠️ No cache found at: {cache_path}")
        return None


def process_data_with_cache(cfg, data_type="train", force_reprocess=False):
    """
    캐시를 활용한 데이터 처리
    
    Args:
        cfg: config.yaml 설정
        data_type: "train" or "test"
        force_reprocess: True면 캐시 무시하고 재처리
    """
    
    # 1️⃣ 캐시 확인
    if not force_reprocess:
        cached_df = load_engineered_features(cfg, data_type)
        if cached_df is not None:
            return cached_df
    
    # 2️⃣ 캐시가 없거나 force_reprocess=True면 새로 처리
    log.info(f"🔧 Processing {data_type} data from scratch...")
    
    if data_type == "train":
        data_path = cfg["paths"]["train_path"]
    else:
        data_path = cfg["paths"]["test_path"]
    
    df = pl.read_parquet(data_path)
    log.info(f"📊 {data_type.capitalize()} shape: {df.shape}")
    
    # 3️⃣ Feature Engineering 수행
    df = advanced_feature_engineering_polars(df)
    engineer = PolarsFeatureEngineer(cfg)
    df = engineer.transform(df)
    log.info(f"📊 {data_type.capitalize()} shape after FE: {df.shape}")
    
    # 4️⃣ 결과 저장
    save_engineered_features(df, cfg, data_type)
    
    return df


# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("catboost_adv")

warnings.filterwarnings("ignore")

# =====================================================
# WandB Setup
# =====================================================
def setup_wandb(cfg):
    wandb_enabled = False
    wandb_cfg = cfg.get("wandb", {})
    api_key = wandb_cfg.get("api_key")
    project = wandb_cfg.get("project")

    if api_key and project:
        try:
            wandb.login(key=api_key)
            wandb.init(project=project, entity=wandb_cfg.get("entity"), config=cfg)
            log.info(f"📡 W&B initialized: project={project}, entity={wandb_cfg.get('entity')}")
            wandb_enabled = True
        except Exception as e:
            log.warning(f"⚠️ W&B initialization failed: {e}")
    else:
        log.info("🚫 W&B disabled (no API key)")

    return wandb_enabled


# =====================================================
# Data Loading & Feature Engineering
# =====================================================
def load_train_data(path):
    df = pl.read_parquet(path)
    log.info(f"📊 Train shape: {df.shape}")
    return df


CAT_FILL = "__MISSING__"
CATEGORICAL_FEATURES = ["gender", "age_group", "inventory_id", "day_of_week", "hour"]
EXCLUDED_COLUMNS = ["seq"]


def advanced_feature_engineering_polars(df: pl.DataFrame) -> pl.DataFrame:
    df = df.clone()

    # -----------------------------
    # 1️⃣ 안전한 타입 변환
    # -----------------------------
    for col in ["hour", "day_of_week"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # -----------------------------
    # 2️⃣ 시간 관련 Feature
    # -----------------------------
    if "hour" in df.columns:
        df = df.with_columns([
            (pl.col("hour") // 4).cast(pl.Utf8).alias("hour_block"),
            ((pl.col("hour") >= 6) & (pl.col("hour") < 12)).cast(pl.Int8).alias("is_morning"),
            ((pl.col("hour") >= 18) & (pl.col("hour") < 24)).cast(pl.Int8).alias("is_evening"),
        ])

    if "day_of_week" in df.columns:
        df = df.with_columns(
            pl.col("day_of_week").is_in([5, 6]).cast(pl.Int8).alias("is_weekend")
        )

    if all(c in df.columns for c in ["day_of_week", "hour"]):
        df = df.with_columns(
            pl.concat_str([
                pl.col("day_of_week").cast(pl.Utf8),
                (pl.col("hour") // 4).cast(pl.Utf8)
            ], separator="_").alias("day_hour_cross")
        )

    # -----------------------------
    # 3️⃣ l_feat 계열
    # -----------------------------
    l_feats = [c for c in df.columns if c.startswith("l_feat_")]
    if l_feats:
        df = df.with_columns([
            pl.sum_horizontal(l_feats).alias("l_feat_sum"),
            pl.mean_horizontal(l_feats).alias("l_feat_mean"),
            pl.min_horizontal(l_feats).alias("l_feat_min"),
            pl.max_horizontal(l_feats).alias("l_feat_max"),
        ])

        # ✅ 표준편차 직접 계산 (apply 없이)
        mean_sq = pl.mean_horizontal([pl.col(c) ** 2 for c in l_feats])
        mean = pl.mean_horizontal(l_feats)
        var_expr = mean_sq - mean**2
        safe_var = pl.when(var_expr > 0).then(var_expr).otherwise(0)  # clip_min(0) 대체
        df = df.with_columns(safe_var.sqrt().alias("l_feat_std"))

        if {"l_feat_1", "l_feat_14"} <= set(df.columns):
            df = df.with_columns(
                (pl.col("l_feat_1") / (pl.col("l_feat_14") + 1e-6)).fill_nan(0).alias("l_feat_ratio_1_14")
            )

        df = df.with_columns([
            ((pl.col("l_feat_max") - pl.col("l_feat_mean")) / (pl.col("l_feat_std") + 1e-6))
            .fill_nan(0)
            .alias("l_feat_zmax"),
            ((pl.col("l_feat_min") - pl.col("l_feat_mean")) / (pl.col("l_feat_std") + 1e-6))
            .fill_nan(0)
            .alias("l_feat_zmin")
        ])

        # -----------------------------
        # 4️⃣ history 계열
        # -----------------------------
        hist_a = [c for c in df.columns if c.startswith("history_a_")]
        hist_b = [c for c in df.columns if c.startswith("history_b_")]

        if len(hist_a) >= 2:
            df = df.with_columns([
                pl.mean_horizontal(hist_a).alias("hist_a_mean"),
                (pl.col("history_a_1") - pl.col("history_a_3")).alias("hist_a_delta_13"),
                (pl.col("history_a_1") / (pl.col("history_a_3") + 1e-6)).fill_nan(0).alias("hist_a_ratio_13"),
            ])

        if len(hist_b) >= 2:
            df = df.with_columns([
                pl.mean_horizontal(hist_b).alias("hist_b_mean"),
                (pl.col("history_b_1") - pl.col("history_b_3")).alias("hist_b_delta_13"),
                (pl.col("history_b_1") / (pl.col("history_b_3") + 1e-6)).fill_nan(0).alias("hist_b_ratio_13"),
            ])

        if {"history_a_1", "history_a_3"} <= set(df.columns):
            df = df.with_columns(
                pl.when((pl.col("history_a_1") - pl.col("history_a_3")) > 0)
                .then(1)
                .when((pl.col("history_a_1") - pl.col("history_a_3")) < 0)
                .then(-1)
                .otherwise(0)
                .alias("history_a_trend")
            )

        if {"history_b_1", "history_b_3"} <= set(df.columns):
            df = df.with_columns(
                pl.when((pl.col("history_b_1") - pl.col("history_b_3")) > 0)
                .then(1)
                .when((pl.col("history_b_1") - pl.col("history_b_3")) < 0)
                .then(-1)
                .otherwise(0)
                .alias("history_b_trend")
            )


    # -----------------------------
    # 5️⃣ 교차 Feature
    # -----------------------------
    if {"inventory_id", "age_group"} <= set(df.columns):
        df = df.with_columns(
            pl.concat_str(["inventory_id", "age_group"], separator="_").alias("inv_age_cross")
        )
    if {"inventory_id", "hour_block"} <= set(df.columns):
        df = df.with_columns(
            pl.concat_str(["inventory_id", "hour_block"], separator="_").alias("inv_hour_cross")
        )
    if {"inventory_id", "gender"} <= set(df.columns):
        df = df.with_columns(
            pl.concat_str(["inventory_id", "gender"], separator="_").alias("inv_gender_cross")
        )

    # -----------------------------
    # 6️⃣ Frequency Encoding
    # -----------------------------
    for col in ["inventory_id", "age_group", "gender"]:
        if col in df.columns:
            freq = df.group_by(col).agg(pl.count().alias("freq"))
            total = freq["freq"].sum()
            freq = freq.with_columns((pl.col("freq") / total).alias(f"{col}_freq"))
            df = df.join(freq.select([col, f"{col}_freq"]), on=col, how="left")

    # -----------------------------
    # 7️⃣ Cleanup
    # -----------------------------
    for c in ["gender", "age_group", "inventory_id", "day_of_week", "hour"]:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Utf8).fill_null("__MISSING__"))

    if "seq" in df.columns:
        df = df.drop("seq")

    df = df.fill_nan(0).fill_null(0)
    return df


# =====================================================
# Utility
# =====================================================
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def print_memory():
    mem = psutil.virtual_memory()
    log.info(f"💾 CPU {mem.used/1e9:.1f}/{mem.total/1e9:.1f}GB ({mem.percent:.1f}%)")

def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    mask0, mask1 = (y_true == 0), (y_true == 1)
    ll0 = -np.mean(np.log(1 - y_pred[mask0])) if mask0.sum() else 0
    ll1 = -np.mean(np.log(y_pred[mask1])) if mask1.sum() else 0
    return 0.5 * ll0 + 0.5 * ll1

def calculate_competition_score(y_true, y_pred):
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    return 0.5 * ap + 0.5 * (1 / (1 + wll)), ap, wll


# =====================================================
# CV Training (with WandB)
# =====================================================
def run_cv(df, cfg, wandb_enabled=False):
    y = df["clicked"].to_numpy()
    X = df.drop("clicked")

    # X = pl.from_pandas(X)
    # X = advanced_feature_engineering_polars(X)
    # engineer = PolarsFeatureEngineer(cfg)
    # X = engineer.transform(X)
    X = X.to_pandas()

    cat_features = [i for i, c in enumerate(X.columns) if str(X[c].dtype) == "object"]
    skf = StratifiedKFold(
        n_splits=cfg["training"]["n_folds"],
        shuffle=True,
        random_state=cfg["training"]["seed"],
    )

    scores = []
    model_dir = cfg["paths"].get("model_dir", "./models")
    os.makedirs(model_dir, exist_ok=True)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        log.info(f"📍 Fold {fold}/{cfg['training']['n_folds']}")
        train_pool = Pool(X.iloc[tr], y[tr], cat_features=cat_features)
        val_pool = Pool(X.iloc[va], y[va], cat_features=cat_features)

        model = CatBoostClassifier(**cfg["catboost"])
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=cfg["catboost"]["early_stopping_rounds"])

        if wandb_enabled:
            evals_result = model.get_evals_result()
            if "validation" in evals_result:
                for metric_name, values in evals_result["validation"].items():
                    for i, v in enumerate(values):
                        wandb.log({f"fold{fold}/{metric_name}": v, "iteration": i})

        y_pred = model.predict_proba(val_pool)[:, 1]
        score, ap, wll = calculate_competition_score(y[va], y_pred)
        log.info(f"   Score={score:.4f}, AP={ap:.4f}, WLL={wll:.4f}")
        scores.append(score)

        if wandb_enabled:
            wandb.log({f"fold{fold}_score": score, f"fold{fold}_AP": ap, f"fold{fold}_WLL": wll})

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{model_dir}/catboost_fold{fold}_{timestamp}.cbm"
        model.save_model(model_path)
        log.info(f"💾 Fold {fold} model saved: {model_path}")

        del model
        gc.collect()
        print_memory()

    mean_score, std_score = np.mean(scores), np.std(scores)
    log.info(f"🏆 Final CV Score: {mean_score:.6f} ± {std_score:.6f}")

    if wandb_enabled:
        wandb.log({"cv_mean_score": mean_score, "cv_std_score": std_score})

    return scores


# =====================================================
# Inference (예측 생성)
# =====================================================
def run_inference(cfg, model_path=None):
    log.info("🚀 Running inference")

    if model_path is None:
        model_dir = cfg["paths"].get("model_dir", "./models")
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".cbm")]
        if not model_files:
            raise FileNotFoundError("❌ No trained model found in model_dir")
        model_path = os.path.join(model_dir, sorted(model_files)[-1])
        log.info(f"📦 Auto-loaded latest model: {model_path}")

    # Load model
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Load test data
    test_path = cfg["paths"]["test_path"]
    log.info(f"📂 Loading test data from: {test_path}")
    df_test = pl.read_parquet(test_path)

    # Extract ID column (required for submission)
    if "ID" not in df_test.columns:
        raise KeyError("❌ Test dataset must contain 'ID' column")

    test_ids = df_test["ID"].to_numpy()
    df_test = df_test.drop("ID")

    # Drop seq column if exists
    if "seq" in df_test.columns:
        df_test = df_test.drop("seq")

    # Apply feature engineering
    df_test = advanced_feature_engineering_polars(df_test)
    fe = PolarsFeatureEngineer(cfg)
    df_test = fe.transform(df_test)

    df_test = df_test.fill_nan(0).fill_null(0)
    df_test = df_test.to_pandas()

    # Identify categorical features
    cat_features = [i for i, c in enumerate(df_test.columns) if str(df_test[c].dtype) == "object"]

    # Run inference
    test_pool = Pool(df_test, cat_features=cat_features)
    preds = model.predict_proba(test_pool)[:, 1]

    # Save submission
    submission_path = cfg["paths"]["submission_path"]
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    pd.DataFrame({"ID": test_ids, "clicked": preds}).to_csv(submission_path, index=False)
    log.info(f"✅ Submission saved: {submission_path}")

    return submission_path


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, default="cv",
                       choices=["cv", "train", "inference", "all"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--force-reprocess", action="store_true")

    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info("📘 Config loaded")

    wandb_enabled = setup_wandb(cfg)

    # Feature Engineering 캐싱
    if (args.mode in ["cv", "train", "all", "focal_train", "weighted_train"] 
        or args.optimize_focal or args.optimize_weights 
        or args.optimize_class_weights or args.grid_search_weights):
        
        force_reprocess = args.force_reprocess or cfg["training"].get("force_reprocess", False)
        df = process_data_with_cache(cfg, data_type="train", force_reprocess=force_reprocess)

    # 🔥 Class Weights 최적
    if args.mode == "cv":
        scores = run_cv(df, cfg, wandb_enabled)
        if wandb_enabled:
            wandb.log({"cv_mean_score": np.mean(scores), "cv_std_score": np.std(scores)})

    elif args.mode == "train":
        model = CatBoostClassifier(**cfg["catboost"])
        y = df["clicked"].to_numpy()
        X = df.drop("clicked").to_pandas()
        cat_features = [i for i, c in enumerate(X.columns) if str(X[c].dtype) == "object"]
        pool = Pool(X, y, cat_features=cat_features)
        model.fit(pool)
        
        model_dir = cfg["paths"].get("model_dir", "./models")
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"catboost_full_{timestamp}.cbm")
        model.save_model(model_path)
        log.info(f"💾 Full model saved: {model_path}")

    elif args.mode == "inference":
        # Test 데이터도 캐싱 적용
        df_test = process_data_with_cache(cfg, data_type="test", force_reprocess=args.force_reprocess)
        
        # ID 컬럼 추출
        if "ID" not in df_test.columns:
            raise KeyError("❌ Test dataset must contain 'ID' column")
        test_ids = df_test["ID"].to_numpy()
        df_test = df_test.drop("ID")
        
        # 모델 로드
        if args.model_path is None:
            model_dir = cfg["paths"].get("model_dir", "./models")
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".cbm")]
            if not model_files:
                raise FileNotFoundError("❌ No trained model found")
            args.model_path = os.path.join(model_dir, sorted(model_files)[-1])
            log.info(f"📦 Auto-loaded: {args.model_path}")
        
        model = CatBoostClassifier()
        model.load_model(args.model_path)
        
        # 추론
        df_test = df_test.to_pandas()
        cat_features = [i for i, c in enumerate(df_test.columns) if str(df_test[c].dtype) == "object"]
        test_pool = Pool(df_test, cat_features=cat_features)
        preds = model.predict_proba(test_pool)[:, 1]
        
        # 제출 파일 저장
        submission_path = cfg["paths"]["submission_path"]
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        pd.DataFrame({"ID": test_ids, "clicked": preds}).to_csv(submission_path, index=False)
        log.info(f"✅ Submission saved: {submission_path}")

    elif args.mode == "all":
        scores = run_cv(df, cfg, wandb_enabled)
        
        # Full training
        model = CatBoostClassifier(**cfg["catboost"])
        y = df["clicked"].to_numpy()
        X = df.drop("clicked").to_pandas()
        cat_features = [i for i, c in enumerate(X.columns) if str(X[c].dtype) == "object"]
        pool = Pool(X, y, cat_features=cat_features)
        model.fit(pool)
        
        model_dir = cfg["paths"].get("model_dir", "./models")
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"catboost_full_{timestamp}.cbm")
        model.save_model(model_path)
        log.info(f"💾 Full model saved: {model_path}")
        
        # Inference
        df_test = process_data_with_cache(cfg, data_type="test", force_reprocess=args.force_reprocess)
        test_ids = df_test["ID"].to_numpy()
        df_test = df_test.drop("ID").to_pandas()
        
        cat_features = [i for i, c in enumerate(df_test.columns) if str(df_test[c].dtype) == "object"]
        test_pool = Pool(df_test, cat_features=cat_features)
        preds = model.predict_proba(test_pool)[:, 1]
        
        submission_path = cfg["paths"]["submission_path"]
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        pd.DataFrame({"ID": test_ids, "clicked": preds}).to_csv(submission_path, index=False)
        log.info(f"✅ Submission saved: {submission_path}")

    if wandb_enabled:
        wandb.finish()