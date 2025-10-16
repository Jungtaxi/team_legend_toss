# feature_engineering.py
import logging
import time
import numpy as np
import polars as pl

log = logging.getLogger("feature_engineering")


class PolarsFeatureEngineer:
    """Feature Engineering for CTR prediction using Polars"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.feat_cfg = cfg["features"]
        self.fe_cfg = cfg.get("feature_engineering", {})
        self.enabled = self.fe_cfg.get("enabled", False)
        log.info(f"🔧 Feature Engineering: {'ENABLED' if self.enabled else 'DISABLED'}")

    def _get_feature_columns(self, prefix):
        if prefix not in self.feat_cfg:
            return []
        return [f"{prefix}_{i}" for i in self.feat_cfg[prefix]]

    # ----------------------------------------------------------------------
    # Temporal Features
    # ----------------------------------------------------------------------
    def apply_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.enabled or not self.fe_cfg.get("temporal", {}).get("enabled", False):
            return df

        start = time.time()
        log.info("⏰ Creating temporal features...")
        temporal_cfg = self.fe_cfg["temporal"]

        # 1️⃣ Hour binning
        if "hour_bins" in temporal_cfg:
            bins = np.array(temporal_cfg["hour_bins"])
            log.info(f"   - Hour bins: {bins.tolist()}")
            df = df.with_columns([
                pl.col("hour")
                .cast(pl.Float32)
                .map_elements(lambda x: np.digitize(x, bins[1:-1]), return_dtype=pl.Int32)
                .alias("hour_period")
            ])

        # 2️⃣ Weekend flag
        if temporal_cfg.get("weekend_flag", False):
            df = df.with_columns([
                (pl.col("day_of_week") >= 5).cast(pl.Int8).alias("is_weekend")
            ])
            log.info("   - Weekend flag added")

        # 3️⃣ Cyclical encoding (hour)
        if temporal_cfg.get("time_of_day_features", False):
            df = df.with_columns([
                (pl.col("hour").cast(pl.Float32, strict=False) * 2 * np.pi / 24).sin().alias("hour_sin"),
                (pl.col("hour").cast(pl.Float32, strict=False) * 2 * np.pi / 24).cos().alias("hour_cos"),
            ])
            log.info("   - Cyclical encoding (sin/cos) added")

        log.info(f"✅ Temporal features done in {time.time() - start:.2f} sec")
        return df

    # ----------------------------------------------------------------------
    # Cross Features
    # ----------------------------------------------------------------------
    def apply_cross_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.enabled or not self.fe_cfg.get("cross_features", {}).get("enabled", False):
            return df

        start = time.time()
        cross_cfg = self.fe_cfg["cross_features"]
        pairs = cross_cfg.get("pairs", [])
        num_buckets = cross_cfg.get("num_buckets", 100_000)

        log.info("🔀 Creating cross features...")
        for col1, col2 in pairs:
            if col1 not in df.columns or col2 not in df.columns:
                log.warning(f"⚠️ Skipping cross feature ({col1}, {col2}) — column not found.")
                continue

            new_name = f"cross_{col1}_{col2}"
            df = df.with_columns([
                (pl.col(col1).cast(pl.Utf8) + "_" + pl.col(col2).cast(pl.Utf8))
                .hash(seed=0)
                .mod(num_buckets)
                .cast(pl.Int32)
                .alias(new_name)
            ])
            log.info(f"   - {new_name} (buckets={num_buckets})")

        log.info(f"✅ Cross features done in {time.time() - start:.2f} sec")
        return df

    # ----------------------------------------------------------------------
    # Aggregation Features
    # ----------------------------------------------------------------------
    def apply_aggregation_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.enabled or not self.fe_cfg.get("aggregations", {}).get("enabled", False):
            return df

        start = time.time()
        log.info("📊 Creating aggregation features...")
        agg_cfg = self.fe_cfg["aggregations"]
        groups = agg_cfg.get("groups", {})

        for group_name, group_cfg in groups.items():
            stats = group_cfg.get("stats", [])
            cols = self._get_feature_columns(group_name)

            if not cols:
                continue

            log.info(f"   - {group_name}: {len(cols)} columns, stats={stats}")

            exprs = []
            if "mean" in stats:
                exprs.append(
                    pl.concat_list([pl.col(c).cast(pl.Float32) for c in cols])
                    .list.mean().alias(f"{group_name}_mean")
                )
            if "std" in stats:
                exprs.append(
                    pl.concat_list([pl.col(c).cast(pl.Float32) for c in cols])
                    .list.std().alias(f"{group_name}_std")
                )
            if "max" in stats:
                exprs.append(
                    pl.concat_list([pl.col(c).cast(pl.Float32) for c in cols])
                    .list.max().alias(f"{group_name}_max")
                )
            if "min" in stats:
                exprs.append(
                    pl.concat_list([pl.col(c).cast(pl.Float32) for c in cols])
                    .list.min().alias(f"{group_name}_min")
                )
            if "sum" in stats:
                exprs.append(
                    pl.concat_list([pl.col(c).cast(pl.Float32) for c in cols])
                    .list.sum().alias(f"{group_name}_sum")
                )

            df = df.with_columns(exprs)

        log.info(f"✅ Aggregation features done in {time.time() - start:.2f} sec")
        return df

    # ----------------------------------------------------------------------
    # 전체 파이프라인
    # ----------------------------------------------------------------------
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        log.info("🚀 Starting Polars Feature Engineering (with timing)")

        df = df.with_columns([
            pl.col("hour").cast(pl.Float32, strict=False),
            pl.col("day_of_week").cast(pl.Int32, strict=False)
        ])

        total_start = time.time()
        df = self.apply_temporal_features(df)
        df = self.apply_cross_features(df)
        df = self.apply_aggregation_features(df)

        # Null 값 처리 - CatBoost는 null을 허용하지 않으므로 fill
        log.info("🧹 Handling null values...")
        null_counts = df.null_count()

        # 숫자형 컬럼은 0으로 채우기
        numeric_cols = []
        for col, dtype in zip(df.columns, df.dtypes):
            if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                if null_counts[col][0] > 0:
                    numeric_cols.append(col)

        if numeric_cols:
            log.info(f"   - Filling {len(numeric_cols)} numeric columns with nulls")
            df = df.with_columns([
                pl.col(c).fill_null(0) for c in numeric_cols
            ])

        # 문자열/카테고리 컬럼은 빈 문자열로 채우기
        string_cols = []
        for col, dtype in zip(df.columns, df.dtypes):
            if dtype == pl.Utf8:
                if null_counts[col][0] > 0:
                    string_cols.append(col)

        if string_cols:
            log.info(f"   - Filling {len(string_cols)} string columns with nulls")
            df = df.with_columns([
                pl.col(c).fill_null("") for c in string_cols
            ])

        log.info(f"🏁 All feature engineering done in {time.time() - total_start:.2f} sec")
        return df