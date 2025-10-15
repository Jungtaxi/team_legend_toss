"""
DIN Model Inference
- Load trained DIN models from models/ directory (5-fold ensemble)
- Perform inference on test data
- Generate submissions/din.csv
"""

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import logging

# ==========================================================
# Constants (must match training)
# ==========================================================
ID_COLUMN = "ID"
TARGET_COLUMN = "clicked"
SEQ_COLUMN = "seq"

CATEGORICAL_FEATURES = ["gender", "inventory_id", "hour", "age_group", "day_of_week",
                        "time_segment", "seq_length_bin_str", "age_hour", "l_feat_14"]

# DIN Hyperparameters
MAX_SEQ_LENGTH = 100
ITEM_EMBEDDING_DIM = 32
DENSE_EMBEDDING_DIM = 16
ATTENTION_HIDDEN_UNITS = [80, 40]
DNN_HIDDEN_UNITS = [256, 128, 64]

# Inference settings
BATCH_SIZE = 16384 * 2
USE_MIXED_PRECISION = True

# Directories
MODEL_DIR = "./models"
DATA_DIR = "./data"
SUBMISSION_DIR = "./submissions"

# ==========================================================
# Feature Engineering (identical to training)
# ==========================================================

def truncate_seq_in_polars(lf: pl.LazyFrame, max_items: int = MAX_SEQ_LENGTH) -> pl.LazyFrame:
    return lf.with_columns([
        pl.col(SEQ_COLUMN)
        .str.split(",")
        .list.head(max_items)
        .list.join(",")
        .alias(SEQ_COLUMN + "_truncated")
    ])


def create_seq_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
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
    ]).with_columns([
        pl.col("seq_length_bin").cast(pl.Utf8).alias("seq_length_bin_str")
    ])


def create_history_features_polars(lf: pl.LazyFrame) -> pl.LazyFrame:
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
    lf = truncate_seq_in_polars(lf, MAX_SEQ_LENGTH)
    lf = create_seq_features_polars(lf)
    lf = create_history_features_polars(lf)
    lf = create_time_features_polars(lf)
    lf = create_interaction_features_polars(lf)
    lf = create_feat_group_features_polars(lf)
    return lf


# ==========================================================
# Dataset and DataLoader
# ==========================================================

class CTRDatasetLazy(Dataset):
    def __init__(self, dense_features, sparse_features, seq_strings, seq_encoder):
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.seq_strings = seq_strings
        self.seq_encoder = seq_encoder

    def __len__(self):
        return len(self.dense_features)

    def __getitem__(self, idx):
        dense = torch.FloatTensor(self.dense_features[idx])
        sparse = {k: torch.LongTensor([v[idx]]) for k, v in self.sparse_features.items()}

        seq_str = self.seq_strings[idx]
        seq_encoded = self.seq_encoder.encode_single(seq_str)
        seq_tensor = torch.LongTensor(seq_encoded) if seq_encoded else torch.LongTensor([0])

        candidate_id = sparse["l_feat_14"]

        return {
            'dense': dense,
            'sparse': sparse,
            'seq': seq_tensor,
            "candidate_id": candidate_id,
        }


def collate_fn_dynamic_padding(batch):
    dense_list = [item['dense'] for item in batch]
    dense_batch = torch.stack(dense_list, dim=0)

    sparse_keys = batch[0]['sparse'].keys()
    sparse_batch = {}
    for key in sparse_keys:
        sparse_values = [item['sparse'][key] for item in batch]
        sparse_batch[key] = torch.cat(sparse_values, dim=0)

    seq_list = [item['seq'] for item in batch]
    seq_padded = pad_sequence(seq_list, batch_first=True, padding_value=0)

    if seq_padded.size(1) > MAX_SEQ_LENGTH:
        seq_padded = seq_padded[:, :MAX_SEQ_LENGTH]

    candidate_ids = torch.cat([item["candidate_id"] for item in batch], dim=0).long()

    return {
        'dense': dense_batch,
        'sparse': sparse_batch,
        'seq': seq_padded,
        "candidate_id": candidate_ids,
    }


# ==========================================================
# DIN Model
# ==========================================================

class LocalActivationUnit(nn.Module):
    def __init__(self, embedding_dim: int, hidden_units: List[int]):
        super().__init__()
        layers = []
        input_dim = embedding_dim * 4

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.PReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.attention = nn.Sequential(*layers)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, candidate, history, mask):
        batch_size, seq_len, emb_dim = history.shape
        candidate_expanded = candidate.unsqueeze(1).expand(-1, seq_len, -1)

        concat_features = torch.cat([
            candidate_expanded,
            history,
            candidate_expanded * history,
            candidate_expanded - history
        ], dim=-1)

        attention_scores = self.attention(concat_features).squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e4)
        attention_weights = F.softmax(attention_scores, dim=1)

        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * history, dim=1)
        return weighted_sum


class DIN(nn.Module):
    def __init__(self, item_vocab_size, sparse_feature_dims, dense_feature_dim,
                 item_embedding_dim=ITEM_EMBEDDING_DIM, dense_embedding_dim=DENSE_EMBEDDING_DIM,
                 attention_hidden_units=ATTENTION_HIDDEN_UNITS, dnn_hidden_units=DNN_HIDDEN_UNITS,
                 dropout=0.2):
        super().__init__()

        self.item_embedding = nn.Embedding(item_vocab_size, item_embedding_dim, padding_idx=0)
        self.sparse_embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, dense_embedding_dim)
            for name, dim in sparse_feature_dims.items()
        })
        self.attention = LocalActivationUnit(item_embedding_dim, attention_hidden_units)
        self.dense_bn = nn.BatchNorm1d(dense_feature_dim)
        self.position_embedding = nn.Embedding(MAX_SEQ_LENGTH, item_embedding_dim)

        dnn_input_dim = dense_feature_dim + len(sparse_feature_dims) * dense_embedding_dim + item_embedding_dim

        dnn_layers = []
        for hidden_dim in dnn_hidden_units:
            dnn_layers.append(nn.Linear(dnn_input_dim, hidden_dim))
            dnn_layers.append(nn.BatchNorm1d(hidden_dim))
            dnn_layers.append(nn.PReLU())
            dnn_layers.append(nn.Dropout(dropout))
            dnn_input_dim = hidden_dim

        dnn_layers.append(nn.Linear(dnn_input_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, dense, sparse, seq, candidate_id):
        seq_emb = self.item_embedding(seq)
        seq_mask = (seq != 0).float()

        seq_len = seq_emb.size(1)
        pos_idx = torch.arange(seq_len - 1, -1, -1, device=seq.device).unsqueeze(0).expand(seq_emb.size(0), -1)
        pos_emb = self.position_embedding(pos_idx)
        seq_emb = seq_emb + pos_emb

        candidate_id = candidate_id.clamp(max=self.item_embedding.num_embeddings - 1)
        candidate_emb = self.item_embedding(candidate_id)
        attention_output = self.attention(candidate_emb, seq_emb, seq_mask)

        sparse_embs = [self.sparse_embeddings[name](sparse[name]) for name in sorted(sparse.keys())]
        sparse_concat = torch.cat(sparse_embs, dim=1)

        dense_normed = self.dense_bn(dense)

        final_input = torch.cat([dense_normed, sparse_concat, attention_output], dim=1)
        logits = self.dnn(final_input)
        return logits.squeeze(1)


# ==========================================================
# Data Preparation
# ==========================================================

def prepare_data_for_din_lazy(lf: pl.LazyFrame, seq_encoder, streaming: bool = True) -> Tuple:
    lf = engineer_all_features_polars(lf)

    engine = "streaming" if streaming else None
    df = lf.collect(engine=engine).to_pandas()

    exclude_cols = {ID_COLUMN, TARGET_COLUMN, SEQ_COLUMN, SEQ_COLUMN + "_truncated"}

    sparse_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    dense_cols = [c for c in df.columns if c not in exclude_cols and c not in sparse_cols]

    X_dense = df[dense_cols].fillna(0).astype(np.float32).values

    X_sparse = {}
    sparse_dims = {}
    for col in sparse_cols:
        df[col] = df[col].astype(str).fillna("__MISSING__")
        le = LabelEncoder()
        X_sparse[col] = le.fit_transform(df[col])
        sparse_dims[col] = len(le.classes_)

    seq_col = SEQ_COLUMN + "_truncated" if SEQ_COLUMN + "_truncated" in df.columns else SEQ_COLUMN
    seq_strings = df[seq_col].astype(str).tolist()

    return X_dense, X_sparse, seq_strings, sparse_dims


# ==========================================================
# Main Inference
# ==========================================================

def main():
    log = logging.getLogger("din_inference")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    log.info("\n" + "=" * 60)
    log.info("DIN Model Inference (5-Fold Ensemble)")
    log.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"\nDevice: {device}")

    # Load metadata
    log.info("\n1. Loading metadata and encoder")
    metadata_path = Path(MODEL_DIR) / "din_metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    seq_encoder = metadata['seq_encoder']
    sparse_dims = metadata['sparse_dims']
    n_splits = metadata['n_splits']
    log.info(f"   Metadata loaded: {n_splits}-fold CV")
    log.info(f"   Training score: {metadata['overall_score']:.5f}")

    # Load test data
    log.info("\n2. Loading and preprocessing test data")
    test_lf = pl.scan_parquet(f"{DATA_DIR}/test.parquet")
    test_ids = test_lf.select([ID_COLUMN]).collect(engine="streaming").to_pandas()[ID_COLUMN]

    X_dense, X_sparse, seq_strings, _ = prepare_data_for_din_lazy(
        test_lf, seq_encoder, streaming=True
    )
    log.info(f"   Test shape: Dense={X_dense.shape}, Seq={len(seq_strings)} strings")

    # Create dataset and dataloader
    test_dataset = CTRDatasetLazy(X_dense, X_sparse, seq_strings, seq_encoder)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=collate_fn_dynamic_padding
    )

    # Load models and perform inference
    log.info(f"\n3. Loading {n_splits} fold models and performing inference")
    test_preds_all = []

    for fold in range(1, n_splits + 1):
        log.info(f"\n   Fold {fold}/{n_splits}")
        model_path = Path(MODEL_DIR) / f"din_fold{fold}.pt"

        if not model_path.exists():
            log.warning(f"   Model not found: {model_path}, skipping...")
            continue

        # Initialize model
        model = DIN(
            item_vocab_size=seq_encoder.vocab_size,
            sparse_feature_dims=sparse_dims,
            dense_feature_dim=X_dense.shape[1],
        ).to(device)

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        log.info(f"   Model loaded: {model_path.name}")

        # Inference
        test_preds_fold = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"   Fold {fold} inference"):
                dense = batch['dense'].to(device)
                sparse = {k: v.to(device) for k, v in batch['sparse'].items()}
                candidate_id = batch["candidate_id"].to(device)
                seq = batch['seq'].to(device)

                if USE_MIXED_PRECISION:
                    with autocast('cuda'):
                        logits = model(dense, sparse, seq, candidate_id)
                else:
                    logits = model(dense, sparse, seq, candidate_id)

                probs = torch.sigmoid(logits).cpu().numpy()
                test_preds_fold.extend(probs)

        test_preds_all.append(np.array(test_preds_fold))
        log.info(f"   Fold {fold} predictions: Mean={np.mean(test_preds_fold):.4f}, Std={np.std(test_preds_fold):.4f}")

        del model
        torch.cuda.empty_cache()

    # Ensemble (average)
    log.info("\n4. Ensemble predictions (averaging)")
    test_preds_mean = np.mean(test_preds_all, axis=0)
    log.info(f"   Final predictions: Mean={test_preds_mean.mean():.4f}, Std={test_preds_mean.std():.4f}")

    # Save submission
    log.info("\n5. Generating submission file")
    Path(SUBMISSION_DIR).mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({ID_COLUMN: test_ids, TARGET_COLUMN: test_preds_mean})
    submission_path = Path(SUBMISSION_DIR) / "din.csv"
    submission.to_csv(submission_path, index=False)

    log.info(f"   Submission file: {submission_path}")
    log.info(f"   Shape: {submission.shape}")

    log.info("\n" + "=" * 60)
    log.info("Inference completed!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
