"""
DIN - Lazy Loading + Dynamic Padding (메모리 극한 최적화)

🚀 핵심 전략:
1. Dataset: seq를 문자열로만 저장 (40GB → 1GB)
2. __getitem__: 요청받을 때만 하나씩 파싱 (Lazy)
3. collate_fn: 배치별 동적 패딩 (전체 max_len 불필요)

예상 메모리:
- Before: 54.5 GB (전체 seq array 저장)
- After:  ~14 GB (문자열만 저장, 동적 파싱)
"""

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
import logging
import time
import datetime
import gc
from functools import lru_cache
from pathlib import Path
import os
import pickle
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm

# ==========================================================
# Constants
# ==========================================================
ID_COLUMN = "ID"
TARGET_COLUMN = "clicked"
SEQ_COLUMN = "seq"

CATEGORICAL_FEATURES = ["gender", "inventory_id", "hour", "age_group", "day_of_week",
                        "time_segment", "seq_length_bin_str", "age_hour", "l_feat_14"]
EXCLUDED_FEATURES = []
BINARY_FEATURES = [
    "l_feat_1", "l_feat_2", "l_feat_8", "l_feat_13",
    "l_feat_16", "l_feat_19", "l_feat_21", "l_feat_22", "l_feat_24"
]

# DIN Hyperparameters
MAX_SEQ_LENGTH = 100  # 최대 길이 (배치 내에서 동적으로 조정)
ITEM_EMBEDDING_DIM = 32
DENSE_EMBEDDING_DIM = 16
ATTENTION_HIDDEN_UNITS = [80, 40]
DNN_HIDDEN_UNITS = [256, 128, 64]

# Training
BATCH_SIZE = 16384
LEARNING_RATE = 0.003
USE_MIXED_PRECISION = True

# Directories
MODEL_DIR = "./models"
DATA_DIR = "./data"
SUBMISSION_DIR = "./submissions"

# ==========================================================
# Feature Engineering
# ==========================================================

def truncate_seq_in_polars(lf: pl.LazyFrame, max_items: int = MAX_SEQ_LENGTH) -> pl.LazyFrame:
    """Polars에서 seq 미리 자르기"""
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
# Seq Encoder (Dict 매핑만, 실제 변환은 Dataset에서)
# ==========================================================

class LazySeqEncoder:
    """Vocabulary만 관리, 실제 인코딩은 Dataset에서 lazy하게"""

    def __init__(self):
        self.item_to_idx = {}
        self.vocab_size = 0
        self.fitted = False

    def fit(self, all_seqs: List[str]):
        """Vocabulary 생성"""
        unique_items = set(["0"])  # Padding

        print("🔧 Building vocabulary...")
        for seq_str in tqdm(all_seqs, desc="Collecting unique items"):
            if pd.notna(seq_str) and seq_str != "":
                items = str(seq_str).split(",")
                unique_items.update(items)

        sorted_items = sorted(list(unique_items))
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted_items)}
        self.vocab_size = len(self.item_to_idx)
        self.fitted = True

        print(f"✅ Vocabulary size: {self.vocab_size}")

    @lru_cache(maxsize=200_000)
    def encode_single(self, seq_str: str) -> List[int]:
        """단일 seq 문자열을 인코딩 (Dataset의 __getitem__에서 사용)"""
        if not self.fitted:
            raise ValueError("Encoder not fitted!")

        if pd.isna(seq_str) or seq_str == "":
            return []

        items = str(seq_str).split(",")
        # Dict lookup (O(1), 빠름!)
        encoded = [self.item_to_idx.get(item, 0) for item in items]
        return encoded


# ==========================================================
# Dataset with Lazy Loading (핵심!)
# ==========================================================

class CTRDatasetLazy(Dataset):
    """
    Lazy Loading Dataset

    핵심 전략:
    1. __init__: seq를 문자열 리스트로만 저장 (메모리 효율)
    2. __getitem__: 요청받을 때만 해당 idx의 seq를 파싱
    3. collate_fn: 배치별 동적 패딩 (전체 max_len 불필요)
    """

    def __init__(self, dense_features, sparse_features, seq_strings, seq_encoder, labels=None):
        """
        Args:
            seq_strings: List[str] - seq 문자열 리스트 (메모리 효율적!)
            seq_encoder: LazySeqEncoder - item_to_idx dict 보유
        """
        self.dense_features = dense_features  # numpy array
        self.sparse_features = sparse_features  # dict of numpy arrays
        self.seq_strings = seq_strings  # List[str] - 문자열만 저장!
        self.seq_encoder = seq_encoder
        self.labels = labels

    def __len__(self):
        return len(self.dense_features)

    def __getitem__(self, idx):
        """
        핵심: 요청받을 때만 seq를 파싱!
        """
        # Dense & Sparse (이미 numpy array)
        dense = torch.FloatTensor(self.dense_features[idx])
        sparse = {k: torch.LongTensor([v[idx]]) for k, v in self.sparse_features.items()}

        # Seq: 이때 비로소 문자열 → 숫자 변환! (Lazy!)
        seq_str = self.seq_strings[idx]
        seq_encoded = self.seq_encoder.encode_single(seq_str)
        seq_tensor = torch.LongTensor(seq_encoded) if seq_encoded else torch.LongTensor([0])

        candidate_id = sparse["l_feat_14"]

        item = {
            'dense': dense,
            'sparse': sparse,
            'seq': seq_tensor,  # 길이가 각각 다름!
            "candidate_id": candidate_id,
        }

        if self.labels is not None:
            item['label'] = torch.FloatTensor([self.labels[idx]])

        return item


def collate_fn_dynamic_padding(batch):
    """
    Dynamic Padding Collate Function

    핵심: 배치 내에서만 max_len을 찾아서 패딩!
    - 전체 데이터의 max_len 불필요
    - 배치마다 텐서 크기가 다름 (효율적!)
    """
    # Dense
    dense_list = [item['dense'] for item in batch]
    dense_batch = torch.stack(dense_list, dim=0)

    # Sparse
    sparse_keys = batch[0]['sparse'].keys()
    sparse_batch = {}
    for key in sparse_keys:
        sparse_values = [item['sparse'][key] for item in batch]
        sparse_batch[key] = torch.cat(sparse_values, dim=0)

    # Seq: Dynamic Padding! (배치 내 max_len만 사용)
    seq_list = [item['seq'] for item in batch]
    # pad_sequence: 자동으로 배치 내 최대 길이로 패딩
    seq_padded = pad_sequence(seq_list, batch_first=True, padding_value=0)

    # 최대 길이 제한 (너무 길면 자르기)
    if seq_padded.size(1) > MAX_SEQ_LENGTH:
        seq_padded = seq_padded[:, :MAX_SEQ_LENGTH]

    candidate_ids = torch.cat([item["candidate_id"] for item in batch], dim=0).long()  # ✅ dtype 보장

    result = {
        'dense': dense_batch,
        'sparse': sparse_batch,
        'seq': seq_padded,
        "candidate_id": candidate_ids,
    }

    # Labels
    if 'label' in batch[0]:
        label_list = [item['label'] for item in batch]
        result['label'] = torch.cat(label_list, dim=0)

    return result


# ==========================================================
# DIN Model (동일)
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
        # FP16 safe: -1e4 instead of -1e9 (FP16 range: -65504 ~ 65504)
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
# Data Preparation (Lazy!)
# ==========================================================

def prepare_data_for_din_lazy(
    lf: pl.LazyFrame,
    seq_encoder: LazySeqEncoder,
    is_train: bool = True,
    streaming: bool = True
) -> Tuple:
    """
    Lazy 방식으로 데이터 준비

    핵심: seq를 문자열 리스트로만 반환!
    """
    lf = engineer_all_features_polars(lf)

    engine = "streaming" if streaming else None
    df = lf.collect(engine=engine).to_pandas()

    y = None
    if is_train and TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].values

    exclude_cols = {ID_COLUMN, TARGET_COLUMN, SEQ_COLUMN, SEQ_COLUMN + "_truncated"}

    sparse_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    dense_cols = [c for c in df.columns if c not in exclude_cols and c not in sparse_cols]

    # Dense features
    X_dense = df[dense_cols].fillna(0).astype(np.float32).values

    # Sparse features
    X_sparse = {}
    sparse_dims = {}
    for col in sparse_cols:
        df[col] = df[col].astype(str).fillna("__MISSING__")
        le = LabelEncoder()
        X_sparse[col] = le.fit_transform(df[col])
        sparse_dims[col] = len(le.classes_)

    # Seq: 문자열로만 저장! (메모리 효율!)
    seq_col = SEQ_COLUMN + "_truncated" if SEQ_COLUMN + "_truncated" in df.columns else SEQ_COLUMN
    seq_strings = df[seq_col].astype(str).tolist()  # List[str]

    print(f"✅ Seq stored as strings: {len(seq_strings)} samples")
    print(f"   Estimated memory: {len(str(seq_strings)) / 1024**2:.2f} MB (vs {len(seq_strings) * MAX_SEQ_LENGTH * 4 / 1024**2:.2f} MB if array)")

    return X_dense, X_sparse, seq_strings, y, sparse_dims


# ==========================================================
# Utilities
# ==========================================================

def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    mask0, mask1 = (y_true == 0), (y_true == 1)
    ll0 = -np.mean(np.log(1 - y_pred[mask0])) if mask0.sum() else 0
    ll1 = -np.mean(np.log(y_pred[mask1])) if mask1.sum() else 0
    return 0.5 * ll0 + 0.5 * ll1


def calculate_competition_score(y_true, y_pred):
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler=None):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        dense = batch['dense'].to(device)
        sparse = {k: v.to(device) for k, v in batch['sparse'].items()}
        seq = batch['seq'].to(device)
        candidate_id = batch["candidate_id"].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        if scaler:
            with autocast('cuda'):
                logits = model(dense, sparse, seq, candidate_id)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(dense, sparse, seq, candidate_id)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        dense = batch['dense'].to(device)
        sparse = {k: v.to(device) for k, v in batch['sparse'].items()}
        seq = batch['seq'].to(device)
        candidate_id = batch["candidate_id"].to(device)
        labels = batch['label'].to(device)

        if USE_MIXED_PRECISION:
            with autocast('cuda'):
                logits = model(dense, sparse, seq, candidate_id)
                loss = criterion(logits, labels)
        else:
            logits = model(dense, sparse, seq, candidate_id)
            loss = criterion(logits, labels)

        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


# ==========================================================
# Main
# ==========================================================

def main():
    log = logging.getLogger("din_lazy_loading")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    log.info("🚀 DIN - Lazy Loading + Dynamic Padding")
    log.info("💡 Seq stored as strings (메모리 효율)")
    log.info("💡 __getitem__에서 하나씩 파싱 (Lazy)")
    log.info("💡 collate_fn으로 배치별 동적 패딩")

    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"\n🖥️  Device: {device}")

    # Create directories
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(SUBMISSION_DIR).mkdir(parents=True, exist_ok=True)

    # Load data
    log.info("\n📂 Loading data...")
    train_lf = pl.scan_parquet(f"{DATA_DIR}/train.parquet")
    test_lf = pl.scan_parquet(f"{DATA_DIR}/test.parquet")

    # Seq Encoder (vocabulary만)
    log.info("🔧 Building seq vocabulary...")
    train_lf_truncated = truncate_seq_in_polars(train_lf, MAX_SEQ_LENGTH)
    test_lf_truncated = truncate_seq_in_polars(test_lf, MAX_SEQ_LENGTH)

    train_seqs = train_lf_truncated.select([SEQ_COLUMN + "_truncated"]).collect(engine="streaming")[SEQ_COLUMN + "_truncated"].to_list()
    test_seqs = test_lf_truncated.select([SEQ_COLUMN + "_truncated"]).collect(engine="streaming")[SEQ_COLUMN + "_truncated"].to_list()

    # ✅ NEW: 광고 ID(l_feat_14)도 vocab에 포함
    train_ads = train_lf.select("l_feat_14").collect()["l_feat_14"].to_list()
    test_ads = test_lf.select("l_feat_14").collect()["l_feat_14"].to_list()

    seq_encoder = LazySeqEncoder()
    seq_encoder.fit(train_seqs + test_seqs + list(map(str, train_ads + test_ads)))

    # Prepare data (Lazy!)
    log.info("🔧 Preparing data (lazy mode)...")
    X_dense, X_sparse, seq_strings, y, sparse_dims = prepare_data_for_din_lazy(
        train_lf, seq_encoder, is_train=True, streaming=True
    )

    log.info(f"✅ Data ready: Dense={X_dense.shape}, Seq={len(seq_strings)} strings")
    gc.collect()

    Xt_dense, Xt_sparse, seq_strings_test, _, _ = prepare_data_for_din_lazy(
        test_lf, seq_encoder, is_train=False, streaming=True
    )

    # CV
    n_splits = 5
    seed = 42
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(X_dense))
    test_preds = np.zeros((n_splits, len(Xt_dense)))

    pos_weight = (1 - y.mean()) / y.mean()
    num_epochs = 10

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dense, y), 1):
        log.info(f"\n{'='*80}")
        log.info(f"🔵 Fold {fold}/{n_splits}")
        log.info(f"{'='*80}")

        # Split
        X_dense_tr, X_dense_va = X_dense[tr_idx], X_dense[va_idx]
        X_sparse_tr = {k: v[tr_idx] for k, v in X_sparse.items()}
        X_sparse_va = {k: v[va_idx] for k, v in X_sparse.items()}
        seq_strings_tr = [seq_strings[i] for i in tr_idx]
        seq_strings_va = [seq_strings[i] for i in va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Datasets (Lazy!)
        train_dataset = CTRDatasetLazy(X_dense_tr, X_sparse_tr, seq_strings_tr, seq_encoder, y_tr)
        val_dataset = CTRDatasetLazy(X_dense_va, X_sparse_va, seq_strings_va, seq_encoder, y_va)
        test_dataset = CTRDatasetLazy(Xt_dense, Xt_sparse, seq_strings_test, seq_encoder)

        # Dataloaders (with dynamic padding!)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True,
            collate_fn=collate_fn_dynamic_padding  # Dynamic padding!
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
            num_workers=4, pin_memory=True,
            collate_fn=collate_fn_dynamic_padding
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
            num_workers=4, pin_memory=True,
            collate_fn=collate_fn_dynamic_padding
        )

        # Model
        model = DIN(
            item_vocab_size=seq_encoder.vocab_size,
            sparse_feature_dims=sparse_dims,
            dense_feature_dim=X_dense.shape[1],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        scaler = GradScaler('cuda') if USE_MIXED_PRECISION else None
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        best_score = 0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
            val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
            val_score, val_ap, val_wll = calculate_competition_score(val_labels, val_preds)

            scheduler.step(val_loss)

            log.info(f"Epoch {epoch}: Train={train_loss:.5f} | Score={val_score:.5f}")

            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(model.state_dict(), f"{MODEL_DIR}/din_fold{fold}.pt")
                log.info(f"✅ Best: {best_score:.5f}")
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    log.info("⚠️  Early stopping")
                    break

        # Load best & predict
        model.load_state_dict(torch.load(f"{MODEL_DIR}/din_fold{fold}.pt"))
        _, val_preds, _ = evaluate(model, val_loader, criterion, device)
        oof_preds[va_idx] = val_preds

        val_score, _, _ = calculate_competition_score(y[va_idx], oof_preds[va_idx])
        log.info(f"🔵 Fold {fold} Final: {val_score:.5f}")

        # Test
        model.eval()
        test_preds_fold = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test"):
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

        test_preds[fold - 1] = np.array(test_preds_fold)

        # Cleanup
        del model, optimizer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    # Overall
    overall_score, overall_ap, overall_wll = calculate_competition_score(y, oof_preds)
    log.info(f"\n{'='*80}")
    log.info(f"📊 Overall: Score={overall_score:.5f}")
    log.info(f"{'='*80}")

    # Save final ensemble model (average of all folds)
    test_ids = test_lf.select([ID_COLUMN]).collect(engine="streaming").to_pandas()[ID_COLUMN]
    test_preds_mean = test_preds.mean(axis=0)

    submission = pd.DataFrame({ID_COLUMN: test_ids, TARGET_COLUMN: test_preds_mean})
    submission_path = Path(SUBMISSION_DIR) / "din.csv"
    submission.to_csv(submission_path, index=False)
    log.info(f"\n💾 Submission saved: {submission_path}")

    # Save metadata for inference
    metadata = {
        'seq_encoder': seq_encoder,
        'sparse_dims': sparse_dims,
        'dense_cols': list(X_dense.shape),
        'n_splits': n_splits,
        'overall_score': overall_score
    }
    metadata_path = Path(MODEL_DIR) / "din_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    log.info(f"💾 Metadata saved: {metadata_path}")

    log.info(f"\n✅ Done in {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    main()
