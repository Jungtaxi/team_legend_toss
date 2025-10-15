"""
Wide & Deep CTR Model Training
- 6 epochs 학습 후 최종 모델만 저장
- 재현성을 위해 고정된 설정 사용
"""

import pandas as pd
import numpy as np
import os, random
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CFG = {
    'BATCH_SIZE': 1024,
    'EPOCHS': 6,
    'LEARNING_RATE': 1e-3,
    'SEED': 42,
    'MODEL_DIR': './models',
    'DATA_DIR': './data'
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 폴더 생성
Path(CFG['MODEL_DIR']).mkdir(parents=True, exist_ok=True)
print(f"Model 폴더: {CFG['MODEL_DIR']}")

# ========================================
# 피처 엔지니어링
# ========================================

def add_features_for_deep_learning(df: pd.DataFrame) -> pd.DataFrame:
    """딥러닝 모델용 핵심 피처 엔지니어링"""
    df = df.copy()

    if 'history_a_1' in df.columns:
        df['history_a_1_log'] = np.log1p(df['history_a_1'].fillna(0))
        df['history_a_1_high'] = (df['history_a_1'] > 0.2).astype(int)
        df['history_a_1_very_high'] = (df['history_a_1'] > 1.0).astype(int)

    if 'history_a_4' in df.columns:
        df['history_a_4_near_zero'] = (df['history_a_4'] > -200).astype(int)

    if 'hour' in df.columns:
        df['is_dawn'] = df['hour'].isin([2, 3, 4]).astype(int)
        df['is_morning_rush'] = df['hour'].isin([7, 8, 9, 10]).astype(int)

    if 'day_of_week' in df.columns:
        df['is_tuesday'] = (df['day_of_week'] == 2).astype(int)

    if 'seq' in df.columns:
        df['seq_length'] = df['seq'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 0
        )
        df['seq_length_log'] = np.log1p(df['seq_length'])
        df['seq_very_short'] = (df['seq_length'] <= 100).astype(int)

        df['seq_unique_count'] = df['seq'].apply(
            lambda x: len(set(str(x).split(','))) if pd.notna(x) else 0
        )
        df['seq_diversity'] = df['seq_unique_count'] / (df['seq_length'] + 1)

    return df

# ========================================
# 데이터 로드
# ========================================

print("\n데이터 로드 시작")
train = pd.read_parquet(f"{CFG['DATA_DIR']}/train.parquet", engine="pyarrow")
test = pd.read_parquet(f"{CFG['DATA_DIR']}/test.parquet", engine="pyarrow")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

train = add_features_for_deep_learning(train)
test = add_features_for_deep_learning(test)
print("피처 엔지니어링 완료")

target_col = "clicked"
seq_col = "seq"
FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]
num_cols = [c for c in feature_cols if c not in cat_cols]
print(f"\nNum features: {len(num_cols)} | Cat features: {len(cat_cols)}")

# 범주형 인코딩
cat_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([train[col], test[col]], axis=0).astype(str).fillna("UNK")
    le.fit(all_values)
    train[col] = le.transform(train[col].astype(str).fillna("UNK"))
    test[col] = le.transform(test[col].astype(str).fillna("UNK"))
    cat_encoders[col] = le

# 인코더 저장 (inference에서 사용)
encoder_path = Path(CFG['MODEL_DIR']) / 'widectr_encoders.pkl'
with open(encoder_path, 'wb') as f:
    pickle.dump(cat_encoders, f)
print(f"인코더 저장: {encoder_path}")

# ========================================
# Dataset
# ========================================

class ClickDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, seq_col, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target
        self.num_X = self.df[self.num_cols].astype(float).fillna(0).values
        self.cat_X = self.df[self.cat_cols].astype(int).values
        self.seq_strings = self.df[self.seq_col].astype(str).values
        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        num_x = torch.tensor(self.num_X[idx], dtype=torch.float)
        cat_x = torch.tensor(self.cat_X[idx], dtype=torch.long)
        s = self.seq_strings[idx]
        if s and s != 'nan':
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        else:
            arr = np.array([0.0], dtype=np.float32)
        seq = torch.from_numpy(arr)
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            return num_x, cat_x, seq, y
        else:
            return num_x, cat_x, seq

def collate_fn_train(batch):
    num_x, cat_x, seqs, ys = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths, ys

def collate_fn_infer(batch):
    num_x, cat_x, seqs = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths

# ========================================
# 모델
# ========================================

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])

    def forward(self, x0):
        x = x0
        for w in self.layers:
            x = x0 * w(x) + x
        return x

class WideDeepCTR(nn.Module):
    def __init__(self, num_features, cat_cardinalities, emb_dim=16, lstm_hidden=64,
                 hidden_units=[512, 256, 128], dropout=[0.1, 0.2, 0.3]):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim) for cardinality in cat_cardinalities
        ])
        cat_input_dim = emb_dim * len(cat_cardinalities)
        self.bn_num = nn.BatchNorm1d(num_features)
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=lstm_hidden,
            num_layers=2, batch_first=True, bidirectional=True
        )
        seq_out_dim = lstm_hidden * 2
        input_dim = num_features + cat_input_dim + seq_out_dim
        self.cross = CrossNetwork(input_dim, num_layers=2)
        layers = []
        for i, h in enumerate(hidden_units):
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout[i % len(dropout)])]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, num_x, cat_x, seqs, seq_lengths):
        num_x = self.bn_num(num_x)
        cat_embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.emb_layers)]
        cat_feat = torch.cat(cat_embs, dim=1)
        seqs = seqs.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            seqs, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        z = torch.cat([num_x, cat_feat, h], dim=1)
        z_cross = self.cross(z)
        out = self.mlp(z_cross)
        return out.squeeze(1)

# ========================================
# 학습 함수
# ========================================

def train_model(train_df, num_cols, cat_cols, seq_col, target_col,
                batch_size, epochs, lr, device):
    """최종 모델만 저장하는 학습 함수"""
    train_dataset = ClickDataset(train_df, num_cols, cat_cols, seq_col, target_col, True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_train, pin_memory=True, num_workers=2
    )

    # 모델 초기화
    cat_cardinalities = [len(cat_encoders[c].classes_) for c in cat_cols]
    model = WideDeepCTR(
        num_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        emb_dim=16, lstm_hidden=64,
        hidden_units=[512, 256, 128],
        dropout=[0.1, 0.2, 0.3]
    ).to(device)

    # 손실 함수
    pos_weight_value = (len(train_df) - train_df[target_col].sum()) / train_df[target_col].sum()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

    print("\n" + "=" * 60)
    print("학습 시작")
    print(f"Total samples: {len(train_dataset):,}")
    print(f"Positive ratio: {train_df[target_col].mean():.4f}")
    print(f"Pos weight: {pos_weight_value:.2f}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # ===== 학습 =====
        model.train()
        total_loss = 0

        for num_x, cat_x, seqs, lens, ys in tqdm(train_loader, desc=f"[Train Epoch {epoch}/{epochs}]"):
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            seqs = seqs.to(device)
            lens = lens.to(device)
            ys = ys.to(device)

            optimizer.zero_grad()
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * ys.size(0)

        avg_loss = total_loss / len(train_dataset)
        print(f"\n[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print("-" * 60)

    # ===== 최종 모델 저장 =====
    model_path = Path(CFG['MODEL_DIR']) / 'widectr.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'cat_cardinalities': [len(cat_encoders[c].classes_) for c in cat_cols],
        'model_config': {
            'num_features': len(num_cols),
            'emb_dim': 16,
            'lstm_hidden': 64,
            'hidden_units': [512, 256, 128],
            'dropout': [0.1, 0.2, 0.3]
        }
    }, model_path)

    print("=" * 60)
    print("✅ 학습 완료!")
    print(f"   모델 저장: {model_path}")
    print("=" * 60)

    return model

# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    model = train_model(
        train_df=train,
        num_cols=num_cols,
        cat_cols=cat_cols,
        seq_col=seq_col,
        target_col=target_col,
        batch_size=CFG['BATCH_SIZE'],
        epochs=CFG['EPOCHS'],
        lr=CFG['LEARNING_RATE'],
        device=device
    )
