"""
Wide & Deep CTR Model Inference
- Load trained model from models/widectr.pt
- Perform inference on test data
- Generate submissions/widectr.csv
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CFG = {
    'BATCH_SIZE': 1024,
    'MODEL_DIR': './models',
    'SUBMISSION_DIR': './submissions',
    'DATA_DIR': './data'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create directories
Path(CFG['SUBMISSION_DIR']).mkdir(parents=True, exist_ok=True)

# ========================================
# Feature Engineering
# ========================================

def add_features_for_deep_learning(df: pd.DataFrame) -> pd.DataFrame:
    """Core feature engineering for deep learning models"""
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
# Model Definition
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
# Dataset
# ========================================

class ClickDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, seq_col):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.num_X = self.df[self.num_cols].astype(float).fillna(0).values
        self.cat_X = self.df[self.cat_cols].astype(int).values
        self.seq_strings = self.df[self.seq_col].astype(str).values

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
        return num_x, cat_x, seq

def collate_fn_infer(batch):
    num_x, cat_x, seqs = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths

# ========================================
# Inference Function
# ========================================

def inference(model, test_df, num_cols, cat_cols, seq_col, batch_size, device):
    """Perform inference on test data"""
    test_dataset = ClickDataset(test_df, num_cols, cat_cols, seq_col)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_infer, pin_memory=True, num_workers=2
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for num_x, cat_x, seqs, lens in test_loader:
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            seqs = seqs.to(device)
            lens = lens.to(device)
            logits = model(num_x, cat_x, seqs, lens)
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu())

    return torch.cat(predictions).numpy()

# ========================================
# Main Execution
# ========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Wide & Deep CTR Model Inference")
    print("=" * 60)

    # 1. Load model and encoders
    print("\n1. Loading model and encoders")
    model_path = Path(CFG['MODEL_DIR']) / 'widectr.pt'
    encoder_path = Path(CFG['MODEL_DIR']) / 'widectr_encoders.pkl'

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    checkpoint = torch.load(model_path, map_location=device)
    with open(encoder_path, 'rb') as f:
        cat_encoders = pickle.load(f)

    num_cols = checkpoint['num_cols']
    cat_cols = checkpoint['cat_cols']
    cat_cardinalities = checkpoint['cat_cardinalities']
    model_config = checkpoint['model_config']

    print(f"   Model: {model_path}")
    print(f"   Encoders: {encoder_path}")

    # 2. Load and preprocess test data
    print("\n2. Loading and preprocessing test data")
    test = pd.read_parquet(f"{CFG['DATA_DIR']}/test.parquet", engine="pyarrow")
    print(f"   Test shape: {test.shape}")

    # Feature engineering
    test = add_features_for_deep_learning(test)
    print("   Feature engineering completed")

    # Categorical encoding
    for col in cat_cols:
        test[col] = cat_encoders[col].transform(test[col].astype(str).fillna("UNK"))
    print("   Categorical encoding completed")

    # 3. Initialize model and load weights
    print("\n3. Initializing model and loading weights")
    model = WideDeepCTR(
        num_features=model_config['num_features'],
        cat_cardinalities=cat_cardinalities,
        emb_dim=model_config['emb_dim'],
        lstm_hidden=model_config['lstm_hidden'],
        hidden_units=model_config['hidden_units'],
        dropout=model_config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print("   Model weights loaded successfully")

    # 4. Perform inference
    print("\n4. Starting inference")
    seq_col = "seq"
    test_preds = inference(model, test, num_cols, cat_cols, seq_col, CFG['BATCH_SIZE'], device)
    print(f"   Prediction Mean: {test_preds.mean():.4f} | Std: {test_preds.std():.4f}")

    # 5. Generate submission file
    print("\n5. Generating submission file")
    sample_submit = pd.read_csv(f"{CFG['DATA_DIR']}/sample_submission.csv")
    sample_submit['clicked'] = test_preds
    submission_path = Path(CFG['SUBMISSION_DIR']) / 'widectr.csv'
    sample_submit.to_csv(submission_path, index=False)

    print(f"   Submission file: {submission_path}")
    print(f"   Shape: {sample_submit.shape}")

    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)
