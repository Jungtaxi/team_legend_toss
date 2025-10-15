# CTR 예측 대회 - 기술 보고서

## Team: Legend Toss

---

## 1. Validation (검증 전략)

### 1.1 두 가지 접근 방식

#### Model 1: DIN (5-Fold Cross Validation)
- **전략**: 5-Fold Stratified Cross Validation
- **근거**:
  - **모델 안정성 확보**: 5개의 독립적인 모델을 학습하여 단일 모델의 과적합 위험 감소
  - **성능 검증**: Out-of-Fold (OOF) 예측을 통해 학습 데이터 전체에 대한 일반화 성능 측정 가능
  - **앙상블 효과**: 테스트 시 5개 모델의 평균을 사용하여 예측 분산 감소 및 robust한 예측
  - **조기 종료**: 각 fold에서 validation loss 기반 early stopping (patience=3)으로 최적 epoch 자동 선택
- **구현**:
  ```python
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dense, y), 1):
      # 각 fold별 학습 및 검증
      # Best validation score 기준으로 모델 저장
  ```
- **결과**: 모든 train 데이터에 대해 OOF 예측 생성 → 과적합 없는 성능 추정

#### Model 2: Wide & Deep CTR (Full Training)
- **전략**: Validation 없이 전체 데이터로 6 epoch 학습
- **근거**:
  - **시간 제약**: 1 epoch당 약 **40분** 소요
    - Validation split 사용 시: 80% train으로 학습 → 더 많은 epoch 필요 → 시간 초과
    - Full training: 100% 데이터로 6 epoch → 실행 가능 시간 내
  - **데이터 최대 활용**: CTR 예측은 데이터가 많을수록 유리 → 전체 train data 활용이 성능 향상에 기여
  - **재현성 확보**: 고정된 seed(42) + 고정된 epoch(6)로 결과 재현 가능
  - **보완책**: DIN 모델의 CV 결과로 전체적인 성능 트렌드 파악 가능
- **구현**:
  ```python
  CFG = {
      'EPOCHS': 6,
      'SEED': 42,
      # 모든 하이퍼파라미터 고정
  }
  seed_everything(CFG['SEED'])  # 재현성 보장
  ```

### 1.2 Validation vs Full Training 선택 근거

| 측면 | 5-Fold CV (DIN) | Full Training (Wide&Deep) |
|------|-----------------|---------------------------|
| **장점** | 성능 검증, 앙상블 효과 | 더 많은 데이터, 시간 절약 |
| **단점** | 학습 시간 5배 증가 | 과적합 위험 |
| **적용 모델** | DIN (복잡한 attention) | Wide&Deep (상대적으로 단순) |
| **최종 선택** | 안정성 우선 | 효율성 우선 |

**결론**: 두 전략을 병행하여 안정성(DIN)과 효율성(Wide&Deep)을 동시에 확보

---

## 2. Feature Engineering (피처 엔지니어링)

### 2.1 l_feat_14 (광고 ID) 활용

#### 가설
- `l_feat_14`는 광고의 고유 ID로, **사용자의 과거 이력(seq)와 현재 광고 간의 유사도**가 클릭 확률에 큰 영향을 미칠 것

#### 구현
1. **DIN 모델**: Local Activation Unit (Attention Mechanism)
   ```python
   # seq의 각 아이템과 candidate(l_feat_14) 간의 attention 계산
   concat_features = [candidate, history, candidate*history, candidate-history]
   attention_scores = attention_network(concat_features)
   weighted_history = sum(attention_weights * history)
   ```
   - **원리**: 현재 광고와 유사한 과거 이력에 높은 가중치 부여
   - **효과**: 사용자별 맞춤 광고 선호도 모델링

2. **Wide & Deep 모델**: Embedding + LSTM
   ```python
   # seq를 LSTM으로 인코딩하여 시간적 패턴 파악
   seq_emb = lstm(seq)
   # l_feat_14를 categorical embedding으로 변환
   ad_emb = embedding(l_feat_14)
   ```

#### 결론
- DIN의 attention 메커니즘이 광고-이력 매칭에 효과적
- l_feat_14를 vocab에 포함시켜 seq와 동일한 embedding space에서 학습

### 2.2 Temporal Features (시간적 특징)

#### 가설
- 클릭 행동은 **시간대별로 다른 패턴**을 보일 것 (예: 출퇴근 시간, 심야 시간)

#### 구현
```python
# Hour 기반 세분화
time_segment = {
    'dawn': [2,3,4],           # 새벽 (수면 시간)
    'morning': [7,8,9,10,11],  # 오전 (출근/업무)
    'afternoon': [12-17],       # 오후 (업무/활동)
    'evening': [18,19,20],      # 저녁 (퇴근/여가)
    'night': [21,22,23],        # 밤 (여가/휴식)
    'midnight': [0,1],          # 자정
    'early': [5,6]              # 이른 아침
}

# 특정 시간대 binary flag
is_dawn = hour in [2,3,4]
is_morning_rush = hour in [7,8,9,10]

# 요일 패턴
is_tuesday = (day_of_week == 2)
is_weekend = day_of_week in [6,7]
```

#### 근거
- **시간대별 사용자 상태**: 새벽/심야는 낮은 클릭률, 출퇴근 시간은 모바일 사용 증가
- **요일 효과**: 주중/주말의 행동 패턴 차이
- **화요일 효과**: EDA 결과 화요일에 특이 패턴 발견 → 별도 feature 추가

#### 결론
- 시간 세그먼트를 categorical feature로 사용하여 non-linear 패턴 학습
- Binary flag는 특정 시간대의 중요성을 명시적으로 모델에 전달

### 2.3 범주형 변수 (Categorical Features)

#### 선택된 Categorical Features
```python
cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]
# DIN 추가: ["hour", "day_of_week", "time_segment", "seq_length_bin_str", "age_hour"]
```

#### 전략
1. **Embedding 사용**
   - One-hot encoding 대신 embedding으로 변환
   - **장점**: 고차원 희소 벡터 → 저차원 밀집 벡터 (메모리 효율 + 표현력 향상)
   - **크기**: 각 categorical feature마다 16차원 embedding (DIN: dense_embedding_dim=16)

2. **Label Encoding**
   ```python
   le = LabelEncoder()
   all_values = pd.concat([train[col], test[col]])
   le.fit(all_values)  # train+test 통합 vocabulary
   ```
   - **중요**: train과 test를 함께 fit하여 unseen category 방지

3. **결측치 처리**
   ```python
   df[col] = df[col].astype(str).fillna("__MISSING__")
   ```
   - 결측값을 별도 카테고리로 처리하여 정보 손실 방지

#### 결론
- Embedding 방식이 one-hot보다 파라미터 효율적이며 feature 간 상호작용 학습 가능

### 2.4 Sequence Features (seq - 사용자 이력)

#### 가설
- **seq 길이**와 **다양성**이 사용자의 활동 패턴과 클릭 확률에 영향

#### 구현
```python
# 1. Sequence 길이
seq_length = len(seq.split(','))
seq_length_log = log1p(seq_length)          # 로그 변환 (skewness 완화)
seq_very_short = (seq_length <= 100)        # 매우 짧은 이력

# 2. Sequence 다양성
seq_unique_count = len(set(seq.split(',')))
seq_diversity = seq_unique_count / (seq_length + 1)  # 고유 아이템 비율

# 3. Sequence 길이 구간화
seq_length_bin = bin(seq_length, bins=[0,100,200,500,1000,inf])
```

#### 근거
- **짧은 seq**: 신규 사용자 또는 낮은 활동성 → 낮은 engagement
- **다양성**: 고유 아이템 수가 많을수록 탐색적 사용자 → 클릭 확률 변화
- **로그 변환**: seq_length의 long-tail 분포 정규화

#### DIN 모델 특화: Dynamic Padding
```python
# Lazy Loading: seq를 문자열로 저장 (메모리 40GB → 1GB)
seq_strings = df['seq'].astype(str).tolist()

# Dynamic Padding: 배치별 최대 길이로 패딩
seq_padded = pad_sequence(seq_list, batch_first=True)
if seq_padded.size(1) > MAX_SEQ_LENGTH:
    seq_padded = seq_padded[:, :MAX_SEQ_LENGTH]  # 100으로 truncate
```

#### 결론
- Sequence 통계량이 사용자 프로필 이해에 유용
- DIN의 lazy loading + dynamic padding으로 메모리 효율성 확보

### 2.5 History Features (history_a_* - 사용자 행동 통계)

#### 가설
- 사용자의 과거 행동 통계(history_a_1~7)가 미래 클릭 행동 예측에 유용

#### 구현
```python
# history_a_1: 가장 중요한 지표 (추정: 클릭률 관련)
history_a_1_log = log1p(history_a_1)
history_a_1_high = (history_a_1 > 0.2)       # 임계값 초과 여부
history_a_1_very_high = (history_a_1 > 1.0)  # 매우 높은 값

# history_a_4: 특정 패턴 (추정: 부정적 지표)
history_a_4_near_zero = (history_a_4 > -200)
history_a_4_clipped = clip(history_a_4, lower=-5000)  # outlier 처리

# 집계 통계
history_a_mean = mean(history_a_1~7)
history_a_std = std(history_a_1~7)
history_a_max = max(history_a_1~7)
```

#### 근거
- **history_a_1 중요성**: EDA 결과 target과 높은 상관관계 확인
- **비선형 변환**: 로그 변환 + 구간화로 non-linear 관계 모델링
- **Outlier 처리**: clipping으로 극단값의 영향 제한
- **다차원 압축**: 7개 feature → 3개 통계량 (mean, std, max)

#### 결론
- history_a_1이 핵심 예측 변수
- 통계량 압축으로 모델의 복잡도 감소 및 일반화 능력 향상

### 2.6 Feature Interaction (피처 상호작용)

#### 구현
```python
# 1. History × Sequence Length
h1_x_seqlen = history_a_1 * log1p(seq_length)
h1_x_seqlen_log = log1p(h1_x_seqlen + 1)

# 2. Age × Hour
age_hour = str(age_group) + "_" + str(hour)  # "30대_09시"
```

#### 근거
- **가설**: 사용자의 활동성(history)과 이력 길이(seq_length)의 조합이 engagement 지표
- **Age-Hour 상호작용**: 연령대별로 활동 시간대가 다를 것 (예: 10대는 저녁, 30-40대는 오전)

### 2.7 Feature Group Aggregation (feat_a, feat_b, ...)

#### 구현
```python
for prefix in ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']:
    cols = [col for col in all_cols if col.startswith(f'{prefix}_')]
    if cols:
        # 통계량 생성
        f'{prefix}_mean' = mean(cols)
        f'{prefix}_max' = max(cols)
        f'{prefix}_min' = min(cols)
        f'{prefix}_std' = std(cols)

# feat_a 특화: sparsity 지표
feat_a_nonzero_count = sum(feat_a_* != 0)
feat_a_is_sparse = (feat_a_nonzero_count == 0)
```

#### 근거
- **차원 축소**: 동일 그룹 내 여러 feature를 통계량으로 압축
- **Sparsity 정보**: 0 값의 패턴이 의미 있는 정보일 수 있음

---

## 3. Model Training (모델 학습)

### 3.1 Model 1: DIN (Deep Interest Network)

#### 가설
"사용자의 클릭 확률은 **현재 광고와 유사한 과거 이력**에 크게 영향을 받는다"

#### Architecture (모델 구조)
```
Input: [Dense Features, Sparse Features, Sequence, Candidate ID]
  ↓
Sparse → Embedding (16d)
Sequence → Item Embedding (32d) + Position Embedding
Candidate → Item Embedding (32d)
  ↓
Attention Module (핵심):
  - Query: Candidate Embedding (현재 광고)
  - Key/Value: Sequence Embeddings (과거 이력)
  - Output: Weighted sum of history (유사한 이력에 높은 가중치)
  ↓
Concatenate: [Dense_BN, Sparse_Emb, Attention_Output]
  ↓
DNN: [256, 128, 64] → BatchNorm → PReLU → Dropout(0.2)
  ↓
Output: Binary Classification (BCEWithLogitsLoss)
```

#### Attention Mechanism 상세
```python
# Local Activation Unit
concat = [candidate, history, candidate*history, candidate-history]  # 4차원 feature
scores = MLP([80,40] → 1)(concat)  # attention scores
weights = softmax(scores.masked_fill(mask==0, -1e4))
output = sum(weights * history)  # weighted sum
```

#### Hyperparameters (하이퍼파라미터)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Item Embedding | 32 | seq vocab 크기 고려 |
| Dense Embedding | 16 | 범주형 변수 차원 |
| Attention Hidden | [80, 40] | 점진적 차원 축소 |
| DNN Hidden | [256, 128, 64] | 깊은 네트워크로 복잡한 패턴 학습 |
| Dropout | 0.2 | 과적합 방지 |
| Batch Size | 16384 | 큰 배치로 안정적 학습 |
| Learning Rate | 0.003 | Adam optimizer 기본값 조정 |
| MAX_SEQ_LENGTH | 100 | 메모리 효율 + 최근 이력 중심 |

#### Training Strategy (학습 전략)
- **Optimizer**: Adam (lr=0.003, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)
- **Loss**: BCEWithLogitsLoss with pos_weight (class imbalance 보정)
  ```python
  pos_weight = (1 - y.mean()) / y.mean()  # 약 19.0
  ```
- **Mixed Precision**: FP16 (학습 속도 향상 + 메모리 절약)
- **Early Stopping**: validation score 기준, patience=3

#### 메모리 최적화
```python
# Before: 54.5 GB (전체 seq를 numpy array로 저장)
# After: ~14 GB (문자열로 저장 + lazy loading)

# Lazy Loading
seq_strings = df['seq'].astype(str).tolist()  # List[str]

# Dynamic Padding (배치별)
seq_padded = pad_sequence(seq_list, batch_first=True)
```

#### 결론
- **장점**: Attention으로 광고-이력 관계 명시적 모델링, 5-Fold 앙상블로 안정성 확보
- **단점**: 학습 시간 길고 메모리 요구량 높음 (하지만 최적화로 해결)
- **최종 성능**: OOF Score 기반 검증 완료

### 3.2 Model 2: Wide & Deep CTR

#### 가설
"CTR 예측은 **저차 feature 조합(Wide)**과 **고차 feature 조합(Deep)**을 동시에 학습해야 한다"

#### Architecture (모델 구조)
```
Input: [Numerical Features, Categorical Features, Sequence]
  ↓
Categorical → Embedding (16d)
Numerical → BatchNorm
Sequence → LSTM(hidden=64, layers=2, bidirectional) → [128d]
  ↓
Concatenate: [Numerical_BN, Cat_Emb, LSTM_output]
  ↓
Cross Network (2 layers) - Wide Part:
  x_{i+1} = x_0 · w_i(x_i) + x_i  # explicit feature crossing
  ↓
Deep Network: [512, 256, 128] → ReLU → Dropout([0.1, 0.2, 0.3])
  ↓
Output: Binary Classification
```

#### Cross Network 역할
```python
# Explicit Feature Crossing (Wide part)
for layer in cross_layers:
    x = x0 * layer(x) + x  # x0와 x의 element-wise product
```
- **효과**: 저차 feature 조합을 명시적으로 학습 (예: gender × age_group)
- **장점**: Deep network가 놓칠 수 있는 간단한 규칙 포착

#### LSTM for Sequence
```python
lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, bidirectional=True)
# Output: hidden state [128d] (64*2)
```
- **역할**: seq의 시간적 패턴 포착 (최근 이력에 더 높은 가중치)
- **Bidirectional**: 과거→현재, 현재→과거 양방향 정보 활용

#### Hyperparameters (하이퍼파라미터)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding Dim | 16 | 범주형 변수 표현 |
| LSTM Hidden | 64 (× 2방향 = 128) | Sequence 압축 표현 |
| Cross Layers | 2 | 2차, 3차 조합까지 학습 |
| DNN Hidden | [512, 256, 128] | 점진적 차원 축소 |
| Dropout | [0.1, 0.2, 0.3] | 깊은 층일수록 강한 regularization |
| Batch Size | 1024 | DIN보다 작은 배치 (메모리 효율) |
| Learning Rate | 1e-3 | AdamW optimizer |
| Epochs | 6 (고정) | 시간 제약 + 재현성 |

#### Training Strategy (학습 전략)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=2, T_mult=2)
  - 주기적으로 learning rate를 감소시켰다가 재시작 → local minima 탈출
- **Loss**: BCEWithLogitsLoss with pos_weight
- **No Validation**: 전체 train data 사용 (시간 제약, 1 epoch = 40분)
- **Seed**: 42 고정 (재현성)

#### 결론
- **장점**: 빠른 학습 속도 (DIN 대비), 전체 데이터 활용, Wide & Deep 조합으로 다양한 패턴 학습
- **단점**: Validation 없어 과적합 위험
- **보완**: DIN의 CV 결과로 간접 검증

---

## 4. Evaluation (평가)

### 4.1 Competition Metric (대회 평가 지표)

#### 공식
```python
Score = 0.5 * AP + 0.5 * (1 / (1 + WLL))

# AP: Average Precision (Precision-Recall 곡선 아래 면적)
# WLL: Weighted Log Loss
WLL = 0.5 * (-mean(log(1 - pred[y==0]))) + 0.5 * (-mean(log(pred[y==1])))
```

#### 의미
- **AP (Average Precision)**: Ranking 성능 (클릭할 사용자를 상위에 잘 배치했는가)
- **WLL (Weighted Log Loss)**: Calibration 성능 (예측 확률이 실제 확률과 유사한가)
- **가중 평균**: Ranking과 Calibration의 균형

#### 구현
```python
def calculate_competition_score(y_true, y_pred):
    ap = average_precision_score(y_true, y_pred)

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    mask0, mask1 = (y_true == 0), (y_true == 1)
    ll0 = -np.mean(np.log(1 - y_pred[mask0])) if mask0.sum() else 0
    ll1 = -np.mean(np.log(y_pred[mask1])) if mask1.sum() else 0
    wll = 0.5 * ll0 + 0.5 * ll1

    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll
```

### 4.2 DIN Model Evaluation (5-Fold CV)

#### Out-of-Fold Validation
- 각 fold에서 validation set으로 성능 측정
- 5개 fold의 결과를 합쳐서 전체 train data에 대한 OOF 예측 생성
- OOF 예측으로 최종 CV Score 계산

#### 장점
- **Unbiased Estimation**: 모든 train data가 정확히 1번씩 validation에 사용
- **Model Selection**: Best epoch을 fold별로 자동 선택 (early stopping)

### 4.3 Wide & Deep Model Evaluation

#### 간접 평가
- DIN의 OOF score와 test score의 관계를 참고
- Train loss의 수렴 양상으로 과적합 여부 판단

---

## 5. Ensemble (앙상블)

### 5.1 DIN Internal Ensemble (5-Fold Averaging)

#### 방법
```python
# 각 fold 모델로 test 예측
test_preds = np.zeros((5, len(test)))
for fold in range(1, 6):
    model.load_state_dict(torch.load(f'din_fold{fold}.pt'))
    test_preds[fold-1] = predict(model, test_loader)

# 평균 앙상블
final_pred = test_preds.mean(axis=0)
```

#### 근거
- **분산 감소**: 5개 독립 모델의 예측을 평균 → 개별 모델의 오차 상쇄
- **Bagging 효과**: 각 fold가 다른 train data로 학습 → 다양성 확보
- **이론적 배경**: Bias-Variance Tradeoff
  - Variance ↓: 여러 모델의 평균
  - Bias ≈ 유지: 동일한 architecture

### 5.2 Model-level Ensemble (DIN + Wide&Deep)

#### 전략
- **Option 1**: Simple Averaging
  ```python
  final = 0.5 * din_pred + 0.5 * widectr_pred
  ```
- **Option 2**: Weighted Averaging (CV score 기반)
  ```python
  w_din = din_cv_score / (din_cv_score + widectr_estimated_score)
  final = w_din * din_pred + (1 - w_din) * widectr_pred
  ```

#### 근거
- **다양성**: 서로 다른 architecture (Attention vs LSTM+Cross)
- **상호 보완**: DIN은 sequence 관계 강점, Wide&Deep은 feature crossing 강점
- **Empirical Evidence**: Kaggle 대회에서 다른 모델 앙상블이 단일 모델보다 거의 항상 우수

### 5.3 최종 제출 전략

#### 제출 파일
1. **submissions/din.csv**: DIN 5-fold ensemble
2. **submissions/widectr.csv**: Wide & Deep single model
3. **(Optional) ensemble.csv**: 두 모델의 가중 평균

#### 제출 선택 기준
- DIN의 OOF score 확인
- Wide & Deep의 train loss 추이 확인
- 가능하면 두 모델의 가중 평균 사용

---

## 6. Implementation Details (구현 세부사항)

### 6.1 재현성 보장

#### Seed 고정
```python
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

#### 고정 설정
- DIN: `random_state=42` in StratifiedKFold
- Wide & Deep: `SEED=42`, `EPOCHS=6` 고정

### 6.2 메모리 최적화

#### DIN: Lazy Loading
```python
# seq를 문자열로 저장 (40GB → 1GB)
seq_strings = df['seq'].astype(str).tolist()

# Dataset에서 lazy하게 변환
def __getitem__(self, idx):
    seq_str = self.seq_strings[idx]
    seq_encoded = self.seq_encoder.encode_single(seq_str)  # O(1) dict lookup
    return seq_encoded
```

#### Dynamic Padding
```python
# 배치별로 max_len 결정
seq_padded = pad_sequence(seq_list, batch_first=True)
if seq_padded.size(1) > MAX_SEQ_LENGTH:
    seq_padded = seq_padded[:, :MAX_SEQ_LENGTH]
```

### 6.3 데이터 처리

#### Polars vs Pandas
- **DIN**: Polars (LazyFrame + streaming) → 메모리 효율
- **Wide & Deep**: Pandas (단순 작업) → 구현 편의

#### 전처리 Pipeline
```python
lf = pl.scan_parquet("data/train.parquet")  # Lazy loading
lf = engineer_all_features_polars(lf)       # Lazy evaluation
df = lf.collect(engine="streaming")          # Streaming execution
```

---

## 7. 코드 구조

```
team_legend_toss/
├── data/
│   ├── train.parquet
│   ├── test.parquet
│   └── sample_submission.csv
├── models/                      # 학습된 모델
│   ├── widectr.pt
│   ├── widectr_encoders.pkl
│   ├── din_fold1.pt ~ din_fold5.pt
│   └── din_metadata.pkl
├── submissions/                 # 제출 파일
│   ├── widectr.csv
│   └── din.csv
├── train/                       # 학습 스크립트
│   ├── widectr.py              # Wide & Deep CTR 학습
│   └── din..py                 # DIN 학습 (5-Fold CV)
├── inference/                   # 추론 스크립트
│   ├── widectr.py              # Wide & Deep 추론
│   └── din.py                  # DIN 추론 (5-fold ensemble)
└── README.MD                    # 본 보고서
```

### 실행 방법

#### 학습
```bash
# Wide & Deep CTR (6 epochs, full training)
python train/widectr.py

# DIN (5-Fold CV, early stopping)
python train/din..py
```

#### 추론
```bash
# Wide & Deep 추론
python inference/widectr.py  # → submissions/widectr.csv

# DIN 추론 (5-fold ensemble)
python inference/din.py      # → submissions/din.csv
```

---

## 8. 핵심 전략 요약

| 측면 | 전략 | 근거 |
|------|------|------|
| **Validation** | DIN: 5-Fold CV<br>Wide&Deep: Full Training | 안정성 vs 효율성 trade-off<br>1 epoch = 40분 시간 제약 |
| **Feature Engineering** | Temporal, Interaction, Aggregation | 다양한 각도에서 사용자 행동 모델링 |
| **Model Architecture** | DIN: Attention<br>Wide&Deep: Cross + LSTM | 광고-이력 관계 vs feature crossing |
| **Memory Optimization** | Lazy Loading + Dynamic Padding | 대용량 sequence 데이터 처리 (40GB→1GB) |
| **Ensemble** | 5-Fold Averaging<br>Model Averaging | 분산 감소 + 상호 보완 |
| **Reproducibility** | Seed=42, 고정 설정 | 코드 제출 재현성 보장 |

---

## 9. 결론

본 프로젝트는 **두 가지 상호 보완적인 모델**을 통해 CTR 예측 문제를 해결했습니다:

1. **DIN (Deep Interest Network)**:
   - Attention 메커니즘으로 광고-사용자 이력 간의 관계를 명시적으로 모델링
   - 5-Fold CV로 안정적인 성능 검증 및 앙상블 효과
   - OOF score를 통한 신뢰할 수 있는 성능 추정

2. **Wide & Deep CTR**:
   - Cross Network와 LSTM으로 feature crossing과 sequence 패턴 학습
   - 전체 데이터 활용으로 최대 성능 추구
   - 6 epoch 고정으로 시간 제약 내 완료 (1 epoch = 40분)

**Feature Engineering**에서는 다음과 같은 다양한 각도로 사용자 행동을 모델링했습니다:
- **Temporal**: 시간대별 사용자 행동 패턴 (출퇴근, 심야 등)
- **l_feat_14**: 광고 ID와 사용자 이력의 유사도
- **Sequence**: 길이, 다양성, 최근성
- **History**: 과거 행동 통계 (history_a_1이 핵심)
- **Interaction**: Age×Hour, History×SeqLength

**메모리 최적화** 기법으로 대용량 데이터를 효율적으로 처리했습니다:
- Lazy Loading: 40GB → 1GB 메모리 사용
- Dynamic Padding: 배치별 최적 패딩

**재현성**을 위해 모든 설정을 고정하고, 학습/추론을 분리한 깔끔한 코드 구조를 유지했습니다.

---

**Team Legend Toss**
*CTR Prediction Competition 2025*
