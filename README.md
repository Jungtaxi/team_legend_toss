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

### 3.2 Model 2: XGBoost + CatBoost Ensemble

#### 가설
"전통적인 GBDT 모델은 **테이블 형태 데이터**에서 여전히 강력하며, 특히 **feature 간 복잡한 상호작용**을 자동으로 학습한다"

#### Architecture (모델 구조)
```
Input: Polars LazyFrame (Out-of-Core Processing)
  ↓
Feature Engineering (Streaming Mode):
  - Sequence Features (길이, 다양성, top events)
  - History Features (통계량, binning, clipping)
  - Time Features (시간대 세그먼트, 요일 패턴)
  - Interaction Features (history × seq_length, age × hour)
  - Group Aggregation (feat_a~e 통계량)
  ↓
5-Fold Stratified CV:
  ├─ XGBoost (tree_method=hist, GPU)
  │   - num_boost_round: 2500
  │   - early_stopping: 100 rounds
  │   - max_depth: 10, lr: 0.01
  │   - Noise Injection: Gaussian(σ=1e-5)
  │
  └─ CatBoost (GPU)
      - iterations: 4000
      - depth: 6, lr: 0.03
      - auto_class_weights: Balanced
      - Noise Injection: Numeric features only
  ↓
Performance-based Weighted Ensemble:
  weight_xgb = score_xgb / (score_xgb + score_cat)
  final = weight_xgb × XGB + weight_cat × CAT
```

#### 핵심 기술

**1. Polars Out-of-Core Processing**
```python
# 메모리 효율적 Lazy Loading
lf = pl.scan_parquet("data/train.parquet")  # 메모리에 안 올림!
lf = engineer_all_features_polars(lf)        # Lazy evaluation
df = lf.collect(engine="streaming")          # Streaming 실행
```
- **장점**: Pandas 대비 ~20배 메모리 절약, 10-30초만에 feature engineering 완료
- **Streaming Mode**: 전체 데이터를 메모리에 올리지 않고 처리

**2. Noise Injection (Data Augmentation)**
```python
# XGBoost: 모든 numeric features
gaussian_noise = np.random.normal(0, 1e-5, X_train.shape)
X_train += gaussian_noise

# CatBoost: numeric columns only (categorical 제외)
for col in numeric_cols:
    X_train[col] += np.random.normal(0, 1e-5, size=len(X_train))
```
- **목적**: 과적합 방지, 일반화 능력 향상
- **σ=1e-5**: 매우 작은 noise로 데이터 분포를 크게 바꾸지 않으면서 regularization 효과

**3. Model Persistence (모델 가중치 저장)**
```python
# XGBoost: JSON 형식 (재사용 가능)
model.save_model(f"outputs/{run_name}/xgb_fold{fold}.json")

# CatBoost: Native format
model.save_model(f"outputs/{run_name}/catboost_fold{fold}.cbm")
```
- 각 fold별 모델 저장 → 재학습 없이 추론 가능
- Submission 파일과 함께 저장되어 재현성 보장

#### Hyperparameters (하이퍼파라미터)

**XGBoost**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| tree_method | hist | GPU 가속 + 메모리 효율 |
| max_depth | 10 | 깊은 트리로 복잡한 패턴 학습 |
| learning_rate | 0.01 | 낮은 lr + 많은 round로 안정적 학습 |
| subsample | 0.8 | Row sampling으로 과적합 방지 |
| colsample_bytree | 0.8 | Column sampling |
| gamma | 0.1 | Min loss reduction (pruning) |
| reg_alpha | 0.1 | L1 regularization |
| reg_lambda | 1.0 | L2 regularization |
| scale_pos_weight | auto | Class imbalance 보정 |
| max_bin | 256 | Histogram bin 개수 |

**CatBoost**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| depth | 6 | 얕은 트리 (XGB와 상호 보완) |
| learning_rate | 0.03 | XGB보다 높은 lr |
| iterations | 4000 | 충분한 boosting rounds |
| l2_leaf_reg | 15.0 | L2 regularization |
| auto_class_weights | Balanced | 자동 클래스 가중치 |
| task_type | GPU | GPU 가속 |

#### Training Strategy (학습 전략)

**5-Fold Cross Validation**
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))

for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    # XGBoost 학습
    xgb_model.train(..., early_stopping_rounds=100)
    oof_xgb[va] = xgb_model.predict(X_val)

    # CatBoost 학습
    cat_model.fit(..., early_stopping_rounds=20)
    oof_cat[va] = cat_model.predict_proba(X_val)[:, 1]
```

**Performance-based Weighting**
```python
# OOF 성능으로 가중치 자동 결정
score_xgb = calculate_competition_score(y, oof_xgb)
score_cat = calculate_competition_score(y, oof_cat)

w_xgb = score_xgb / (score_xgb + score_cat)
w_cat = score_cat / (score_xgb + score_cat)

final = w_xgb * pred_xgb + w_cat * pred_cat
```

**Test-time Aggregation**
```python
# Median aggregation (robust against outliers)
test_xgb_final = np.median(test_preds_5folds, axis=0)
test_cat_final = np.median(test_preds_5folds, axis=0)
```
- Mean 대신 Median 사용 → Outlier fold에 강건

#### W&B Integration (실험 추적)
```python
wandb.init(project="CTR_Polars_OOC_Stack", name=run_name)
wandb.log({
    "xgb/fold1_score": score,
    "cat/fold1_score": score,
    "weighted/score": ensemble_score,
    "weighted/xgb_weight": w_xgb,
})
```
- 모든 fold별 성능, 가중치, 런타임 자동 기록
- 실험 재현 및 비교 용이

#### 결론
- **장점**:
  - Feature engineering이 매우 빠름 (Polars streaming)
  - XGBoost + CatBoost의 상호 보완적 학습 (depth, lr 다름)
  - Noise injection으로 일반화 능력 향상
  - 5-fold CV로 robust한 성능 검증
  - 모델 가중치 저장으로 재사용 가능
- **단점**:
  - Sequence를 직접 모델링하지 않음 (통계량으로만 활용)
  - Deep learning 모델보다 feature engineering 의존도 높음
- **적용 시나리오**: 빠른 실험, 안정적인 baseline, 딥러닝 모델과 앙상블

#### CatBoost Hybrid 변형 (Modular Feature Engineering)

본 프로젝트는 향상된 feature engineering을 위한 **모듈화된 접근법**도 개발했습니다 (`train/catboost_hybrid/`).

**핵심 컴포넌트: `PolarsFeatureEngineer` 클래스** (train/catboost_hybrid/feature_engineering.py:10)

이 클래스는 설정 기반(config-driven) 방식으로 세 가지 고급 feature engineering 기법을 제공합니다:

**1. Temporal Features (시간적 특징 강화)**
```python
class PolarsFeatureEngineer:
    def apply_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Hour binning (설정 가능한 구간)
        df = df.with_columns([
            pl.col("hour").cast(pl.Float32)
            .map_elements(lambda x: np.digitize(x, bins), return_dtype=pl.Int32)
            .alias("hour_period")
        ])

        # Weekend flag
        df = df.with_columns([
            (pl.col("day_of_week") >= 5).cast(pl.Int8).alias("is_weekend")
        ])

        # Cyclical encoding (sin/cos 변환)
        df = df.with_columns([
            (pl.col("hour") * 2 * np.pi / 24).sin().alias("hour_sin"),
            (pl.col("hour") * 2 * np.pi / 24).cos().alias("hour_cos"),
        ])

        return df
```
- **시간의 주기성 포착**: sin/cos 변환으로 23시와 0시의 유사성 학습
- **유연한 구간화**: config.yaml에서 hour_bins 설정 가능
- **주말 효과**: 주중/주말 행동 패턴 차이 명시적 표현

**2. Cross Features (교차 특징)**
```python
def apply_cross_features(self, df: pl.DataFrame) -> pl.DataFrame:
    # 예: gender × age_group, hour × inventory_id
    for col1, col2 in pairs:
        new_name = f"cross_{col1}_{col2}"
        df = df.with_columns([
            (pl.col(col1).cast(pl.Utf8) + "_" + pl.col(col2).cast(pl.Utf8))
            .hash(seed=0)
            .mod(num_buckets)  # Hash trick으로 메모리 절약
            .alias(new_name)
        ])
    return df
```
- **Hash trick**: 무한한 조합을 고정된 버킷 수(default: 100,000)로 매핑
- **메모리 효율**: One-hot encoding 대비 공간 절약
- **설정 기반**: config.yaml에서 교차할 feature 쌍 지정

**3. Aggregation Features (집계 특징)**
```python
def apply_aggregation_features(self, df: pl.DataFrame) -> pl.DataFrame:
    # feat_a, feat_b 등 그룹별 통계
    for group_name, group_cfg in groups.items():
        cols = self._get_feature_columns(group_name)

        # 통계량 계산
        exprs.append(
            pl.concat_list([pl.col(c).cast(pl.Float32) for c in cols])
            .list.mean().alias(f"{group_name}_mean")
        )
        # std, max, min, sum도 가능

    return df.with_columns(exprs)
```
- **그룹 통계**: feat_a_1~feat_a_N → feat_a_mean, feat_a_std 등
- **차원 축소**: 여러 관련 feature를 소수의 통계량으로 압축

**설정 기반 접근법의 장점**

`config.yaml`에서 모든 feature engineering 설정을 관리:
```yaml
feature_engineering:
  enabled: true
  temporal:
    enabled: true
    hour_bins: [0, 6, 12, 18, 24]
    weekend_flag: true
    time_of_day_features: true  # sin/cos encoding
  cross_features:
    enabled: true
    pairs:
      - ["gender", "age_group"]
      - ["hour", "inventory_id"]
    num_buckets: 100000
  aggregations:
    enabled: true
    groups:
      feat_a:
        stats: ["mean", "std", "max", "min"]
```

**실제 적용**

이 모듈화된 접근법은 `xgboost_catboost.py` 스크립트에 인라인 형태로 통합되었습니다.

CatBoost Hybrid 변형의 제출 파일(`submissions/catboost_hybrid.csv`)은 이러한 강화된 feature engineering을 포함한 모델로 생성되었으며, 최종 앙상블에서 6.67%의 기여도를 차지합니다.

**결론**:
- **모듈화**: 재사용 가능한 feature engineering 컴포넌트
- **설정 기반**: 코드 수정 없이 실험 가능
- **Polars 네이티브**: 고속 처리 유지
- **확장성**: 새로운 feature 유형 쉽게 추가 가능

---

### 3.3 Model 3: Wide & Deep CTR

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

### 5.2 GBDT Internal Ensemble (XGBoost + CatBoost)

#### 전략
```python
# 1. OOF 성능 기반 가중치 계산
score_xgb, _, _ = calculate_competition_score(y_train, oof_xgb)
score_cat, _, _ = calculate_competition_score(y_train, oof_cat)

w_xgb = score_xgb / (score_xgb + score_cat)
w_cat = score_cat / (score_xgb + score_cat)

# 2. Test predictions (5-fold median)
test_xgb = np.median([fold1_pred, ..., fold5_pred], axis=0)
test_cat = np.median([fold1_pred, ..., fold5_pred], axis=0)

# 3. Weighted ensemble
final_pred = w_xgb * test_xgb + w_cat * test_cat
```

#### 근거
- **자동 가중치**: OOF 성능으로 가중치 결정 → 수동 튜닝 불필요
- **다양성**: XGBoost(깊은 트리) vs CatBoost(얕은 트리)의 상호 보완
- **Median Aggregation**: Mean 대신 median으로 outlier fold 영향 감소
- **검증 가능**: 전체 OOF prediction으로 앙상블 성능 사전 검증

#### XGBoost vs CatBoost 차이점
| 측면 | XGBoost | CatBoost |
|------|---------|----------|
| **Depth** | 10 (깊은 트리) | 6 (얕은 트리) |
| **Learning Rate** | 0.01 (낮음) | 0.03 (높음) |
| **Categorical** | Label encoding | Native 지원 |
| **Class Weights** | scale_pos_weight | auto_class_weights |
| **특징** | 복잡한 패턴, 고차 상호작용 | 단순 패턴, 범주형 변수 강점 |

### 5.3 Model-level Ensemble (DIN + GBDT + Wide&Deep)

#### 전략
- **Option 1**: Simple Averaging
  ```python
  final = 0.33 * din_pred + 0.33 * gbdt_pred + 0.33 * widectr_pred
  ```
- **Option 2**: Weighted Averaging (CV score 기반)
  ```python
  total = din_cv + gbdt_cv + widectr_est
  w_din = din_cv / total
  w_gbdt = gbdt_cv / total
  w_wide = widectr_est / total
  final = w_din * din_pred + w_gbdt * gbdt_pred + w_wide * widectr_pred
  ```
- **Option 3**: Stacking (Meta-learner)
  ```python
  # Level 1: Base models의 OOF predictions
  meta_features = np.column_stack([oof_din, oof_gbdt, oof_wide])

  # Level 2: Logistic Regression
  meta_model = LogisticRegression()
  meta_model.fit(meta_features, y_train)

  # Test prediction
  test_meta = np.column_stack([din_pred, gbdt_pred, wide_pred])
  final = meta_model.predict_proba(test_meta)[:, 1]
  ```

#### 근거
- **최대 다양성**: 3가지 완전히 다른 접근법
  - DIN: Attention mechanism (광고-이력 유사도)
  - GBDT: Tree-based (자동 feature interaction)
  - Wide&Deep: Neural network (feature crossing + deep learning)
- **상호 보완**:
  - DIN → Sequence 모델링 강점
  - GBDT → Tabular data 강점, 빠른 학습
  - Wide&Deep → Feature crossing 강점
- **Empirical Evidence**: Kaggle 대회에서 서로 다른 모델 앙상블이 단일 모델보다 거의 항상 우수

### 5.4 최종 제출 전략

#### 제출 파일
1. **submissions/din.csv**: DIN 5-fold ensemble
2. **outputs/{run_name}/submission_xgb.csv**: XGBoost 5-fold median
3. **outputs/{run_name}/submission_cat.csv**: CatBoost 5-fold median
4. **outputs/{run_name}/submission_weighted.csv**: XGB+CAT weighted ensemble
5. **submissions/widectr.csv**: Wide & Deep single model
6. **(Optional) ensemble.csv**: 3개 모델의 가중 평균 또는 stacking

#### 제출 선택 기준
| 우선순위 | 파일 | 근거 |
|---------|------|------|
| 1 | **3-Model Ensemble** | 최대 다양성, 최고 성능 기대 |
| 2 | **GBDT Weighted** | 검증된 OOF score, 자동 가중치 |
| 3 | **DIN 5-fold** | Attention 메커니즘, 검증된 성능 |
| 4 | **Wide & Deep** | 전체 데이터 활용 |

#### 실행 가이드
```bash
# 1. GBDT 모델 학습 (가장 빠름, ~30분)
python train/xgboost_catboost.py
# → outputs/{run_name}/submission_weighted.csv

# 2. DIN 학습 (5-fold, ~2-3시간)
python train/din.py
# → submissions/din.csv

# 3. Wide & Deep 학습 (~4시간)
python train/widectr.py
# → submissions/widectr.csv

# 4. (Optional) 앙상블
python ensemble.py --weights auto  # CV score 기반
```

### 5.5 실제 앙상블 구현

본 프로젝트에서는 **계층적 앙상블(Hierarchical Ensemble)** 전략을 사용하여 최종 제출 파일을 생성했습니다.

#### 앙상블 구조

```
Level 1: 모델 계열별 앙상블
├─ Tree 계열 앙상블
│   ├─ submissions/catboost.csv (1/3)
│   ├─ submissions/xgboost.csv (1/3)
│   └─ submissions/catboost_hybrid.csv (1/3)
│
└─ Deep 계열 앙상블
    ├─ submissions/din.csv (1/2)
    └─ submissions/widectr_epoch_5.csv (1/2)

Level 2: 최종 앙상블
├─ Tree 앙상블 결과 (1/5 = 20%)
├─ Deep 앙상블 결과 (2.5/5 = 50%)
└─ submissions/widectr_epoch_6.csv (1.5/5 = 30%)
```

#### 구현 코드

**Step 1: Tree 계열 앙상블**
```python
import pandas as pd
import numpy as np

# Tree 모델 예측 로드
catboost = pd.read_csv('submissions/catboost.csv')
xgboost = pd.read_csv('submissions/xgboost.csv')
catboost_hybrid = pd.read_csv('submissions/catboost_hybrid.csv')

# 1:1:1 비율로 앙상블
tree_ensemble = (
    catboost['clicked'].values * (1/3) +
    xgboost['clicked'].values * (1/3) +
    catboost_hybrid['clicked'].values * (1/3)
)
```

**Step 2: Deep 계열 앙상블**
```python
# Deep learning 모델 예측 로드
din = pd.read_csv('submissions/din.csv')
widectr_epoch_5 = pd.read_csv('submissions/widectr_epoch_5.csv')

# 1:1 비율로 앙상블
deep_ensemble = (
    din['clicked'].values * 0.5 +
    widectr_epoch_5['clicked'].values * 0.5
)
```

**Step 3: 최종 앙상블**
```python
# Wide & Deep epoch 6 예측 로드
widectr_epoch_6 = pd.read_csv('submissions/widectr_epoch_6.csv')

# 1:2.5:1.5 비율로 최종 앙상블
# 총합 = 1 + 2.5 + 1.5 = 5.0
final_predictions = (
    tree_ensemble * (1.0 / 5.0) +      # 20%
    deep_ensemble * (2.5 / 5.0) +      # 50%
    widectr_epoch_6['clicked'].values * (1.5 / 5.0)  # 30%
)

# 제출 파일 생성
submission = pd.DataFrame({
    'ID': catboost['ID'],
    'clicked': final_predictions
})
submission.to_csv('submissions/final_ensemble.csv', index=False)
```

#### 앙상블 가중치 선택 근거

| 구성 요소 | 가중치 | 근거 |
|----------|--------|------|
| **Tree 앙상블** | 20% | GBDT 모델의 안정성과 빠른 학습, 5-Fold CV로 검증된 성능 |
| **Deep 앙상블** | 50% | DIN의 attention mechanism과 Wide&Deep epoch 5의 조합으로 가장 높은 가중치 부여 |
| **Wide&Deep epoch 6** | 30% | 전체 데이터로 학습한 최종 epoch의 예측을 추가하여 다양성 확보 |

#### 모델별 기여도 분해

최종 앙상블에서 각 개별 모델의 실질적 기여도:

```python
# Tree 계열
catboost:         1/3 × 1/5 = 6.67%
xgboost:          1/3 × 1/5 = 6.67%
catboost_hybrid:  1/3 × 1/5 = 6.67%

# Deep 계열
din:              1/2 × 2.5/5 = 25.0%
widectr_epoch_5:  1/2 × 2.5/5 = 25.0%

# 최종 모델
widectr_epoch_6:  1.5/5 = 30.0%

# 총합: 100%
```

#### 앙상블 효과

- **다양성**: Tree-based + Attention-based + Cross Network 모델의 조합
- **안정성**: 각 계열 내부에서 먼저 앙상블하여 노이즈 감소
- **성능 균형**: Deep learning 모델에 높은 가중치(80%)를 부여하되, GBDT의 안정성도 활용

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
├── models/                      # 학습된 모델 (DIN, Wide&Deep)
│   ├── widectr.pt
│   ├── widectr_encoders.pkl
│   ├── din_fold1.pt ~ din_fold5.pt
│   └── din_metadata.pkl
├── outputs/                     # GBDT 모델 출력 (자동 생성)
│   └── {run_name}/             # 예: polars-ooc-stack-phase2-noise-20250117-143022
│       ├── xgb_fold1.json ~ xgb_fold5.json      # XGBoost 모델
│       ├── catboost_fold1.cbm ~ catboost_fold5.cbm  # CatBoost 모델
│       ├── submission_xgb.csv
│       ├── submission_cat.csv
│       └── submission_weighted.csv  # ⭐ Best GBDT submission
├── submissions/                 # 제출 파일
│   ├── widectr.csv             # Wide & Deep
│   └── din.csv                 # DIN
├── train/                       # 학습 스크립트
│   ├── xgboost_catboost.py     # ⭐ XGBoost + CatBoost 학습 (5-Fold CV)
│   ├── widectr.py              # Wide & Deep CTR 학습
│   ├── din..py                  # DIN 학습 (5-Fold CV)
│   └── catboost_hybrid/        # 모듈화된 Feature Engineering
│       ├── feature_engineering.py  # PolarsFeatureEngineer 클래스
│       ├── config.yaml         # Feature engineering 설정
│       └── catboost_hybrid.py  # (향후 독립 학습 스크립트용)
├── inference/                   # 추론 스크립트
│   ├── widectr.py              # Wide & Deep 추론
│   ├── din.py                  # DIN 추론 (5-fold ensemble)
│   └── xgboost_catboost_inference.py  # XGBoost + CatBoost 추론 (5-fold ensemble)
└── README.md                    # 본 보고서
```

### 실행 방법

#### 학습
```bash
# 1. XGBoost + CatBoost (가장 빠름, ~30분)
python train/xgboost_catboost.py
# 출력:
# - outputs/{run_name}/submission_weighted.csv (최종 제출용)
# - outputs/{run_name}/xgb_fold*.json (모델 가중치)
# - outputs/{run_name}/catboost_fold*.cbm (모델 가중치)
# - W&B 로그: https://wandb.ai/your-project/CTR_Polars_OOC_Stack

# 2. Wide & Deep CTR (6 epochs, full training, ~4시간)
python train/widectr.py

# 3. DIN (5-Fold CV, early stopping, ~2-3시간)
python train/din.py
```

#### 추론
```bash
# Wide & Deep 추론
python inference/widectr.py  # → submissions/widectr.csv

# DIN 추론 (5-fold ensemble)
python inference/din.py      # → submissions/din.csv

# XGBoost + CatBoost 추론 (5-fold ensemble)
# 학습 시 자동 생성되지만, 독립 실행도 가능
python inference/xgboost_catboost_inference.py \
    --model-dir outputs/polars-ooc-stack-phase2-noise-20250117-143022 \
    --data-dir ./data \
    --output-dir ./submissions
# → submissions/xgboost.csv
# → submissions/catboost.csv
# → submissions/xgb_cat_weighted.csv
```

#### 모델 재사용 (저장된 가중치 로드)
```python
# XGBoost 모델 로드
import xgboost as xgb
model = xgb.Booster()
model.load_model("outputs/{run_name}/xgb_fold1.json")

# CatBoost 모델 로드
from catboost import CatBoostClassifier
model = CatBoostClassifier()
model.load_model("outputs/{run_name}/catboost_fold1.cbm")
```

---

## 8. 핵심 전략 요약

| 측면 | 전략 | 근거 |
|------|------|------|
| **Validation** | DIN: 5-Fold CV<br>GBDT: 5-Fold CV<br>Wide&Deep: Full Training | 안정성 vs 효율성 trade-off<br>1 epoch = 40분 시간 제약 |
| **Feature Engineering** | Polars Streaming (GBDT)<br>Temporal, Interaction, Aggregation | 10-30초 고속 처리 (~20배 빠름)<br>다양한 각도에서 사용자 행동 모델링 |
| **Model Architecture** | DIN: Attention<br>GBDT: XGBoost + CatBoost<br>Wide&Deep: Cross + LSTM | 광고-이력 관계 학습<br>자동 feature interaction<br>저차+고차 feature crossing |
| **Memory Optimization** | DIN: Lazy Loading + Dynamic Padding<br>GBDT: Polars Out-of-Core | 대용량 sequence 데이터 처리 (40GB→1GB)<br>Streaming mode로 메모리 절약 |
| **Regularization** | GBDT: Noise Injection (σ=1e-5)<br>DNN: Dropout, BatchNorm | Data augmentation으로 과적합 방지<br>신경망 정규화 |
| **Ensemble** | GBDT: Performance-based Weighting<br>DIN: 5-Fold Averaging<br>Final: 3-Model Stacking | OOF 성능 기반 자동 가중치<br>분산 감소<br>최대 다양성 확보 |
| **Reproducibility** | Seed=42, 고정 설정<br>모델 가중치 저장 | 코드 제출 재현성 보장<br>재학습 없이 추론 가능 |
| **Experiment Tracking** | W&B (GBDT)<br>수동 로깅 (DIN, Wide&Deep) | 자동 실험 추적, 비교 용이 |

---

## 9. 결론

본 프로젝트는 **세 가지 상호 보완적인 모델**을 통해 CTR 예측 문제를 다각도로 해결했습니다:

### 9.1 모델별 강점

1. **DIN (Deep Interest Network)**:
   - Attention 메커니즘으로 광고-사용자 이력 간의 관계를 명시적으로 모델링
   - 5-Fold CV로 안정적인 성능 검증 및 앙상블 효과
   - OOF score를 통한 신뢰할 수 있는 성능 추정
   - **강점**: Sequence 모델링, 사용자별 맞춤 광고 추천

2. **XGBoost + CatBoost Ensemble**:
   - Polars streaming으로 **10-30초 만에 feature engineering** 완료 (Pandas 대비 ~20배 빠름)
   - 5-Fold CV + 성능 기반 자동 가중치로 **검증된 성능**
   - Noise injection (σ=1e-5)으로 일반화 능력 향상
   - 모델 가중치 저장으로 **재학습 없이 추론 가능**
   - W&B 자동 실험 추적으로 **재현성 보장**
   - **강점**: 빠른 학습 속도 (~30분), 안정적인 baseline, 자동 feature interaction

3. **Wide & Deep CTR**:
   - Cross Network와 LSTM으로 feature crossing과 sequence 패턴 학습
   - 전체 데이터 활용으로 최대 성능 추구
   - 6 epoch 고정으로 시간 제약 내 완료 (1 epoch = 40분)
   - **강점**: Feature crossing, 전체 데이터 활용

### 9.2 Feature Engineering

**Polars Out-of-Core Processing** (GBDT 모델):
- Pandas 대비 ~20배 메모리 절약
- 10-30초만에 feature engineering 완료 (Pandas는 5-10분)
- Streaming mode로 대용량 데이터 처리

**다양한 각도의 사용자 행동 모델링**:
- **Temporal**: 시간대별 사용자 행동 패턴 (출퇴근, 심야 등)
- **l_feat_14**: 광고 ID와 사용자 이력의 유사도
- **Sequence**: 길이, 다양성, top events 포함 여부
- **History**: 과거 행동 통계 (history_a_1이 핵심), outlier clipping
- **Interaction**: Age×Hour, History×SeqLength
- **Group Aggregation**: feat_a~e 통계량 (mean, max, min, std)

### 9.3 메모리 최적화

| 모델 | 기법 | 효과 |
|------|------|------|
| DIN | Lazy Loading + Dynamic Padding | 40GB → 1GB |
| GBDT | Polars Out-of-Core Streaming | Pandas 대비 ~20배 절약 |
| Wide&Deep | 일반 Pandas | - |

### 9.4 앙상블 전략

**계층적 앙상블 구조** (실제 구현):

1. **Level 1 - 모델 계열별 앙상블**:
   - **Tree 계열**: CatBoost + XGBoost + CatBoost Hybrid (1:1:1 비율)
   - **Deep 계열**: DIN + Wide&Deep epoch 5 (1:1 비율)

2. **Level 2 - 최종 앙상블**:
   - Tree 앙상블: 20%
   - Deep 앙상블: 50%
   - Wide&Deep epoch 6: 30%

**개별 모델 최종 기여도**:
- CatBoost: 6.67% | XGBoost: 6.67% | CatBoost Hybrid: 6.67%
- DIN: 25.0% | Wide&Deep epoch 5: 25.0%
- Wide&Deep epoch 6: 30.0%

**최대 다양성 확보**:
- DIN (Attention) + GBDT (Tree-based) + Wide&Deep (Neural Network)
- Sequence 직접 모델링 (DIN) + 통계량 활용 (GBDT) + LSTM 인코딩 (Wide&Deep)
- Deep learning 모델에 80% 가중치를 부여하여 복잡한 패턴 학습 강조
- GBDT의 안정성으로 예측 분산 감소

### 9.5 재현성 및 실험 관리

**재현성 보장**:
- 모든 모델에서 Seed=42 고정
- GBDT: 각 fold별 모델 가중치 저장 (xgb_fold*.json, catboost_fold*.cbm)
- 학습/추론 스크립트 분리

**실험 추적**:
- GBDT: W&B 자동 로깅 (fold별 성능, 가중치, 런타임)
- 하이퍼파라미터, 성능 지표, 모델 파일 모두 자동 저장

### 9.6 최종 제출 전략

**우선순위**:
1. **3-Model Ensemble** (DIN + GBDT + Wide&Deep) → 최고 성능 기대
2. **GBDT Weighted** → 검증된 OOF score, 가장 안정적
3. **DIN 5-fold** → Attention 메커니즘, 검증된 성능
4. **Wide & Deep** → 전체 데이터 활용

**실행 시간**:
- GBDT: ~30분 (가장 빠름) ⚡
- DIN: ~2-3시간
- Wide & Deep: ~4시간
- **Total: ~7시간** (병렬 실행 시 ~4시간)

---

**Team Legend Toss**
*CTR Prediction Competition 2025*
