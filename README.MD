```
team_legend_toss/
├── models/                    # 모델 파일 저장 (루트)
│   ├── widectr.pt
│   ├── widectr_encoders.pkl
│   ├── din_fold1.pt ~ din_fold5.pt
│   └── din_metadata.pkl
├── submissions/               # 제출 파일 저장
│   ├── widectr.csv
│   └── din.csv
├── train/
│   ├── widectr.py
│   └── din..py
└── inference/
    ├── widectr.py
    └── din.py
```

# 1. 학습 (6 epochs, 최종 모델만 저장)
python train/widectr.py

# 생성되는 파일:
# - models/widectr.pt
# - models/widectr_encoders.pkl

# 2. 추론 (제출 파일 생성)
python inference/widectr.py

# 생성되는 파일:
# - submissions/widectr.csv