# Book Rating Prediction

## 1. 프로젝트 소개

사용자의 도서 평점 이력을 바탕으로, 특정 사용자가 특정 도서에 얼마나 높은 평점을 줄지 예측하는 추천 시스템 프로젝트입니다.

이 프로젝트에서는 단순한 `user-item` 상호작용만 사용하는 대신, 저자, 출판사, 카테고리, 지역, 나이와 같은 메타데이터를 함께 활용해 평점 예측 성능을 높이는 데 집중했습니다. 데이터 희소성과 범주형 피처 비중이 높은 환경에서 어떤 모델이 실제로 효과적인지 검증하고, 전처리와 피처 엔지니어링이 성능에 미치는 영향을 체계적으로 비교했습니다.

## 2. 프로젝트 목표

- 도서 평점 예측 문제에 적합한 모델 구조 탐색
- 범주형 중심 데이터에서의 Feature Engineering 효과 검증
- 위치 정보 정제와 통계 기반 피처를 포함한 전처리 파이프라인 구축
- CatBoost, LightGBM, FM 계열, BERT 기반 모델을 비교해 최종 전략 도출

## 3. 문제 정의와 접근

### 3.1. 데이터 특성
- 상호작용 데이터가 매우 희소하며, 대부분의 상호작용이 소수 엔티티에 집중되어 있습니다.
- `user`, `book`, `author`, `location`, `category` 등 범주형 피처 비중이 높습니다.
- 단순한 딥러닝 모델보다 범주형 변수와 상호작용을 잘 처리하는 모델이 유리할 가능성이 큰 환경이었습니다.

### 3.2. 주요 실험
- Feature importance 및 residual heatmap 기반 상호작용 분석
- FM, FFM, DeepFM, NCF, WDN, DCN 등 다양한 추천 모델 비교
- LightGBM, CatBoost 기반 트리 모델 실험
- BERT 기반 rating prediction 모델 실험
- Stratified K-Fold CV 기반 앙상블 적용
- 위치 문자열 정제와 평점 개수 기반 통계 피처 추가

### 3.3. 핵심 관찰
- `지역`, `저자`, `출판사`, `카테고리`가 평점에 큰 영향을 미쳤습니다.
- 희소한 범주형 데이터에서는 FM 계열보다 GBDT 계열이 더 안정적인 성능을 보였습니다.
- 위치 정보 정제와 상호작용 횟수 기반 피처가 성능 개선에 직접 기여했습니다.
- BERT 기반 접근은 표현력은 있었지만, 데이터 희소성과 임베딩 학습 밀도 문제로 효율이 낮았습니다.

## 4. 실험 결과

### 4.1. 최종 성능

| 모델 | Private RMSE |
|------|--------------|
| CatBoost Base | 2.1409 |
| CatBoost Final | 2.1160 |

### 4.2. 결과 요약
- Stratified K-Fold Cross Validation 적용으로 기존 CatBoost 대비 약 `0.5%` 성능 향상을 확인했습니다.
- 유저, 도서, 저자별 평점 개수 피처를 추가한 뒤 CatBoost 기준 `RMSE 2.474 -> 2.1318` 수준의 개선을 확인했습니다.
- 위치 정제 파이프라인 적용 후 GeoMap 기준 유효 데이터 비율이 `City 86.5% -> 89.1%`, `State 83.9% -> 97.3%`, `Country 95.7% -> 99.9%`로 향상됐습니다.

### 4.3. 최종 선택 전략
- 최종 모델은 `CatBoost` 기반 접근입니다.
- 범주형 피처 수가 많고 각 범주의 cardinality가 큰 환경에서 안정적으로 동작했습니다.
- 모델 복잡도를 높이는 것보다, 데이터 정제와 통계 기반 피처 추가가 더 큰 성능 개선으로 이어졌습니다.

## 5. 저장소 구조

```text
pro-recsys-bookratingprediction-recsys-06
├─ config/              # 모델별 실행/실험 설정
├─ src/
│  ├─ data/             # 데이터 로딩, 분할, 전처리
│  ├─ models/           # FM, DeepFM, CatBoost, BERT 등 모델 정의
│  ├─ train/            # 학습 및 추론 로직
│  ├─ loss/             # 커스텀 loss
│  └─ ensembles/        # 앙상블 유틸
├─ results/             # EDA 및 분석 결과물
├─ main.py              # 학습/추론 진입점
├─ ensemble.py          # 앙상블 실행 스크립트
└─ EDA_*.py, ipynb      # 탐색 및 실험 노트
```

## 6. 기술 스택

- Python
- PyTorch
- Scikit-learn
- CatBoost
- Pandas
- Weights & Biases
- Hugging Face Transformers

## 7. 실행 방법

### 7.1. 설치

```bash
python3 -m pip install -r requirements.txt
```

### 7.2. 학습

```bash
python3 main.py --config config/config_best.yaml
```

### 7.3. 예측

```bash
python3 main.py --config config/config_best.yaml --predict True --checkpoint saved/checkpoints/bert_best.pt
```

실행 시 YAML 설정을 우선 사용하고, CLI 인자로 일부 값을 덮어쓸 수 있습니다.

## 8. 팀

| 이름 | 역할 |
|------|------|
| 김태형 | 팀장 |
| 김민재 | EDA, CatBoost baseline 전처리, Category/Location Feature Engineering |
| 석찬휘 | EDA, Category Feature Engineering |
| 조형동 | BERT 기반 모델 설계 및 실험 |
| 최영진 | LightGBM, CatBoost, Stratified K-Fold CV 구현 및 실험 |
