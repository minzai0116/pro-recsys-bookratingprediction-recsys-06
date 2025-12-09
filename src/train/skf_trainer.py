import numpy as np
import os
import joblib
import pandas as pd
import glob
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .sklearn_trainer import normalize_metric_name, calculate_metrics, log_feature_importance # 나중에 헬퍼 파일로 따로 모으는 것을 고려
from sklearn.model_selection import StratifiedKFold

METRIC_FUNCTIONS = {
    'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'rmseloss': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'mseloss': mean_squared_error,
    'maeloss': mean_absolute_error,
}


def save_model(args, model, setting, fold):
    """모델 저장"""
    if not args.train.save_best_model:
        return

    os.makedirs(args.train.ckpt_dir, exist_ok=True)
    # 저장명 변경 {setting.save_time}_{args.model}.pkl -> {setting.save_time}_{args.model}_{fold+1}.pkl
    model_path = f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_{fold+1}.pkl' 
    joblib.dump(model, model_path)
    print(f'Model saved: {model_path}')


def train(args, model, data, logger, setting):
    """
    sklearn 모델 학습 및 검증

    Parameters
    ----------
    args : argparse.Namespace
        설정 파라미터
    model : sklearn 모델
        학습할 모델 (fit 메서드 필요)
    data : dict
        학습/검증 데이터 딕셔너리
    logger : Logger
        학습 로그 기록 객체
    setting : Setting
        로거 설정인듯?

    Returns
    -------
    model : sklearn 모델
        학습된 모델
    """
    if args.wandb:
        import wandb

    X_train, y_train = data['X_train'], data['y_train']

    print(f'Training {args.model}...')

    # 1. fit 파라미터 준비 및 학습
    # fit_params = prepare_fit_params(args, data)

    # Stratified K Fold 설정
    skf = StratifiedKFold(n_splits=args.stratifiedkfold.n_splits, shuffle=args.stratifiedkfold.shuffle, random_state=args.seed)

    # Stratified K Fold 루프 실행
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        print(f'Training {fold+1}...')

        # Train, Valid 데이터 분할
        # 데이터 분할을 여기서 하면, 기존의 data_split 함수를 어떻게 사용하지 않게 하지? -> 어떻게 효율적으로 구성할 수 있지?
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        # 모델 학습
        current_model = clone(model)
        current_model.fit(X_t, y_t, eval_set=[(X_v, y_v)])

        # Train metric 계산 -> 필요할까? 필요하겠지? 일단 넣어봐
        main_metric = args.metrics[-1] # MSELoss 아닌가? RMSE를 봐야하는거 아닌가? -> args.metrics[-1]로 수정
        train_results = calculate_metrics(model, X_train, y_train, [main_metric])
        train_score = train_results[normalize_metric_name(main_metric)]

        msg = f'\tTrain {normalize_metric_name(main_metric).upper()}: {train_score:.3f}'

        # Validation score
        valid_score = valid(model, X_v, y_v, )

        print(msg)
        logger.log(epoch=1, train_loss=train_score)

        if args.wandb:
            wandb.log({f'Train {normalize_metric_name(main_metric).upper()}': train_score})

        # fold별 모델 저장
        save_model(args, model, fold)

    # 4. Feature Importance
    log_feature_importance(args, model, data)

    logger.close()

    return current_model # 앙상블을 하기 때문에 큰 의미는 없음


def valid(model, X_valid, y_valid, metric): # 이거 필요한 함수인가?
    """
    검증 데이터로 모델 평가

    Parameters
    ----------
    model : sklearn 모델
        평가할 학습된 모델
    X_valid : pd.DataFrame
        검증 입력 데이터
    y_valid : pd.Series
        검증 정답 레이블
    metric : str
        평가 메트릭 이름

    Returns
    -------
    score : float
        계산된 메트릭 값
    """
    y_pred = model.predict(X_valid)
    metric_fn = METRIC_FUNCTIONS[metric]
    score = metric_fn(y_valid, y_pred)
    return score


def test(args, model, data, setting, checkpoint=None):
    """
    테스트 데이터로 예측 수행

    Parameters
    ----------
    args : argparse.Namespace
        설정 파라미터
    model : sklearn 모델
        예측에 사용할 모델 (checkpoint 없을 경우)
    data : dict
        테스트 데이터 딕셔너리 ('test' 키 포함)
    setting : Setting
        실험 설정
    checkpoint : str, optional
        저장된 모델 경로 (None이면 방금 학습 된 모델 사용)

    Returns
    -------
    predicts : list
        예측값 리스트
    """
    if checkpoint:
        model = joblib.load(checkpoint)
    else:
        if args.train.save_best_model:
            # 각 fold별로 학습된 모델을 불러오기
            model_path_list = sorted(glob.glob(f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_*.pkl'))
            print(f"불러올 모델 개수: {len(model_path_list)}")

    X_test = data['test']
    fold_predictions = [] # 예측값 저장할 리스트

    # 모델을 하나씩 불러와서 예측하기
    for path in model_path_list:
        model = joblib.load(path)
        predicts = model.predict(X_test)
        fold_predictions.append(predicts)
    
    # 평균으로 앙상블 하기
    predicts = np.mean(fold_predictions, axis = 0)

    return predicts