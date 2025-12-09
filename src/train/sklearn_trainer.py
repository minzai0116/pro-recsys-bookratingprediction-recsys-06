import numpy as np
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

METRIC_FUNCTIONS = {
    'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'rmseloss': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'mseloss': mean_squared_error,
    'maeloss': mean_absolute_error,
}


def normalize_metric_name(metric):
    """
    메트릭 이름 정규화

    Parameters
    ----------
    metric : str
        메트릭 이름 (예: 'RMSELoss', 'rmse')

    Returns
    -------
    str
        정규화된 메트릭 이름 (소문자, 'loss' 제거)
    """
    return metric.lower().replace('loss', '')


def prepare_fit_params(args, data):
    """
    모델 학습을 위한 fit 파라미터 준비

    Parameters
    ----------
    args : argparse.Namespace
        설정 파라미터
    data : dict
        학습/검증 데이터 딕셔너리

    Returns
    -------
    fit_params : dict
        모델별 fit에 전달할 파라미터 (eval_set 등)
    """

    fit_params = {}

    if args.dataset.valid_ratio != 0:
        X_valid, y_valid = data['X_valid'], data['y_valid']

        if args.model in ['CatBoost', 'LightGBM']:
            fit_params['eval_set'] = [(X_valid, y_valid)]

    return fit_params


def calculate_metrics(model, X, y, metrics):
    """
    여러 메트릭 계산

    Parameters
    ----------
    model : sklearn 모델
        예측을 수행할 학습된 모델
    X : pd.DataFrame
        입력 데이터
    y : pd.Series
        정답 레이블
    metrics : list
        계산할 메트릭 이름 리스트

    Returns
    -------
    results : dict
        메트릭명을 key, 계산값을 value로 하는 딕셔너리
    """

    pred = model.predict(X)
    results = {}

    for metric in metrics:
        metric_normalized = normalize_metric_name(metric)
        metric_fn = METRIC_FUNCTIONS[metric_normalized]
        results[metric_normalized] = metric_fn(y, pred)

    return results


def log_feature_importance(args, model, data, importance_values=None):
    """
    Feature Importance 계산 및 로깅 (CatBoost, LightGBM만 지원)

    Parameters
    ----------
    args : argparse.Namespace
        설정 파라미터
    model : sklearn 모델
        학습된 모델 (feature_importances_ 속성 필요)
    data : dict
        학습 데이터 딕셔너리 (feature_names 포함)

    Returns
    -------
    None
    """
    if args.model not in ['CatBoost', 'LightGBM']:
        return

    # 중요도 값 결정 로직
    if importance_values is not None:
        # 직접 넣어준 값이 있으면 그걸 씀 
        importance = importance_values
    elif hasattr(model, 'feature_importances_'):
        # 없으면 모델에서 꺼내 씀 
        importance = model.feature_importances_
    else:
        # 둘 다 없으면 그냥 종료
        return

    print("\n>>> Top 10 Feature Importance")

    if importance_values is None:
        X_train = data['X_train']
    else:
        X_train = data['train']
    feature_names = data.get('feature_names', X_train.columns.tolist())

    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    fi_df = fi_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

    print(fi_df.head(10))

    # WandB에 전송
    if args.wandb:
        import wandb
        wandb.log({
            "feature_importance": wandb.plot.bar(
                wandb.Table(dataframe=fi_df.head(10)),
                "feature",
                "importance",
                title="Top 10 Feature Importance"
            )
        })


def save_model(args, model, setting):
    """모델 저장"""
    if not args.train.save_best_model:
        return

    os.makedirs(args.train.ckpt_dir, exist_ok=True)
    model_path = f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}.pkl'
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
    fit_params = prepare_fit_params(args, data)
    model.fit(X_train, y_train, **fit_params)

    # 2. Train metric 계산
    main_metric = args.metrics[0]
    train_results = calculate_metrics(model, X_train, y_train, [main_metric])
    train_score = train_results[normalize_metric_name(main_metric)]

    msg = f'\tTrain {normalize_metric_name(main_metric).upper()}: {train_score:.3f}'

    # 3. Validation (있을 경우)
    if args.dataset.valid_ratio != 0:
        X_valid, y_valid = data['X_valid'], data['y_valid']

        # 메인 metric
        valid_results = calculate_metrics(model, X_valid, y_valid, args.metrics)
        main_metric_normalized = normalize_metric_name(main_metric)
        valid_score = valid_results[main_metric_normalized]

        msg += f'\n\tValid {main_metric_normalized.upper()}: {valid_score:.3f}'

        # 추가 metrics
        valid_metrics = {}
        for metric in args.metrics[1:]:
            metric_normalized = normalize_metric_name(metric)
            valid_metrics[f'Valid {metric_normalized.upper()}'] = valid_results[metric_normalized]

        for metric_name, value in valid_metrics.items():
            msg += f' | {metric_name}: {value:.3f}'

        print(msg)

        # Logger
        logger.log(epoch=1, train_loss=train_score, valid_loss=valid_score, valid_metrics=valid_metrics)

        # WandB
        if args.wandb:
            wandb.log({
                f'Train {main_metric_normalized.upper()}': train_score,
                f'Valid {main_metric_normalized.upper()}': valid_score,
                **valid_metrics
            })
    else:
        print(msg)
        logger.log(epoch=1, train_loss=train_score)

        if args.wandb:
            wandb.log({f'Train {normalize_metric_name(main_metric).upper()}': train_score})

    # 4. Feature Importance
    log_feature_importance(args, model, data)

    # 5. 모델 저장
    save_model(args, model, setting)

    logger.close()

    return model


def valid(model, X_valid, y_valid, metric):
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
            model_path = f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}.pkl'
            model = joblib.load(model_path)

    X_test = data['test']
    predicts = model.predict(X_test)

    return predicts.tolist()