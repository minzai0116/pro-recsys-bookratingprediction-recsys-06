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
    """메트릭 이름 정규화 (RMSELoss -> rmse, rmse -> rmse)"""
    return metric.lower().replace('loss', '')


def prepare_fit_params(args, data):
    """fit 파라미터 준비"""
    fit_params = {}

    if args.dataset.valid_ratio != 0:
        X_valid, y_valid = data['X_valid'], data['y_valid']

        if args.model in ['CatBoost', 'LightGBM']:
            fit_params['eval_set'] = [(X_valid, y_valid)]

    return fit_params


def calculate_metrics(model, X, y, metrics):
    """메트릭 계산 (여러 개)"""
    pred = model.predict(X)
    results = {}

    for metric in metrics:
        metric_normalized = normalize_metric_name(metric)
        metric_fn = METRIC_FUNCTIONS[metric_normalized]
        results[metric_normalized] = metric_fn(y, pred)

    return results


def log_feature_importance(args, model, data):
    """Feature Importance 계산 및 로깅"""
    if args.model not in ['CatBoost', 'LightGBM']:
        return

    if not hasattr(model, 'feature_importances_'):
        return

    print("\n>>> Top 10 Feature Importance")

    X_train = data['X_train']
    feature_names = data.get('feature_names', X_train.columns.tolist())
    importance = model.feature_importances_

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
    sklearn 모델 학습

    Args:
        model: sklearn 모델 (fit 메서드 있어야 함)
        data: dict with 'X_train', 'y_train', 'X_valid', 'y_valid'
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
    """Validation 데이터로 평가"""
    y_pred = model.predict(X_valid)
    metric_fn = METRIC_FUNCTIONS[metric]
    score = metric_fn(y_valid, y_pred)
    return score


def test(args, model, data, setting, checkpoint=None):
    """
    테스트 데이터로 예측

    Args:
        data: dict with 'test' - 테스트 데이터
        checkpoint: 저장된 모델 경로 (None이면 방금 학습된 모델 사용)
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