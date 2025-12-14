import catboost

def CatBoost(args, data):
    """CatBoost 모델 생성"""

    # data에서 numeric_features 가져오기 (전처리에서 미리 결정됨)
    numeric_features = data.get('numeric_features', [])
    all_features = data['feature_names']
    cat_features = [f for f in all_features if f not in numeric_features]

    model = catboost.CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        cat_features=cat_features,
        loss_function=args.loss_function,
        early_stopping_rounds=args.early_stopping_rounds,
        eval_metric=args.eval_metric,
        verbose=args.verbose
    )

    return model
