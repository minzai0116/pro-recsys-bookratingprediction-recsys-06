import pandas as pd
import numpy as np
import catboost
import wandb
from catboost import Pool # feature importance 생성할 때 사용하는 데이터
from sklearn.metrics import mean_squared_error

def run_catboost(args, data, setting):
    '''
    CatBoost의 전체 프로세스 실행 (학습 -> 예측 -> 저장)
    '''
    print(f"-------- {args.model} PROCESSING --------")


    # 1. 데이터 준비
    print(">>> Loading Data...")

    # 모든 피처 리스트 생성
    all_features = data['categorical_features'] + data['numeric_features']

    X_train = data['X_train'][all_features]
    y_train = data['y_train']

    ## validation 데이터가 있는지 확인
    if args.dataset.valid_ratio != 0:
        X_val = data['X_valid'][data['field_names']]
        y_val = data['y_valid']
        eval_set = [(X_val, y_val)]
    else:
        eval_set = None
    
    X_test = data['test'][data['field_names']]

    ## 범주형 변수만 지정 (숫자형은 제외)
    cat_features = data['categorical_features']
    print(f">>> Categorical Features: {cat_features}")
    print(f">>> Numeric Features: {data['numeric_features']}")

    # 2. 파라미터 로드
    catboost_params = args.model_args['CatBoost']

    # 3. 모델 정의
    model = catboost.CatBoostRegressor(
        iterations=catboost_params.iterations,
        learning_rate=catboost_params.learning_rate,
        depth=catboost_params.depth,
        cat_features=cat_features,  # 범주형 변수만 전달
        loss_function=catboost_params.loss_function,
        early_stopping_rounds=catboost_params.early_stopping_rounds,
        eval_metric=catboost_params.eval_metric,
        verbose=catboost_params.verbose
    )

    # 4. 학습
    print(f">>> Start Training CatBoost (Metric: {catboost_params.eval_metric})")
    model.fit(
        X_train, y_train,
        eval_set = eval_set,
        use_best_model = catboost_params.use_best_model
    )

    ## CatBoost 학습 기록을 WandB에 전송
    if args.wandb:
        evals_result = model.get_evals_result()
        metric_name = catboost_params.eval_metric
        # evals_result 구조: {'learn': {'RMSE': [values...]}, 'validation': {'RMSE': [values...]}}

        train_history = evals_result['learn'][metric_name]
        valid_history = evals_result['validation'][metric_name] if 'validation' in evals_result else []

        for i in range(len(train_history)):
            log_dict = {
                'epoch': i + 1,
                f'Train {metric_name}': train_history[i]
            }
            if valid_history:
                log_dict[f'Valid {metric_name}'] = valid_history[i]

            wandb.log(log_dict)

    # Feature Importance 상위 10개 출력 (참고용)
    print(">>> Top 10 Feature Importance")
    val_pool = Pool(X_val, y_val, cat_features = cat_features)
    importance = model.get_feature_importance(data = val_pool, type = 'LossFunctionChange')
    feature_names = model.feature_names_

    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    fi_df = fi_df.sort_values(by = 'importance', ascending = False).reset_index(drop = True)

    print(fi_df.head(10))

    ## Feaeature Importance를 WandB에 그래프로 저장
    if args.wandb:
        wandb.log({
            "feature_importance": wandb.plot.bar(
                wandb.Table(dataframe = fi_df.head(10)), 
                "feature", # X축
                "importance", # Y축
                title = "Feature Importance"
            )
        })

    # 5. 예측 및 후처리
    print(">>> Predicting...")
    predicts = model.predict(X_test)
    predicts = np.clip(predicts, 1, 10)

    # 검증 점수 출력 (참고용)
    if eval_set:
        val_preds = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"⭐️ >>> Final Validation RMSE: {val_rmse:.4f}")
    
    # 6. 결과 저장
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')
    submission['rating'] = predicts

    filename = setting.get_submit_filename(args)
    print(f"Save Predict: {filename}")
    submission.to_csv(filename, index = False)

    print(f">>> {args.model} Process Finished")

