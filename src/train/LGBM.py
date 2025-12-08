import pandas as pd
import numpy as np
import lightgbm as lgb
import wandb
from wandb.integration.lightgbm import wandb_callback
from sklearn.metrics import mean_squared_error


def run_lightgbm(args, data, setting):
    '''
    LigthGBM의 전체 프로세스 실행 (학습 -> 예측 -> 저장)
    '''
    print(f"-------- {args.model} PROCESSING --------")

    # 1. 데이터 준비
    print(">>> Loading Data...")
    X_train = data['X_train'][data['field_names']]
    y_train = data['y_train']

    ## validation 데이터가 있는지 확인
    if args.dataset.valid_ratio != 0:
        X_val = data['X_valid'][data['field_names']]
        y_val = data['y_valid']
        eval_set = [(X_val, y_val)]
    else:
        eval_set = None

    X_test = data['test'][data['field_names']]

    ## 범주형 변수 지정
    cat_features = data['field_names']
    print(f">>> Categorical Features: {cat_features}")

    # 2. 파라미터 로드
    lgbm_params = args.model_args['LightGBM']

    # 3. 모델 정의
    model = lgb.LGBMRegressor(
        n_estimators = lgbm_params.n_estimators,
        learning_rate = lgbm_params.learning_rate,
        num_leaves = lgbm_params.num_leaves,
        max_depth = lgbm_params.max_depth,
        objective = lgbm_params.objective,
        random_state = args.seed,
        verbosity = lgbm_params.verbosity,
        n_jobs = -1
    )

    ## WandB Callbacks 설정
    callbacks = [
        lgb.early_stopping(stopping_rounds = lgbm_params.early_stopping_rounds),
        lgb.log_evaluation(period = 100)
    ]

    if args.wandb:
        callbacks.append(wandb_callback())

    # 4. 학습
    print(f">>> Start Training LightGBM (Metric: {lgbm_params.metric})")
    model.fit(
        X_train, y_train,
        eval_set = eval_set,
        eval_metric = lgbm_params.metric,
        categorical_feature = cat_features,
        callbacks = callbacks
    )

    ## Feature Importance 상위 10개 출력 (참고용)
    print(">>> Top 10 Feature Importance")
    importance = model.booster_.feature_importance(importance_type = 'gain') # lgbm의 feature importance 계산 방식의 default 값은 split
    feature_names = model.feature_name_

    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    fi_df = fi_df.sort_values(by = 'importance', ascending = False).reset_index(drop = True)

    print(fi_df.head(10))

    ## Feature Importance를 WandB에 그래프로 저장
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
    predicts = np.clip(predicts, 1, 10) # clip하는게 좋을까?

    ## 검증 점수 출력 (참고용)
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

    print(">>> LightGBM Process Finished")