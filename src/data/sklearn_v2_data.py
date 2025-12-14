import pandas as pd
import numpy as np
import regex
from sklearn.model_selection import train_test_split

def sklearn_v2_data_load(args):
    '''
    Parameters
    ----------
    args : argparse.Namespace
        설정 파라미터 (dataset, model_args, data_args 등 포함)

    Returns
    -------
    data : dict
        전처리된 데이터 딕셔너리
        - train : pd.DataFrame, 학습 데이터 (features)
        - train_y : pd.Series, 학습 데이터 레이블
        - test : pd.DataFrame, 테스트 데이터 (features)
        - feature_names : list, 전체 피처명 리스트
        - categorical_features : list, 범주형 피처명 리스트
        - numeric_features : list, 숫자형 피처명 리스트
        - sub : pd.DataFrame, 제출 파일 템플릿
    '''
    # 1. 데이터 로드
    print(">>> Loading Data...")
    train_df = pd.read_csv(args.dataset.data_path + 'preprocessed_data_v1.csv')
    test_df = pd.read_csv(args.dataset.data_path + 'preprocessed_test_data_v1.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    # 2. 피처 정의 -> args에서 지정으로 따로 빼려다가,
    # 아예 이 파트만 하드코딩하고 나머지를 재사용 하기로 함 팀장이 args에서 이래저래 길게 주는거 싫어함
    categorical_cols = ['user_id', 'loc_country', 'loc_state', 'loc_city', 'isbn', 'book_title', 'book_author', 'publisher', 'language', 'category']
    numeric_cols = ['year_of_publication', 'age', 'book_rating_count', 'user_rating_count', 'author_rating_count']

    all_cols = categorical_cols + numeric_cols

    # 3. 데이터 타입 변환
    train_X = train_df[all_cols].copy()
    test_X = test_df[all_cols].copy()

    # 범주형: 문자열로 변환
    for col in categorical_cols:
        train_X[col] = train_X[col].astype(str)
        test_X[col] = test_X[col].astype(str)

    # 숫자형: 숫자 타입 유지... 즉 아무것도 안 함

    data = {
        'train': train_X,
        'train_y': train_df['rating'],
        'test': test_X,
        'feature_names': all_cols,
        'categorical_features': categorical_cols,
        'numeric_features': numeric_cols,
        'sub': sub
    }

    return data