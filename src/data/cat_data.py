import pandas as pd
import numpy as np
import regex
import re
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. Helper Functions (lightgbm_data.py와 100% 동일하므로 복사 사용)
# =============================================================================
def str2list(x: str) -> list:
    return x[1:-1].split(', ')

def split_location(x: str) -> list:
    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]
    res = [i if i not in ['n/a', ''] else np.nan for i in res]
    res.reverse()
    for i in range(len(res)-1, 0, -1):
        if (res[i] in res[:i]) and (not pd.isna(res[i])):
            res.pop(i)
    return res

def process_context_data(users, books):
    users_ = users.copy()
    books_ = books.copy()

    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else np.nan)
    books_['language'] = books_['language'].fillna(books_['language'].mode()[0])
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)

    users_['age'] = users_['age'].fillna(users_['age'].mode()[0])
    users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10)

    users_['location_list'] = users_['location'].apply(lambda x: split_location(x)) 
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0])
    users_['location_state'] = users_['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    users_['location_city'] = users_['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
    
    for idx, row in users_.iterrows():
        if (not pd.isna(row['location_state'])) and pd.isna(row['location_country']):
            fill_country = users_[users_['location_state'] == row['location_state']]['location_country'].mode()
            fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
            users_.loc[idx, 'location_country'] = fill_country
        elif (not pd.isna(row['location_city'])) and pd.isna(row['location_state']):
            if not pd.isna(row['location_country']):
                fill_state = users_[(users_['location_country'] == row['location_country']) 
                                    & (users_['location_city'] == row['location_city'])]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                users_.loc[idx, 'location_state'] = fill_state
            else:
                fill_state = users_[users_['location_city'] == row['location_city']]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                fill_country = users_[users_['location_city'] == row['location_city']]['location_country'].mode()
                fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
                users_.loc[idx, 'location_country'] = fill_country
                users_.loc[idx, 'location_state'] = fill_state

    users_ = users_.drop(['location'], axis=1)
    return users_, books_

# =============================================================================
# 2. Main Data Load (CatBoost용)
# =============================================================================
def catboost_data_load(args):
    """
    CatBoost용 데이터 로드 및 전처리
    - context_data.py의 로직 그대로 사용
    - 결과물: DataFrame (CatBoost에 맞게 String으로 변환)
    """
    # 1. 데이터 로드
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    # 2. 베이스라인 전처리 수행 (context_data.py logic)
    users_, books_ = process_context_data(users, books)

    # 3. 데이터 병합
    user_features = ['user_id', 'age_range', 'location_country', 'location_state', 'location_city']
    book_features = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'category', 'publication_range']
    sparse_cols = list(set(user_features + book_features))

    train_df = train.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')
    test_df = test.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')

    # 4. [CatBoost용 변환] String 변환
    # Label Encoding을 하지 않고, 원본 문자열을 그대로 사용하는 것이 CatBoost에 유리함
    # (단, 전처리에 의해 생성된 age_range 같은 숫자는 문자로 변환)
    train_X = train_df[sparse_cols].copy()
    test_X = test_df[sparse_cols].copy()

    for col in sparse_cols:
        train_X[col] = train_X[col].fillna('unknown').astype(str)
        test_X[col] = test_X[col].fillna('unknown').astype(str)

    data = {
        'train': train_X,
        'train_y': train_df['rating'],
        'test': test_X,
        'field_names': sparse_cols, # CatBoost에 알려줄 범주형 변수 리스트
        'sub': sub
    }
    return data

def catboost_data_split(args, data):
    """basic_data_split과 동일 로직"""
    X_train, X_valid, y_train, y_valid = train_test_split(
        data['train'],
        data['train_y'],
        test_size=args.dataset.valid_ratio,
        random_state=args.seed,
        shuffle=True
    )
    data['X_train'], data['X_valid'] = X_train, X_valid
    data['y_train'], data['y_valid'] = y_train, y_valid
    return data