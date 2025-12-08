import pandas as pd
import numpy as np
import regex
import re
from sklearn.model_selection import train_test_split

# 1. Helper Functions (context_data.py 로직 100% 동일)

def str2list(x: str) -> list:
    '''문자열을 리스트로 변환하는 함수'''
    return x[1:-1].split(', ')

def split_location(x: str) -> list:
    '''location 데이터를 나눈 뒤, 정제한 결과를 반환합니다.'''
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
    '''context_data.py의 핵심 전처리 로직'''
    users_ = users.copy()
    books_ = books.copy()

    # Books 전처리
    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else np.nan)
    books_['language'] = books_['language'].fillna(books_['language'].mode()[0])
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)

    # Users 전처리
    users_['age'] = users_['age'].fillna(users_['age'].mode()[0])
    users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10)

    # Location 전처리 & 결측치 보간 (기존 로직 유지)
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

# 2. Main Data Load (LightGBM용)

def lightgbm_data_load(args):
    """
    LightGBM용 데이터 로드 및 전처리
    - context_data.py의 로직을 그대로 사용
    - 결과물: DataFrame (Label Encoding 적용된 상태)
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
    # context_data.py에서 정의한 컬럼 리스트
    user_features = ['user_id', 'age_range', 'location_country', 'location_state', 'location_city']
    book_features = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'category', 'publication_range']
    sparse_cols = list(set(user_features + book_features))

    train_df = train.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')
    test_df = test.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')

    # 4. [LightGBM용 변환] Label Encoding
    # basic_data.py와 context_data.py에서 사용하는 방식과 동일하게 처리
    all_df = pd.concat([train_df[sparse_cols], test_df[sparse_cols]], axis=0)
    
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        # LightGBM은 정수형(Label Encoding)을 선호하므로 cat.codes 사용
        all_df[col] = all_df[col].astype("category").cat.codes

    # 다시 Train/Test 분리
    train_X = all_df.iloc[:len(train_df)].reset_index(drop=True)
    test_X = all_df.iloc[len(train_df):].reset_index(drop=True)

    data = {
        'train': train_X,
        'train_y': train_df['rating'],
        'test': test_X,
        'field_names': sparse_cols, # 모든 컬럼이 인코딩된 범주형
        'sub': sub
    }
    return data

def lightgbm_data_split(args, data):
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