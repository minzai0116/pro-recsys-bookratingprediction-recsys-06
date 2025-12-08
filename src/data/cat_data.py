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
    for i in range(len(res) - 1, 0, -1):
        if (res[i] in res[:i]) and (not pd.isna(res[i])):
            res.pop(i)
    return res


def process_context_data(users, books):
    users_ = users.copy()
    books_ = books.copy()

    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else np.nan)
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
    - 범주형/숫자형 변수를 명확히 구분하여 처리
    """
    # 1. 데이터 로드
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    emb_cols = []   # 임베딩 피쳐 사용시 필요해서 미리 선언, 사용 안해도 빈 리스트라서 문제 없음
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    # 1.1 요약 임베딩 로드 (books와 동일 순서로 정렬 해놨어요~) + 붙여주기~
    if args.use_summary_feature:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        summary_embeddings = np.load(args.dataset.data_path + 'text_vector/summary_embeddings.npy')

        # 결측치 찾기 (summary 있는 책만)
        has_summary = books['summary'].notna()
        print(f">>> Summary 있는 책: {has_summary.sum()} / {len(books)} ({has_summary.mean() * 100:.2f}%)")

        # 1단계: summary 있는 것만으로 scaler fit
        scaler = StandardScaler()
        scaler.fit(summary_embeddings[has_summary])
        summary_embeddings_scaled = scaler.transform(summary_embeddings)

        # 2단계: summary 있는 것만으로 PCA fit
        pca = PCA(n_components=50)
        pca.fit(summary_embeddings_scaled[has_summary])
        summary_embeddings_pca = pca.transform(summary_embeddings_scaled)

        print(f">>> Embedding Processing")
        print(f"원본 차원: {summary_embeddings.shape[1]}")
        print(f"PCA 후 차원: {summary_embeddings_pca.shape[1]}")
        print(f"설명 분산 비율: {pca.explained_variance_ratio_.sum():.4f}")

        # DataFrame 변환 후 결측치는 NaN으로
        embedding_dim = summary_embeddings_pca.shape[1]
        emb_cols = [f'summary_pca_{i}' for i in range(embedding_dim)]
        emb_df = pd.DataFrame(summary_embeddings_pca, columns=emb_cols)
        emb_df.loc[~has_summary] = np.nan  # 결측치는 NaN

        books = pd.concat([books.reset_index(drop=True), emb_df], axis=1)

    # 2. 베이스라인 전처리 수행 (context_data.py logic)
    users_, books_ = process_context_data(users, books)

    # 3. 피처 정의 (범주형 vs 연속형 명확히 구분)
    user_categorical = ['user_id', 'location_country', 'location_state', 'location_city']
    user_numeric = ['age']

    book_categorical = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'category']
    book_numeric = ['year_of_publication']

    categorical_cols = user_categorical + book_categorical
    numeric_cols = user_numeric + book_numeric
    all_cols = categorical_cols + numeric_cols

    # 4. 데이터 병합
    train_df = train.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')
    test_df = test.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')

    # 5. [CatBoost용 변환] 범주형/숫자형 분리 처리
    train_X = train_df[all_cols].copy()
    test_X = test_df[all_cols].copy()

    # 범주형: 문자열로 변환 (CatBoost가 자동으로 처리)
    for col in categorical_cols:
        train_X[col] = train_X[col].fillna('unknown').astype(str)
        test_X[col] = test_X[col].fillna('unknown').astype(str)

    # 숫자형: 숫자 타입 유지, 결측치만 처리
    for col in numeric_cols:
        train_X[col] = train_X[col].fillna(-1).astype(int)
        test_X[col] = test_X[col].fillna(-1).astype(int)

    data = {
        'train': train_X,
        'train_y': train_df['rating'],
        'test': test_X,
        'categorical_features': categorical_cols,  # CatBoost cat_features 파라미터용
        'numeric_features': numeric_cols,
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