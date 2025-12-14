import pandas as pd
import numpy as np
import regex
from sklearn.model_selection import train_test_split

def sklearn_v1_data_load(args):
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
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    # 솔직히 여기가 제일마음에 안들긴 함, 뭐냐면 각 데이터 타입에서의 아규먼트 쓰겠다는거임, 그대로 안 준 이유는
    # 오직 위에서 데이터셋을 써야 하기 때문임
    datatype = args.model_args[args.model]['datatype'] # 이 데이터 타입
    args = args.data_args[datatype] # 그래서 이 데이터 타입에 할당된 아규먼트

    # 요약 임베딩 로드 (옵션)
    emb_cols = []
    if args.use_summary_feature:
        books, emb_cols = load_summary_embeddings(args, books)

    # 2. 전처리 수행
    print(">>> Processing Context Data...")
    users_, books_ = process_sklearn_v1_data(users, books)

    if args.remove_noise:
        users_, books_ = remove_noise_features(users_, books_, train, threshold=args.threshold)

    # 3. 피처 정의 -> args에서 지정으로 따로 빼려다가,
    # 아예 이 파트만 하드코딩하고 나머지를 재사용 하기로 함 팀장이 args에서 이래저래 길게 주는거 싫어함
    user_categorical = ['user_id', 'location_country', 'location_state', 'location_city']
    user_numeric = ['age']
    book_categorical = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'category']
    book_numeric = ['year_of_publication']

    categorical_cols = user_categorical + book_categorical
    numeric_cols = user_numeric + book_numeric + emb_cols
    all_cols = categorical_cols + numeric_cols

    # 4. 데이터 병합
    train_df = train.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')
    test_df = test.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')

    # 5. 데이터 타입 변환
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


def sklearn_v1_data_split(args, data):
    """
    학습/검증 데이터 분리

    Parameters
    ----------
    args : argparse.Namespace
        설정 파라미터
    data : dict
        전체 데이터 딕셔너리

    Returns
    -------
    data : dict
        분리된 데이터가 추가된 딕셔너리
    """
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

def str2list(x: str) -> list:
    '''문자열을 리스트로 변환하는 함수'''
    return x[1:-1].split(', ')


def split_location(x: str) -> list:
    '''
    Parameters
    ----------
    x : str
        location 데이터

    Returns
    -------
    res : list
        location 데이터를 나눈 뒤, 정제한 결과를 반환합니다.
        순서는 country, state, city, ... 입니다.
    '''
    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]
    res = [i if i not in ['n/a', ''] else np.nan for i in res]
    res.reverse()
    for i in range(len(res) - 1, 0, -1):
        if (res[i] in res[:i]) and (not pd.isna(res[i])):
            res.pop(i)
    return res


def remove_noise_features(users, books, train_ratings, threshold=1):
    """
    train_ratings 기준으로 상호작용이 threshold 이하인 카테고리 값 others로 
    """
    users_ = users.copy()
    books_ = books.copy()

    print(f">>> Removing noise features (rating <= {threshold})...")

    train_merged = train_ratings.merge(users_, on='user_id', how='left') \
        .merge(books_, on='isbn', how='left')

    categorical_features = ['category', 'location_country', 'location_state', 'location_city',
                            'book_author', 'publisher', 'language']

    for feature in categorical_features:
        if feature in train_merged.columns:
            value_counts = train_merged[feature].value_counts()
            noise_values = value_counts[value_counts <= threshold].index.tolist()

            print(f"    {feature}: {len(noise_values)}개 노이즈 값 제거 (전체 {len(value_counts)}개 중)")

            if feature in users_.columns:
                users_.loc[users_[feature].isin(noise_values), feature] = np.nan
            elif feature in books_.columns:
                books_.loc[books_[feature].isin(noise_values), feature] = np.nan

    return users_, books_


def process_sklearn_v1_data(users, books):
    """
    지역 -> 국가,주,도시

    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터

    Returns
    -------
    users_, books_ : tuple
        전처리된 users, books 데이터프레임
    """
    users_ = users.copy()
    books_ = books.copy()

    users_['location_list'] = users_['location'].apply(lambda x: split_location(x))
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0])
    users_['location_state'] = users_['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    users_['location_city'] = users_['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)

    # location 결측치 보완
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

    users_ = users_.drop(['location', 'location_list'], axis=1)

    # book 결측치 보완
    books_['book_author'] = books_['book_author'].replace('Not Applicable (Na )', np.nan)

    return users_, books_


def load_summary_embeddings(args, books):
    """
    요약 임베딩 로드 및 PCA 적용

    Parameters
    ----------
    args : argparse.Namespace
        설정 파라미터
    books : pd.DataFrame
        책 데이터프레임

    Returns
    -------
    books_with_emb : pd.DataFrame
        임베딩이 추가된 책 데이터프레임
    emb_cols : list
        임베딩 컬럼명 리스트
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    summary_embeddings = np.load(args.dataset.data_path + 'text_vector/summary_embeddings.npy')

    # 결측치 찾기
    has_summary = books['summary'].notna()
    print(f">>> Summary 있는 책: {has_summary.sum()} / {len(books)} ({has_summary.mean() * 100:.2f}%)")

    # StandardScaler 적용
    scaler = StandardScaler()
    scaler.fit(summary_embeddings[has_summary])
    summary_embeddings_scaled = scaler.transform(summary_embeddings)

    # PCA 적용
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
    emb_df.loc[~has_summary] = np.nan

    books_with_emb = pd.concat([books.reset_index(drop=True), emb_df], axis=1)

    return books_with_emb, emb_cols

