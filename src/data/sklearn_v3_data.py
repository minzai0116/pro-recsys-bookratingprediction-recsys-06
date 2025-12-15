import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''
김태형 코드 리뷰 요청에서, 필요한 부분이 있다고 판단되어 추가하였습니다.
'''

"""
지역 전처리가 이미 된 데이터를 받습니다. 
스레드홀드 이하의 상호작용이 있는 피쳐를 무시합니다.
"""


def sklearn_v3_data_load(args):
    '''
    데이터 로드만 수행
    '''
    print(">>> Loading Data...")
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    data = {
        'users': users,
        'books': books,
        'train': train,
        'test': test,
        'sub': sub
    }

    return data


def sklearn_v3_data_split(args, data):
    """
    학습/검증 데이터 분리 (user_id, isbn, rating만)
    """
    # rating 분리
    X = data['train'][['user_id', 'isbn']]
    y = data['train']['rating']

    if args.dataset.valid_ratio:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=args.dataset.valid_ratio,
            random_state=args.seed,
            shuffle=True
        )
        data['X_train'], data['X_valid'] = X_train, X_valid
        data['y_train'], data['y_valid'] = y_train, y_valid
    else:
        # valid_ratio가 0이면 전체 데이터를 train으로 사용
        data['X_train'] = X
        data['y_train'] = y
        data['X_valid'] = None
        data['y_valid'] = None

    return data


def remove_noise_features(users, books, train_ratings, threshold=1):
    """
    train_ratings 기준으로 상호작용이 threshold 이하인 카테고리 값 제거
    """
    users_ = users.copy()
    books_ = books.copy()

    print(f">>> Removing noise features (rating count <= {threshold})...")

    # train_ratings과 merge
    train_merged = train_ratings.merge(users_, on='user_id', how='left') \
        .merge(books_, on='isbn', how='left')

    # 실제 컬럼명에 맞게 수정
    user_categorical_features = ['loc_country', 'loc_state', 'loc_city']
    book_categorical_features = ['category', 'book_author', 'publisher', 'language']

    all_features = user_categorical_features + book_categorical_features

    for feature in all_features:
        if feature in train_merged.columns:
            # 각 카테고리 값의 등장 횟수 계산
            value_counts = train_merged[feature].value_counts()
            # threshold 이하인 값들 찾기
            noise_values = value_counts[value_counts <= threshold].index.tolist()

            if len(noise_values) > 0:
                print(f"    {feature}: {len(noise_values)}개 노이즈 값 제거 (전체 {len(value_counts)}개 중)")

                # users 또는 books에서 해당 값을 NaN으로 변경
                if feature in users_.columns:
                    users_.loc[users_[feature].isin(noise_values), feature] = np.nan
                elif feature in books_.columns:
                    books_.loc[books_[feature].isin(noise_values), feature] = np.nan

    return users_, books_


def add_count_features(X_train, X_valid, test, books, train_full):
    """
    train 기준으로 user, book, author의 rating count를 피처로 추가

    Parameters:
    - X_train, X_valid, test: 분할된 데이터셋 (이미 book_author 포함)
    - books: book 메타데이터 (book_author 포함)
    - train_full: 전체 train 데이터 (count 계산용)
    """
    print(">>> Adding count features...")

    # 1. Train에서 count 계산 (data leakage 방지)
    user_counts = train_full['user_id'].value_counts().to_dict()
    book_counts = train_full['isbn'].value_counts().to_dict()

    # 저자별 count를 위해 isbn -> author 매핑 필요
    train_with_author = train_full.merge(books[['isbn', 'book_author']], on='isbn', how='left')
    author_counts = train_with_author['book_author'].value_counts().to_dict()

    print(f"    User count range: {min(user_counts.values())} ~ {max(user_counts.values())}")
    print(f"    Book count range: {min(book_counts.values())} ~ {max(book_counts.values())}")
    print(f"    Author count range: {min(author_counts.values())} ~ {max(author_counts.values())}")

    # 2. 각 데이터셋에 count 피처 추가하는 함수
    def apply_counts(df):
        df = df.copy()

        # User rating count
        df['user_rating_count'] = df['user_id'].map(user_counts).fillna(0).astype(int)

        # Book rating count
        df['book_rating_count'] = df['isbn'].map(book_counts).fillna(0).astype(int)

        # Author rating count (이미 존재하는 book_author 사용)
        if 'book_author' in df.columns:
            df['author_rating_count'] = df['book_author'].map(author_counts).fillna(0).astype(int)
        else:
            print("    Warning: book_author not found in dataframe")
            df['author_rating_count'] = 0

        return df

    # 3. 각 데이터셋에 적용
    X_train = apply_counts(X_train)
    if X_valid is not None:
        X_valid = apply_counts(X_valid)
    test = apply_counts(test)

    return X_train, X_valid, test


def sklearn_v3_data_preprocess(args, data):
    """
    전처리 메인 함수 - 각 단계를 호출
    """
    print(">>> Processing Context Data...")

    # 1. 노이즈 피처 제거 (옵션)
    threshold = getattr(args, 'threshold', None)
    if threshold is not None and threshold > 0:
        users_processed, books_processed = remove_noise_features(
            data['users'],
            data['books'],
            data['train'],
            threshold=threshold
        )
    else:
        users_processed = data['users'].copy()
        books_processed = data['books'].copy()

    # 2. 기본 전처리
    users_processed = users_processed.drop('location', axis=1)

    # 3. 피처 정의
    user_categorical = ['user_id', 'loc_city', 'loc_state', 'loc_country']
    user_numeric = ['age']
    book_categorical = ['isbn', 'book_author', 'publisher', 'language', 'category']
    book_numeric = ['year_of_publication']

    # 4. Train 데이터 merge
    X_train_merged = data['X_train'].merge(users_processed, on='user_id', how='left').merge(
        books_processed, on='isbn', how='left'
    )

    # 5. Valid 데이터 merge (있는 경우만)
    if data['X_valid'] is not None:
        X_valid_merged = data['X_valid'].merge(users_processed, on='user_id', how='left').merge(
            books_processed, on='isbn', how='left'
        )
        y_valid = data['y_valid']
    else:
        X_valid_merged = None
        y_valid = None

    # 6. Test 데이터 merge
    test_merged = data['test'][['user_id', 'isbn']].merge(
        users_processed, on='user_id', how='left'
    ).merge(books_processed, on='isbn', how='left')

    # 7. Count 피처 추가 (옵션) ⭐
    add_counts = getattr(args, 'add_count_features', False)
    if add_counts:
        X_train_merged, X_valid_merged, test_merged = add_count_features(
            X_train_merged,
            X_valid_merged,
            test_merged,
            books_processed,
            data['train']
        )
        count_features = ['user_rating_count', 'book_rating_count', 'author_rating_count']
    else:
        count_features = []

    # 8. 최종 피처 목록
    categorical_cols = user_categorical + book_categorical
    numeric_cols = user_numeric + book_numeric + count_features
    all_cols = categorical_cols + numeric_cols

    # 9. 최종 데이터 준비
    data.update(prepare_final_data(
        X_train_merged,
        X_valid_merged,
        test_merged,
        data['y_train'],
        y_valid,
        categorical_cols,
        numeric_cols,
        all_cols
    ))

    print(f">>> Total features: {len(data['feature_names'])}")

    return data


def prepare_final_data(X_train, X_valid, test, y_train, y_valid, categorical_cols, numeric_cols, all_cols):
    """
    최종 데이터 타입 변환 및 정리
    """
    print("- Preparing final data...")

    # 데이터 타입 변환
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        if X_valid is not None:
            X_valid[col] = X_valid[col].astype(str)
        test[col] = test[col].astype(str)

    result = {
        'X_train': X_train[all_cols],
        'X_valid': X_valid[all_cols] if X_valid is not None else None,
        'test': test[all_cols],
        'train_y': y_train,
        'valid_y': y_valid,
        'feature_names': all_cols,
        'categorical_features': categorical_cols,
        'numeric_features': numeric_cols
    }

    return result