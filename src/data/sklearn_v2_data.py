import pandas as pd
import numpy as np
import regex
from sklearn.model_selection import train_test_split
"""
LOO, 즉 상호작용 마다 해당 책,아이템,저자의 자기 제외 평균 점수를 피쳐로 줍니다.
"""


def sklearn_v2_data_load(args):
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


def sklearn_v2_data_split(args, data):
    """
    학습/검증 데이터 분리 (user_id, isbn, rating만)
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        data['train'][['user_id', 'isbn']],
        data['train']['rating'],
        test_size=args.dataset.valid_ratio,
        random_state=args.seed,
        shuffle=True
    )

    data['X_train'] = X_train
    data['X_valid'] = X_valid
    data['y_train'] = y_train
    data['y_valid'] = y_valid

    return data


def sklearn_v2_data_preprocess(args, data):
    """
    전처리 메인 함수 - 각 단계를 호출
    """
    print(">>> Processing Context Data...")

    # 1. Users & Books 전처리
    users_processed = preprocess_users_location(data['users'])
    books_processed = data['books'].copy()

    # 2. 피처 정의
    user_categorical = ['user_id', 'location_country', 'location_state', 'location_city']
    user_numeric = ['age']
    book_categorical = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'category']
    book_numeric = ['year_of_publication']

    categorical_cols = user_categorical + book_categorical
    numeric_cols = user_numeric + book_numeric

    # 3. LOO 통계 계산 (Train용 LOO + Valid/Test용 전체 평균)
    train_loo_stats, valid_test_stats, global_mean = compute_rating_statistics_loo(
        data['X_train'],
        data['y_train'],
        books_processed,
        user_threshold=args.user_threshold,
        isbn_threshold=args.isbn_threshold,
        author_threshold=args.author_threshold
    )

    # 4. Train: LOO 통계 사용
    X_train_merged = data['X_train'].merge(users_processed, on='user_id', how='left').merge(books_processed, on='isbn', how='left')
    X_train_merged['user_mean_rating'] = train_loo_stats['user_mean_loo'].values
    X_train_merged['book_mean_rating'] = train_loo_stats['book_mean_loo'].values
    X_train_merged['author_mean_rating'] = train_loo_stats['author_mean_loo'].values

    # 5. Valid/Test: 전체 평균 사용
    X_valid_merged = merge_and_add_features(
        data['X_valid'],
        users_processed,
        books_processed,
        valid_test_stats,
        global_mean
    )

    test_merged = merge_and_add_features(
        data['test'][['user_id', 'isbn']],
        users_processed,
        books_processed,
        valid_test_stats,
        global_mean
    )

    # 6. 최종 데이터 준비
    all_cols = categorical_cols + numeric_cols + ['user_mean_rating', 'book_mean_rating', 'author_mean_rating']

    data.update(prepare_final_data(
        X_train_merged,
        X_valid_merged,
        test_merged,
        data['y_train'],
        categorical_cols,
        numeric_cols,
        all_cols
    ))

    print(f">>> Total features: {len(data['feature_names'])}")

    return data


# ========== 분리된 전처리 함수들 ==========

def preprocess_users_location(users):
    """
    Users location 전처리
    """
    print("    - Processing users location...")
    users_ = users.copy()

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

    return users_


def compute_rating_statistics_loo(X_train, y_train, books, user_threshold, isbn_threshold, author_threshold):
    """
    Train용 LOO 통계 + Valid/Test용 전체 평균 통계 계산
    """
    print("    - Computing rating statistics (LOO for train)...")

    train_with_rating = X_train.copy()
    train_with_rating['rating'] = y_train.values

    # Author 매핑
    isbn_to_author = books[['isbn', 'book_author']].set_index('isbn')['book_author'].to_dict()
    train_with_rating['author'] = train_with_rating['isbn'].map(isbn_to_author)

    global_mean = train_with_rating['rating'].mean()
    print(f"      Global mean: {global_mean:.4f}")

    # ===== Train용 LOO 통계 =====
    def loo_mean(group, threshold):
        # threshold < 0 이면 사용 안 함
        if threshold < 0:
            return np.nan

        if len(group) >= threshold:
            return (group.sum() - group) / (len(group) - 1)
        return np.nan

    train_with_rating['user_mean_loo'] = train_with_rating.groupby('user_id')['rating'].transform(
        lambda x: loo_mean(x, user_threshold)
    )
    train_with_rating['book_mean_loo'] = train_with_rating.groupby('isbn')['rating'].transform(
        lambda x: loo_mean(x, isbn_threshold)
    )
    train_with_rating['author_mean_loo'] = train_with_rating.groupby('author')['rating'].transform(
        lambda x: loo_mean(x, author_threshold)
    )

    train_loo_stats = train_with_rating[['user_mean_loo', 'book_mean_loo', 'author_mean_loo']]

    # ===== Valid/Test용 전체 평균 통계 =====
    # User
    user_rating_counts = train_with_rating.groupby('user_id').size()
    valid_users = user_rating_counts[user_rating_counts >= user_threshold].index
    user_mean_rating = train_with_rating[train_with_rating['user_id'].isin(valid_users)].groupby('user_id')[
        'rating'].mean().reset_index()
    user_mean_rating.columns = ['user_id', 'user_mean_rating']

    # ISBN
    isbn_rating_counts = train_with_rating.groupby('isbn').size()
    valid_isbns = isbn_rating_counts[isbn_rating_counts >= isbn_threshold].index
    book_mean_rating = train_with_rating[train_with_rating['isbn'].isin(valid_isbns)].groupby('isbn')[
        'rating'].mean().reset_index()
    book_mean_rating.columns = ['isbn', 'book_mean_rating']

    # Author
    author_rating_counts = train_with_rating.groupby('author').size()
    valid_authors = author_rating_counts[author_rating_counts >= author_threshold].index
    author_mean_rating = train_with_rating[train_with_rating['author'].isin(valid_authors)].groupby('author')[
        'rating'].mean().reset_index()
    author_mean_rating.columns = ['author', 'author_mean_rating']

    print(f"      Valid users: {len(user_mean_rating)}")
    print(f"      Valid books: {len(book_mean_rating)}")
    print(f"      Valid authors: {len(author_mean_rating)}")

    valid_test_stats = {
        'user': user_mean_rating,
        'book': book_mean_rating,
        'author': author_mean_rating
    }

    return train_loo_stats, valid_test_stats, global_mean


def merge_and_add_features(X, users, books, rating_stats, global_mean):
    """
    Context merge 및 rating feature 추가
    """
    X_merged = X.merge(users, on='user_id', how='left').merge(books, on='isbn', how='left')
    X_merged = add_rating_features(
        X_merged,
        rating_stats['user'],
        rating_stats['book'],
        rating_stats['author'],
        global_mean
    )
    return X_merged


def prepare_final_data(X_train, X_valid, test, y_train, categorical_cols, numeric_cols, all_cols):
    """
    최종 데이터 타입 변환 및 정리
    """
    print("    - Preparing final data...")

    # 데이터 타입 변환
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_valid[col] = X_valid[col].astype(str)
        test[col] = test[col].astype(str)

    result = {
        'X_train': X_train[all_cols],
        'X_valid': X_valid[all_cols],
        'test': test[all_cols],
        'train_y': y_train,
        'feature_names': all_cols,
        'categorical_features': categorical_cols,
        'numeric_features': numeric_cols + ['user_mean_rating', 'book_mean_rating', 'author_mean_rating']
    }

    return result


# ========== Helper Functions ==========

def split_location(x: str) -> list:
    '''
    location 데이터를 나눈 뒤, 정제한 결과를 반환
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


def add_rating_features(X, user_stats, book_stats, author_stats, global_mean):
    """
    User/Book/Author 평균 평점 feature 추가
    """
    X_new = X.copy()

    temp_df = pd.DataFrame({
        'user_id': X_new['user_id'],
        'isbn': X_new['isbn'],
        'book_author': X_new['book_author']
    })

    temp_df = temp_df.merge(user_stats, on='user_id', how='left')
    temp_df = temp_df.merge(book_stats, on='isbn', how='left')
    temp_df = temp_df.merge(author_stats, left_on='book_author', right_on='author', how='left')

    X_new['user_mean_rating'] = temp_df['user_mean_rating'].values
    X_new['book_mean_rating'] = temp_df['book_mean_rating'].values
    X_new['author_mean_rating'] = temp_df['author_mean_rating'].values

    return X_new