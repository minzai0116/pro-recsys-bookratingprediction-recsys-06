import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import regex
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer, AutoModel
from .basic_data import basic_data_split

import pickle


### process_context_data 전용 fragment function ###

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

def text_preprocessing(summary):
    """
    Parameters
    ----------
    summary : pd.Series
        정규화와 같은 기본적인 전처리를 하기 위한 텍스트 데이터를 입력합니다.
    
    Returns
    -------
    summary : pd.Series
        전처리된 텍스트 데이터를 반환합니다.
        베이스라인에서는 특수문자 제거, 공백 제거를 진행합니다.
    """
    summary = re.sub("[^0-9a-zA-Z.,!?]", " ", summary)  # .,!?를 제외한 특수문자 제거
    summary = re.sub("\s+", " ", summary)  # 중복 공백 제거

    return summary

# 참고로 여기서 summary index 생성
def process_context_data(users, books):
    '''
    
    ### input ###
        users = df.read_csv("books.csv")
        books = df.read_csv("users.csv")
        
    ### output ###
        * users_
            - location을 city, state, country로 나눠서 새로운 열로 할당
            - age는 기준년 단위(10)로 나눠서 카테고리화
        
        * books_
            - year_of_publication를 기준년 단위(10)로 나눠서 카테고리화
            - isbn을 자연수로 변환 -> 나중에 summary의 index를 찾아오기 위함
    '''
    users_ = users.copy()
    books_ = books.copy()
    
    age_categorizer = 10
    year_of_publication_categorizer = 10

    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else np.nan)
    books_['language'] = books_['language'].fillna(books_['language'].mode()[0])
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // year_of_publication_categorizer * year_of_publication_categorizer)

    users_['age'] = users_['age'].fillna(users_['age'].mode()[0])
    users_['age_range'] = users_['age'].apply(lambda x: x // age_categorizer * age_categorizer)

    users_['location_list'] = users_['location'].apply(lambda x: split_location(x)) 
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0])
    users_['location_state'] = users_['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    users_['location_city'] = users_['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
    
    for idx, row in users_.iterrows():
        # state는 있는데 country가 없으면, 같은 state의 사람들의 최빈값으로 채움
        if (not pd.isna(row['location_state'])) and pd.isna(row['location_country']):
            fill_country = users_[users_['location_state'] == row['location_state']]['location_country'].mode()
            fill_country = fill_country[0] if len(fill_country) > 0 else "Unknown"
            users_.loc[idx, 'location_country'] = fill_country
        # city는 있는데 state가 없으면
        elif (not pd.isna(row['location_city'])) and pd.isna(row['location_state']):
            # country가 있다면 city와 country가 같은 사람들의 최빈값으로 채움
            if not pd.isna(row['location_country']):
                fill_state = users_[(users_['location_country'] == row['location_country']) 
                                    & (users_['location_city'] == row['location_city'])]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else "Unknown"
                users_.loc[idx, 'location_state'] = fill_state
            # country도 없으면, 같은 city 사람들의 state와 country 최빈값으로 채움.
            else:
                fill_state = users_[users_['location_city'] == row['location_city']]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else "Unknown"
                fill_country = users_[users_['location_city'] == row['location_city']]['location_country'].mode()
                fill_country = fill_country[0] if len(fill_country) > 0 else "Unknown"
                users_.loc[idx, 'location_country'] = fill_country
                users_.loc[idx, 'location_state'] = fill_state
        
    # category index로 변환해도 문제 없게 nan 처리
    users_ = users_.replace({np.nan: "unknown"})
    books_ = books_.replace({np.nan: "unknown"})
    
    # summary를 찾기위한 index 추가.
    books_["summary_index"] = books_["isbn"].astype("category").cat.codes
    
    return users_, books_


# 학습에 입력시킬 열만 남기기 + [CLS] 토큰 추가 + 각 원소를 고유 index로 변환
def remain_train_features_only(rating):
    '''
    ##################
    
    input = rating에 books와 users를 병합한 dataframe
    
    output = [summary_index, CLS, rating, user_id, country, state, city..... ]
    
    ##################
    
    특이사항 :   summary_index는 맨 앞에, CLS는 그 뒤에 추가됨.
                summary index == bert에서 summary_vector 탐색용 index
                CLS == 출력 vector를 편하게 찾기 위해 맨 앞에 붙임. 
    '''
    
    
    # 남길 열    
    # summary는 미리 부여한 고유한 값이라 바뀌면 안 됨.
    # rating(=score)는 ground_truth(=Y)값이므로 바뀌면 안 됨.
    summary_index = rating['summary_index']
    scores = rating['rating']
    
    col_to_left = ['user_id', 
                   'age_range', 
                   'location_country', 
                   'location_state', 
                   'location_city', 
                   'book_author', 
                   'publisher',
                   'language', 
                   'category', 
                   'publication_range',
                   ]

    rating = rating.drop(columns=[c for c in rating if c not in col_to_left])

    # cls 토큰 미리 입력
    rating.insert(0, "CLS", 0)
    
    # category index로 변경 : 각 열에서의 고유값 할당
    rating = rating.apply(lambda col: col.astype("category").cat.codes)
    
    
    # 각 열의 고유값 갯수를 list로 == offset 계산을 위해
    col_len_set = rating.nunique().tolist()
    
    # 각 열의 실제 offset 계산
    total = 1
    off_set = [0]
    for x in col_len_set[:-1]:
        off_set.append(off_set[-1] + x)
    
    df_offset = rating.copy()
    for col, offset in zip(rating.columns, off_set):
        df_offset[col] = (rating[col].astype("int32") + offset).astype("int32")
    
    # offset = 현 dataframe에서의 고유 index를 부여하는 행위
    # 이 고유값은 embedding index로 활용된다.
    df_offset.insert(0, "summary_index", summary_index)
    df_offset.insert(2, "rating", scores)
    
    # 맨 처음이 summary_index, 다음은 cls token
    # df_offset = [summary_index, cls, rating, user_id, country, state, city.....]
    return df_offset


# only once per model
# summary를 미리 vector로 저장해놓는 행위
# 나중에 summary -> vector로 검색할 수 있게
# isbn으로 고유 번호 지정
# book_summary_vector_list[isbn] = sumnary_vector
def text_to_vector(books, args):
    print('--------------- Preparing Summary Vector ---------------')
    
    """
    Parameters
    ----------
    text : str
        `summary_merge()`를 통해 병합된 요약 데이터
    tokenizer : Tokenizer
        텍스트 데이터를 `model`에 입력하기 위한 토크나이저
    model : 사전학습된 언어 모델
        텍스트 데이터를 벡터로 임베딩하기 위한 모델
    ----------
    """
    
    books_ = books.copy()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_args.bert_rec.pretrained_model)
    model = AutoModel.from_pretrained(args.model_args.bert_rec.pretrained_model).to(device=args.device)
    model.eval()
    book_summary_vector_list = {}
    
    test_idx = 0
    
    for title, summary, book_id in tqdm(zip(books_['book_title'], books_['summary'], books_['summary_index']), total=len(books_)):
        # 책에 대한 텍스트 프롬프트는 아래와 같이 구성됨
        # '''
        # Book Title: {title}
        # Summary: {summary}
        # '''
        
        prompt_ = f'Book Title: {title}\n Summary: {summary}\n'
        prompt_ = "[CLS] " + prompt_ + " [SEP]"
        
        tokenized = tokenizer.encode(prompt_, add_special_tokens=True)
        token_tensor = torch.tensor([tokenized], device=model.device)
        
        with torch.no_grad():
            outputs = model(token_tensor)  # attention_mask를 사용하지 않아도 됨
            ### BERT 모델의 경우, 최종 출력물의 사이즈가 (토큰길이, 임베딩=768)이므로, 이를 평균내어 사용하거나 pooler_output을 사용하여 [CLS] 토큰의 임베딩만 사용
            # sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0)  # 방법1) 모든 토큰의 임베딩을 평균내어 사용
            sentence_embedding = outputs.pooler_output.squeeze(0).cpu().detach().numpy()
            #print(sentence_embedding)
            
            book_summary_vector_list[book_id] = sentence_embedding
        
        test = False
        test_idx+=1
        if test_idx > 3 and test:
            for idx, i in enumerate(book_summary_vector_list):
                print(i, book_summary_vector_list[i][:5])
            break
            
    with open(f"{args.dataset.data_path}" + "/summary_vector/summaries.pkl", "wb") as f:
        pickle.dump(book_summary_vector_list, f)
    #np.save('./data/text_vector/book_summary_vector.npy', book_summary_vector_list)  
        
    return book_summary_vector_list

# 나눠주는거 main에서도 split 불러와서 해주고 있음. (???) 뭐지
def bert_data_split(args, data):
    '''data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.'''
    return basic_data_split(args, data)

# 원본 데이터를 bert에 맞게 전처리해서 반환
def bert_data_load(args):
    '''
        - csv 불러와서, split까지 하는 함수
        1) csv 읽어오고
        2) 전처리함수 거치고 ( + summart_vect가 없으면 생성)
        3) rating에 user_id, isbn 기준으로 merge
        4) 
    '''
    
    # 1. 데이터 로드
    users_ = pd.read_csv(args.dataset.data_path + 'users.csv')
    books_ = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')


    # 2. 베이스라인 전처리 수행
    users_, books_ = process_context_data(users_, books_)

    # summary를 vector화 하는 함수, 여기는 summary가 있으면 실행 안 됨.
    if not args.model_args.bert_rec.prepared_summary:
        text_to_vector(books_, args)

    # 3. 데이터 병합
    train_df = train.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')
    test_df = test.merge(users_, on='user_id', how='left').merge(books_, on='isbn', how='left')
    
    # 4. 필요없는 column drop + index로 변환 (offset 적용상태)
    train_df = remain_train_features_only(train_df)
    test_df = remain_train_features_only(test_df)
    
    test_df = test_df.drop('rating', axis = 1)
    
    # 5. data 구성하기
    data = {
            'train':train_df,
            'test':test_df,
            'sub':sub,
            }

    # test, vaild = 8:2
    return data


def bert_data_loader(args, data):
    """
    Parameters
    ----------
    args.dataloader.batch_size : int
        데이터 batch에 사용할 데이터 사이즈
    args.dataloader.shuffle : bool
        data shuffle 여부
    args.dataloader.num_workers: int
        dataloader에서 사용할 멀티프로세서 수
    args.dataset.valid_ratio : float
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용합니다.
    data : dict
        context_data_load 함수에서 반환된 데이터
    
    Returns
    -------
    data : dict
        DataLoader가 추가된 데이터를 반환합니다.
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values)) if args.dataset.valid_ratio != 0 else None
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data