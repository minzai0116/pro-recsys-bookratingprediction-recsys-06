import pandas as pd
import os
import json
import re
import time
import requests
import concurrent.futures
from tqdm import tqdm
'''
민재님의 코드리뷰 요청 파일입니다.
'''
# [설정] 환경 및 하드웨어 설정
class Config:
    # 1. 데이터 경로 (본인의 경로에 맞게 수정하세요)
    DATA_PATH = r"" 
    
    # 2. Ollama API 설정
    API_KEY = "ollama"
    MODEL_NAME = "llama3.2" 
    API_URL = "http://localhost:11434/v1/chat/completions"
    
    LOC_BATCH_SIZE = 10
    LOC_WORKERS = 3
    CAT_BATCH_SIZE = 20
    CAT_WORKERS = 2

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {Config.API_KEY}"
}

# 아마존 표준 카테고리 목록
AMAZON_TARGET_CATEGORIES = [
    "Comics & Manga", "Literature & Fiction", "Mystery, Thriller & Suspense",
    "Romance", "Science Fiction & Fantasy", "Teen & Young Adult", "Arts & Photography",
    "Biographies & Memoirs", "Business & Money", "Computers & Technology",
    "Cookbooks, Food & Wine", "Education & Teaching", "History",
    "Humor & Entertainment", "Religion & Spirituality", "Science & Math", 
    "Self-help", "Sports & Outdoors", "Travel", "Unknown"
]

# 유틸리티 함수
def clean_text(val):
    """텍스트 전처리 및 결측치 제거"""
    if pd.isna(val): return 'Unknown'
    s = str(val).strip()
    if s.lower() in ['n/a', 'nan', 'null', 'unknown', '', 'none', '?', 'undefined']:
        return 'Unknown'
    if re.match(r'^[,.\s]+$', s): return 'Unknown'
    return s.title()

def call_ollama_api(messages):
    """Ollama API 호출 (재시도 로직 포함)"""
    payload = {
        "model": Config.MODEL_NAME,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.1 
    }
    try:
        response = requests.post(Config.API_URL, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        if "```" in content:
            content = re.sub(r"```json|```", "", content).strip()
        return content
    except requests.exceptions.ConnectionError:
        print("\n❌ [Fatal] Ollama 서버가 꺼져있습니다. 'ollama serve'를 실행하세요.")
        raise
    except Exception as e:
        time.sleep(1)
        raise e

# Location 처리
def get_location_mapping(unique_list, cache_path):
    # 캐시 로드
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        except:
            mapping = {}
    else:
        mapping = {}

    target_list = [loc for loc in unique_list if loc not in mapping]
    print(f"    [Location] Total Unique: {len(unique_list)}, To Process: {len(target_list)}")
    
    api_targets = []
    
    # 1차: 규칙 기반 파싱
    for loc in target_list:
        loc_str = str(loc).strip()
        parts = [p.strip() for p in loc_str.split(',')]
        
        if len(parts) == 3: 
            state, country = parts[1], parts[2]
            if state and country:
                mapping[loc] = {"state": state.title(), "country": country.title()}
            else:
                api_targets.append(loc)
        else:
            api_targets.append(loc)

    print(f"    [Location] Auto-parsed: {len(target_list) - len(api_targets)} items")
    
    if not api_targets:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        return mapping

    # 2차: LLM 추론
    batches = [api_targets[i:i+Config.LOC_BATCH_SIZE] for i in range(0, len(api_targets), Config.LOC_BATCH_SIZE)]
    
    # 프롬프트
    system_prompt = """
    Analyze location strings. Return JSON: {"input_string": {"state": "...", "country": "..."}}
    Rules:
    1. Infer State and Country based on the input text.
    2. Even if a City is provided, DO NOT output the City. Only output State and Country.
    3. If State is not applicable (e.g., small countries), use 'n/a' or the Country name.
    4. Use "Unknown" if impossible to guess.
    """

    def process_batch(batch_data):
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(batch_data)}]
        for _ in range(3):
            try:
                return json.loads(call_ollama_api(msgs))
            except:
                time.sleep(1)
        return None

    print(f"    [Location] Calling Ollama for inference ({len(api_targets)} items)...")
    processed_cnt = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.LOC_WORKERS) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(batches), desc="Loc Inference"):
            res = future.result()
            if res:
                mapping.update(res)
                processed_cnt += 1
                if processed_cnt % 10 == 0:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(mapping, f, ensure_ascii=False, indent=2)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    return mapping

def process_location(users, cache_path):
    col = next((c for c in users.columns if c.lower() == 'location'), None)
    if not col: return users

    print("[Location] Mapping Start...")
    unique_vals = users[col].dropna().astype(str).unique().tolist()
    # 매핑 함수 호출
    mapping = get_location_mapping(unique_vals, cache_path)
    
    # State와 Country만 생성
    users['state'] = users[col].map(lambda x: mapping.get(str(x), {}).get('state'))
    users['country'] = users[col].map(lambda x: mapping.get(str(x), {}).get('country'))
    
    for c in ['state', 'country']:
        users[c] = users[c].apply(clean_text)
    
    country_map = {"Usa": "USA", "United States": "USA", "Uk": "United Kingdom", "United Kingdom": "United Kingdom"}
    users['country'] = users['country'].replace(country_map)
    
    users.drop(col, axis=1, inplace=True)
    return users

# Category 처리
def get_category_mapping_smart(category_data_list, cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        except:
            mapping = {}
    else:
        mapping = {}

    target_list = [item for item in category_data_list if item['text'] not in mapping]
    print(f"    [Category] Total Unique Categories: {len(category_data_list)}, To Process: {len(target_list)}")
    
    if not target_list: return mapping

    batches = [target_list[i:i+Config.CAT_BATCH_SIZE] for i in range(0, len(target_list), Config.CAT_BATCH_SIZE)]
    
    system_prompt = f"""
    Map the input category string to ONE of these Amazon standard categories: {json.dumps(AMAZON_TARGET_CATEGORIES)}.
    Input JSON: [{{"text": "Raw Category String"}}, ...]
    Output JSON: {{"Raw Category String": "StandardCategoryName", ...}}
    
    Rules:
    1. Find the semantically closest standard category based on the input text.
    2. Example: "Sci-Fi" -> "Science Fiction & Fantasy", "Investing" -> "Business & Money".
    3. If the input is "Unknown", "NaN", or gibberish, map to "Unknown".
    """

    def process_batch(batch_data):
        minimized = [{"text": b['text']} for b in batch_data]
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(minimized)}]
        for _ in range(3):
            try:
                return json.loads(call_ollama_api(msgs))
            except:
                time.sleep(1)
        return None

    print(f"    [Category] Calling Ollama (Category Standardization)...")
    processed_cnt = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.CAT_WORKERS) as executor:
        futures = {executor.submit(process_batch, b): b for b in batches}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(batches), desc="Standardizing Cats"):
            res = future.result()
            if res:
                mapping.update(res)
                processed_cnt += 1
                if processed_cnt % 5 == 0:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(mapping, f, ensure_ascii=False, indent=2)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    return mapping

def process_category(books, cache_path):
    cat_col = next((c for c in books.columns if c.lower() in ['category', 'categories', 'cat']), None)

    if not cat_col:
        print("⚠️ [Warning] 'Category' 컬럼을 찾을 수 없어 처리를 건너뜁니다.")
        return books

    print("[Category] Preparing Raw Category data...")
    
    books['temp_category'] = books[cat_col].apply(clean_text)
    unique_categories = books['temp_category'].unique().tolist()
    data_to_process = [{'text': cat} for cat in unique_categories]
    
    mapping = get_category_mapping_smart(data_to_process, cache_path)
    
    print("    [Category] Applying standardized map...")
    books['category_standard'] = books['temp_category'].map(lambda x: mapping.get(str(x), 'Unknown'))
    
    books.drop(['temp_category'], axis=1, inplace=True)
    return books

# [Main] 메인 실행 함수
def main():
    print(f">>> [Start] Data Preprocessing with Ollama (Llama 3.2)")
    print(f">>> Path: {Config.DATA_PATH}\n")
    
    u_path = os.path.join(Config.DATA_PATH, 'users.csv')
    b_path = os.path.join(Config.DATA_PATH, 'books.csv')
    
    if not os.path.exists(u_path) or not os.path.exists(b_path):
        print(f"❌ Error: 파일이 없습니다.\n   {u_path}\n   {b_path}")
        return

    try:
        users = pd.read_csv(u_path)
        books = pd.read_csv(b_path)
    except UnicodeDecodeError:
        print("⚠️ utf-8 로드 실패, cp949(latin1)로 재시도합니다.")
        users = pd.read_csv(u_path, encoding='cp949')
        books = pd.read_csv(b_path, encoding='cp949')
        
    print(f"✅ Data Loaded - Users: {len(users)}, Books: {len(books)}")

    # Location 처리
    print("\n" + "="*50)
    print(">>> [Step 1] Processing Location (State, Country only)")
    print("="*50)
    loc_cache = os.path.join(Config.DATA_PATH, 'location_std_map_cache.json')
    users_processed = process_location(users, loc_cache)

    # Category 처리
    print("\n" + "="*50)
    print(">>> [Step 2] Processing Category (Standardization)")
    print("="*50)
    cat_cache = os.path.join(Config.DATA_PATH, 'category_std_map_cache.json')
    books_processed = process_category(books, cat_cache)

    # 저장
    print("\n" + "="*50)
    print(">>> [Step 3] Saving Results...")
    print("="*50)
    
    u_save_path = os.path.join(Config.DATA_PATH, 'users_processed.csv')
    b_save_path = os.path.join(Config.DATA_PATH, 'books_processed.csv')
    
    users_processed.to_csv(u_save_path, index=False, encoding='utf-8-sig')
    books_processed.to_csv(b_save_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 저장 완료!")
    print(f"   - {u_save_path}")
    print(f"   - {b_save_path}")

if __name__ == "__main__":
    main()