import torch
import torch.nn as nn
import numpy as np
import pickle

# done
class FeedForward(nn.Module):
    ''' 
        - multihead_attenion된 x를 차원을 늘렸다 줄이는 곳입니다.
    
        ######################
        
        * input     :   tensor([batches, features, embedding_dim])
        * output    :   tensor([batches, features, embedding_dim])
        
        ######################
        
        1) x_extend = extend_linear(x)          : [batch, features, embedding_dim] -> [batch, features, extended_dim]
        2) x_extend = active(x_extend)          : silu 적용
        3) x_shrink = shrink_linear(x_extend)   : [batch, features, extended_dim] -> [batch, features, embedding_dim]
        
        return x_shrink
        
    '''
    def __init__(self, args, test = False):
        super().__init__()
        
        self.test = test
        
        self.extend = nn.Linear(args.embed_dim, args.extend_dim)
        self.activation = select_activation(args)
        self.shrink = nn.Linear(args.extend_dim, args.embed_dim)
        
    def forward(self, x: torch.Tensor):
        
        x_extend = self.extend(x)
        x_extend = self.activation(x_extend)
        x_shrink = self.shrink(x_extend)
        return x_shrink

# done
class attention_head(nn.Module):
    ''' 
        - self_attention이 실제 일어나는 곳입니다.
    
        ######################
        
        * input     :   tensor([batches, features, embedding_dim])
        * output    :   tensor([batches, features, latent_dim])
        
        ######################
        
        1) attention = dot(Q.T,K) / embed_dim^(-0.5)    : 커진 차원에 대한 scalar_normalization 적용
        2) attention_map = softmax(attention)           
        3) latent_state = attention_map @ V
        
        return latent_state
    
    '''
    
    def __init__(self, args, test = False):
        super().__init__()
        
        self.test = test
        
        self.threshold = args.threshold
        self.num_heads = args.num_heads
        self.head_dims = args.embed_dim // args.num_heads
        self.scalar_factor = args.embed_dim ** 0.5
        
        self.Q_w = nn.Linear(args.embed_dim, self.head_dims)
        self.K_w = nn.Linear(args.embed_dim, self.head_dims)
        self.V_w = nn.Linear(args.embed_dim, self.head_dims)
        
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, x: torch.Tensor):
        
        Q = self.Q_w(x)
        K = self.K_w(x)
        V = self.V_w(x)
        
        self_attention = Q@K.transpose(-1, -2) / self.scalar_factor
        attention_map = thresholded_softmax(self_attention, threshold = self.threshold, dim = -1)
        latent_state = attention_map@V
        
        return latent_state, attention_map

# done
class multihead_attention(nn.Module):
    ''' 
        - self_attention된 latent vector들을 모아 반환해주는 layer입니다.
    
        ######################
        
        * input     :   tensor([batches, features, embedding_dim])
        * output    :   tensor([batches, features, embedding_dim])
        
        ######################
        
        1) latents = [head_1(x), head_2(x) ..., head_n(x)] = [latent_1, latent_2 ...., latent_n]
        2) residual = concat(latents)
        3) x_res = x + residual
        
        return x_res
        
    '''
    def __init__(self, args, test = False):
        super().__init__()
        
        self.test = test
        
        # head 선언
        self.Heads = nn.ModuleList([attention_head(args) for _ in range(args.num_heads)])
        # FF
        self.FF = FeedForward(args)
        # batch_norm
        #self.batch = nn.BatchNorm1d()
        # drop_out
        self.drop = nn.Dropout(args.dropout)
        
        
    def forward(self, x: torch.Tensor):
        
        # state에서 latent 세트를 num_heads의 갯수만큼 생성 : latent = [latent_1, latent_2 ....]
        latents = []
        attention_map = []
        for l in self.Heads:
            latent, att_map = l(x)
            latents.append(latent)
            attention_map.append(att_map)
            
        # latent를 concat: x = latent_1 + lantent_2 ....
        residual = torch.cat(latents, axis=-1)
        
        # x_res : gradient 지름길 제공 + batch + drop
        #x_res = x + self.drop(self.batch(residual))
        x_res = x + self.drop(residual)
        
        return x_res, attention_map

# done
class encoder(nn.Module):
    '''
        - encoder는 multihead_layer의 집합입니다.
        
        ######################
        
        * input     :   tensor([batches, features, embedding_dim])
        * output    :   tensor([batches, features, embedding_dim])
        
        ######################
        
        1) x = layer_norm(x)
        2) state = x + multihead_attention(x)  
        3) hidden_state = state + FF(state)
        
        return hidden_state
        
    '''
    
    def __init__(self, args, test = False):
        super().__init__()
        
        self.test = test
        
        # layer_norm
        # self.layer_norm = nn.LayerNorm([args.batch, args.num_features, args.embed_dim])
        self.layer_norm_1 = nn.LayerNorm(args.embed_dim)
        # 인코더 layers 생성 = head_attention * num_layers
        self.multihead = multihead_attention(args)
        self.drop_1 = nn.Dropout(args.dropout)
        
        #layer_nomr
        self.layer_norm_2 = nn.LayerNorm(args.embed_dim)
        # FF layer
        self.FF = FeedForward(args)
        self.drop_2 = nn.Dropout(args.dropout)

    
    def forward(self, x: torch.Tensor):
        
        x_norm = self.layer_norm_1(x)
        res, attention_map = self.multihead(x_norm)
        x = x + self.drop_1(res)
        
        x_norm = self.layer_norm_2(x)
        x = x + self.drop_2(self.FF(x_norm))
        
        return x, attention_map

# done
class tab_rec(nn.Module):
    '''
        - 입력을 받아 추천을 하는 layer
        ######################
        
        * input     :   tensor([batches, features-2, embedding_dim]) + embedded_vector["summary", "image"]
        * output    :   tensor([batches, features, embedding_dim])
        
        ######################
        
        1) x = layer_norm(x)
        2) state = x + multihead_attention(x)  
        3) hidden_state = state + FF(state)
        
        return hidden_state
    '''
    def __init__(self, args, test = False):
        super().__init__()
        
        self.test = test
        self.mlp_hidden = 128
        
        self.summary_layer = nn.Sequential(
            nn.Linear(args.summary_dim, args.summary_dim//2),
            nn.SiLU(),
            nn.Linear(args.summary_dim//2, args.embed_dim)
            # <<-- 여기 활성화 함수를 꼭 넣어야 하는가?
            # 아니라곤 하는데, 나중에 실험할 예정
            )
        
        # 임베딩 관리는 따로
        self.embedding = nn.ModuleList([nn.Embedding(cardi, args.embed_dim) for cardi in args.cardinality])
        self.encoders = nn.ModuleList([encoder(args) for _ in range(args.num_layers)])
        
        self.output_layer = nn.Sequential(
            # +1 은 summary를 위해. -> summary 안 쓰는게 더 성능 잘 나오는듯?
            #nn.Linear(args.embed_dim*(args.dim_feature+1), self.mlp_hidden)
            nn.Linear(args.embed_dim*(args.dim_feature), self.mlp_hidden),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden, 1)
            )
        
        '''
            summary_vector 불러와서 embedding으로 저장하기
        '''
        self.summary_embedding = load_summary_vector(args)

    def forward(self, x: torch.Tensor):
        
        # [ summary_index,  cls,  user,.....]
        
        # summary_index를 추출하고
        summary_index = x[:,0]
        # 저장된 embedding을 불러와서
        summary_embedding = self.summary_embedding(summary_index)
        # 차원도 맞출 겸, linear를 한 번 거친다. 768 -> embedding_dim
        summary_state = self.summary_layer(summary_embedding)
        
        # 나머지는 사실상 원본
        state = x[:,1:]
        
        # Embedding을 뽑아오고
        embs = []
        for i, emb in enumerate(self.embedding):
            embs.append(emb(state[:, i]))        

        # batch로 병합
        x = torch.stack(embs, dim=1)
        
        # encoding 실행
        attention_map = []
        for l in self.encoders:
            x, att_map = l(x)
            attention_map.append(att_map)

        x = x.flatten(1)

        # 나가기 전에 linaer 한 번 거치기: embedding_dim -> 단일 출력 변환
        output = self.output_layer(x)
        return output.squeeze(-1), attention_map
    
# summary_vector initializer
def load_summary_vector(args):
    with open(f"{args.summary_path}" + "/summary_vector/summaries.pkl", "rb") as f:
        kv_dict = pickle.load(f)   # {key: np.ndarray} 구조라고 가정
    
    
    num_embedding = len(kv_dict)
    dim_embedding = len(kv_dict[next(iter(kv_dict))])
    
    vector_rag = nn.Embedding(num_embedding, dim_embedding)
    
    with torch.no_grad():
        for k in kv_dict:
            vector_rag.weight[k] = torch.tensor(kv_dict[k], dtype=torch.float32)
        
    vector_rag.weight.requires_grad_(False)
    
    return vector_rag

# activation에 대한 모델 성능 실험을 위한 함수
def select_activation(args):
    if args.activation.lower() == "relu":
        return nn.ReLU()
    if args.activation.lower() == "silu":
        return nn.SiLU()
    
def thresholded_softmax(x, threshold, dim=-1):
    s = torch.nn.functional.softmax(x, dim=dim)
    s = torch.where(s < threshold, 0, s)
    s = s / (s.sum(dim=dim, keepdim=True) + 1e-12)
    return s

# 현재
# 1. location 전처리 = 기본으로 해치움
# 2. Nan값 Unknown으로 처리 -> category로 바꿔도 -1 안나오고 index 할당된거 확인했음.
# 3. summary index 부여해서 summary 다시 찾을 수 있음
# 4. n/a 값들 처리 -> 된듯? 안나오게 바꿈 -> 혹시 모르니 인지하고 있을 것.

# data 전처리 먼저 해야함 
# 1. summary vector 만들고, 다시 찾을 수 있게 index랑 묶어서 pickle로 저장해놓기 -> text_to_vector
# 2. 입력으로 쓸 수 있게 필요없는 열은 자르고, category index로 변환 -> convert_to_index, remain_train_features_only
# 3. 정규식으로 이상한 str 삭제하고 ex) //n/a//" 같은거 -> 선택사항
# 4. 그리고 나서 고유값 찾고, offset 만들기. 맨 앞에 0 붙이고 나머지에 +1 하는거 잊지말기 -> 반드시
# 5. bert_init에 summary_vector tensor 넣어주기,summary linear 작성 -> 768 to args.emb_dim
# 6. summary 다시 만들기. (bool_id랑 같이 저장해서 불러오기 편하게)

# 전처리 사항 (입력으로 쓸 feature들만)
# 1. location -> city, state, country로 바꾸기
# 2. age_range -> 10단위로 카테고리

# 3. summary_vector = summary_index
# 4. pub_range
# 5. language
# 6. publisher
# 7. category
# 8. author

# 안 쓸 feature
# 1. age
# 2. location
# 3. summary
# 4. image
# 5. title
# 6. isbn
# 7. publication of year

# 안 쓸거는 탈락 시키고
# embedding index로 변환