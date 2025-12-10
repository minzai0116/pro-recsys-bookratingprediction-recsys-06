import argparse
import ast
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module
from src.utils import Logger, Setting
import src.data as data_module
import src.train as train_module
import src.models as model_module
'''
지금 sklearn이랑 torch 모델 둘이라서 중간에 if문 좀 많은데, 과해지면 헬퍼함수로 빼거나, 대격변 패치로 data 다루듯이 바꾸겠습니다만
굳이 안 불편하면 뺐을때 더 꼴뵈기 싫어짐 ㅎㅎ
'''

def main(args):
    Setting.seed_everything(args.seed)

    ######################## LOAD DATA
    datatype = args.model_args[args.model].datatype
    data_load_fn = getattr(data_module, f'{datatype}_data_load')  # e.g. basic_data_load()
    data_split_fn = getattr(data_module, f'{datatype}_data_split')  # e.g. basic_data_split()
    data_loader_fn = getattr(data_module, f'{datatype}_data_loader', None)  # Default -> None

    print(f'--------------- {args.model} Load Data ---------------')
    data = data_load_fn(args)


    print(f'--------------- {args.model} Train/Valid Split ---------------')
    data = data_split_fn(args, data)
    if data_loader_fn: # 해당 데이터 모듈에 Data_loader 있을때 에만(딥러닝 모델) 데이터로더 사용
        data = data_loader_fn(args, data)

    ####################### Setting for Log
    setting = Setting()
    
    if args.predict == False:
        log_path = setting.get_log_path(args)
        logger = Logger(args, log_path)
        logger.save_args()


    ######################## Model
    print(f'--------------- INIT {args.model} ---------------')
    # models > __init__.py 에 저장된 모델만 사용 가능
    # model = FM(args.model_args.FM, data).to('cuda')와 동일한 코드
    model = getattr(model_module, args.model)(args.model_args[args.model], data)

    # PyTorch 모델만 device로 이동, sklearn은 그런거 못함
    if not args.model_args[args.model].is_sklearn:
        model = model.to(args.device)

    # 만일 기존의 모델을 불러와서 학습을 시작하려면 resume을 true로 설정하고 resume_path에 모델을 지정하면 됨
    # sklearn은 그 뭐냐 joblib써서 함 굿굿
    if args.train.resume:
        if args.model_args[args.model].is_sklearn:
            import joblib
            model = joblib.load(args.train.resume_path)
        else:
            model.load_state_dict(torch.load(args.train.resume_path, weights_only=True))

    ######################## TRAIN
    # nn모듈일때랑 sklearn일때 train 함수가 각각 다르니까 알아서 잘 지정임 위에서 getattr 한거랑 비슷한거
    if args.model_args[args.model].is_sklearn:
        train = train_module.sklearn_train
        test = train_module.sklearn_test
    else:
        train = train_module.train
        test = train_module.test

    if not args.predict:
        print(f'--------------- {args.model} TRAINING ---------------')
        model = train(args, model, data, logger, setting)

    ######################## INFERENCE
    if not args.predict:
        print(f'--------------- {args.model} PREDICT ---------------')
        predicts = test(args, model, data, setting)
    else:
        print(f'--------------- {args.model} PREDICT ---------------')
        predicts = test(args, model, data, setting, args.checkpoint)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')
    submission['rating'] = predicts

    filename = setting.get_submit_filename(args)
    print(f'Save Predict: {filename}')
    submission.to_csv(filename, index=False)


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    

    arg = parser.add_argument
    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}

    # add basic arguments (no default value)
    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', required=True)
    arg('--predict', '-p', '--p', '--pred', type=ast.literal_eval, 
        help='학습을 생략할지 여부를 설정할 수 있습니다.')
    arg('--checkpoint', '-ckpt', '--ckpt', type=str, 
        help='학습을 생략할 때 사용할 모델을 설정할 수 있습니다. 단, 하이퍼파라미터 세팅을 모두 정확하게 입력해야 합니다.')
    arg('--model', '-m', '--m', type=str, 
        choices=['FM', 'FFM', 'DeepFM', 'NCF', 'WDN', 'DCN', 'Image_FM', 'Image_DeepFM', 'Text_FM', 'Text_DeepFM', 'ResNet_DeepFM', 'LightGBM', 'CatBoost', 'bert_rec', 'tab_rec'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--seed', '-s', '--s', type=int,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--wandb', '--w', '-w', type=ast.literal_eval, 
        help='wandb를 사용할지 여부를 설정할 수 있습니다.')
    arg('--wandb_project', '--wp', '-wp', type=str,
        help='wandb 프로젝트 이름을 설정할 수 있습니다.')
    arg('--run_name', '--rn', '-rn', '--r', '-r', type=str,
        help='wandb에서 사용할 run 이름을 설정할 수 있습니다.')
    
    arg('--STE', type=bool)
    arg('--threshold', type=float)                       # softmax threshold : 이 값 이하의 softmax는 탈락시킴
    arg('--memo', type=str)                                         # test group 명시화를 위해 memo 변경
    arg('--regularization', type=ast.literal_eval)                  # Lasso 적용을 위한 실험
    arg('--regularize_lambda', type=ast.literal_eval)               
    arg('--model_args', '--ma', '-ma', type=ast.literal_eval)
    arg('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    arg('--dataset', '--dset', '-dset', type=ast.literal_eval)
    arg('--optimizer', '-opt', '--opt', type=ast.literal_eval)
    arg('--loss', '-l', '--l', type=str)
    arg('--lr_scheduler', '-lr', '--lr', type=ast.literal_eval)
    arg('--metrics', '-met', '--met', type=ast.literal_eval)
    arg('--train', '-t', '--t', type=ast.literal_eval)              

    
    args = parser.parse_args()

    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]
    #config_yaml = OmegaConf.merge(config_yaml, config_args)
    

    # 사용되지 않는 정보 삭제 (학습 시에만)
    if config_yaml.predict == False:
        del config_yaml.checkpoint
    
        if config_yaml.wandb == False:
            del config_yaml.wandb_project, config_yaml.run_name
        
        config_yaml.model_args = OmegaConf.create({config_yaml.model : config_yaml.model_args[config_yaml.model]})
        
        config_yaml.optimizer.args = {k: v for k, v in config_yaml.optimizer.args.items() 
                                    if k in getattr(optimizer_module, config_yaml.optimizer.type).__init__.__code__.co_varnames}
        
        if config_yaml.lr_scheduler.use == False:
            del config_yaml.lr_scheduler.type, config_yaml.lr_scheduler.args
        else:
            config_yaml.lr_scheduler.args = {k: v for k, v in config_yaml.lr_scheduler.args.items() 
                                            if k in getattr(scheduler_module, config_yaml.lr_scheduler.type).__init__.__code__.co_varnames}
        
        if config_yaml.train.resume == False:
            del config_yaml.train.resume_path

    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(config_yaml))
    
    ######################## W&B
    if config_yaml.wandb:
        import wandb
        # wandb.require("core")
        # https://docs.wandb.ai/ref/python/init 참고
        wandb.init(project=config_yaml.wandb_project, 
                   config=OmegaConf.to_container(config_yaml, resolve=True),
                   name=config_yaml.run_name if config_yaml.run_name else None,
                   notes=config_yaml.memo if hasattr(config_yaml, 'memo') else None,
                   tags=[config_yaml.model],
                   resume="allow")
        config_yaml.run_href = wandb.run.get_url()

        wandb.run.log_code("./src")  # src 내의 모든 파일을 업로드. Artifacts에서 확인 가능

    ######################## MAIN
    main(config_yaml)

    if config_yaml.wandb:
        wandb.finish()