import os, random
import numpy as np
import torch
import argparse
import wandb

from train import train
from train_msAttnFlow import train_OurFlow
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parsing_args(c):
    parser = argparse.ArgumentParser(description='msflow')
    parser.add_argument('--dataset', default='mvtec', type=str, 
                        choices=['mvtec', 'visa'], help='dataset name')
    parser.add_argument('--mode', default='train', type=str, 
                        help='train or test.')
    parser.add_argument('--amp_enable', action='store_true', default=False, 
                        help='use amp or not.')
    parser.add_argument('--wandb_enable', action='store_true', default=False, 
                        help='use wandb for result logging or not.')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='resume training or not.')
    parser.add_argument('--eval_ckpt', default='', type=str, 
                        help='checkpoint path for evaluation.')
    parser.add_argument('--class-names', default=['all'], type=str, nargs='+', 
                        help='class names for training')
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help='learning rate')
    parser.add_argument('--batch-size', default=4, type=int, 
                        help='train batch size')
    parser.add_argument('--meta-epochs', default=25, type=int,
                        help='number of meta epochs to train')
    parser.add_argument('--sub-epochs', default=1, type=int,
                        help='number of sub epochs to train')
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str, 
                        help='feature extractor')
    parser.add_argument('--pool-type', default='avg', type=str, 
                        help='pool type for extracted feature maps')
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, metavar='L', nargs='+',
                        help='number of flow blocks used in parallel flows.')
    parser.add_argument('--pro-eval', action='store_true', default=False, 
                        help='evaluate the pro score or not.')
    parser.add_argument('--pro-eval-interval', default=4, type=int, 
                        help='interval for pro evaluation.')
    
    parser.add_argument('--num-transformer-blocks', default=4, type=int, 
                        help='.')

    args = parser.parse_args()

    for k, v in vars(args).items():
        setattr(c, k, v)
    
    if c.dataset == 'mvtec':
        from datasets import MVTEC_CLASS_NAMES
        setattr(c, 'data_path', './data/MVTec')
        if c.class_names == ['all']:
            setattr(c, 'class_names', MVTEC_CLASS_NAMES)
    elif c.dataset == 'visa':
        from datasets import VISA_CLASS_NAMES
        setattr(c, 'data_path', './data/VisA_pytorch/1cls')
        if c.class_names == ['all']:
            setattr(c, 'class_names', VISA_CLASS_NAMES)
        
    c.input_size = (256, 256) if c.class_name == 'transistor' else (512, 512)

    return c
import json
def main(c):
    c = parsing_args(c)
    init_seeds(seed=c.seed)
    # c.class_names=["transistor"]
    c.mode="train"
    c.flow_name="msAttnFlow"
    # c.version_name = 'msflow_{}_{}pool_pl{}'.format(c.extractor, c.pool_type, "".join([str(x) for x in c.parallel_blocks]))
    c.version_name=f"{c.flow_name}_pool421_allChannel_noNeck"
    c.batch_size=4
    print(c.class_names)
    results={}
    c.meta_epochs=50
    for class_name in c.class_names:
        c.class_name = class_name
        c.eval_ckpt=f"work_dirs/{c.version_name}/mvtec/{c.class_name}/best_det_auroc.pt"
        print('-+'*5, class_name, '+-'*5)
        c.ckpt_dir = os.path.join(c.work_dir, c.version_name, c.dataset, c.class_name)
        c.use_attn=True
        c.use_conv=False
        c.use_ffn=False
        c.use_norm=False
        c.attn_all=False
        c.use_all_channels=True
        c.resample_args={"resample":False,
                          "thresholds":[0.4,0.5,0.6],
                          "stragety":"auto"}
        c.num_transformer_blocks=4
        c.pro_eval=False
        det_auroc_obs,loc_auroc_obs,loc_pro_obs=train_OurFlow(c)
        
        result={"det_auroc":[det_auroc_obs.max_score,det_auroc_obs.max_epoch],
                "loc_auroc":[loc_auroc_obs.max_score,loc_auroc_obs.max_epoch]}
        results[class_name]=result

    if c.mode=="train":
        loc_ave=sum([i["loc_auroc"][0] for i in results.values()])/len(results.values())
        det_ave=sum([i["det_auroc"][0] for i in results.values()])/len(results.values())
        results["average"]={"det_auroc":[det_ave,0],
                            "loc_auroc":[loc_ave,0]}
        with open(os.path.join(c.work_dir, c.version_name, c.dataset,"results.json"),"w") as fp:
            json.dump(results,fp)

    with open(os.path.join(c.work_dir, c.version_name, c.dataset,"results.json"),"r") as fp:
        results=json.load(fp)
    print(c.version_name)
    print("loc auroc")
    print("|||")
    print("|---|---|")
    for k,v in results.items():
        print(f"|{k}|{round(v['loc_auroc'][0],4)}|")
    print("det auroc")
    print("|||")
    print("|---|---|")
    for k,v in results.items():
        print(f"|{k}|{round(v['det_auroc'][0],4)}|")



if __name__ == '__main__':
    import default as c
    main(c)