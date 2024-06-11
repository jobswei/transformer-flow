import os
import time
import datetime
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from Datasets.datasets import MVTecDataset, VisADataset
from Datasets.datasets_rotate import MVTecDatasetRotate
from Datasets.datasets_diffusion import MVTecDatasetDiffusion
from models.extractors import build_extractor
from models.flow_models import *
from post_process import post_process,post_process_resampled
from utils.utils import Score_Observer, t2np, positionalencoding2d, save_weights, load_weights
from evaluations import eval_det_loc

import tqdm
def get_pool_layer(c):
    if c.pool_type == 'avg':
        # pool_layer = nn.AvgPool2d(2,2,0)
        # pool_layer = nn.AvgPool2d(3, 2, 1)
        pool_layer = nn.AvgPool2d(4,2,1)
        # pool_layer=nn.Identity()
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()
    return pool_layer
def model_forward(c, extractor,  conv_neck,msAttn_flow, fusion_flow, image):
    h_list = extractor(image)
    pool_layer=get_pool_layer(c)
    cond_lis=[]
    c_cond=c.c_conds[0]
    for idx,h in enumerate(h_list):
        h=pool_layer(h)
        h_list[idx]=h
        B,_,H,W=h.shape
        cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        cond_lis.append(cond)
    if c.use_conv:
        h_list=conv_neck(h_list)

    # import gc
    # objs = set()
    # for obj in gc.get_objects():
    #     if torch.is_tensor(obj) and not isinstance(obj, torch.nn.parameter.Parameter):
    #         objs.add(obj)
    # print(len(objs))
    z_list,jac_lis=msAttn_flow(h_list,[cond_lis,])
    # for obj in gc.get_objects():
    #     if torch.is_tensor(obj) and not isinstance(obj, torch.nn.parameter.Parameter):
    #         if obj not in objs:
    #             objs.add(obj)
    #             referer = gc.get_referrers(obj)
    #             pass
    # print(len(objs))
    # z_list=z_list[0]
    # jac=jac_lis[0]
    z_list, fuse_jac = fusion_flow(z_list)
    jac = fuse_jac + sum(jac_lis)

    # return [z_list[0]], jac
    return z_list,jac,cond_lis

def train_meta_epoch(c, epoch, loader, extractor, conv_neck,msAttn_flow, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler=None):
    msAttn_flow=msAttn_flow.train()
    fusion_flow = fusion_flow.train()

    for sub_epoch in range(c.sub_epochs):
        epoch_loss = 0.
        image_count = 0
        for idx, (image, _, _) in tqdm.tqdm(enumerate(loader),total=len(loader)):
            # print(idx,end="\r")
            optimizer.zero_grad()
            image = image.to(c.device)
            if scaler:
                with autocast():
                    z_list, jac = model_forward(c, extractor, msAttn_flow, fusion_flow, image)
                    loss = 0.
                    for z in z_list:
                        loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                    loss = loss - jac
                    loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 2)
                scaler.step(optimizer)
                scaler.update()
            else:
                z_list, jac,_ = model_forward(c, extractor,  conv_neck,msAttn_flow, fusion_flow, image)
                loss = 0.
                for z in z_list:
                    loss += 0.5 * torch.sum(z**2, (1, 2, 3))
                loss = loss - jac
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2)
                optimizer.step()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if warmup_scheduler:
            warmup_scheduler.step()
        if decay_scheduler:
            decay_scheduler.step()

        mean_epoch_loss = epoch_loss / image_count
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}.{:d} train loss: {:.3e}\tlr={:.2e}'.format(
                epoch, sub_epoch, mean_epoch_loss, lr))
        

def inference_meta_epoch(c, epoch, loader,  extractor,conv_neck, msAttn_flow, fusion_flow):
    conv_neck.eval()
    msAttn_flow=msAttn_flow.eval()
    fusion_flow = fusion_flow.eval()
    epoch_loss = 0.
    image_count = 0
    gt_label_list = list()
    gt_mask_list = list()
    # outputs_list = [list() for _ in range(1)]
    outputs_lists = {"noflip":[list() for _ in range(len(c.output_channels))],
                     "flip":[list() for _ in range(len(c.output_channels))]}
    hidden_variables = [list() for _ in range(len(c.output_channels))]
    size_list = []
    cond_list=[]
    start = time.time()
    with torch.no_grad():
        for idx, (image, label, mask) in tqdm.tqdm(enumerate(loader),total=len(loader)):
            image = image.to(c.device)
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))

            z_list, jac,cond_lis = model_forward(c, extractor, conv_neck, msAttn_flow, fusion_flow, image)
            cond_list.append(cond_lis)

            loss = 0.
            for lvl, z in enumerate(z_list):
                if idx == 0:
                    size_list.append(list(z.shape[-2:]))
                logp = - 0.5 * torch.mean(z**2, 1)
                outputs_lists["noflip"][lvl].append(logp)
                loss += 0.5 * torch.sum(z**2, (1, 2, 3))

                hidden_variables[lvl].append(z)

            if c.post_augmentation:
                image=image.flip(3)
                z_list, jac,cond_lis = model_forward(c, extractor, conv_neck, msAttn_flow, fusion_flow, image)
                for lvl, z in enumerate(z_list):
                    z=z.flip(3)
                    logp = - 0.5 * torch.mean(z**2, 1)
                    outputs_lists["flip"][lvl].append(logp)
                image=image.flip(3)
            loss = loss - jac
            loss = loss.mean()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]

        mean_epoch_loss = epoch_loss / image_count
        fps = len(loader.dataset) / (time.time() - start)
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
            'Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                epoch, mean_epoch_loss, fps))

    return gt_label_list, gt_mask_list, outputs_lists, size_list, hidden_variables,cond_list
def save_weights(epoch,conv_neck, msAttn_flow, fusion_flow, model_name, ckpt_dir, optimizer=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    file_name = '{}.pt'.format(model_name)
    file_path = os.path.join(ckpt_dir, file_name)
    print('Saving weights to {}'.format(file_path))
    state = {'epoch': epoch,
             'conv_neck':conv_neck.state_dict(),
             'fusion_flow': fusion_flow.state_dict(),
             'msAttn_flow': msAttn_flow.state_dict()}
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    torch.save(state, file_path)


def load_weights(conv_neck,msAttn_flow,fusion_flow, ckpt_path, optimizer=None):
    print('Loading weights from {}'.format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    
    fusion_state = state_dict['fusion_flow']
    if isinstance(conv_neck , nn.Module):
        maps = {}
        for i in range(len(fusion_flow.module_list)):
            try:
                maps[fusion_flow.module_list[i].perm.shape[0]] = i
            except:
                continue
        temp = dict()
        for k, v in fusion_state.items():
            if 'perm' not in k:
                continue
            temp[k.replace(k.split('.')[1], str(maps[v.shape[0]]))] = v
        for k, v in temp.items():
            fusion_state[k] = v
    fusion_flow.load_state_dict(fusion_state, strict=True)
    conv_neck.load_state_dict(state_dict['conv_neck'], strict=True)
    # for parallel_flow, state in zip(parallel_flows, state_dict['parallel_flows']):
    msAttn_flow.load_state_dict(state_dict['msAttn_flow'], strict=True)

    if optimizer:
        optimizer.load_state_dict(state_dict['optimizer'])

    return state_dict['epoch']

def test_flow(c,conv_neck,msAttn_flow, fusion_flow, extractor,test_loader, eval_ckpt,det_auroc_obs, loc_auroc_obs, loc_pro_obs):
    start_epoch = load_weights(conv_neck,msAttn_flow, fusion_flow,eval_ckpt)
    epoch = start_epoch + 1
    gt_label_list, gt_mask_list, outputs_lists, size_list, hidden_variables,cond_list = inference_meta_epoch(c, epoch, test_loader, extractor, conv_neck,msAttn_flow, fusion_flow)

    if not c.post_augmentation:
        outputs_lists=[outputs_lists["noflip"]]
    else:
        outputs_lists=list(outputs_lists.values())

    results=[]
    for outputs_list in outputs_lists:
        if c.resample_args["resample"]:
            res\
                = post_process_resampled(c, size_list, outputs_list,
                                        msAttn_flow,fusion_flow,hidden_variables,cond_list)
        else:
            res = post_process(c, size_list, outputs_list)
        results.append(res)
    y=lambda i:sum([res[i] for res in results])/len(results)
    anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = y(0),y(1),y(2)
    det_auroc, loc_auroc, loc_pro_auc, \
        best_det_auroc, best_loc_auroc, best_loc_pro =  eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.pro_eval)
    return det_auroc, loc_auroc, loc_pro_auc
def train_OurFlow(c):
    
    if c.wandb_enable:
        wandb.finish()
        wandb_run = wandb.init(
            project='65001-msflow', 
            group=c.version_name,
            name=c.class_name)
    if c.peer_augmentation:
        if c.peer_type=="1":
            Dataset=MVTecDatasetRotate
        else:
            Dataset=MVTecDatasetDiffusion
    else:
        Dataset=MVTecDataset if c.dataset == 'mvtec' else VisADataset
        
    train_dataset = Dataset(c, is_train=True)
    test_dataset  = Dataset(c, is_train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True)

    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    pool_layer=get_pool_layer(c)  
    for image,_,_ in train_loader:
        image=image.to(c.device)
        image=extractor(image)
        image=[pool_layer(i) for i in image]
        dims=[list(i.shape) for i in image]
        break
    output_channels=[i[1] for i in dims]
    c.output_channels=output_channels
    if c.flow_name=="msAttnFlow":
        conv_neck,msAttn_flow,fusion_flow=build_ms_attn_flow_model(c,output_channels,dims)
    elif c.flow_name=="TransformerFlow":
        conv_neck,msAttn_flow,fusion_flow=build_transformer_flow_model(c,output_channels,dims)
    conv_neck=conv_neck.to(c.device)
    msAttn_flow=msAttn_flow.to(c.device)
    fusion_flow = fusion_flow.to(c.device)
    # if c.wandb_enable:
    #     for idx, parallel_flow in enumerate(parallel_flows):
    #         wandb.watch(parallel_flow, log='all', log_freq=100, idx=idx)
    #     wandb.watch(fusion_flow, log='all', log_freq=100, idx=len(parallel_flows))
    params = list(conv_neck.parameters())+list(fusion_flow.parameters())+list(msAttn_flow.parameters())
        
    optimizer = torch.optim.Adam(params, lr=c.lr)
    if c.amp_enable:
        scaler = GradScaler()

    det_auroc_obs = Score_Observer('Det.AUROC', c.meta_epochs)
    loc_auroc_obs = Score_Observer('Loc.AUROC', c.meta_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', c.meta_epochs)

    start_epoch = 0
    if c.mode == 'test':
        c.pro_eval=False
        eval_ckpt=c.eval_ckpts["det"]
        test_flow(c,conv_neck,msAttn_flow, fusion_flow, extractor,test_loader, eval_ckpt,det_auroc_obs, loc_auroc_obs, loc_pro_obs)
        eval_ckpt=c.eval_ckpts["loc"]
        test_flow(c,conv_neck,msAttn_flow, fusion_flow, extractor,test_loader, eval_ckpt,det_auroc_obs, loc_auroc_obs, loc_pro_obs)
        c.pro_eval=True
        eval_ckpt=c.eval_ckpts["pro"]
        test_flow(c,conv_neck,msAttn_flow, fusion_flow, extractor,test_loader, eval_ckpt,det_auroc_obs, loc_auroc_obs, loc_pro_obs)
        return det_auroc_obs,loc_auroc_obs,loc_pro_obs
    if c.resume:
        last_epoch = load_weights(conv_neck,msAttn_flow, fusion_flow, os.path.join(c.ckpt_dir, 'last.pt'), optimizer)
        start_epoch = last_epoch + 1
        print('Resume from epoch {}'.format(start_epoch))

    if c.lr_warmup and start_epoch < c.lr_warmup_epochs:
        if start_epoch == 0:
            start_factor = c.lr_warmup_from
            end_factor = 1.0
        else:
            start_factor = 1.0
            end_factor = c.lr / optimizer.state_dict()['param_groups'][0]['lr']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=(c.lr_warmup_epochs - start_epoch)*c.sub_epochs)
    else:
        warmup_scheduler = None

    mile_stones = [milestone - start_epoch for milestone in c.lr_decay_milestones if milestone > start_epoch]

    if mile_stones:
        decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, c.lr_decay_gamma)
    else:
        decay_scheduler = None

    for epoch in range(start_epoch, c.meta_epochs):
        print()
        train_meta_epoch(c, epoch, train_loader, extractor,  conv_neck,msAttn_flow, fusion_flow, params, optimizer, warmup_scheduler, decay_scheduler, scaler if c.amp_enable else None)

        gt_label_list, gt_mask_list, outputs_lists, size_list,hidden_variables,cond_list = inference_meta_epoch(c, epoch, test_loader, extractor, conv_neck,msAttn_flow, fusion_flow)

        if not c.post_augmentation:
            outputs_lists=[outputs_lists["noflip"]]
        else:
            outputs_lists=list(outputs_lists.values())

        results=[]
        for outputs_list in outputs_lists:
            if c.resample_args["resample"]:
                res\
                    = post_process_resampled(c, size_list, outputs_list,
                                            msAttn_flow,fusion_flow,hidden_variables,cond_list)
            else:
                res = post_process(c, size_list, outputs_list)
            results.append(res)
        y=lambda i:sum([res[i] for res in results])/len(results)
        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = y(0),y(1),y(2)
        if c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0):
            pro_eval = True
        else:
            pro_eval = False

        det_auroc, loc_auroc, loc_pro_auc, \
            best_det_auroc, best_loc_auroc, best_loc_pro = \
                eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, pro_eval)
        if c.wandb_enable:
            wandb_run.log(
                {
                    'Det.AUROC': det_auroc,
                    'Loc.AUROC': loc_auroc,
                    'Loc.PRO': loc_pro_auc
                },
                step=epoch
            )

        # save_weights(epoch,conv_neck,msAttn_flow, fusion_flow, 'last', c.ckpt_dir, optimizer)
        # if best_det_auroc and c.mode == 'train':
        #     save_weights(epoch,conv_neck,msAttn_flow, fusion_flow, 'best_det_auroc', c.ckpt_dir)
        # if best_loc_auroc and c.mode == 'train':
        #     save_weights(epoch,conv_neck,msAttn_flow, fusion_flow, 'best_loc_auroc', c.ckpt_dir)
        # if best_loc_pro and c.mode == 'train':
        #     save_weights(epoch,conv_neck, msAttn_flow, fusion_flow, 'best_loc_pro', c.ckpt_dir)
        import os.path as osp
        os.makedirs(c.ckpt_dir,exist_ok=True)
        with open(osp.join(c.ckpt_dir,"auroc.txt"),"a+") as fp:
            fp.write(f"Det.AUROC: {det_auroc} Loc.AUROC: {loc_auroc} Loc.PRO: {loc_pro_auc}\n")
    return det_auroc_obs,loc_auroc_obs,loc_pro_obs