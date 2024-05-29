import numpy as np
import torch
import torch.nn.functional as F
import tqdm
def post_process(c, size_list, outputs_list):
    print('Multi-scale sizes:', size_list)
    logp_maps = [list() for _ in size_list]
    prop_maps = [list() for _ in size_list]
    for l, outputs in enumerate(outputs_list):
        # output = torch.tensor(output, dtype=torch.double)
        outputs = torch.cat(outputs, 0)
        logp_maps[l] = F.interpolate(outputs.unsqueeze(1),
                size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
        output_norm = outputs - outputs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        prob_map = torch.exp(output_norm) # convert to probs in range [0:1]
        prop_maps[l] = F.interpolate(prob_map.unsqueeze(1),
                size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
    
    logp_map = sum(logp_maps)
    logp_map-= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    prop_map_mul = torch.exp(logp_map)
    anomaly_score_map_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
    batch = anomaly_score_map_mul.shape[0]
    top_k = int(c.input_size[0] * c.input_size[1] * c.top_k)
    anomaly_score = np.mean(
        anomaly_score_map_mul.reshape(batch, -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
        axis=1)

    prop_map_add = sum(prop_maps)
    prop_map_add = prop_map_add.detach().cpu().numpy()
    anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add

    return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul.detach().cpu().numpy()
def resample(hidden_variable,prob_map,thre:float,stragety:str):
    if stragety == "auto":
        def sample():
            return float(np.random.normal(0, 1, 1)[0])
    elif stragety == "fix":
        def sample():
            return 0.
    prob = prob_map
    mask=prob<thre
    mask=mask.unsqueeze(1)
    mask=torch.repeat_interleave(mask, hidden_variable.shape[1],dim=1)
    resampled_variable=hidden_variable.detach().clone()
    resampled_variable[mask]=sample()
    return resampled_variable
def multi_thre_resample(hidden_variables,prob_map,thresholds,stragety):
    res=[]
    for thre in thresholds:
        res.append(resample(hidden_variables,prob_map,thre,stragety))
    return res
def flow_reverse(resampled_variables,ms_attn_flow,fusion_flow,cond=None,fuse=True,batch=1):
    if not fuse:
        resampled_variables=[[i] for i in resampled_variables]
    num_feature=len(resampled_variables)
    num_thre=len(resampled_variables[0])
    b,_,_,_=resampled_variables[0][0].shape

    resampled_features=[]
    # for i in flow_models:
    #     i.eval()
    for iter_thre in range(num_thre):
        x=[resampled_variables[i][iter_thre] for i in range(num_feature)]
        resampled_feature,_=fusion_flow(x,rev=True)
        # import gc
        # objs = set()
        # for obj in gc.get_objects():
        #     if torch.is_tensor(obj) and not isinstance(obj, torch.nn.parameter.Parameter):
        #         objs.add(obj)
        resampled_feature,_=ms_attn_flow(list(resampled_feature),[cond,],rev=True)
        # for obj in gc.get_objects():
        #     if torch.is_tensor(obj) and not isinstance(obj, torch.nn.parameter.Parameter):
        #         if obj not in objs:
        #             referers = gc.get_referrers(obj)
        resampled_features.append(resampled_feature)
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    resampled_feature2=[sum([resampled_features[i][j] for i in range(num_thre)])/num_thre for j in range(num_feature)]

    return resampled_feature2
def cos_distance(tensor1, tensor2):
    tensor1_flat = tensor1.view(tensor1.size()[:-1] + (-1,))
    tensor2_flat = tensor2.view(tensor2.size()[:-1] + (-1,))
    
    # 计算余弦相似度
    dot_product = torch.sum(tensor1_flat * tensor2_flat, dim=-1)
    norm_tensor1 = torch.norm(tensor1_flat, dim=-1)
    norm_tensor2 = torch.norm(tensor2_flat, dim=-1)
    similarity:torch.Tensor = dot_product / (norm_tensor1 * norm_tensor2)
    
    return similarity.reshape(tensor1.shape[:-1])
def feature_heatmap(feature1,feature2):
    assert feature1.device==feature2.device
    dis=cos_distance(feature1,feature2)
    return dis
def post_process_resampled(c, size_list, outputs_list, ms_attn_flow,fusion_flow, hidden_variables,cond_list=None):
    print("++++RESAMPLE ! ! !+++++++++++++++ ")
    thresholds=c.resample_args["thresholds"]
    stragety=c.resample_args["stragety"]
    num_feature=len(outputs_list)
    num_batch=len(outputs_list[0])

    print('Multi-scale sizes:', size_list)
    logp_maps = [list() for _ in size_list]
    prop_maps = [list() for _ in size_list]

    for iter_batch in tqdm.tqdm(range(num_batch)):
        origin_variables=[hidden_variables[i][iter_batch] for i in range(num_feature)]
        resampled_variables=[]
        for iter_feature in range(num_feature):
            output=outputs_list[iter_feature][iter_batch]
            hidden_variable=hidden_variables[iter_feature][iter_batch]
            output_norm = output - output.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
            prob_map = torch.exp(output_norm) # convert to probs in range [0:1]
            resampled_variable=multi_thre_resample(hidden_variable,prob_map,thresholds,stragety)
            resampled_variables.append(resampled_variable)

        resampled_features=flow_reverse(resampled_variables,ms_attn_flow,fusion_flow,cond_list[iter_batch],fuse=True)
        origin_features=flow_reverse(origin_variables,ms_attn_flow,fusion_flow,cond_list[iter_batch],fuse=False)

        for iter_feat in range(num_feature):
            score=feature_heatmap(origin_features[iter_feat].detach().permute(0,2,3,1),resampled_features[iter_feat].detach().permute(0,2,3,1))
            score=(1+score)/2
            # score[score<1e-2]=1e-2
            prob_map=score
            prop_maps[iter_feat].append(F.interpolate(prob_map.unsqueeze(1),
                size=c.input_size, mode='bilinear', align_corners=True).squeeze(1))
            logp_map=torch.log(prob_map)
            logp_maps[iter_feat].append(F.interpolate(logp_map.unsqueeze(1),
                size=c.input_size, mode='bilinear', align_corners=True).squeeze(1))
    prop_maps=[torch.cat(i,0) for i in prop_maps]
    logp_maps=[torch.cat(i,0) for i in logp_maps]
    logp_map = sum(logp_maps)
    # if torch.any(torch.isnan(logp_map)):
    #     print()

    logp_map-= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    prop_map_mul = torch.exp(logp_map)
    anomaly_score_map_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
    batch = anomaly_score_map_mul.shape[0]
    top_k = int(c.input_size[0] * c.input_size[1] * c.top_k)
    anomaly_score = np.mean(
        anomaly_score_map_mul.reshape(batch, -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
        axis=1)

    prop_map_add = sum(prop_maps)
    prop_map_add = prop_map_add.detach().cpu().numpy()
    anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add

    return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul.detach().cpu().numpy()

if __name__=="__main__":
    a=torch.tensor([[[1,0]]],dtype=torch.float32)
    b=torch.tensor([[[-1,0]]],dtype=torch.float32)
    c=feature_heatmap(a,b)
    print()