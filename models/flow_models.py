import math

import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from .freia_utils import FusionCouplingLayer
from .transformer_flow_blocks.transformer_blocks import TransformFlowBlock
from .transformer_flow_blocks.msAttnFlow_blocks import *
class ConvNeck(nn.Module):
    def __init__(self, in_channels,out_channels) -> None:
        super().__init__()
        self.convs=nn.ModuleList()
        for c_in,c_out in zip(in_channels,out_channels):
            self.convs.append(nn.Conv2d(c_in,c_out,1))
    def forward(self,x):
        assert len(x)==len(self.convs),f"{len(x)}=={len(self.convs)}"
        out=[]
        for num,i in enumerate(x):
            out.append(self.convs[num](i))
        return out
def build_transformer_flow_model(c, c_feats, all_dims):
    c_conds = c.c_conds
    n_block = c.num_transformer_blocks
    clamp_alpha = c.clamp_alpha

    # mid_c=sum(c_feats)//len(c_feats)
    mid_c=1024
    mid_c=8*(mid_c//8)  # 和多头注意力的head对齐
    c_feats_new=[mid_c]*len(c_feats)
    print('Build Conv Neck: in_channels:{}, out_channels:{}'.format(c_feats, c_feats_new))
    conv_neck=ConvNeck(c_feats,c_feats_new)

    print('Build transformer flow: channels:{}, block:{}, cond:{}'.format(c_feats_new, n_block, None))
    for i in all_dims:
        i[1]=mid_c
    transformer_flow=Ff.SequenceINN(*all_dims,force_tuple_output=True)
    for k in range(n_block):
        transformer_flow.append(TransformFlowBlock)

    print("Build fusion flow with channels", c_feats_new)
    nodes = list()
    n_inputs = len(c_feats_new)
    for idx, c_feat in enumerate(c_feats_new):
        nodes.append(Ff.InputNode(c_feat, 1, 1, name='input{}'.format(idx)))
    for idx in range(n_inputs):
        nodes.append(Ff.Node(nodes[-n_inputs], Fm.PermuteRandom, {}, name='permute_{}'.format(idx)))
    nodes.append(Ff.Node([(nodes[-n_inputs+i], 0) for i in range(n_inputs)], FusionCouplingLayer, {'clamp': clamp_alpha}, name='fusion flow'))
    for idx, c_feat in enumerate(c_feats_new):
        nodes.append(Ff.OutputNode(eval('nodes[-idx-1].out{}'.format(idx)), name='output_{}'.format(idx)))
    fusion_flow = Ff.GraphINN(nodes)

    return conv_neck,transformer_flow, fusion_flow

def build_ms_attn_flow_model(c, c_feats, all_dims):
    c_conds = c.c_conds
    n_block = c.num_transformer_blocks
    clamp_alpha = c.clamp_alpha
    
    if c.use_conv:
        mid_c=sum(c_feats)//len(c_feats)
        mid_c=8*(mid_c//8)  # 和多头注意力的head对齐
        c_feats_new=[mid_c]*len(c_feats)
        print('Build Conv Neck: in_channels:{}, out_channels:{}'.format(c_feats, c_feats_new))
        conv_neck=ConvNeck(c_feats,c_feats_new)
        c_feats=c_feats_new
        for i in all_dims:
            i[1]=mid_c
    else:
        conv_neck=nn.Module()

    msAttn_flow=Ff.SequenceINN(*all_dims,force_tuple_output=True)
    print('Build msAttn flow: channels:{}, block:{}, cond:{}'.format(c_feats, n_block, c_conds))
    for i in range(n_block):
        if c.attn_all:
            attn_block=AttentionAll
        else:
            if i//2==0:
                attn_block=AttentionBU
            else:
                attn_block=AttentionTD
        msAttn_flow.append(MSAttnFlowBlock,cond=0,cond_shape=(c_conds[0],1,1),
                        use_attn=c.use_attn,use_ffn=c.use_ffn,use_norm=c.use_norm,
                        attn_block=attn_block,subnet_constructor=subnet_conv_ln, affine_clamping=clamp_alpha,global_affine_type='SOFTPLUS')

    print("Build fusion flow with channels", c_feats)
    nodes = list()
    n_inputs = len(c_feats)
    for idx, c_feat in enumerate(c_feats):
        nodes.append(Ff.InputNode(c_feat, 1, 1, name='input{}'.format(idx)))
    for idx in range(n_inputs):
        nodes.append(Ff.Node(nodes[-n_inputs], Fm.PermuteRandom, {}, name='permute_{}'.format(idx)))
    nodes.append(Ff.Node([(nodes[-n_inputs+i], 0) for i in range(n_inputs)], FusionCouplingLayer, {'clamp': clamp_alpha}, name='fusion flow'))
    for idx, c_feat in enumerate(c_feats):
        nodes.append(Ff.OutputNode(eval('nodes[-idx-1].out{}'.format(idx)), name='output_{}'.format(idx)))
    fusion_flow = Ff.GraphINN(nodes)

    return conv_neck, msAttn_flow, fusion_flow

def subnet_conv(dims_in, dims_out):
    return nn.Sequential(nn.Conv2d(dims_in, dims_in, 3, 1, 1), nn.ReLU(True), nn.Conv2d(dims_in, dims_out, 3, 1, 1))

def subnet_conv_bn(dims_in, dims_out):
    return nn.Sequential(nn.Conv2d(dims_in, dims_in, 3, 1, 1), nn.BatchNorm2d(dims_in), nn.ReLU(True), nn.Conv2d(dims_in, dims_out, 3, 1, 1))

class subnet_conv_ln(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_mid = dim_in
        self.conv1 = nn.Conv2d(dim_in, dim_mid, 3, 1, 1)
        self.ln = nn.LayerNorm(dim_mid)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(dim_mid, dim_out, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.conv2(out)

        return out

def single_parallel_flows(c_feat, c_cond, n_block, clamp_alpha, subnet=subnet_conv_ln):
    flows = Ff.SequenceINN(c_feat, 1, 1)
    print('Build parallel flows: channels:{}, block:{}, cond:{}'.format(c_feat, n_block, c_cond))
    for k in range(n_block):
        flows.append(Fm.AllInOneBlock, cond=0, cond_shape=(c_cond, 1, 1), subnet_constructor=subnet, affine_clamping=clamp_alpha,
            global_affine_type='SOFTPLUS')
    return flows

def build_msflow_model(c, c_feats):
    c_conds = c.c_conds
    n_blocks = c.parallel_blocks
    clamp_alpha = c.clamp_alpha
    parallel_flows = []
    for c_feat, c_cond, n_block in zip(c_feats, c_conds, n_blocks):
        parallel_flows.append(
            single_parallel_flows(c_feat, c_cond, n_block, clamp_alpha, subnet=subnet_conv_ln))
    
    print("Build fusion flow with channels", c_feats)
    nodes = list()
    n_inputs = len(c_feats)
    for idx, c_feat in enumerate(c_feats):
        nodes.append(Ff.InputNode(c_feat, 1, 1, name='input{}'.format(idx)))
    for idx in range(n_inputs):
        nodes.append(Ff.Node(nodes[-n_inputs], Fm.PermuteRandom, {}, name='permute_{}'.format(idx)))
    nodes.append(Ff.Node([(nodes[-n_inputs+i], 0) for i in range(n_inputs)], FusionCouplingLayer, {'clamp': clamp_alpha}, name='fusion flow'))
    for idx, c_feat in enumerate(c_feats):
        nodes.append(Ff.OutputNode(eval('nodes[-idx-1].out{}'.format(idx)), name='output_{}'.format(idx)))
    fusion_flow = Ff.GraphINN(nodes)

    return parallel_flows, fusion_flow
