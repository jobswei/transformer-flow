import sys
sys.path.append("/root/transformer-flow")
from models.transformer_flow_blocks.transformer_blocks import *


import FrEIA.modules as Fm

import torch
import torch.nn as nn










class MSAttnFlowBlock(nn.Module):
    def __init__(self,variable_dims:tuple[tuple[tuple[int]]],use_attn=True,use_all_channels=False,attn_block=None,
                 use_ffn=False,ffn_residual=False,use_norm=False,reverse_blocks=False,
                 **kwargs) -> None:
        super().__init__()
        self.variable_dims=list(variable_dims[0])
        self.use_attn=use_attn
        self.use_ffn=use_ffn
        self.use_norm=use_norm
        self.reverse_blocks=reverse_blocks
        self.flow_blocks=nn.ModuleList()
        for num,dim in enumerate(self.variable_dims):
            dims_in=[(dim[1],1,1)]
            flow_block=Fm.AllInOneBlock(dims_in,**kwargs)
            self.flow_blocks.append(flow_block)
        if self.use_attn:
            self.attn_block=attn_block(self.variable_dims,use_all_channels)
        if self.use_ffn:
            residual = ffn_residual
            self.ffn=FeedForward(self.variable_dims,residual=residual)
            # self.ffn=InvConv2dLU(self.variable_dims[0][1])
        if self.use_norm:
            self.norm=Normalize(self.variable_dims[0][1])
    def output_dims(self,dim_in):
        return dim_in
    def forward(self,hidden_variables:list[torch.tensor],c=[[None,None,None]],jac:bool=True,rev:bool=False):
        if rev:
            return self.forward_rev(hidden_variables,c,jac)
        if not self.reverse_blocks:
            h=[]
            jac_lis=[]
            for i,x in enumerate(hidden_variables):
                y,jac=self.flow_blocks[i]((x,),[c[0][i]],rev=rev,jac=jac)
                h.append(y[0])
                jac_lis.append(jac)
            jac_lis=torch.stack(jac_lis,dim=0)
            res=h
            if self.use_attn:
                res,attn_jac=self.attn_block(h)
            if self.use_ffn:
                res,ffn_jac=self.ffn(res)
                jac_lis+=ffn_jac
            if self.use_norm:
                res,norm_jac=self.norm_forward(res)
                jac_lis+=norm_jac
            return res,jac_lis
        else:
            jac_lis=torch.zeros((len(hidden_variables),hidden_variables[0].shape[0])).to(hidden_variables[0].device)
            if self.use_attn:
                res,attn_jac=self.attn_block(hidden_variables)
            if self.use_ffn:
                res,ffn_jac=self.ffn(res)
                jac_lis+=ffn_jac
            if self.use_norm:
                res,norm_jac=self.norm_forward(res)
                jac_lis+=norm_jac
            h=[]
            temp=[]
            for i,x in enumerate(res):
                y,jac=self.flow_blocks[i]((x,),[c[0][i]],rev=rev,jac=jac)
                h.append(y[0])
                temp.append(jac)
            jac_lis+=torch.stack(temp,dim=0)
            res=h
            return res,jac_lis
    def forward_rev(self,features:list[torch.tensor],c=[],jac:bool=True):
        if self.use_norm:
            features,norm_jac=self.norm_forward(features,rev=True)
            # jac_lis+=norm_jac
        if self.use_ffn:
            features,ffn_jac=self.ffn(features,rev=True)
            # jac_lis+=ffn_jac
        if self.use_attn:
            features,attn_jac=self.attn_block(features,rev=True)
        res=[]
        for i,x in enumerate(features):
            y,jac=self.flow_blocks[i]((x,),[c[0][i]],rev=True,jac=jac)
            res.append(y[0])
        return res,0
    # def ffn_forward(self,x,rev=False):
    #     jac_lis=[]
    #     results=[]
    #     for num,attn in enumerate(x):
    #         attn,ffn_log_jac=self.ffn(attn,rev=rev)
    #         results.append(attn)
    #         jac_lis.append(ffn_log_jac)
    #     return results,torch.stack(jac_lis,dim=0)
    def norm_forward(self,x,rev=False):
        jac_lis=[]
        results=[]
        for num,attn in enumerate(x):
            attn,ffn_log_jac=self.norm(attn,rev=rev)
            results.append(attn)
            jac_lis.append(ffn_log_jac)
        return results,torch.stack(jac_lis,dim=0)

if __name__=="__main__":
    from models.flow_models import *
    models=[MSAttnFlowBlock(([[3,256,16,16],[3,128,32,32]],),attn_block=AttentionTD,use_all_channels=True,dims_c=[(64,1,1)],subnet_constructor=subnet_conv_ln,affine_clamping=1.9,global_affine_type='SOFTPLUS',use_attn=True,use_ffn=False,use_norm=False,)\
            for _ in range(6)]
    x=[255*torch.rand([3,256,16,16]),255*torch.rand([3,128,32,32])]
    cond=[torch.rand([3,64,16,16]),torch.rand([3,64,32,32])]
    y=x
    for model in models:
        y,_=model(y,(cond,))
    z=y
    for model in models[::-1]:
        z,_=model(z,(cond,),rev=True)
    print(z[0]-x[0])