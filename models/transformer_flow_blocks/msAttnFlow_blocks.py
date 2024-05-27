from .transformer_blocks import *


import FrEIA.modules as Fm

import torch
import torch.nn as nn










class MSAttnFlowBlock(nn.Module):
    def __init__(self,variable_dims:tuple[tuple[int]],use_attn=True,attn_block=None,
                 use_ffn=False,use_norm=False,
                 **kwargs) -> None:
        super().__init__()
        self.variable_dims=list(variable_dims[0])
        self.use_attn=use_attn
        self.use_ffn=use_ffn
        self.use_norm=use_norm
        self.flow_blocks=nn.ModuleList()
        for num,dim in enumerate(self.variable_dims):
            dims_in=[(dim[1],1,1)]
            flow_block=Fm.AllInOneBlock(dims_in,**kwargs)
            self.flow_blocks.append(flow_block)
        if self.use_attn:
            self.attn_block=attn_block(self.variable_dims)
        if self.use_ffn:
            self.ffn=FeedForward(self.variable_dims[0][1],self.variable_dims[0][1])
        if self.use_norm:
            self.norm=Normalize(self.variable_dims[0][1])
    def output_dims(self,dim_in):
        return dim_in
    def forward(self,hidden_variables:list[torch.tensor],c=[],jac:bool=True,rev:bool=False):
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
            res,ffn_jac=self.ffn_forward(res)
            jac_lis+=ffn_jac
        if self.use_norm:
            res,norm_jac=self.norm_forward(res)
            jac_lis+=norm_jac
        return res,jac_lis
    def ffn_forward(self,x,rev=False):
        jac_lis=[]
        results=[]
        for num,attn in enumerate(x):
            attn,ffn_log_jac=self.ffn(attn,rev=rev)
            results.append(attn)
            jac_lis.append(ffn_log_jac)
        return results,torch.stack(jac_lis,dim=0)
    def norm_forward(self,x,rev=False):
        jac_lis=[]
        results=[]
        for num,attn in enumerate(x):
            attn,ffn_log_jac=self.norm(attn,rev=rev)
            results.append(attn)
            jac_lis.append(ffn_log_jac)
        return results,torch.stack(jac_lis,dim=0)