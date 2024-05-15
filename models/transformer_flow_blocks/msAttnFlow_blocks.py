from .transformer_blocks import *


import FrEIA.modules as Fm

import torch
import torch.nn as nn










class MSAttnFlowBlock(nn.Module):
    def __init__(self,variable_dims:tuple[tuple[int]],attn_block,
                 **kwargs) -> None:
        super().__init__()
        self.variable_dims=list(variable_dims[0])
        self.flow_blocks=nn.ModuleList()
        for num,dim in enumerate(self.variable_dims):
            dims_in=[(dim[1],1,1)]
            flow_block=Fm.AllInOneBlock(dims_in,**kwargs)
            self.flow_blocks.append(flow_block)
        self.attn_block=attn_block(self.variable_dims)
    def output_dims(self,dim_in):
        return dim_in
    def forward(self,hidden_variables:list[torch.tensor],c=[],jac:bool=True,rev:bool=False):
        h=[]
        jac_lis=[]
        for i,x in enumerate(hidden_variables):
            y,jac=self.flow_blocks[i]((x,),[c[0][i]],rev=rev,jac=jac)
            h.append(y[0])
            jac_lis.append(jac)
        res,attn_jac=self.attn_block(h)
        return res,torch.tensor(jac_lis)