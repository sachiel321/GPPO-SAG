import os
import math
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from einops import repeat,rearrange
from copy import deepcopy

'''Attention Block'''
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1   
    resid_pdrop = 0.1  
    attn_pdrop = 0.1   
    mask = False

    def __init__(self,block_size, **kwargs):
        self.block_size = block_size    
        for k,v in kwargs.items():
            setattr(self, k, v)   
class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768
    init_gru_gate_bias: float = 2.0


class TransformerEncoder(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=0.00001, batch_first=False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, device, dtype)
    def forward(self, q, k, v, src_mask = None, src_key_padding_mask = None):

        src2 = self.self_attn(q, k, v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = q + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class BlockGTrXLTS(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        configS = deepcopy(config)
        configT = deepcopy(config)
        configS.n_embd = config.n_embdS
        # configS.block_size = configT.n_embd

        self.TrXL_T1 = TransformerEncoder(d_model=configT.n_embd, nhead=8, dim_feedforward=512, dropout=configT.resid_pdrop)
        self.TrXL_T2 = TransformerEncoder(d_model=configT.n_embd, nhead=8, dim_feedforward=512, dropout=configT.resid_pdrop)
        self.TrXL_T3 = TransformerEncoder(d_model=configT.n_embd, nhead=8, dim_feedforward=512, dropout=configT.resid_pdrop)

        self.transformT2S = nn.Linear(32, 1)
        self.transformS2T = nn.Linear(1, 32)

        self.TrXL_S = TransformerEncoder(d_model=configS.n_embd, nhead=8, dim_feedforward=512, dropout=configS.resid_pdrop)

    def forward(self, s_a, g, attn_bias=None):
        s_a = s_a.permute(1, 0, 2)
        q = self.TrXL_T1(s_a, s_a, s_a)
        q = self.TrXL_T2(q, q, q)
        q = self.TrXL_T3(q, q, q).permute(1, 0, 2)

        batch_size = q.shape[0]
        q = rearrange(q,'b t c -> b c t') # (b,32,c) -> (b,c,32)
        q = self.transformT2S(q)
        q = rearrange(q,'b t c -> b c t') # (b,c,1) -> (b,1,c)

        q = self.TrXL_S(q.permute(1, 0, 2), g.permute(1, 0, 2), g.permute(1, 0, 2)).permute(1, 0, 2)
        # q = rearrange(q,'b t c -> b c t') 
        # q = self.transformS2T(q)
        # q = rearrange(q,'b t c -> b c t') 
        return q

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        configS = deepcopy(config)
        configT = deepcopy(config)
        configS.n_embd = config.n_embdS
        # configS.block_size = configT.n_embd

        self.TrXL_T1 = TransformerEncoder(d_model=configT.n_embd, nhead=8, dim_feedforward=512, dropout=configT.resid_pdrop, batch_first=True)
        self.TrXL_T2 = TransformerEncoder(d_model=configT.n_embd, nhead=8, dim_feedforward=512, dropout=configT.resid_pdrop, batch_first=True)


    def forward(self, s_a, g, attn_bias=None):
        q = self.TrXL_T1(s_a, s_a, s_a)
        q = self.TrXL_T2(q, q, q)
        return q

class BlockSeq(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.layer = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

    def forward(self,  q, k, attn_bias=None):

        for _, layer_module in enumerate(self.layer):
            q = layer_module( q, k, attn_bias)

        return q


class ReverseGRD(nn.Module):
    """  the reverse general robot dynamics model, without encoder and decoder """   

    def __init__(self, config):
        super().__init__()

        self.embd = config.n_embd
        self.tok_emb_s = nn.Linear(config.state_dim, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, 100, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.att_bias_encoder = nn.Linear(config.state_dim,config.n_embd)
        # transformer
        self.blocks = BlockSeq(config)
        self.block_size = 100
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, q,k=None,att_bias=None):
        '''
        BatchSize, TimeSequence, StateEmbedding = input.size()
        Output with the same size
        BatchSize,  T_q, T_k, 1 = att_bias.size()
        '''
        data_dim = q.dim()
        if k == None:
            k=q
        if data_dim == 3:
            b, t_q, c = q.size()
            b, t_k, c = k.size()
        elif data_dim == 4:
            t, b, t_q, c = q.size()
            t, b, t_k, c = k.size()
            q = q.reshape(-1,t_q,c)
            k = k.reshape(-1,t_k,c)
        assert t_k <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings_q = self.pos_emb[:, :t_q, :] # each position maps to a (learnable) vector
        position_embeddings_kv = self.pos_emb[:, :t_k, :] # each position maps to a (learnable) vector
        if att_bias is not None:
            att_bias = repeat(att_bias,'q k c -> b q k c',b=b)
            att_bias = self.att_bias_encoder(att_bias.to(q.device))
        # q = self.drop(self.tok_emb_s(q) + position_embeddings_q)
        # k = self.drop(self.tok_emb_g(k) + position_embeddings_kv)

        q = self.drop(q + position_embeddings_q)
        k = self.drop(k + position_embeddings_kv)

        # x, encoded_layers, layer_atts = self.blocks(q,k,v,att_bias)
        x = self.blocks(q,k,att_bias)
        x = self.ln_f(x)

        output = x.mean(1).squeeze(1)
        if data_dim == 4:
            output = output.reshape(t,b,-1)
        return output
        
"""utils"""


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)

class ConcatNet(nn.Module):  # concatenate
    def __init__(self, mid_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.dense2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.dense3 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.dense4 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.out_dim = mid_dim * 4

    def forward(self, x0):
        x1 = self.dense1(x0)
        x2 = self.dense2(x0)
        x3 = self.dense3(x0)
        x4 = self.dense4(x0)

        return torch.cat((x1, x2, x3, x4), dim=1)

class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, mid_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(mid_dim // 2, mid_dim // 2), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(mid_dim * 1, mid_dim * 1), nn.Hardswish())
        self.inp_dim = mid_dim // 2
        self.out_dim = mid_dim * 2

    def forward(self, x1):  # x1.shape == (-1, mid_dim // 2)
        x2 = torch.cat((x1, self.dense1(x1)), dim=2)
        x3 = torch.cat((x2, self.dense2(x2)), dim=2)
        return x3  # x3.shape == (-1, mid_dim * 2)


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
