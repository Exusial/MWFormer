import jittor as jt 
from jittor import nn
from jittor.misc import _pair
from functools import partial
import math
import copy
import numpy as np
from jmesh.graph_utils import *
from jmesh.models.transformer.utils import trunc_normal,load_pretrained,_conv_filter
from jmesh.models.transformer.config import default_cfgs
from jmesh.models.meshformers.meshconv import *

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x):
        b,n,c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c// self.num_heads).permute(2, 0, 3, 1, 4)
        
        q,k,v = qkv[0],qkv[1],qkv[2]

        # attn = nn.bmm(q,k.transpose(0,1,3,2))*self.scale
        attn = nn.bmm_transpose(q, k)*self.scale
        
        attn = nn.softmax(attn,dim=-1)

        attn = self.attn_drop(attn)

        out = nn.bmm(attn,v)
        out = out.transpose(0,2,1,3).reshape(b,n,c)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if self.drop_prob == 0. or not self.is_training():
            return x
        keep_prob = 1-self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + jt.random(shape, dtype=x.dtype)
        random_tensor = jt.floor(random_tensor)  # binarize
        output = (x / keep_prob) * random_tensor
        return output


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block,self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, 
                              num_heads=num_heads, 
                              qkv_bias=qkv_bias, 
                              qk_scale=qk_scale, 
                              attn_drop=attn_drop, 
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, 
                       hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, 
                       drop=drop)

    def execute(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def enhance_subconv_global(feats, mlp):
    gfeature = feats[:,0,:]
    return mlp(jt.concat([gfeature, (gfeature[:,None,:] - feats[:,1:,:]).sum(1)], -1)).unsqueeze(1)

def enhance_face_global(feats, res_faces, adjdict, oriadjdict, patch_num, mlp):
    B, N, C = feats.shape
    fs = oriadjdict.shape[1]
    gfeature = feats[:,0,:]
    pfeature = feats[:,1:patch_num+1,:]
    sfeature = feats[:,patch_num+1:,:]
    patch_source_f = pfeature - sfeature.reindex_reduce("add", [B, patch_num, C], 
    ["i0", "@e0(i0, i1)", "i2"], extras=[res_faces])
    l = adjdict.shape[-1]
    patch_patch_f = (pfeature[...,None,:] - pfeature.reindex( [B, patch_num, l, C], ["i0", "@e0(i0, i1, i2)", "i3"],
    extras=[adjdict], overflow_conditions=["@e0(i0, i1, i2)==-1"])).sum(2)
    epfeature = mlp(jt.concat([pfeature, pfeature-gfeature[:,None,:]+patch_source_f+patch_patch_f], -1))
    source_patch_f = sfeature - pfeature.reindex([B, fs, C], 
    ["i0", "@e0(i0, i1)", "i2"], extras=[res_faces])
    l = oriadjdict.shape[-1]
    source_source_f = (sfeature[...,None,:]-sfeature.reindex([B, fs, l, C], ["i0", "@e0(i0, i1, i2)", "i3"],
    extras=[oriadjdict])).sum(2)
    esfeature = mlp(jt.concat([sfeature, sfeature-gfeature[:,None,:]+source_patch_f+source_source_f],-1))
    return epfeature, esfeature

class ShapeBlock(nn.Module):
    def __init__(self, dim=128, patch_size=16, num_heads=4, qkv_bias=False, qk_scale=None, depth=4,
    drop=0., attn_drop=0., drop_path=0., act_layer=nn.gelu, norm_layer=nn.LayerNorm):
        self.dim = dim
        self.norm = norm_layer(dim)
        self.enhance_mlp = nn.Linear(dim*2, dim)
        self.attn = Attention(dim, 
                              num_heads=num_heads, 
                              qkv_bias=qkv_bias, 
                              qk_scale=qk_scale, 
                              attn_drop=attn_drop, 
                              proj_drop=drop)

    def execute(self, x, res_faces, adjdict, oriadjdict):
        # patch_num = adjdict.shape[1]
        # en_x = jt.concat([enhance_subconv_global(x, self.enhance_mlp), 
        # *enhance_face_global(x, res_faces, adjdict, oriadjdict, patch_num, self.enhance_mlp)], 1)
        return x + self.attn(self.norm(x))

def get_qkv(feat, qkv):
    if len(feat.shape) > 2:
        b, n, c = feat.shape
    else:
        b, c = feat.shape
        n = 1
    qkv = qkv(feat).reshape(b, n, 3, c).permute(2, 0, 1, 3)
    return qkv[[0,2]],qkv[1]

class TopologyBlock(nn.Module):
    def __init__(self, dim=128, qkv_bias=False, qk_scale=None,
    drop=0., attn_drop=0., drop_path=0., act_layer=nn.gelu, norm_layer=nn.LayerNorm):
        self.act_fn = act_layer
        self.global_qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.patch_qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.source_qkv = nn.Linear(dim, dim*3, bias=qkv_bias)

    def get_patch_feature(self, qvg, kp, qvp, qvs, adjdict, p2f):
        b, patch_num, c = kp.shape
        fs = qvs.shape[2]
        psnqv = qvs.reindex([2, b, patch_num, p2f.shape[-1], c], ["i0", "i1", "@e0(i1, i2, i3)", "i4"], extras=[p2f],
        overflow_conditions=["@e0(i1, i2, i3) == -1"]) 
        ppnqv = qvp.reindex([2, b, patch_num, adjdict.shape[-1], c], ["i0", "i1", "@e0(i1, i2, i3)", "i4"], extras=[adjdict],
        overflow_conditions=["@e0(i1, i2, i3) == -1"]) 
        psnq, psnv = psnqv[0], psnqv[1]
        ppnq, ppnv = ppnqv[0], ppnqv[1]
        qg, vg = qvg[0], qvg[1]
        edge_ps = (psnv * nn.softmax(psnq * kp[...,None,:])).sum(2)
        edge_pp = (ppnv * nn.softmax(ppnq * kp[...,None,:])).sum(2)
        edge_pg = (vg * nn.softmax(qg * kp))
        return edge_ps + edge_pp + edge_pg
    
    def get_source_feature(self, qvg, qvp, ks, qvs, oriadjdict, res_faces):
        b, fs, c = ks.shape
        patch_num = qvp.shape[2]
        psnqv = qvs.reindex([2, b, fs, c], ["i0", "i1", "@e0(i1, i2)", "i3"], extras=[res_faces],
        overflow_conditions=["@e0(i1, i2) == -1"]) 
        ppnqv = qvp.reindex([2, b, fs, oriadjdict.shape[-1], c], ["i0", "i1", "@e0(i1, i2, i3)", "i4"], extras=[oriadjdict],
        overflow_conditions=["@e0(i1, i2, i3) == -1"]) 
        psnq, psnv = psnqv[0], psnqv[1]
        ppnq, ppnv = ppnqv[0], ppnqv[1]
        qg, vg = qvg[0], qvg[1]
        edge_ps = (psnv * nn.softmax(psnq * ks))
        edge_pp = (ppnv * nn.softmax(ppnq * ks[...,None,:])).sum(2)
        edge_pg = (vg * nn.softmax(qg * ks))
        return edge_ps + edge_pp + edge_pg

    def execute(self, x, res_faces, adjdict, oriadjdict, p2f):
        B, patch_num, _ = adjdict.shape
        global_feature = x[:,0,:]
        patch_feature = x[:,1:1+patch_num,:]
        source_feature = x[:,1+patch_num:,:]
        qvg, kg = get_qkv(global_feature, self.global_qkv)
        qvp, kp = get_qkv(patch_feature, self.patch_qkv)
        qvs, ks = get_qkv(source_feature, self.source_qkv)
        # process global feature
        # print(qvg.shape, kg.shape, qvp.shape, global_feature.shape, (jt.concat([qvp[1], qvs[1]], 1) * nn.softmax(kg * jt.concat([qvp[0], qvs[0]], 1))).sum(1, keepdims=True).shape)
        global_feature = global_feature[:,None,:] + self.act_fn((jt.concat([qvp[1], qvs[1]], 1) * nn.softmax(kg * jt.concat([qvp[0], qvs[0]], 1))).sum(1, keepdims=True))
        # process patch feature
        patch_feature = patch_feature + self.act_fn(self.get_patch_feature(qvg, kp, qvp, qvs, adjdict, p2f))
        #process source_feature
        source_feature = source_feature + self.act_fn(self.get_source_feature(qvg, qvp, ks, qvs, oriadjdict, res_faces))
        # print("finish: ", global_feature.shape, patch_feature.shape, source_feature.shape)
        return jt.concat([global_feature, patch_feature, source_feature], 1)

class Block(nn.Module):
    def __init__(self, dim=128, num_heads=4, qkv_bias=False, qk_scale=None,
    drop=0., attn_drop=0., act_layer=nn.gelu, norm_layer=nn.LayerNorm):
        self.shape_attn = ShapeBlock(dim=dim, num_heads=num_heads, qkv_bias=False, qk_scale=None, drop=drop, attn_drop=attn_drop, act_layer=act_layer, norm_layer=norm_layer)
        self.top_attn = TopologyBlock(dim=dim, qkv_bias=False, qk_scale=None, act_layer=act_layer, norm_layer=norm_layer)
    
    def execute(self, x, res_faces, adjdict, oriadjdict, p2f):
        x = self.shape_attn(x, res_faces, adjdict, oriadjdict)
        # x = self.top_attn(x, res_faces, adjdict, oriadjdict, p2f)
        return x

class Graphomer(nn.Module):
    def __init__(self, dim=[128], num_heads=[4], depth=6, qkv_bias=False, qk_scale=None,
    drop=0., attn_drop=0., act_layer=nn.gelu, norm_layer=nn.LayerNorm, task_type="seg", num_classes=5):
        depth = depth[0]
        self.blocks = nn.ModuleList([Block(
            dim=dim[i], num_heads=num_heads[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, act_layer=act_layer, norm_layer=norm_layer) for i in range(depth)])
        self.task_type = task_type
        if task_type == "cls":
            self.head = nn.Linear(dim[-1], num_classes)
        else:
            self.attn_norm = norm_layer(dim[-1] * 3)
            self.attn_head = Attention(dim[-1]*3, num_heads=1, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
            self.head = nn.Linear(dim[-1], num_classes)
            self.seg_head = nn.Linear(dim[-1]*3, num_classes)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def execute(self, x, res_faces, adjdict, oriadjdict, p2f):
        fs = oriadjdict.shape[1]
        patch_num = adjdict.shape[1]
        # print(x.shape, p2f.shape, adjdict.shape, oriadjdict.shape)
        for block in self.blocks:
            x = block(x, res_faces, adjdict, oriadjdict, p2f)
        if self.task_type == "seg":
            b, n, c = x.shape
            gf = x[:,0,:].unsqueeze(1).expand(1,fs,1)
            f2p = x[:,1:1+patch_num,:].reindex([b, fs, c], ["i0","@e0(i0,i1)","i2"], extras=[res_faces])
            f_new = jt.concat([x[:,1+patch_num:,:], f2p, gf],2)
            # qkv = self.head(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
            # q, k, v = qkv[0], qkv[1], qkv[2]
            # f_new = v * nn.softmax(q * k)
            f_new = self.attn_head(self.attn_norm(f_new))
            # f_new = x[:,1+patch_num:,:]
            return self.seg_head(f_new).permute(0,2,1)
        else:
            return self.head(x[:,0,...])

