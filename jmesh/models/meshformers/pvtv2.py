import jittor as jt
import jittor.nn as nn
import jittor.init as init
import numpy as np
from functools import partial
from jmesh.graph_utils import *
from jmesh.models.meshformers.meshconv import SimSubConv, SubConv
from jmesh.models.transformer.utils import trunc_normal
import math

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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def execute(self, x, k, kset):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        # x = self.dwconv(x.permute(0, 2, 1), k, kset).permute(0, 2, 1)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = SimSubConv(dim, dim)
                self.norm = nn.LayerNorm(dim)
        else:
            # self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def execute(self, x, k, kset, init_faces):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if not self.linear:
            if self.sr_ratio > 1:
                patch_num = N // self.sr_ratio
                # problem here: kset contains index not in x_. 
                x_ = x.reindex([B, C, patch_num], ['i0', '@e0(i0, i2)', 'i1'], extras=[init_faces])
                x_ = self.sr(x_ , self.sr_ratio, kset, x.permute(0, 2, 1)).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            if self.sr_ratio > 1:
                patch_num = N // self.sr_ratio
                x_ = x.reindex([B, C, patch_num, k], ['i0', '@e0(i0, i2, i3)', 'i1'], extras=[kset])
                cx_ = x.reindex([B, C, patch_num], ['i0', '@e0(i0, i2)', 'i1'], extras=[init_faces])
                x_ = jt.concat([x_, cx_.unsqueeze(-1)], -1).mean(-1).permute(0, 2, 1)
                x_ = self.sr(x_)
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False, use_conv=0, knn=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        self.use_conv = use_conv
        self.knn = knn
        if use_conv:
            self.conv = SubConv(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def execute(self, x, k, kset, init_faces, orikset):
        # print(x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x), k, kset, init_faces))
        x = x + self.drop_path(self.mlp(self.norm2(x), k, orikset))
        if self.use_conv:
            x = self.conv(x.permute(0, 2, 1), self.knn, orikset).permute(0, 2, 1)
        return x

def online_process_k(center, patch_num, k, adjdict, kset, need_fp=True):
    N, F, _ = center.shape
    init_faces = get_patches_ffs(center, F, patch_num)
    # centers = center.reindex([N, int(patch_num), 3], ['i0', '@e0(i0, i1)', 'i2'], extras=[init_faces])
    return kset.reindex([N, int(patch_num), k], ['i0', '@e0(i0, i1)', 'i2'], extras=[init_faces]), jt.array(init_faces)

def online_process_downsample(x, center, patch_num, k, adjdict, kset, patch_size, next_size, use_euc=0):
    N, F, _ = center.shape
    init_faces = get_patches_ffs(center, F, patch_num)
    # centers = center[init_faces]
    centers = center.reindex([N, int(patch_num), 3], ['i0', '@e0(i0, i1)', 'i2'], extras=[init_faces])
    res_mat, res_mat_counter = unoverlapping_bfs_patch_cpu(adjdict, F, init_faces)
    # x = patch_aggregation("select", x, init_faces, patch_num)
    x = patch_aggregation("add", x, res_mat, patch_num)
    # x = x.reindex_reduce(op="add",
    #     shape=[N, patch_num, x.shape[2]], # (N, F, C)
    #     indexes=[
    #         'i0',
    #         '@e0(i0,i1)',
    #         'i2'
    #     ],
    #     extras=[res_mat],
    # )
    centers = center.reindex([N, int(patch_num), 3], ['i0', '@e0(i0, i1)', 'i2'], extras=[init_faces])
    if use_euc == 0:
        adjdict = get_hierar_FAF_cpu(patch_num, adjdict, res_mat, next_size)
        kset = k_bfs_patch_cpu(adjdict, patch_num, patch_size, False)
    else:
        adjdict = knn_indices_func_gpu(centers, centers, next_size, 1)
        kset = adjdict[:,:patch_size]
    return x, kset, res_mat, centers, adjdict

class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, patch_size=[8, 8, 8, 8], patch_nums=[512, 128, 32, 8, 8], in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, seg=False, down=1, use_conv=0, knn=3):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.seg = seg
        self.down = down
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            uc = 0
            if use_conv == 1 and i != num_stages - 1:
                uc = 1
            if i < num_stages - 2:
                setattr(self, f"next_size{i+1}", patch_nums[i+2])
                next_size = patch_nums[i+2]
            elif i != num_stages - 1:
                setattr(self, f"next_size{i+1}", patch_size[i+1])
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear,  use_conv=uc, knn=knn)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_num{i + 1}", patch_nums[i])
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"psize{i + 1}", patch_size[i])
            setattr(self, f"sr{i+1}", sr_ratios[i])
        setattr(self, f"patch_num{len(patch_nums)}", patch_nums[-1])
        # classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, kset, k, center, adjdict, res_face, init_faces, preprocess, use_euc=0):
        B = x.shape[0]
        ksets = []
        res_faces_list = []
        feats = []
        for i in range(self.num_stages):
            # patch_embed = getattr(self, f"patch_embed{i + 1}")
            N = x.shape[1]
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            if getattr(self, f"sr{i + 1}") > 1:
                patch_num = N // getattr(self, f"sr{i + 1}")
                if preprocess == 0:
                    kset_, init_face_ = online_process_k(center, patch_num, getattr(self, f"psize{i + 1}"), adjdict, kset)
                else:
                    kset_ = jt.array(kset[i+1])
                    init_face_ = jt.array(init_faces[i])
                    # print(kset_.shape)
                    # print(init_face_.shape)
                for blk in block:
                    x = blk(x, k, kset_, init_face_, kset)
                x = norm(x)
            else:
                for blk in block:
                    x = blk(x, k, None, None, None)
                x = norm(x)
            if self.seg == 1:
                feats.append(x)
            if i != self.num_stages - 1 and self.down == 1:
                # downsample
                if preprocess == 0:
                    x, kf, res_face, center, adjdict = online_process_downsample(x, center, getattr(self, f"patch_num{i + 2}"), k, adjdict, kset, getattr(self, f"psize{i+1}"), getattr(self, f"next_size{i+1}"), use_euc=use_euc)
                    if self.seg:
                        ksets.append(kf)
                        res_faces_list.append(res_face)
                else:
                    N = x.shape[0]
                    x = x.reindex_reduce(op="add",
                        shape=[N, getattr(self, f"patch_num{i + 2}"), x.shape[2]], # (N, F, C)
                        indexes=[
                            'i0',
                            '@e0(i0,i1)',
                            'i2'
                        ],
                        extras=[jt.array(res_face[i+1])],
                    )
        if self.seg:
            return x, ksets, feats, res_faces_list
        else:   
            return x.mean(dim=1), None, None, None

    def execute(self, x, kset, k, center, adjdict, res_face, init_faces, preprocess=0, use_euc=0):
        if preprocess == 0:
            x, ksets, feats, res_faces_list = self.forward_features(x, kset, k, center, adjdict, res_face, None, preprocess, use_euc=use_euc)
        else:
            x, ksets, feats, res_faces_list = self.forward_features(x, kset, k, center, adjdict, res_face, init_faces, preprocess, use_euc=use_euc)
        x = self.head(x)

        return x, ksets, feats, res_faces_list


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = SimSubConv(dim, dim)

    def execute(self, x, k, kset):
        return self.dwconv(x, k, kset)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

