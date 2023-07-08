import logging
import math
from copy import deepcopy
from typing import Optional

import jittor as jt
import jittor.nn as nn
import numpy as np
from jmesh.models.meshformers.meshconv import *
from jmesh.models.transformer.utils import trunc_normal
from jmesh.graph_utils import *

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
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

def window_partition(x, nw, k, kset):
    """
    Args:
        x: (B, N, C)
        kset: (nw*B, k)
        k (int): window size
    Returns:
        windows: (nw*B, k, C)
    """
    B, N, C = x.shape
    # jt.sync_all()
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    x_ = x.reindex([nw*B, k ,C], ['i0 / @e1(0)', '@e0(i0, i1)', 'i2'], extras=[kset, jt.array(nw)])
    return x_


def window_reverse(windows, window_size: int, nw, patch_num, win_kset):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        win_kset: nw * B, k
    Returns:
        x: (B, patch_num, C)
    """
    B = int(windows.shape[0] / (nw))
    _, k, C = windows.shape
    # x = windows.view(B, nw, window_size, window_size, -1)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return windows.reindex_reduce(op='add', 
        shape=[B, patch_num, C], 
        indexes=['i0 / @e1(0)', '@e0(i0, i1)', 'i2'], 
        extras=[win_kset, jt.array(nw)])


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # k
        self.relative_encoder = nn.Linear(3, 1)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.apply(self._init_weights)

    def execute(self, x, win_center, win_kset, mask = None):
        """
        Args:
            win_center [B, W, 3] win_kset [nw * B, window_size]
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # todo: add mlp rel pos bias ?
        # ori_center = win_center.reindex([B_, self.window_size, self.window_size, 3], ['i0', '@e0(i0, i2)', 'i3'], extras=[win_kset])
        # k_center = win_center.reindex([B_, self.window_size, self.window_size, 3], ['i0', '@e0(i0, i1)', 'i3'], extras=[win_kset])
        # rel_encode = self.relative_encoder(ori_center - k_center).squeeze(-1)
        # attn = attn + rel_encode.unsqueeze(1).expand(-1, self.num_heads, -1 , -1)
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.softmax(attn)
        # else:
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, patch_num, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.patch_num = patch_num
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.patch_num <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = patch_num
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        # self.attn = Attention(dim, num_heads=num_heads,qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  
        # here we set shift_size to 1 for now
        if self.shift_size > 0:
            pass
        self.apply(self._init_weights)
        
    def execute(self, x, center, win_kset, init_faces, nw):
        # init_faces contains window_num face id.
        B, L, C = x.shape
        assert L == self.patch_num
        shortcut = x
        x = self.norm1(x)

        # win_kset = kset.reindex([nw * B, self.window_size, k], ['i0 // @e1(0)', '@e0(i0 // @e1(0), i0 % @e1(0), i1)', 'i2'], extras=[init_faces, jt.array(nw)])
        # win_kset = kset.reindex([nw * B, k], ['i0 // @e1(0)', '@e0(i0 // @e1(0), i0 % @e1(0))', 'i1'], extras=[init_faces, jt.array(nw)])
        # partition windows
        x_windows = window_partition(x, nw, self.window_size, win_kset)  # nW*B, k, C
        # W-MSA/SW-MSA
        # kset B, patch_num, k -> nw * B, W, k
        attn_windows = self.attn(x_windows, center, win_kset)  # nW*B, k, C

        # merge windows
        x = window_reverse(attn_windows, self.window_size, nw, self.patch_num, win_kset)  # B H' W' C
        # print(x.shape)
        # x = self.attn(x)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def online_process(feats, res_faces, oriadjlist, patch_num, center, preprocess=0):
    N = feats.shape[0]
    F = feats.shape[1]
    res_mat = jt.zeros((N, feats.shape[1]))
    # adjlist = []
    centers = []
    init_faces = get_patches_ffs(center, F, patch_num)
    # for i in range(N):
    #     assert patch_num == len(list(init_faces[0].numpy()))
    centers = center[init_faces]
    res_mat, res_mat_counter = unoverlapping_bfs_patch_cpu(oriadjlist, F, init_faces)
    patches = feats.reindex_reduce(op="add",
        shape=[N, patch_num, feats.shape[2]], # (N, F, C)
        indexes=[
            'i0',
            '@e0(i0,i1)',
            'i2'
        ],
        extras=[res_mat],
    )
    # return patches, adjlist, res_mat
    return patches, res_mat, centers, res_mat_counter

def downsample_k_non_overlapping(center, patch_num, down_num, kset, k, res_face, adjdict, next_size, window_size, use_euc=0, preprocess=0):
    N = center.shape[0]
    # init_faces = []
    # centers = []
    # centers = np.zeros((N, down_num, 3))
    # print("start get centers")
    st = None
    if preprocess == 1: st = 0
    init_faces = get_patches_ffs(center, patch_num, down_num, st)
    centers = center.reindex([N, int(down_num), 3], ['i0', '@e0(i0, i1)', 'i2'], extras=[init_faces])
    res_mat, res_mat_counter = unoverlapping_bfs_patch_cpu(adjdict, patch_num, init_faces)
    # centers = center.reindex_reduce("add", [N, int(down_num), 3], ['i0', '@e0(i0, i1)', 'i2'], extras=[res_mat])
    # centers = centers / res_mat_counter
    if use_euc == 0:
        adjdict = get_hierar_FAF_cpu(down_num, adjdict, res_mat, next_size)
        kset = k_bfs_patch_cpu(adjdict, down_num, window_size, False)
    else:
        adjdict = knn_indices_func_gpu(centers, centers, next_size, 1)
        kset = adjdict[:,:window_size]
    # adjlist, kset = get_jt_kset(adjdict, res_mat, down_num, k)
    return kset, centers, adjdict, res_mat, res_mat_counter

def online_process_k(center, kset, nw, window_size, preprocess=0):
    # generate the pos of windows and shifted windows here.
    k = window_size
    # print(center.shape)
    N, F, _ = center.shape
    init_faces = []
    s_init_faces = []
    # kset = knn_indices_func_gpu(center, center, k)
    st = None
    if preprocess == 1: st = 0
    init_faces = get_patches_ffs(center, F, nw, st=st)
    # s_init_faces = jt.array(s_init_faces)
    centers = center.reindex([N * int(nw), window_size, 3], ['i0 / @e2(0)', '@e1(i0 / @e2(0), @e0(i0 / @e2(0), i0 % @e2(0)), i1)', 'i2'], extras=[init_faces, kset, jt.array(nw)])
    # scenters = center.reindex([N * int(nw), window_size, 3], ['i0 / @e2(0)', '@e1(i0 / @e2(0), @e0(i0 / @e2(0), i0 % @e2(0)), i1)', 'i2'], extras=[s_init_faces, kset, jt.array(nw)])
    win_kset = kset.reindex([nw * N, k], ['i0 / @e1(0)', '@e0(i0 / @e1(0), i0 % @e1(0))', 'i1'], extras=[init_faces, jt.array(nw)])
    # s_win_kset = kset.reindex([nw * N, k], ['i0 / @e1(0)', '@e0(i0 / @e1(0), i0 % @e1(0))', 'i1'], extras=[s_init_faces, jt.array(nw)])
    # return win_kset, s_win_kset, init_faces, s_init_faces, centers, scenters
    return win_kset, None, init_faces, None, centers, None

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, nw, dim, patch_num, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, seg=0, use_checkpoint=False, next_size=16, use_conv=0, knn=3, next_dim=256):

        super().__init__()
        self.dim = dim
        self.patch_num = patch_num
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.downsample = downsample
        self.nw = nw
        self.seg = seg
        self.window_size = window_size
        self.next_size = next_size
        self.use_conv = use_conv
        self.knn = knn
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, patch_num=patch_num, num_heads=num_heads, window_size=window_size,
                shift_size=0, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])
        if use_conv:
            self.conv = SubConv(dim, dim, 2)
        if dim != next_dim:
            self.ups = nn.Linear(dim, next_dim)
        else:
            self.ups = nn.Identity()
        self.apply(self._init_weights)

    def execute(self, x, kset, center, res_face, adjdict, feats, down=True, preprocess=0, init_faces=None, use_euc=0):
        # if preprocess == 0:
        win_kset, s_win_kset, init_faces, s_init_faces, centers, s_centers = online_process_k(center, kset, self.nw, self.window_size, preprocess=preprocess)
        # else:
        #     win_kset = jt.array(kset).reshape(-1, self.window_size)
        #     centers = center
        for idx, blk in enumerate(self.blocks):
            x = blk(x, centers, win_kset, init_faces, self.nw)
            # if idx % 2 == 0:
            #     x = blk(x, centers, win_kset, init_faces, self.nw)
            # else:
            #     x = blk(x, s_centers, s_win_kset, s_init_faces, self.nw)
        if self.use_conv:
            x = self.conv(x.permute(0, 2, 1), self.knn, kset).permute(0, 2, 1)
        if self.seg == 1:
            feats.append(x)
        if down:
            kset, center, adjdict, res_face, counter = downsample_k_non_overlapping(center, self.patch_num, self.downsample, kset, self.window_size, res_face, adjdict, self.next_size, self.window_size, use_euc, preprocess)
            # downsampling
            N = x.shape[0]
            x = x.reindex_reduce(op='add', shape=[N, self.downsample, x.shape[-1]], 
                indexes=['i0', '@e0(i0, i1)', 'i2'], 
                extras=[jt.array(res_face)])
            # x /= counter
        x = self.ups(x)
        return x, center, kset, adjdict, res_face

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, nw=[32 ,8], patch_size=[128, 32], in_chans=3, num_classes=30,
                 embed_dim=[96, 192], depths=(3, 3, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=[4, 4], qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, preprocess=0, seg=0, weight_init='', use_conv=0, knn=3,down=0,**kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.seg = seg
        self.preprocess = preprocess
        if isinstance(mlp_ratio, int):
            mlp_ratio = [mlp_ratio for _ in range(self.num_layers)]
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        patch_size.append(patch_size[-1])
        self.pos_mlp = nn.Linear(3, embed_dim[0])
        self.absolute_pos_embed = jt.zeros((1, patch_size[0], embed_dim[-1]))
        trunc_normal(self.absolute_pos_embed, std=.02)
        self.absolute_pos_embed.requires_grad = True
        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        # self.patch_grid = self.patch_embed.grid_size

        # absolute position embedding
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.down = down
        # stochastic depth
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        layers = []
        for i_layer in range(self.num_layers):
            uc = 0
            if i_layer != self.num_layers - 1 and use_conv == 1:
                uc = 1
            layers += [BasicLayer(
                nw=nw[i_layer],
                dim=embed_dim[i_layer],
                patch_num=patch_size[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio[i_layer],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=patch_size[i_layer+1],
                seg=seg,
                use_checkpoint=use_checkpoint,
                next_size=patch_size[i_layer+2],
                use_conv=uc,
                knn=3,
                next_dim=embed_dim[min(i_layer+1, len(embed_dim)-1)]
                )
            ]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
    
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         init.gauss_(m.weight, 0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             init.zero_(m.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, kset, center, res_face, adjdict, init_faces=None, use_euc=0, preprocess=0):
        # x = self.patch_embed(x)
        # if self.absolute_pos_embed is not None:
        #     x = x + self.absolute_pos_embed
        x = x + self.pos_mlp(center)
        x = self.pos_drop(x)
        ksets = []
        res_faces_list = []
        feats = []
        for idx, layer in enumerate(self.layers):
            down = self.down
            if idx == len(self.layers) - 1:
                down = False
            x, center, kset, adjdict, res_face = layer(x, kset, center, res_face, adjdict, feats, down, preprocess, use_euc=use_euc)
            if self.seg == 1:
                ksets.append(kset)
                res_faces_list.append(res_face)
        x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = x.permute(0, 2, 1).mean(-1)
        x = jt.flatten(x, 1)
        return x, ksets, feats, res_faces_list

    def execute(self, x, kset, center, res_face, adjdict, init_faces=None, use_euc=0, preprocess=0):
        x, ksets, feats, res_faces_list = self.forward_features(x, kset, center, res_face, adjdict, init_faces, use_euc=use_euc, preprocess=preprocess)
        x = self.head(x)
        return x, ksets, feats, res_faces_list


def _create_swin_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        SwinTransformer, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model
