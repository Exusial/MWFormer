#coding=utf-8
import jittor as jt 
from jittor import nn
from jittor.misc import _pair
from functools import partial
import math
import numpy as np

from .utils import trunc_normal,load_pretrained,_conv_filter
from .config import default_cfgs

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
    return windows.reindex_reduce(op='add', 
        shape=[B, patch_num, C], 
        indexes=['i0 / @e1(0)', '@e0(i0, i1)', 'i2'], 
        extras=[win_kset, jt.array(nw)])

class Block2(nn.Module):
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
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        # self.attn = Attention(dim, num_heads=num_heads,qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # here we set shift_size to 1 for now
        if self.shift_size > 0:
            pass
        self.apply(self._init_weights)
        
    def execute(self, x, win_kset, nw):
        # init_faces contains window_num face id.
        B, L, C = x.shape
        print(x.shape)
        print(self.patch_num)
        assert L == self.patch_num
        shortcut = x
        x = self.norm1(x)
        x_windows = window_partition(x, nw, self.window_size, win_kset)  # nW*B, k, C
        attn_windows = self.attn(x_windows)  # nW*B, k, C
        x = window_reverse(attn_windows, self.window_size, nw, self.patch_num, win_kset)  # B H' W' C
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


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbed,self).__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def execute(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(0, 2, 1)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super(HybridEmbed,self).__init__()
        assert isinstance(backbone, nn.Module)
        img_size = _pair(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with jt.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.is_training()
                if training:
                    backbone.eval()
                o = self.backbone(jt.zeros((1, in_chans, img_size[0], img_size[1])))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train()
        else:
            feature_size = _pair(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def execute(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(0,2,1)
        x = self.proj(x)
        return x

class PatchHieTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 num_patches=256,
                 num_classes=30, 
                 embed_dim=128, 
                 depth=4,
                 num_heads=4, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 hybrid_backbone=None, 
                 norm_layer=nn.LayerNorm,
                 nw=32,
                 window_size=7):
        super(PatchHieTransformer,self).__init__()
 
        # self.pos_embed = jt.zeros((1, num_patches + 1, embed_dim))
        # self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block2(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, window_size=window_size, patch_num=num_patches)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # print(embed_dim, num_classes)
        self.nw = nw
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def apply(self,fn):
        for m in self.modules():
            fn(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def execute(self, x, win_kset):
        # x = x + self.pos_embed
        # x = self.pos_drop(x)
        # print(x.shape)
        for idx, blk in enumerate(self.blocks):
            x = blk(x, win_kset, self.nw)

        x = self.norm(x)
        x = self.head(x)
        return x


class PatchAvgTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 num_patches=256,
                 num_classes=30, 
                 embed_dim=128, 
                 depth=4,
                 num_heads=4, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 hybrid_backbone=None, 
                 norm_layer=nn.LayerNorm):
        super(PatchAvgTransformer,self).__init__()
 
        # self.pos_embed = jt.zeros((1, num_patches + 1, embed_dim))
        # self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

        # self.pos_embed = trunc_normal(self.pos_embed, std=.02)
        self.apply(self._init_weights)
    
    def apply(self,fn):
        for m in self.modules():
            fn(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def execute(self, x):
        # x = x + self.pos_embed
        # x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = jt.mean(x, dim=1)
        x = self.head(x)
        return x



