import jittor as jt
from jittor import nn
import numpy as np
from .meshconv import SubConv
from .graphomer import Attention
from jmesh.models.transformer.utils import trunc_normal,load_pretrained,_conv_filter

def unpool(feats, res_faces, Fs): # nearest.
    N, F, C = feats.shape
    up_feats = feats.reindex([N, Fs, C], indexes=['i0', '@e0(i0, i1)', 'i2'], extras=[res_faces], overflow_value=-1, overflow_conditions=["@e0(i0, i1) == -1"])
    return up_feats

def unpool2(feats, res_faces, Fs): # nearest.
    N, C, F = feats.shape
    up_feats = feats.reindex([N, C, Fs], indexes=['i0', 'i1', '@e0(i0, i2)'], extras=[res_faces])
    return up_feats

class U_Net(nn.Module):
    def __init__(self, embed_dim, flist, num_classes):
        super(U_Net, self).__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim * (len(flist) + 1), num_classes)

    def execute(self, feats, res_faces_list, flist):
        # for feat in feats:
        #     print(feat.shape)
        x = self.linear(feats[-1])
        # print(len(res_faces_list))
        # print(len(flist))
        for i in range(len(flist) - 2, -1, -1):
            x = unpool(x, res_faces_list[i], flist[i])
            x = jt.concat([x, feats[i]], dim=-1)
            # print(x.shape)
        # x : N, mesh_faces_num, C
        return self.fc(x)

class CrossUNet(nn.Module):
    def __init__(self, embed_dim, flist, num_classes, k, args=None):
        super(CrossUNet, self).__init__()
        # self.linear = nn.Linear(args.n_classes[-2], embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.linears = nn.ModuleList([nn.Linear(embed_dim * 2, embed_dim) for _ in range(len(flist))])
        self.convs = nn.ModuleList([SubConv(embed_dim*2, embed_dim, use_roll=False if _ > 0 else False) for _ in range(len(flist))])
        self.k = k
        self.args = args
        self.agg_conv = SubConv(embed_dim, embed_dim)
        self.cross_attn = Attention(dim=embed_dim, num_heads=4, qkv_bias=True)
        self.apply(self._init_weights)
        self.norm = nn.LayerNorm(embed_dim)

    def execute(self, feats, res_faces_list, flist, ksets, Fs=None):
        patch_f = jt.concat(feats[1:], 1) + self.norm(nn.gelu(self.cross_attn(jt.concat(feats[1:], 1))))
        # x = patch_f[:,:flist[0],:]
        # print(x.shape, ksets[0].shape)
        # x = self.agg_conv(x.permute(0, 2, 1), self.k, ksets[0]).permute(0, 2, 1)
        # enhance with nei.
        sdim = np.sum(flist[1:])
        # print(patch_f.shape, sdim)
        last_dim = sdim - flist[-1]
        x = patch_f[:,last_dim:,:]
        for i in range(len(flist) - 2, -1, -1):
            if flist[i+1] != flist[i]:
                idx = i if len(res_faces_list) > 1 else 0
                x = unpool(x, res_faces_list[idx], flist[i])
            if i > 0:
                x = jt.concat([x, patch_f[:,last_dim-flist[i]:last_dim,:]], dim=-1)
                last_dim = last_dim-flist[i]
            else:
                x = jt.concat([x, feats[0]], dim=-1)
            # print("iter ", i)
            # print(x.shape)
            # print(ksets[i].shape)
            # print(res_faces_list[i].shape)
            # if self.args.unet_conv == 1:
            #     x = self.convs[i](x.permute(0, 2, 1), self.k, ksets[i]).permute(0, 2, 1)
            # else:
            # if i > 0:
            x = self.convs[i](x.permute(0,2,1), 3, ksets[i]).permute(0,2,1)
        return self.fc(x) 
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)    

class U_ConvNet(nn.Module):
    def __init__(self, embed_dim, flist, num_classes, k, args=None):
        super(U_ConvNet, self).__init__()
        self.linear = nn.Linear(args.n_classes[-2], embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.linears = nn.ModuleList([nn.Linear(embed_dim * 2, embed_dim) for _ in range(len(flist))])
        self.convs = nn.ModuleList([SubConv(embed_dim*2, embed_dim) for _ in range(len(flist))])
        self.k = k
        print("embed: ", embed_dim)
        self.args = args
        
    def execute(self, feats, res_faces_list, flist, ksets, Fs=None):
        # print("feats")
        # for feat in feats:
        #     print(feat.shape)
        # print("kset")
        # for kset in ksets:
        #     print(kset.shape)
        # print("Face")
        # for ff in flist:
        #     print(ff)
        # print("res_face")
        # for res_face in res_faces_list:
        #     print(res_face.shape)
        x = self.linear(feats[-1])
        # print(len(res_faces_list))
        # print(flist)
        for i in range(len(flist) - 2, -1, -1):
            if flist[i+1] != flist[i]:
                # if self.args.merge_batch > 0 and x.shape[1] != self.args.patch_num[0]:
                #     N = Fs.shape[0] * self.args.merge_batch
                #     Fs = Fs.reshape((N // self.args.merge_batch, -1))
                #     x = x.reindex((N, self.args.patch_num[0], x.shape[-1]), ["i0 / @e1(0)", "i0 % @e1(0) * @e2(0) + i1", "i2"], extras=[Fs, jt.array([self.args.merge_batch]), jt.array([self.args.patch_num[0]])], overflow_conditions=["i1 >= @e0(i0 / @e1(0), i0 % @e1(0))"], overflow_value=0)
                idx = i if len(res_faces_list) > 1 else 0
                x = unpool(x, res_faces_list[idx], flist[i])
            # print(x.shape, feats[i-1].shape, feats[i].shape, flist[i-1],  ksets[i-1].shape, idx)
            x = jt.concat([x, feats[i]], dim=-1)
            # print("iter ", i)
            # print(x.shape)
            # print(ksets[i].shape)
            # print(res_faces_list[i].shape)
            if self.args.unet_conv == 1:
                x = x + self.convs[i](x.permute(0, 2, 1), self.k, ksets[i]).permute(0, 2, 1)
            else:
                x = self.linears[i](x)
            # print(x.shape)
        # x : N, mesh_faces_num, C
        return self.fc(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class U_NoConvNet(nn.Module):
    def __init__(self, embed_dim, up_len, num_classes, k, args=None):
        super(U_NoConvNet, self).__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.linears = nn.ModuleList([nn.Linear(embed_dim * 2, embed_dim) for _ in range(up_len)])
        # self.convs = nn.ModuleList([SubConv(embed_dim, embed_dim) for _ in range(up_len)])
        self.k = k

    def execute(self, feats, res_faces_list, flist, ksets):
        x = self.linear(feats[-1])
        print(len(res_faces_list))
        print(flist)
        for i in range(len(flist) - 2, -1, -1):
            x = unpool(x, res_faces_list[i], flist[i])
            x = jt.concat([x, feats[i]], dim=-1)
            # print("iter ", i)
            # print(x.shape)
            # print(ksets[i].shape)
            # print(res_faces_list[i].shape)
            x = self.linears[i](x)
            # x = self.convs[i](x, self.k, ksets[i]).permute(0, 2, 1)
            # print(x.shape)
        # x : N, mesh_faces_num, C
        return self.fc(x)

def fl2el(preds, gt, edge2face):
    acc = 0
    N, F = preds.shape
    _, e = gt.shape
    for i in range(N):
        for edge in range(e):
            face_ids = edge2face[i][edge]
            for face_id in face_ids:
                if preds[i, face_id] == gt[i, edge]:
                    acc += 1
                    break
    return acc / e