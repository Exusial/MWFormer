import jittor as jt
from jittor import nn
import numpy as np
from jmesh.graph_utils import get_roll_feature

class MLP_forward(nn.Module):
    def __init__(self, input_channel, channels):
        self.fc1 = nn.Conv1d(input_channel, channels[0], kernel_size=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm(channels[0])
        self.fc_list = nn.ModuleList()
        for ic in range(0, len(channels)-1):
            self.fc_list.append(nn.Conv1d(channels[ic], channels[ic+1], kernel_size=1))
            self.fc_list.append(nn.ReLU())
            self.fc_list.append(nn.BatchNorm(channels[ic+1]))
        
    def execute(self, x, kfaces=None, fn=None, args=None, k=3):
        x = self.bn1(self.relu1(self.fc1(x)))
        return self.fc_list(x).permute(0,2,1)

class SimSubConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimSubConv, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 3))

    def execute(self, x, k, res_faces, ori_x): 
        B, C, F = x.shape
        neighbor = ori_x.reindex([B, C, F, k], ['i0', 'i1', '@e0(i0, i2, i3)'], extras=[res_faces])
        abs_residual = jt.abs(neighbor - x.unsqueeze(-1).expand(B, C, F, k)).sum(-1, keepdims=True)
        neighbor = neighbor.sum(-1, keepdims=True)
        x = jt.concat([x.unsqueeze(-1), neighbor, abs_residual], dim=-1)
        x = self.conv1(x)
        x = self.relu(self.bn1(x)).squeeze(-1)
        return x

class SubConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_relu=1, use_roll=False, tl=True):
        super(SubConv, self).__init__()
        self.use_relu = use_relu
        if use_relu == 0:
            self.relu = nn.Identity()
            self.relu2 = nn.Identity()
        elif use_relu == 1:
            self.relu = nn.ReLU()
            self.relu2 = nn.ReLU()
        elif use_relu == 2:
            self.relu = nn.gelu
            self.relu2 = nn.gelu
        self.bn1 = nn.BatchNorm(out_channels)
        self.bn2 = nn.BatchNorm(out_channels)
        # self.bn1 = nn.LayerNorm(out_channels)
        # self.bn2 = nn.LayerNorm(out_channels)
        self.use_roll = use_roll
        self.tl = tl
        if use_roll:
            self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 4))
            self.conv2 = nn.Conv2d(out_channels, out_channels, (1, 4))
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 3))
            self.conv2 = nn.Conv2d(out_channels, out_channels, (1, 3))

    def execute(self, x, k, res_faces): 
        B, C, F = x.shape
        k_faces = res_faces[:, :, :k]
        neighbor = x.reindex([B, C, F, k], ['i0', 'i1', '@e0(i0, i2, i3)'], extras=[k_faces], overflow_conditions=['@e0(i0,i2,i3) < 0'], overflow_value=0)
        abs_residual = jt.abs(neighbor - x.unsqueeze(-1).expand(B, C, F, k)).sum(-1, keepdims=True)
        neighbor = neighbor.sum(-1, keepdims=True)
        if self.use_roll:
            roll_neighbor = get_roll_feature(x.permute(0,2,1), k_faces)
            x = jt.concat([x.unsqueeze(-1), neighbor, abs_residual, roll_neighbor], dim=-1)
        else:
            x = jt.concat([x.unsqueeze(-1), neighbor, abs_residual], dim=-1)
        x = self.conv1(x).squeeze(-1)
        x = self.bn1(x)
        x = self.relu(x)
        if self.tl:
            B, C, F = x.shape
            neighbor = x.reindex([B, C, F, k], ['i0', 'i1', '@e0(i0, i2, i3)'], extras=[k_faces], overflow_conditions=['@e0(i0,i2,i3) < 0'], overflow_value=0)
            abs_residual = jt.abs(neighbor - x.unsqueeze(-1).expand(B, C, F, k)).sum(-1, keepdims=True)
            neighbor = neighbor.sum(-1, keepdims=True)
            if self.use_roll:
                roll_neighbor = get_roll_feature(x.permute(0,2,1), k_faces)
                x = jt.concat([x.unsqueeze(-1), neighbor, abs_residual, roll_neighbor], dim=-1)
            else:
                x = jt.concat([x.unsqueeze(-1), neighbor, abs_residual], dim=-1)
            x = self.conv2(x).squeeze(-1)
            x = self.bn2(x)
            x = self.relu2(x)
            # print(x[0])
            # exit(0)
        return x

class SubMeshConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SubMeshConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 3))

    def execute(self, x, k, res_faces): 
        B, C, F = x.shape
        print("x", x.shape)
        print("kset", res_faces.shape)
        neighbor = x.reindex([B, C, F, k], ['i0', 'i1', '@e0(i0, i2, i3)'], extras=[res_faces[:, :, :k]])
        abs_residual = jt.abs(neighbor - x.unsqueeze(-1).expand(B, C, F, k)).sum(-1, keepdims=True)
        neighbor = neighbor.sum(-1, keepdims=True)
        x = jt.concat([x.unsqueeze(-1), neighbor, abs_residual], dim=-1)
        return self.conv1(x).squeeze(-1)

class DgcnnConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DgcnnConv, self).__init__()
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm(in_channels)
        self.bn2 = nn.BatchNorm(out_channels)
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels * 2, out_channels, 1)

    def execute(self, x, k, res_faces): 
        B, C, F = x.shape
        neighbor = x.reindex([B, C, F, k], ['i0', 'i1', '@e0(i0, i2, i3)'], extras=[res_faces])
        x = x.unsqueeze(-1).expand(-1, -1, -1, k)
        x = jt.concat([x - neighbor, x], 1) # N, 2C, F, K
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = x.max(-1)
        B, C, F = x.shape
        neighbor = x.reindex([B, C, F, k], ['i0', 'i1', '@e0(i0, i2, i3)'], extras=[res_faces])
        x = x.unsqueeze(-1).expand(-1, -1, -1, k)
        x = jt.concat([x - neighbor, x], 1) # N, 2C, F, K
        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        return x.max(-1)

class DGCNN(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        if args.use_relu == 1:
            self.relu = nn.ReLU()
        elif args.use_relu == 0:
            self.relu = nn.Identity()
        elif args.use_relu == 2:
            self.relu = nn.gelu
        self.out_channels = out_channels
        self.linear1 = nn.Linear(in_channels, args.channels[0])
        if args.agg_method == 2:
            self.convs = nn.ModuleList([DgcnnConv(args.channels[i], args.channels[i+1]) for i in range(len(args.channels) - 1)])
            self.convs.append(DgcnnConv(args.channels[-1], args.channels[-1]))
            self.linear2 = nn.Linear(concat_dim, args.channels[-1])
        concat_dim = 0
        for i in range(1, len(args.channels)):
            concat_dim += args.channels[i]
        concat_dim += args.channels[-1]
        # for subdiv like aggregation
        if args.agg_method == 1:
            self.agg = nn.ModuleList([SubConv(args.channels[i], args.channels[i+1], use_relu=args.use_relu, use_roll=False, tl=True) for i in range(len(args.channels) - 1)])
            self.agg.append(SubConv(args.channels[-1], args.channels[-1], use_relu=args.use_relu, use_roll=False, tl=True))
 
    def agg_neighbor(self, x ,neighbor, kind, idx):
        B, C, F, K = neighbor.shape
        if kind == 2:
            x = x.unsqueeze(-1).expand(-1, -1, -1, K)
            return jt.concat([x - neighbor, x], 1) # N, 2C, F, K
        elif kind == 1:
            abs_residual = jt.abs(neighbor - x.unsqueeze(-1).expand(B, C, F, K)).sum(-1, keepdims=True)
            neighbor = neighbor.sum(-1, keepdims=True)
            return self.agg[idx](jt.concat([x.unsqueeze(-1), neighbor, abs_residual], dim=-1))           
    
    def execute(self, x, res_faces, patch_num, args, k=None):
        if not k is None: # x N, F, inc
            x = self.linear1(x.permute(0, 2, 1)).permute(0, 2, 1) # x N C F
            x = self.relu(x)
            # res_faces N P K
            B, C, F = x.shape
            if args.agg_method == 2:
                feature_list = []
                for idx, blk in enumerate(self.convs):
                    x = blk(x, k, res_faces)
                    feature_list.append(x)
                x = jt.concat(feature_list, dim = 1)
                return self.linear2(x.permute(0, 2, 1)) # x N F C
            elif args.agg_method == 1:
                B, C, F = x.shape
                for idx, blk in enumerate(self.agg):
                    x = blk(x, k ,res_faces)
                    if idx != len(args.channels) - 1:
                        C = args.channels[idx + 1]
                return x.permute(0, 2, 1)

class MLP_NET(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(MLP_NET, self).__init__()
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.linear1 = nn.Linear(in_channels, args.channels[0])
        # for subdiv like aggregation
        self.agg = nn.ModuleList()
        for i in range(len(args.channels) - 1):
            self.agg.append(nn.Conv1d(args.channels[i], args.channels[i+1], 1))
            self.agg.append(nn.BatchNorm(args.channels[i+1]))
            self.agg.append(nn.ReLU())
        self.agg.append(nn.Conv1d(args.channels[-1], args.channels[-1], 1))
        self.agg.append(nn.BatchNorm(args.channels[-1]))
        self.agg.append(nn.ReLU())

    def agg_neighbor(self, x ,neighbor, kind, idx):
        B, C, F, K = neighbor.shape
        if kind == 2:
            x = x.unsqueeze(-1).expand(-1, -1, -1, K)
            return jt.concat([x - neighbor, x], 1) # N, 2C, F, K
        elif kind == 1:
            abs_residual = jt.abs(neighbor - x.unsqueeze(-1).expand(B, C, F, K)).sum(-1, keepdims=True)
            neighbor = neighbor.sum(-1, keepdims=True)
            return self.agg[idx](jt.concat([x.unsqueeze(-1), neighbor, abs_residual], dim=-1))           
    
    def execute(self, x, res_faces, patch_num, args, k=None):
        if not k is None: # x N, F, inc
            x = self.linear1(x.permute(0, 2, 1))
            x = self.relu(x)
            x = self.agg(x.permute(0, 2, 1))
            return x.permute(0,2,1)
