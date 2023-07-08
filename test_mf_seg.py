import os
import json
import argparse
from tensorboardX import SummaryWriter

import jittor as jt
import jittor.nn as nn
from jittor.optim import Adam, AdamW
from jittor.optim import SGD
from jittor.lr_scheduler import MultiStepLR

import numpy as np
import pymeshlab
from tqdm import tqdm

from jmesh.dataset import SegmentationMapDataset, load_mesh
from jmesh.network import MeshNet
from jmesh.utils import to_mesh_tensor
from jmesh.utils import ClassificationMajorityVoting
from jmesh.models.transformer.vision_transformer import PatchTransformer, PatchAvgTransformer, PatchHieTransformer
from jmesh.models.meshformers.meshconv import *
from jmesh.models.meshformers.seg_swin import SwinSegTransformer
from jmesh.models.meshformers.seg_pvt import PvTSegTransformer
from jmesh.models.meshformers.unet import U_ConvNet, U_NoConvNet, CrossUNet
from jmesh.graph_utils import *
from jmesh.visualize import visual_segmentation, save_seg_patches
# new
# from utils.bfs_patch import tensor_to_mesh, get_adjacent_dict, get_patches, bfs_patch, get_random_patches, get_hie_adjacent_dict, k_dis_bfs_patch, k_bfs_patch, get_patches_ffs

error_stat = None
acc_map = {}
def online_process(feats, res_faces, oriadjlist, patch_num, center, next_patch, preprocess):
    N = feats.shape[0]
    F = feats.shape[1]
    res_mat = jt.zeros((N, feats.shape[1]))
    # adjlist = []
    centers = []
    ns = None 
    if preprocess == 1: ns = 1
    init_faces = get_patches_ffs(center, F, patch_num, st=ns)
    # for i in range(N):
    #     assert patch_num == len(list(init_faces[0].numpy()))
    centers = center.reindex((N, patch_num, 3), ("i0", "@e0(i0, i1)", "i2"), extras=[init_faces])
    if 1:
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
        patches /= res_mat_counter
    # else:
    #     res_mat, res_mat_counter = None, None
    #     patches = feats.reindex(
    #         shape=[N, patch_num, feats.shape[2]], # (N, F, C)
    #         indexes=[
    #             'i0',
    #             '@e0(i0,i1)',
    #             'i2'
    #         ],
    #         extras=[init_faces],
    #     )
    # return patches, adjlist, res_mat
    if next_patch > 0:
        adjdict = get_hierar_FAF_cpu(patch_num, oriadjlist, res_mat, next_patch)
        # print("adjdict shape: ", adjdict.shape)
    else:
        adjdict = None
    return patches, res_mat, centers, adjdict, res_mat_counter

def online_process_k(feats, res_faces, adjdict, patch_num, k):
    N, F, _ = feats.shape
    # print(adjdict.shape, F)
    res_mat = k_bfs_patch_cpu(adjdict, F, args.knn, False)
    return res_mat

class MutilScaleTransformer(nn.Module):
    def __init__(self,
                 num_patches=[128],
                 num_classes=[30], 
                 embed_dim=128, 
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 depth=4,
                 down=1):
        super(MutilScaleTransformer,self).__init__()
        self.num_patches = num_patches
        embed_dims = [embed_dim]
        for i in range(len(num_classes) - 1):
            embed_dims.append(num_classes[i])
        # print(self.num_patches)
        self.transformers = nn.ModuleList([
            PatchHieTransformer(
                num_patches=num_patches[i], num_classes=num_classes[i], embed_dim=embed_dims[i], depth=depth[i], attn_drop_rate=attn_drop_rate, drop_rate=drop_rate,                 drop_path_rate=drop_path_rate
            ) for i in range(len(num_patches))
        ])
        self.convs = nn.ModuleList([
            SubConv(num_classes[i], num_classes[i]) for i in range(len(num_patches))
        ])
        if down == 0:
            self.fc = nn.Linear(embed_dims[-1], num_classes[-1])
            # self.transformers.append(
            #     PatchTransformer(
            #         num_patches=num_patches[-1], num_classes=num_classes[-1], embed_dim=embed_dims[-1], depth=depth[-1], attn_drop_rate=attn_drop_rate, drop_rate=drop_rate,                  drop_path_rate=drop_path_rate
            #     )
            # )
            self.down = down

    def execute(self, x, res_faces, adjdict, k, center, feats, ksets, preprocess=0):
        res_faces_list = []
        res_faces_list.append(res_faces)
        for idx, transformer in enumerate(self.transformers):
            # print(f"do {idx} transformer")
            # print(x.shape, f"{idx}")
            x = transformer(x) # (N, patch_num, F)
            # print("x.shape: ", x.shape)
            feats.append(x)
            if idx != len(self.transformers) - 1:
                kset = online_process_k(x, res_faces, adjdict, self.num_patches[idx], k)
                # print("lalala")
                # kset.sync()
                # print("lalala2")
                ksets.append(kset)
                x = self.convs[idx](x.permute(0, 2, 1), k, kset).permute(0, 2, 1)
            if idx == len(self.transformers) - 1:
                if args.down == 0:
                    x = self.fc(x).permute(0,2,1)
                break
            if args.down == 1:
                next_patch = self.num_patches[idx + 1] if idx + 1 <= len(self.transformers) else -1
                x, res_faces, center, adjdict, _ = online_process(x, res_faces, adjdict, self.num_patches[idx + 1], center, next_patch, preprocess)
                res_faces_list.append(res_faces)
                # print(x)
        return x, feats, res_faces_list, ksets

def train(net, transformer, unet, optim, train_dataset, writer, epoch, patch_num, args):
    net.train()
    transformer.train()
    unet.train()
    n_correct = 0
    n_samples = 0
    acc = 0
    disable_tqdm = jt.rank != 0
    for meshes, labels, _ in tqdm(train_dataset, desc=f'Train {epoch}', disable=disable_tqdm):
        mesh_tensor = to_mesh_tensor(meshes)
        mesh_labels = jt.int32(labels)
        feats = []
        ksets = []
        # print(_)
        kfaces = jt.array(meshes['k_faces'])
        ksets.append(kfaces)
        outputs = net(mesh_tensor.feats, kfaces, mesh_tensor.faces.shape[1], args, meshes['k_faces'][0].shape[-1])
        feats.append(outputs)
        if args.debug_sub == 0:
            transformer.train()
            # get patches.
            N = outputs.shape[0]
            # print(meshes["res_faces"].shape)
            if args.down == 1:
                patches = outputs.reindex_reduce(op="add",
                    shape=[N, patch_num, outputs.shape[2]], # (N, F, C)
                    indexes=[
                        'i0',
                        '@e0(i0,i1)',
                        'i2'
                    ],
                    extras=[jt.array(meshes['res_faces'])],
                    overflow_conditions=['@e0(i0,i1) < 0']
                )
            else:
                patches = outputs
            # patches /= jt.array(meshes["res_face_counter"])
            if args.transformer_type == "vit":
                _, feats, res_faces_list, ksets = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, jt.array(meshes['centers']), feats, ksets)
                # jt.sync_all()
                # print("pass sync")
                outputs = unet(feats, res_faces_list, [mesh_tensor.faces.shape[1]] + args.patch_num, ksets).permute(0, 2, 1)
            elif args.transformer_type == "pvt":
                outputs, feats, res_faces_list = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, feats, ksets, jt.array(meshes['centers']))
                jt.sync_all()
                if args.down == 1:
                    outputs = unet(feats, res_faces_list, [mesh_tensor.faces.shape[1]] + args.patch_num, ksets).permute(0, 2, 1)
            elif args.transformer_type == "swin":
                _, feats, res_faces_list, last_res_face = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, feats, ksets, jt.array(meshes['centers']), use_conv=args.use_conv, preprocess=args.preprocess)
                outputs = unet(feats, res_faces_list, [mesh_tensor.faces.shape[1]] + args.patch_num, ksets).permute(0, 2, 1)
        loss = nn.cross_entropy_loss(outputs.unsqueeze(dim=-1), mesh_labels.unsqueeze(dim=-1), ignore_index=-1)
        jt.sync_all()
        optim.step(loss)
        print(loss)
        jt.sync_all()

        preds = np.argmax(outputs.data, axis=1)
        acc += np.sum((labels == preds).sum(axis=1) / meshes['Fs'])
        writer.add_scalar('loss', loss.data[0], global_step=train.step)
        train.step += 1
    if jt.rank == 0:
        acc /= train_dataset.total_len
        print('train acc = ', acc)
        writer.add_scalar('train-acc', acc, global_step=epoch)

@jt.single_process_scope()
def test(net, transformer, unet, test_dataset, writer, epoch, args, patch_num):
    global error_stat, acc_map
    net.eval()
    unet.eval()
    acc = 0
    voted = ClassificationMajorityVoting(args.n_classes[-1])
    mps = []
    with jt.no_grad():
        for meshes, labels, mesh_paths in tqdm(test_dataset, desc=f'Test {epoch}'):
            mpath = list(mesh_paths["mesh_paths"])[0]
            acc_map[mpath] = 0
            mps += list(mesh_paths["mesh_paths"])
            # print(meshes['faces'])
            mesh_tensor = to_mesh_tensor(meshes)
            feats = []
            ksets = []
            kfaces = jt.array(meshes['k_faces'])
            ksets.append(kfaces)
            outputs = net(mesh_tensor.feats, kfaces, mesh_tensor.faces.shape[1],  args, meshes['k_faces'][0].shape[-1])
            feats.append(outputs)
            if args.debug_sub == 0:
                transformer.eval()
                # get patches.
                N = outputs.shape[0]
                if args.down == 1:
                    patches = outputs.reindex_reduce(op="add",
                        shape=[N, patch_num, outputs.shape[2]], # (N, C, F)
                        indexes=[
                            'i0',
                            '@e0(i0,i1)',
                            'i2'
                        ],
                        extras=[jt.array(meshes['res_faces'])],
                        overflow_conditions=['@e0(i0,i1) < 0']
                    )
                else:
                    patches = outputs
                # patches /= jt.array(meshes['res_face_counter'])
                # send to transformer
                if args.transformer_type == "vit":
                    outputs, feats, res_faces_list, ksets = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, jt.array(meshes['centers']), feats, ksets)
                    # print(feats)
                    outputs = unet(feats, res_faces_list, [mesh_tensor.faces.shape[1]] + args.patch_num, ksets).permute(0, 2, 1)
                elif args.transformer_type == "pvt":
                    outputs, feats, res_faces_list = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, feats, ksets, jt.array(meshes['centers']))
                    if args.down == 1:
                        outputs = unet(feats, res_faces_list, [mesh_tensor.faces.shape[1]] + args.patch_num, ksets).permute(0, 2, 1)
                elif args.transformer_type == "swin":
                    _, feats, res_faces_list, last_res_face = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, feats, ksets, jt.array(meshes['centers']), use_conv=args.use_conv, preprocess=args.preprocess)
                    outputs = unet(feats, res_faces_list, [mesh_tensor.faces.shape[1]] + args.patch_num, ksets).permute(0, 2, 1)
            mesh_labels = jt.int32(labels)
            loss = nn.cross_entropy_loss(outputs.unsqueeze(dim=-1), mesh_labels.unsqueeze(dim=-1), ignore_index=-1)
            print("test: ", loss)
            preds = np.argmax(outputs.data, axis=1)
            # calc edge_acc
            # e_acc += fl2el(preds, meshes['edge_label'], meshes['e2f'])
            batch_acc = (labels == preds).sum(axis=1) / meshes['Fs']
            # batch_oacc = compute_original_accuracy(mesh_infos, preds, mesh_labels)
            acc += np.sum(batch_acc)
            if not str(mpath) in acc_map or float(batch_acc[0]) > acc_map[str(mpath)]:
                acc_map[str(mpath)] = float(batch_acc[0])
                out_path = os.path.join("visual", args.name, mpath.split("/")[-1].split(".")[0] + ".ply")
                out_face_path = os.path.join("visual", args.name, mpath.split("/")[-1].split(".")[0] + "_patch.ply")
                out_dir = os.path.dirname(out_path)
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                # res_faces_list + [last_res_face]
                save_seg_patches(mpath, res_faces_list + [last_res_face], out_face_path)
                visual_segmentation(mpath, preds, out_path)
            # oacc += np.sum(batch_oacc)
            # update_label_accuracy(preds, mesh_labels, label_acc)
            # voted.vote(mesh_infos, preds, mesh_labels)
    # print(test_dataset.total_len, meshes['Fs'])
    acc /= test_dataset.total_len
    # e_acc /= test_dataset.total_len
    # oacc /= test_dataset.total_len
    # voacc = voted.compute_accuracy(save_results=True)
    print('test acc = ', acc)
    print("test best acc = ", test.best_oacc)
    writer.add_scalar('test-acc', acc, global_step=epoch)
    if test.best_oacc < acc:
        # comp = len(set(mps)) == len(mps)
        # if not comp:
        #     return
        # print(sorted(test.best_paths), , len(test.best_paths))
        test.best_oacc = acc
        test.best_paths = mps
        net.save(os.path.join('checkpoints', name, f'{args.ckpt_name}_nacc-{acc:.4f}.pkl'))
        transformer.save(os.path.join('checkpoints', name, f'{args.ckpt_name}_acc-{acc:.4f}.pkl'))
        unet.save(os.path.join('checkpoints', name, f'{args.ckpt_name}_uacc-{acc:.4f}.pkl'))
    # if test.best_voacc < voacc:
    #     test.best_voacc = voacc
    #     net.save(os.path.join('checkpoints', name, f'{args.ckpt_name}_nvacc-{voacc:.4f}.pkl'))
    #     transformer.save(os.path.join('checkpoints', name, f'{args.ckpt_name}_vacc-{voacc:.4f}.pkl'))
    #     unet.save(os.path.join('checkpoints', name, f'{args.ckpt_name}_uacc-{acc:.4f}.pkl'))
    
@jt.single_process_scope()
def visual(net, transformer, unet, test_dataset, writer, epoch, args, patch_num):
    global acc_map
    acc_map = {}
    net.eval()
    unet.eval()
    acc = 0
    with jt.no_grad():
        for meshes, labels, mesh_paths in tqdm(test_dataset, desc=f'Test {epoch}'):
            mpath = list(mesh_paths["mesh_paths"])[0]
            acc_map[mpath] = 0
            mesh_tensor = to_mesh_tensor(meshes)
            feats = []
            ksets = []
            kfaces = jt.array(meshes['k_faces'])
            ksets.append(kfaces)
            outputs = net(mesh_tensor.feats, kfaces, mesh_tensor.faces.shape[1],  args, meshes['k_faces'][0].shape[-1])
            feats.append(outputs)
            if args.debug_sub == 0:
                transformer.eval()
                N = outputs.shape[0]
                if args.down == 1:
                    patches = outputs.reindex_reduce(op="add",
                        shape=[N, patch_num, outputs.shape[2]], # (N, C, F)
                        indexes=[
                            'i0',
                            '@e0(i0,i1)',
                            'i2'
                        ],
                        extras=[jt.array(meshes['res_faces'])],
                        overflow_conditions=['@e0(i0,i1) < 0']
                    )
                else:
                    patches = outputs
                if args.transformer_type == "vit":
                    outputs, feats, res_faces_list, ksets = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, jt.array(meshes['centers']), feats, ksets)
                    outputs = unet(feats, res_faces_list, [mesh_tensor.faces.shape[1]] + args.patch_num, ksets).permute(0, 2, 1)
                elif args.transformer_type == "pvt":
                    outputs, feats, res_faces_list = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, feats, ksets, jt.array(meshes['centers']))
                    if args.down == 1:
                        outputs = unet(feats, res_faces_list, [mesh_tensor.faces.shape[1]] + args.patch_num, ksets).permute(0, 2, 1)
                elif args.transformer_type == "swin":
                    _, feats, res_faces_list, last_res_face = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, feats, ksets, jt.array(meshes['centers']), use_conv=args.use_conv, preprocess=args.preprocess)
                    outputs = unet(feats, res_faces_list, [mesh_tensor.faces.shape[1]] + args.patch_num, ksets).permute(0, 2, 1)
            mesh_labels = jt.int32(labels)
            loss = nn.cross_entropy_loss(outputs.unsqueeze(dim=-1), mesh_labels.unsqueeze(dim=-1), ignore_index=-1)
            print("test: ", loss)
            preds = np.argmax(outputs.data, axis=1)
            batch_acc = (labels == preds).sum(axis=1) / meshes['Fs']
            acc_map[str(mpath)] = float(batch_acc[0])
            out_path = os.path.join("visual", args.name, mpath.split("/")[-1].split(".")[0] + ".ply")
            out_dir = os.path.dirname(out_path)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            visual_segmentation(mpath, labels, out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, nargs='+')
    parser.add_argument('--n_classes', type=int, nargs='+', required=True)
    parser.add_argument('--depth', type=int, nargs='+', required=True)
    parser.add_argument('--optim', choices=['adam', 'sgd', "adamw"], default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=None)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--channels', type=int, nargs='+', required=True)
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--blocks', type=int, nargs='+', default=None)
    parser.add_argument('--no_center_diff', action='store_true')
    parser.add_argument('--n_dropout', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=4)
    parser.add_argument('--use_xyz', action='store_true')
    parser.add_argument('--use_normal', action='store_true')
    parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--patch_num', type=int, nargs='+', default=96)
    parser.add_argument('--transformer_depth', type=int, nargs='+', default=[3, 3])
    parser.add_argument('--online_bfs', type=int, default=0)
    parser.add_argument('--no_cls', type=int, default=0)
    parser.add_argument('--use_hierarchy', type=int, default=0)
    parser.add_argument('--debug_sub', type=int, default=0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.)
    parser.add_argument('--drop_rate', type=float, default=0.)
    parser.add_argument('--drop_path_rate', type=float, default=0.)
    parser.add_argument('--knn', type=int, default=15)
    parser.add_argument('--agg_method', type=int, default=2)
    parser.add_argument('--ckpt_name',type=str)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--nw',type=int, nargs='+', default=None)
    parser.add_argument('--mlp_ratios', type=int, nargs='+', default=None)
    parser.add_argument('--num_heads', type=int, nargs='+', default=None)
    parser.add_argument('--patch_size',type=int, nargs='+', default=None)
    parser.add_argument('--sr_ratio',type=int, nargs='+', default=None)
    parser.add_argument('--preprocess', type=int, default=0)
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--sample_method',type=str,default='random')
    parser.add_argument('--transformer_type',type=str,default="vit")
    parser.add_argument('--use_ema',type=int, default=0)
    parser.add_argument('--stage',type=int, default=2)
    parser.add_argument('--parts',type=int, default=8)
    parser.add_argument('--label',type=str,default="edge")
    parser.add_argument('--down',type=int,default=1)
    parser.add_argument('--euc', type=int, default=1)
    parser.add_argument('--check_miou',type=int,default=0)
    parser.add_argument('--use_conv',type=int,default=1)
    parser.add_argument('--use_k_neig',type=int,default=0)
    parser.add_argument('--merge_batch',type=int,default=1)
    parser.add_argument('--unet_conv', type=int, default=1)
    parser.add_argument('--eval_lap', type=int, default=0)
    parser.add_argument('--use_relu',type=int,default=0)
    args = parser.parse_args()
    mode = args.mode
    name = args.name
    dataroot = args.dataroot
    if args.seed is not None:
        jt.set_global_seed(args.seed)

    # ========== Dataset ==========
    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')
    extra={"mode": "bfs"}
    train_dataset = SegmentationMapDataset(dataroot, batch_size=args.batch_size, 
        shuffle=True, train=True, num_workers=args.n_worker, extra=extra, args=args)
    test_dataset = SegmentationMapDataset(dataroot, batch_size=1, shuffle=False, train=False, num_workers=args.n_worker, extra=extra, args=args)

    jt.flags.use_cuda = 1
    input_channels = 7
    if args.use_xyz:
        train_dataset.feats.append('center')
        test_dataset.feats.append('center')
        input_channels += 3
    if args.use_normal:
        train_dataset.feats.append('normal')
        test_dataset.feats.append('normal')
        input_channels += 3
    # ========== Network ==========
    net = DGCNN(input_channels, out_channels=args.channels[-1], args=args)
    # unet = U_ConvNet(args.channels[-1], args.patch_num, args.n_classes[-1], args.knn, args)
    unet = CrossUNet(args.channels[-1], args.patch_num, args.n_classes[-1], args.knn, args)
    # net.load("checkpoints/nacc-1.0000.pkl")
    # net = MeshNet(input_channels, out_channels=args.n_classes[-1], depth=args.depth, 
    #     layer_channels=args.channels, residual=args.residual, 
    #     blocks=args.blocks, n_dropout=args.n_dropout)
    transformer = None
    # if args.use_hierarchy == 1:
    if args.transformer_type == "vit":
        transformer = MutilScaleTransformer(num_patches=args.patch_num,
            num_classes=args.n_classes, 
            embed_dim=args.channels[-1],
            depth=args.transformer_depth,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_path_rate,
            down=args.down)
    elif args.transformer_type == "pvt":
        transformer = PvTSegTransformer(num_patches=args.patch_num,
            num_classes=args.n_classes, 
            embed_dim=args.channels[-1],
            depth=args.depth,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path=args.drop_path_rate,
            sr_ratio=args.sr_ratio,
            patch_size=args.patch_size,
            down=args.down)
    elif args.transformer_type == "swin":
        transformer = SwinSegTransformer(num_patches=args.patch_num,
            num_classes=args.n_classes, 
            embed_dim=args.channels[-1],
            depth=args.depth,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path=args.drop_path_rate,
            nw=args.nw,
            window_size=args.window_size)
    # else:
    #     if args.no_cls == 0:
    #         transformer = PatchTransformer(num_patches=args.patch_num[0], num_classes=args.n_classes[0], embed_dim=args.channels[-1], depth=args.transformer_depth)
    #     else:
    #         transformer = PatchAvgTransformer(num_patches=args.patch_num[0], num_classes=args.n_classes[0], embed_dim=args.channels[-1], depth=args.transformer_depth)
    # ========== Optimizer ==========
    if args.optim == 'adam':
        optim = Adam(net.parameters() + transformer.parameters() + unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optim_trans = Adam(transformer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "sgd":
        optim = SGD(net.parameters() + transformer.parameters() + unet.parameters(), lr=args.lr, momentum=0.9)
    else:
        optim = AdamW(net.parameters() + transformer.parameters() + unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_milestones is not None:
        scheduler = MultiStepLR(optim, milestones=args.lr_milestones, gamma=0.1)
    else:
        scheduler = MultiStepLR(optim, milestones=[])

    # ========== MISC ==========
    if jt.rank == 0:
        writer = SummaryWriter("logs/" + name)
    else:
        writer = None
    
    checkpoint_path = os.path.join('checkpoints', name)
    checkpoint_name = os.path.join(checkpoint_path, name + '-latest.pkl')
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.checkpoint is not None:
        unet.load(args.checkpoint[2])
        net.load(args.checkpoint[1])
        transformer.load(args.checkpoint[0])
    train.step = 0
    test.best_acc = 0
    test.best_oacc = 0
    test.best_paths = []
    # ========== Start Training ==========
    # init bfs faces
    if jt.rank == 0:
        print('name: ', name)

    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            train(net, transformer, unet, optim, train_dataset, writer, epoch, args.patch_num[0], args)
            test(net, transformer, unet, test_dataset, writer, epoch, args, args.patch_num[0])
            # check_data(train_dataset, test_dataset, net, transformer, args.patch_num[0])
            scheduler.step()
            jt.sync_all()
            if jt.rank == 0:
                net.save(checkpoint_name)
        jpath = os.path.join("visual", args.name, "acc.json")
        json.dump(acc_map, open(jpath, "w"))
    else:
        test_dataset.set_attrs(batch_size=1)
        # test(net, transformer, unet, test_dataset, writer, 0, args, args.patch_num[0])
        visual(net, transformer, unet, test_dataset, writer, 0, args, args.patch_num[0])
        jpath = os.path.join("visual", args.name, "acc.json")
        json.dump(acc_map, open(jpath, "w"))
