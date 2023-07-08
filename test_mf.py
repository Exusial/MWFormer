import os
import argparse
from tensorboardX import SummaryWriter

import jittor as jt
import jittor.nn as nn
from jittor.optim import Adam
from jittor.optim import SGD
from jittor.lr_scheduler import MultiStepLR

import numpy as np
import pymeshlab
from tqdm import tqdm

from jmesh.dataset import ClassificationDataset, load_mesh
from jmesh.network import MeshNet
from jmesh.utils import to_mesh_tensor
from jmesh.utils import ClassificationMajorityVoting
from jmesh.models.transformer.vision_transformer import PatchTransformer, PatchAvgTransformer, PatchHieTransformer
from jmesh.models.utils.ema import Ema
from jmesh.models.meshformers.meshconv import *
from jmesh.models.meshformers.swin import SwinTransformer
from jmesh.models.meshformers.pvtv2 import PyramidVisionTransformerV2
from jmesh.graph_utils import *
# new
# from utils.bfs_patch import tensor_to_mesh, get_adjacent_dict, get_patches, bfs_patch, get_random_patches, get_hie_adjacent_dict, k_dis_bfs_patch, k_bfs_patch, get_patches_ffs

error_stat = None
def online_process(feats, res_faces, oriadjlist, patch_num, center, next_patch, use_euc=0):
    N = feats.shape[0]
    F = feats.shape[1]
    res_mat = jt.zeros((N, feats.shape[1]))
    # adjlist = []
    centers = []
    init_faces = get_patches_ffs(center, F, patch_num)
    # for i in range(N):
    #     assert patch_num == len(list(init_faces[0].numpy()))
    centers = center[init_faces]
    if next_patch > 0:
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
    else:
        res_mat, res_mat_counter = None, None
        patches = feats.reindex(
            shape=[N, patch_num, feats.shape[2]], # (N, F, C)
            indexes=[
                'i0',
                '@e0(i0,i1)',
                'i2'
            ],
            extras=[init_faces],
        )
    # return patches, adjlist, res_mat
    if next_patch > 0:
        if use_euc == 0:
            adjdict = get_hierar_FAF_cpu(F, oriadjlist, res_mat, patch_num)
        else:
            adjdict = knn_indices_func_gpu(centers, centers, patch_num, 1)
    else:
        adjdict = None
    return patches, res_mat, centers, adjdict, res_mat_counter

def online_process_k(feats, res_faces, adjlist, patch_num, k):
    N, F, _ = feats.shape
    res_mat = k_bfs_patch_cpu(adjlist, F, args.knn, False)
    return res_mat

class MutilScaleTransformer(nn.Module):
    def __init__(self,
                 num_patches=[128],
                 num_classes=[30], 
                 embed_dim=128, 
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 depth=4):
        super(MutilScaleTransformer,self).__init__()
        self.num_patches = num_patches
        embed_dims = [embed_dim]
        for i in range(len(num_classes) - 1):
            embed_dims.append(num_classes[i])
        self.transformers = nn.ModuleList([
            PatchHieTransformer(
                num_patches=num_patches[i], num_classes=num_classes[i], embed_dim=embed_dims[i], depth=depth[i], attn_drop_rate=attn_drop_rate, drop_rate=drop_rate,                 drop_path_rate=drop_path_rate
            ) for i in range(len(num_patches) - 1)
        ])
        self.convs = nn.ModuleList([
            SubConv(num_classes[i], num_classes[i]) for i in range(len(num_patches) - 1)
        ])
        self.transformers.append(
            PatchTransformer(
                num_patches=num_patches[-1], num_classes=num_classes[-1], embed_dim=embed_dims[-1], depth=depth[-1], attn_drop_rate=attn_drop_rate, drop_rate=drop_rate,                  drop_path_rate=drop_path_rate
            )
        )

    def execute(self, x, res_faces, adjdict, k, center,use_euc=0):
        for idx, transformer in enumerate(self.transformers):
            x = transformer(x) # (N, patch_num, F)
            if idx != len(self.transformers) - 1:
                if use_euc == 0:
                    kset = online_process_k(x, res_faces, adjdict, self.num_patches[idx], k)
                else:
                    kset = adjdict[:,:k]
                x = self.convs[idx](x.permute(0, 2, 1), k, kset).permute(0, 2, 1)
            if idx == len(self.transformers) - 1:
                break
            next_patch = self.num_patches[idx + 1] if idx + 1 <= len(self.transformers) else -1
            x, res_faces, center, adjdict, _ = online_process(x, res_faces, adjdict, self.num_patches[idx + 1], center, next_patch,use_euc=use_euc)
        return x

def train(net, transformer, soptim, train_dataset, writer, epoch, patch_num, args, ema=None):
    net.train()

    n_correct = 0
    n_samples = 0

    disable_tqdm = jt.rank != 0
    for meshes, labels, _ in tqdm(train_dataset, desc=f'Train {epoch}', disable=disable_tqdm):
        mesh_tensor = to_mesh_tensor(meshes)
        # if mesh_tensor.faces.shape[1] != 500:
        #     continue
        mesh_labels = jt.int32(labels)
        outputs = net(mesh_tensor.feats, jt.array(meshes['k_faces']), mesh_tensor.faces.shape[1], args, meshes['k_faces'][0].shape[-1])
        if args.debug_sub == 0:
            transformer.train()
            # get patches.
            N = outputs.shape[0]
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
            # patches /= jt.array(meshes["res_face_counter"])
            if args.transformer_type == "vit":
                outputs = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, jt.array(meshes['centers']),use_euc=args.euc)
            elif args.transformer_type == "pvt":
                NFAF = jt.array(meshes['NFAF'])
                if args.euc == 0:
                    kf = k_bfs_patch_cpu(NFAF, args.patch_num[0], args.window_size, False)  
                else:
                    kf = NFAF[:,:args.window_size]
                outputs,_,_,_ = transformer(patches, kf, args.knn, jt.array(meshes['centers']), NFAF, jt.array(meshes['res_faces']), args.preprocess,use_euc=args.euc)
            elif args.transformer_type == "swin":
                NFAF = jt.array(meshes['NFAF'])
                if args.euc == 0:
                    kf = k_bfs_patch_cpu(NFAF, args.patch_num[0], args.window_size, False)  
                else:
                    kf = NFAF[:,:args.window_size]
                outputs,_,_,_ = transformer(patches, kf, jt.array(meshes['centers']), jt.array(meshes['res_faces']), jt.array(meshes['NFAF']),use_euc=args.euc)
        loss = nn.cross_entropy_loss(outputs, mesh_labels, weight=jt.array(train_dataset.weights) if train_dataset.weights is not None else None)
        jt.sync_all()
        optim.step(loss)
        print(loss)
        if args.use_ema:
            ema.update()
        jt.sync_all()

        preds = np.argmax(outputs.data, axis=1)
        n_correct += np.sum(labels == preds)
        n_samples += outputs.shape[0]

        loss = loss.item()
        if jt.rank == 0:
            writer.add_scalar('loss', loss, global_step=train.step)

        train.step += 1
    jt.sync_all(True)

    if jt.rank == 0:
        acc = n_correct / n_samples
        print('train acc = ', acc)
        writer.add_scalar('train-acc', acc, global_step=epoch)

@jt.single_process_scope()
def test(net, transformer, test_dataset, writer, epoch, args, patch_num, ema=None):
    global error_stat
    net.eval()
    acc = 0
    if args.use_ema:
        ema.apply_shadow()
    voted = ClassificationMajorityVoting(args.n_classes[-1])
    wrong_path = []
    with jt.no_grad():
        for meshes, labels, mesh_paths in tqdm(test_dataset, desc=f'Test {epoch}'):
            mesh_tensor = to_mesh_tensor(meshes)
            # outputs = net(mesh_tensor)
            outputs = net(mesh_tensor.feats, jt.array(meshes['k_faces']), mesh_tensor.faces.shape[1],  args, meshes['k_faces'][0].shape[-1])
            if args.debug_sub == 0:
                transformer.eval()
                # get patches.
                N = outputs.shape[0]
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
                # patches /= jt.array(meshes['res_face_counter'])
                # send to transformer
                if args.transformer_type == "vit":
                    outputs = transformer(patches, jt.array(meshes['res_faces']), jt.array(meshes['NFAF']), args.knn, jt.array(meshes['centers']),use_euc=args.euc)
                elif args.transformer_type == "pvt":
                    NFAF = jt.array(meshes['NFAF'])
                    if args.euc == 0:
                        kf = k_bfs_patch_cpu(NFAF, args.patch_num[0], args.window_size, False) 
                    else:
                        kf = NFAF[:,:args.window_size]
                    outputs,_,_,_ = transformer(patches, kf, args.knn, jt.array(meshes['centers']), NFAF, jt.array(meshes['res_faces']), args.preprocess,use_euc=args.euc)
                elif args.transformer_type == "swin":
                    NFAF = jt.array(meshes['NFAF'])
                    if args.euc == 0:
                        kf = k_bfs_patch_cpu(NFAF, args.patch_num[0], args.window_size, False)
                    else:
                        kf = NFAF[:,:args.window_size]
                    outputs,_,_,_ = transformer(patches, kf, jt.array(meshes['centers']), jt.array(meshes['res_faces']), NFAF,use_euc=args.euc,preprocess=args.preprocess)
            mesh_labels = jt.int32(labels)
            loss = nn.cross_entropy_loss(outputs, mesh_labels)
            print("test: ", loss)
            preds = np.argmax(outputs.data, axis=1)
            acc += np.sum(labels == preds)
            if (labels != preds).sum() != 0:
                wrong_path.append(np.array(mesh_paths)[labels != preds])
            # for i in range(len(labels)):
            #     if mesh_labels[i] != preds[i]:
            #         error_stat[int(mesh_labels[i].data)] += 1
    acc /= test_dataset.total_len
    # vacc = voted.compute_accuracy()
    print('test acc = ', acc)
    print('test best acc = ', test.best_acc)
    writer.add_scalar('test-acc', acc, global_step=epoch)
    # writer.add_scalar('test-vacc', vacc, global_step=epoch)
    # print(error_stat)
    # # error_stat = jt.zeros((args.n_classes[-1]))
    # print(wrong_path)
    if test.best_acc < acc:
        test.best_acc = acc
        net.save(os.path.join('checkpoints', name, f'nacc-{acc:.4f}.pkl'))
        transformer.save(os.path.join('checkpoints', name, f'acc-{acc:.4f}.pkl'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, nargs='+')
    parser.add_argument('--n_classes', type=int, nargs='+', required=True)
    parser.add_argument('--depth', type=int, nargs='+', required=True)
    parser.add_argument('--optim', choices=['adam', 'sgd'], default='adam')
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
    parser.add_argument('--use_conv',type=int, default=0)
    parser.add_argument('--euc', type=int, default=0)
    parser.add_argument('--check_miou',type=int,default=0)
    parser.add_argument('--use_k_neig',type=int,default=0)
    parser.add_argument('--use_relu',type=int,default=0)
    parser.add_argument('--down',type=int,default=1)
    args = parser.parse_args()
    mode = args.mode
    name = args.name
    dataroot = args.dataroot
    error_stat = jt.zeros((args.n_classes[-1]))
    if args.seed is not None:
        jt.set_global_seed(args.seed)

    # ========== Dataset ==========
    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')
    extra={"mode": "bfs"}
    train_dataset = ClassificationDataset(dataroot, batch_size=args.batch_size, 
        shuffle=True, train=True, num_workers=args.n_worker, augment=augments, extra=extra, args=args)
    test_dataset = ClassificationDataset(dataroot, batch_size=args.batch_size,  shuffle=False, train=False, num_workers=args.n_worker, extra=extra, args=args)

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
        
    # process_dset(train_dataset, size=2048)
    # process_dset(test_dataset, size=2048)
    # ========== Network ==========
    net = DGCNN(input_channels, out_channels=args.channels[-1], args=args)
    # net = MLP_NET(input_channels, out_channels=args.channels[-1], args=args)
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
            drop_path_rate=args.drop_path_rate)
    elif args.transformer_type == "pvt":
        transformer = PyramidVisionTransformerV2(args.patch_size, args.patch_num, in_chans=args.channels[-1], num_classes=args.n_classes[-1], embed_dims=args.n_classes[:-1], num_heads=args.num_heads, drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate,depths=args.depth, sr_ratios=args.sr_ratio, num_stages=args.stage, mlp_ratios=args.mlp_ratios, linear=args.linear, use_conv=args.use_conv, knn=args.knn)
    elif args.transformer_type == "swin":
        transformer = SwinTransformer(nw=args.nw, patch_size=args.patch_num, in_chans=args.channels[-1], num_classes=args.n_classes[-1], embed_dim=args.n_classes[:-1], num_heads=args.num_heads, drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate,depths=args.depth, mlp_ratios=args.mlp_ratios, use_conv=args.use_conv, knn=args.knn, down=args.down)
    if args.use_ema:
        ema = Ema(transformer, decay=0.99)
        ema.register()
    else:
        ema = None
    # else:
    #     if args.no_cls == 0:
    #         transformer = PatchTransformer(num_patches=args.patch_num[0], num_classes=args.n_classes[0], embed_dim=args.channels[-1], depth=args.transformer_depth)
    #     else:
    #         transformer = PatchAvgTransformer(num_patches=args.patch_num[0], num_classes=args.n_classes[0], embed_dim=args.channels[-1], depth=args.transformer_depth)
    # ========== Optimizer ==========
    if args.optim == 'adam':
        optim = Adam(net.parameters() + transformer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optim_trans = Adam(transformer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = SGD(net.parameters() + transformer.parameters(), lr=args.lr, momentum=0.9)

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
        net.load(args.checkpoint[1])
        transformer.load(args.checkpoint[0])
    train.step = 0
    test.best_acc = 0
    test.best_vacc = 0

    # ========== Start Training ==========
    # init bfs faces
    if jt.rank == 0:
        print('name: ', name)

    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            train(net, transformer, optim, train_dataset, writer, epoch, args.patch_num[0], args, ema)
            test(net, transformer, test_dataset, writer, epoch, args, args.patch_num[0], ema)
            # check_data(train_dataset, test_dataset, net, transformer, args.patch_num[0])
            scheduler.step()

            jt.sync_all()
            if jt.rank == 0:
                net.save(checkpoint_name)
            if args.use_ema:
                ema.restore()
    else:
        test(net, test_dataset, writer, 0, args, ema)
