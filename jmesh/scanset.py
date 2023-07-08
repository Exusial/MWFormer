import json
import random
from pathlib import Path
import tqdm
import jittor as jt
from jittor.dataset import Dataset
import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .graph_utils import process_bfs, check_ops

def augment_points(pts):
    # scale
    pts = pts * np.random.uniform(0.8, 1.25)

    # translation
    translation = np.random.uniform(-0.1, 0.1)
    pts = pts + translation

    return pts


def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    axis_seq = ''.join(random.sample('xy', 2))
    angles = [random.choice([0, 90, 180, 270]) for _ in range(2)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh.vertices = rotation.apply(mesh.vertices)
    return mesh


def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh


def mesh_normalize(mesh: trimesh.Trimesh):
    vertices = mesh.vertices - mesh.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh.vertices = vertices
    return mesh


def load_mesh(path, normalize=False, augments=[], request=[]):
    mesh = trimesh.load_mesh(path, process=False)

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)

    if normalize:
        mesh = mesh_normalize(mesh)

    F = mesh.faces
    V = mesh.vertices
    Fs = mesh.faces.shape[0]
    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    # corner = V[F.flatten()].reshape(-1, 3, 3) - face_center[:, np.newaxis, :]
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])
    
    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))
    if len(request) > 0:
        feats = np.vstack(feats)

    return mesh.faces, feats, Fs, V, face_center

def load_mesh_norm(path, normalize=False):
    mesh = trimesh.load_mesh(path, process=False)
    if normalize:
        mesh = mesh_normalize(mesh)
    return mesh

def load_segment(path):
    with open(path) as f:
        segment = json.load(f)
    raw_labels = np.array(segment['raw_labels']) - 1
    sub_labels = np.array(segment['sub_labels']) - 1
    raw_to_sub = np.array(segment['raw_to_sub'])

    return raw_labels, sub_labels, raw_to_sub

def process_extra(extra, mesh_paths, augments, feats, args, save_dir=None):
    '''
    Used to store extra information of meshes.
    '''
    from copy import deepcopy
    if extra is None:
        return None
    mode = extra["mode"]
    if mode == "bfs":
        attribList = []
        for idx in tqdm.tqdm(range(len(mesh_paths))):
            # print(mesh_paths[idx])
            if save_dir is not None:
                name_prefix = mesh_paths[idx].split("/")[-1].split(".")[0]
                if os.path.exists(os.path.join(save_dir, name_prefix+".npy")):
                    continue
            faces, _, Fs, V, face_center = load_mesh(mesh_paths[idx], normalize=False, augments=[], request=[])
            # result = process_bfs(faces.view(np.ndarray), Fs, V.view(np.ndarray), face_center.view(np.ndarray), args)
            # result0 = process_bfs_ori(faces.view(np.ndarray), Fs, V.view(np.ndarray), face_center.view(np.ndarray), args)
            # check_ops(result, result0)
            attribList.append(process_bfs(faces.view(np.ndarray), Fs, V.view(np.ndarray), face_center.view(np.ndarray), args))
            if save_dir is not None:
                name_prefix = mesh_paths[idx].split("/")[-1].split(".")[0]
                np.save(os.path.join(save_dir, name_prefix), attribList[-1])
    return attribList    


def expand_data(extra_data, key, max_f, N, patch_num=0):
    '''
    Used for meshes with different face numbers.
    '''  
    if key == "res_faces":
        np_faces = -np.ones((N, max_f), dtype=np.int32)
        for i in range(N):
            Fs = extra_data[i][key].shape[0]
            np_faces[i,:Fs] = extra_data[i][key]
        return np_faces
    elif key == "k_faces":
        k = extra_data[0][key].shape[-1]
        np_faces = -np.ones((N, max_f, k), dtype=np.int32)
        for i in range(N):
            Fs = extra_data[i][key].shape[0]
            np_faces[i,:Fs,:] = extra_data[i][key]
        return np_faces
    elif key == "centers":
        k = extra_data[0][key].shape[-1]
        np_faces = 16384 * np.ones((N, patch_num, 3), dtype=np.float32)
        for i in range(N):
            Fs = extra_data[i][key].shape[0]
            Fs = min(Fs, patch_num)
            np_faces[i,:Fs,:] = extra_data[i][key]
        return np_faces

class ScanDataset(Dataset):
    def __init__(self, dataroot, batch_size, train='train', shuffle=False, num_workers=0, augments=None, in_memory=False, extra=None, drop_last=False, args=None):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, keep_numpy_array=True, buffer_size=134217728, drop_last=drop_last)
        self.batch_size = batch_size
        self.in_memory = in_memory
        self.dataroot = dataroot
        self.meshroot = dataroot
        self.augments = []
        if train and augments:
            self.augments = augments

        self.mode = train
        if self.mode != "train" and args.eval_lap == 1:
            self.save_dir = os.path.join(self.dataroot, "npy3_" + self.mode)
        else:
            self.save_dir = os.path.join(self.dataroot, "npy2_" + self.mode)
        self.feats = ['area', 'face_angles', 'curvs', 'center', 'normal']

        self.current_path = None
        self.current_mesh = None
        self.mesh_paths = []
        self.seg_paths = []
        self.npy_paths = []
        # self.all_mesh_paths = []
        # self.all_seg_paths = []
        # self.all_npy_paths = []
        self.total_length = 0
        self.slice_count = 0
        self.disk_io = 100000 # if too many input mesh, process meshes in turns.
        self.slice_id = 0
        # if self.disk_io:
        #     self.mesh_paths = self.all_mesh_paths[:disk_io]
        #     self.seg_paths = self.all_seg_paths[:disk_io]
        self.patch_num = args.patch_num[0]
        self.k_nn = args.use_k_neig
        self.args = args
        self.ignore_count = 0
        self.browse_dataroot()
        self.extra_data = process_extra(extra, self.mesh_paths, self.augments, self.feats, args, self.save_dir)
        self.set_attrs(total_len=len(self.mesh_paths))

    def browse_dataroot(self):
        self.total_length = len(os.listdir(os.path.join(self.dataroot, self.mode)))
        # for data in os.listdir(os.path.join(self.dataroot, "crop_"+self.mode)):
        #     if data.endswith(".npy"):
        #         self.npy_paths.append(os.path.join(self.dataroot, "crop_"+self.mode, data))
        #         self.seg_paths.append(os.path.join(self.dataroot, "crop_"+self.mode, data.split(".")[0] + ".pkl"))
        if self.mode != "train" and self.args.eval_lap == 1:
            prefix = "crop3_"
        else:
            prefix = "crop2_"
        for data in os.listdir(os.path.join(self.meshroot, prefix+self.mode)):
            if data.endswith(".obj"):
                # mesh = trimesh.load(os.path.join(self.meshroot, "crop_"+self.mode, data), process=False)
                # if mesh.faces.shape[0] < self.patch_num:
                #     self.ignore_count += 1
                #     print(self.ignore_count)
                #     continue
                self.mesh_paths.append(os.path.join(self.meshroot, prefix+self.mode, data))
                self.seg_paths.append(os.path.join(self.dataroot, prefix+self.mode, data.split(".")[0] + ".pkl"))

    def next_slice(self):
        if self.slice_id + self.disk_io >= self.total_length:
            self.slice_id = 0
        else:
            self.slice_id += self.disk_io
        self.seg_paths = self.all_seg_paths[self.slice_id:min(self.total_length, self.slice_id + self.disk_io)]
        self.mesh_paths = self.all_mesh_paths[self.slice_id:min(self.total_length, self.slice_id + self.disk_io)]
        self.extra_data = process_extra(extra, self.mesh_paths, self.augments, self.feats, args, self.save_dir)

    def __getitem__(self, idx):
        # npy_file = self.npy_paths[idx]
        # mesh_path = os.path.join(self.meshroot, self.mode, "_".join(npy_file.split("_")[1:]).replace(".npy", ".obj"))
        # if current_path is None or current_path != mesh_path:
        #     current_path = mesh_path
        #     current_mesh = load_mesh_norm(current_path, normalize=True)
        _, feats, Fs, V, _ = load_mesh(self.mesh_paths[idx], 
                                     normalize=False, 
                                     augments=self.augments,
                                     request=self.feats)
        if Fs == 1:
            print(self.mesh_paths[idx])
        if self.mode != "test":
            raw_labels = jt.load(self.seg_paths[idx])
        else:
            raw_labels = None
        extra_data = None
        if self.save_dir is not None:
            name_prefix = self.mesh_paths[idx].split("/")[-1].split(".")[0] + ".npy"
            extra_data = np.load(os.path.join(self.save_dir, name_prefix), allow_pickle=True).item()
        else:
            extra_data = self.extra_data[idx]
        return feats, Fs, raw_labels, self.mesh_paths[idx], extra_data

    def collate_batch(self, batch):
        feats, Fs, raw_labels, mesh_paths, extra_data = zip(*batch)
        N = len(batch)        
        max_f = max(Fs)
        self.slice_count += N
        np_feats = np.zeros((N, feats[0].shape[0], max_f), dtype=np.float32)
        np_Fs = np.int32(Fs)
        np_sub_labels = np.zeros((N, max_f), dtype=np.int32)
        for i in range(N):
            np_feats[i, :, :Fs[i]] = feats[i]
            if self.mode != "test":
                np_sub_labels[i, :Fs[i]] = raw_labels[i]
        meshes = {
            'feats': np_feats,
            'Fs': np_Fs,
        }
        if extra_data[0] is not None:
            for key in extra_data[0]:
                if key == "res_face_counter":
                    continue
                if isinstance(extra_data[0][key], np.ndarray):
                    if key == "res_faces" or key == "k_faces" or key == "centers":
                        meshes[key] = expand_data(extra_data, key, max_f, N, self.patch_num)
                    elif key == "NFAF":
                        meshes[key] = np.array([])
                    else: 
                        print(key)
                        meshes[key] = np.stack([data[key] for data in extra_data], 0)
                elif isinstance(extra_data[0][key], dict):
                    meshes[key] = [data[key] for data in extra_data]
                elif isinstance(extra_data[0][key], type(None)):
                    meshes[key] = np.array([])
                else:
                    print("Unsupported Type: ", type(extra_data[0][key]))
        labels = np_sub_labels
        mesh_info = {
            'mesh_paths': mesh_paths,
        } 
        return meshes, labels, mesh_info