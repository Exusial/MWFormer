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
from .graph_utils import process_bfs

def augment_points(pts):
    # scale
    n = pts.shape[0]
    pts = pts * np.random.uniform(0.8, 1.25, size=(1, 3))

    # translation
    translation = np.random.uniform(-0.1, 0.1, size=(n, 3))
    pts = pts + translation

    return pts


def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
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
            # mesh = random_scale(mesh)
            mesh.vertices = augment_points(augment_points)
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

    feats = np.vstack(feats)

    return mesh.faces, feats, Fs, V, face_center, mesh


def load_segment(path):
    with open(path) as f:
        segment = json.load(f)
    raw_labels = np.array(segment['raw_labels']) - 1
    sub_labels = np.array(segment['sub_labels']) - 1
    raw_to_sub = np.array(segment['raw_to_sub'])

    return raw_labels, sub_labels, raw_to_sub

def process_extra(extra, mesh_paths, augments, feats, args, npz_paths=None):
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
            faces, _, Fs, V, face_center, mesh = load_mesh(mesh_paths[idx], normalize=True, augments=augments, request=feats)
            npz_data = None
            if not npz_paths is None:
                npz_data = np.load(npz_paths[idx])
            save_path = os.path.join(os.path.dirname(mesh_paths[idx]), f"{idx}.npz")
            init_faces = None
            # if os.path.exists(save_path):
            #     init_faces = jt.array(np.load(save_path)["init_faces"])
            extras = process_bfs(faces.view(np.ndarray), Fs, V.view(np.ndarray), face_center.view(np.ndarray), args, npz_data, mesh, init_faces)
            if "init_faces" in extras:
                np.savez(save_path, init_faces=extras["init_faces"]) 
            attribList.append(extras)
    return attribList    

def expand_data(extra_data, key, max_f, N):
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

class ClassificationDataset(Dataset):
    def __init__(self, dataroot, batch_size, train=True, shuffle=False, num_workers=0, augment=False, in_memory=False, extra=None, args=None):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, keep_numpy_array=True, buffer_size=134217728*2)

        self.batch_size = batch_size
        self.augment = augment
        self.in_memory = in_memory
        self.dataroot = Path(dataroot)
        self.augments = []
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs']
        self.num_classes = args.n_classes[-1]
        self.weights = None
        self.mesh_paths = []
        self.labels = []
        self.npz_paths = []
        self.browse_dataroot()
        self.extra_data = process_extra(extra, self.mesh_paths, self.augments, self.feats, args, self.npz_paths)
        self.set_attrs(total_len=len(self.mesh_paths))


    def browse_dataroot(self):
        self.weights = np.zeros((self.num_classes))
        # self.shape_classes = [x.name for x in self.dataroot.iterdir() if x.is_dir()]
        self.shape_classes = sorted([x.name for x in self.dataroot.iterdir() if x.is_dir()])

        for obj_class in self.dataroot.iterdir():
            if obj_class.is_dir():
                label = self.shape_classes.index(obj_class.name)
                cls_path = os.path.join(obj_class, self.mode)
                for obj_path in (obj_class / self.mode).iterdir():
                    if obj_path.is_file():
                        if obj_path.suffix == ".obj":
                            self.mesh_paths.append(obj_path)
                            self.labels.append(label)
                            self.weights[label] += 1
                            basename = os.path.basename(str(obj_path))
                            if os.path.exists(os.path.join(cls_path, basename.split(".")[0]+ ".npz")):
                                self.npz_paths.append(os.path.join(cls_path, basename.split(".")[0] + ".npz"))
        self.mesh_paths = np.array(self.mesh_paths)
        self.labels = np.array(self.labels)
        self.weights /= np.sum(self.weights)
        self.weights = 1 / np.log(self.weights + 1.2)
        if len(self.npz_paths) > 0:
            self.npz_paths = np.array(self.npz_paths)
        else:
            self.npz_paths = None

    def __getitem__(self, idx):
        faces, feats, Fs, _, _,_ = load_mesh(self.mesh_paths[idx],
                                     normalize=True,
                                     augments=self.augments,
                                     request=self.feats)
        label = self.labels[idx]
        return faces, feats, Fs, label, self.mesh_paths[idx], self.extra_data[idx]

    def collate_batch(self, batch):
        faces, feats, Fs, labels, mesh_paths, extra_data = zip(*batch)
        N = len(batch)        
        max_f = max(Fs)

        np_faces = np.zeros((N, max_f, 3), dtype=np.int32)
        np_feats = np.zeros((N, feats[0].shape[0], max_f), dtype=np.float32)
        np_Fs = np.int32(Fs)

        for i in range(N):
            np_faces[i, :Fs[i]] = faces[i]
            np_feats[i, :, :Fs[i]] = feats[i]
        
        meshes = {
            'faces': np_faces,
            'feats': np_feats,
            'Fs': np_Fs
        }
        if extra_data[0] is not None:
            for key in extra_data[0]:
                if isinstance(extra_data[0][key], np.ndarray):
                    if key == "res_faces" or key == "k_faces" or key == "res_face_counter" or key == "p2f":
                        meshes[key] = expand_data(extra_data, key, max_f, N)
                    else: 
                        meshes[key] = np.stack([data[key] for data in extra_data], 0)
                elif isinstance(extra_data[0][key], dict):
                    meshes[key] = [data[key] for data in extra_data]
                elif isinstance(extra_data[0][key], type(None)):
                    meshes[key] = np.array([])
                elif key != "init_faces":
                    print("Unsupported Type: ", type(extra_data[0][key]))
        labels = np.array(labels)

        return meshes, labels, mesh_paths

def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64') - 1
    return seg_labels

def build_gemm(faces, ori_labels):
    face_label = []
    edge2key = dict()
    edge2face = dict()
    edges = []
    edges_count = 0
    for face_id, face in enumerate(faces):
        faces_edges = []
        fl = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edge2face[edges_count] = [face_id]
                edges_count += 1
            else:
                edge2face[edge2key[edge]].append(face_id)
            fl.append(ori_labels[edge2key[edge]])
        count = np.bincount(fl)
        face_label.append(np.argmax(count))
    return face_label, edge2key, edge2face

class SegmentationMapDataset(Dataset):
    def __init__(self, dataroot, batch_size, train=True, shuffle=False, num_workers=0, augments=None, in_memory=False, extra=None, args=None):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, keep_numpy_array=True, buffer_size=134217728)
        self.batch_size = batch_size
        self.in_memory = in_memory
        self.dataroot = dataroot

        self.augments = []
        if train and augments:
            self.augments = augments

        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs']

        self.mesh_paths = []
        self.raw_paths = []
        self.seg_paths = []
        self.browse_dataroot()
        self.extra_data = process_extra(extra, self.mesh_paths, self.augments, self.feats, args)
        self.set_attrs(total_len=len(self.mesh_paths))

    def browse_dataroot(self):
        for dataset in (Path(self.dataroot) / self.mode).iterdir():
            if not dataset.is_dir():
                continue
            for obj_path in dataset.iterdir():
                if obj_path.suffix == '.obj':
                    obj_name = obj_path.stem
                    seg_path = obj_path.parent / (obj_name + '.json')

                    raw_name = obj_name.rsplit('-', 1)[0]
                    raw_path = list(Path(self.dataroot).glob(f'raw/{raw_name}.*'))[0]
                    self.mesh_paths.append(str(obj_path))
                    self.raw_paths.append(str(raw_path))
                    self.seg_paths.append(str(seg_path))
        self.mesh_paths = np.array(self.mesh_paths)
        self.raw_paths = np.array(self.raw_paths)
        self.seg_paths = np.array(self.seg_paths)

    def __getitem__(self, idx):
        faces, feats, Fs, _, _,_ = load_mesh(self.mesh_paths[idx], 
                                     normalize=True, 
                                     augments=self.augments,
                                     request=self.feats)
        raw_labels, sub_labels, raw_to_sub = load_segment(self.seg_paths[idx])

        return faces, feats, Fs, raw_labels, sub_labels, raw_to_sub, self.mesh_paths[idx], self.raw_paths[idx]

    def collate_batch(self, batch):
        faces, feats, Fs, raw_labels, sub_labels, raw_to_sub, mesh_paths, raw_paths = zip(*batch)
        N = len(batch)        
        max_f = max(Fs)

        np_faces = np.zeros((N, max_f, 3), dtype=np.int32)
        np_feats = np.zeros((N, feats[0].shape[0], max_f), dtype=np.float32)
        np_Fs = np.int32(Fs)
        np_sub_labels = np.ones((N, max_f), dtype=np.int32) * -1

        for i in range(N):
            np_faces[i, :Fs[i]] = faces[i]
            np_feats[i, :, :Fs[i]] = feats[i]
            np_sub_labels[i, :Fs[i]] = sub_labels[i]
        
        meshes = {
            'faces': np_faces,
            'feats': np_feats,
            'Fs': np_Fs
        }
        if extra_data[0] is not None:
            for key in extra_data[0]:
                if isinstance(extra_data[0][key], np.ndarray):
                    if key == "res_faces" or key == "k_faces":
                        meshes[key] = expand_data(extra_data, key, max_f, N)
                    else: 
                        meshes[key] = np.stack([data[key] for data in extra_data], 0)
                elif isinstance(extra_data[0][key], dict):
                    meshes[key] = [data[key] for data in extra_data]
                elif isinstance(extra_data[0][key], type(None)):
                    meshes[key] = np.array([])
                elif key != "init_faces":
                    print("Unsupported Type: ", type(extra_data[0][key]))
        labels = np_sub_labels
        mesh_info = {
            'raw_labels': raw_labels,
            'raw_to_sub': raw_to_sub,
            'mesh_paths': mesh_paths,
            'raw_paths': raw_paths,
        } 
        return meshes, labels, mesh_info

class SegmentationMapDataset(Dataset):
    def __init__(self, dataroot, batch_size, train=True, shuffle=False, num_workers=0, augments=None, in_memory=False, extra=None, args=None):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, keep_numpy_array=True, buffer_size=134217728)
        self.batch_size = batch_size
        self.in_memory = in_memory
        self.dataroot = dataroot

        self.augments = []
        if train and augments:
            self.augments = augments

        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs']

        self.mesh_paths = []
        self.raw_paths = []
        self.seg_paths = []
        self.dir = os.path.join(dataroot, self.mode)
        self.paths = sorted(self.make_dataset(self.dir))
        self.seg_paths = self.get_seg_files(self.paths, os.path.join(dataroot, 'seg'), seg_ext='.eseg')
        # self.sseg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'sseg'), seg_ext='.seseg')
        self.classes, self.offset = self.get_n_segs(os.path.join(dataroot, 'classes.txt'), self.seg_paths)
        self.nclasses = len(self.classes)
        self.raw_label = []
        self.e2f = []
        self.max_e = 2250
        self.edge_label = []

        for idx in range(len(self.seg_paths)):
            faces, feats, Fs, V, _,_ = load_mesh(self.paths[idx], 
                                     normalize=True, 
                                     augments=self.augments,
                                     request=self.feats)
            seg_labels = read_seg(self.seg_paths[idx])
            if args.label == "edge":
                self.edge_label.append(seg_labels)
                self.max_e = np.maximum(self.max_e, len(seg_labels))
                rl, _, e2f = build_gemm(faces, seg_labels)
                self.raw_label.append(rl)
                self.e2f.append(e2f)
            else: # face label
                self.raw_label.append(rl)
            # self.raw_label.append(build_gemm(faces, seg_labels))
        self.extra_data = process_extra(extra, self.paths, self.augments, self.feats, args)
        self.set_attrs(total_len=len(self.paths))

    @staticmethod
    def get_seg_files(paths, seg_dir, seg_ext='.seg'):
        segs = []
        for path in paths:
            segfile = os.path.join(seg_dir, os.path.splitext(os.path.basename(path))[0] + seg_ext)
            assert(os.path.isfile(segfile))
            segs.append(segfile)
        return segs

    @staticmethod
    def get_n_segs(classes_file, seg_files):
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype='float64')
            for seg in seg_files:
                all_segs = np.concatenate((all_segs, read_seg(seg)))
            segnames = np.unique(all_segs)
            np.savetxt(classes_file, segnames, fmt='%d')
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset

    @staticmethod
    def make_dataset(path):
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if fname.endswith(".obj"):
                    path = os.path.join(root, fname)
                    meshes.append(path)

        return meshes

    def __getitem__(self, idx):
        faces, feats, Fs, _, _,_ = load_mesh(self.paths[idx], 
                                     normalize=True, 
                                     augments=self.augments,
                                     request=self.feats)

        return faces, feats, Fs, self.raw_label[idx], self.paths[idx], self.extra_data[idx]

    def collate_batch(self, batch):
        faces, feats, Fs, raw_labels, mesh_paths, extra_data = zip(*batch)
        N = len(batch)        
        max_f = max(Fs)

        np_faces = np.zeros((N, max_f, 3), dtype=np.int32)
        np_feats = np.zeros((N, feats[0].shape[0], max_f), dtype=np.float32)
        np_Fs = np.int32(Fs)
        np_raw_labels = np.ones((N, max_f), dtype=np.int32) * -1
        for i in range(N):
            np_faces[i, :Fs[i]] = faces[i]
            np_feats[i, :, :Fs[i]] = feats[i]
            np_raw_labels[i, :Fs[i]] = raw_labels[i]
        # print(np_raw_labels)
        meshes = {
            'faces': np_faces,
            'feats': np_feats,
            'Fs': np_Fs
        }
        if extra_data[0] is not None:
            for key in extra_data[0]:
                if isinstance(extra_data[0][key], np.ndarray):
                    if key == "res_faces" or key == "k_faces":
                        meshes[key] = expand_data(extra_data, key, max_f, N)
                    else: 
                        meshes[key] = np.stack([data[key] for data in extra_data], 0)
                elif isinstance(extra_data[0][key], dict):
                    meshes[key] = [data[key] for data in extra_data]
                elif isinstance(extra_data[0][key], type(None)):
                    meshes[key] = np.array([])
                elif key != "init_faces":
                    print("Unsupported Type: ", type(extra_data[0][key]))
        labels = np_raw_labels
        mesh_info = {
            'mesh_paths': mesh_paths
        } 
        return meshes, labels, mesh_info