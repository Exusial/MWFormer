import argparse
from argparse import RawTextHelpFormatter
import os
import tqdm
import open3d
import trimesh
import time
from trimesh.base import Trimesh
import numpy as np
import jittor as jt
from jmesh.graph_utils import sample_faces_on_mesh, unoverlapping_bfs_patch_cpu, k_bfs_patch_cpu, get_hierar_FAF_cpu

output_path = ''
mesh_counter = 0
fcounter = 0
def get_sampling_positions(original_vertices: np.ndarray, stride: float):
    """returns center points of uniform sampling with certain stride length along the ground plane

    Arguments:
        original_vertices {np.ndarray} -- 3D coordinates of vertices
        stride {float} -- distance between center points of sampling

    Returns:
        Tuple[np.ndarray, np.ndarray] -- returns sampling positions in x and y direction (always full height!)
    """
    mins_xyz = original_vertices[:, :3].min(axis=0)
    maxs_xyz = original_vertices[:, :3].max(axis=0)

    sampling_positions_x = np.arange(mins_xyz[0], maxs_xyz[0], stride)
    offset_x = (maxs_xyz[0] - sampling_positions_x[-1]) / 2
    sampling_positions_x = sampling_positions_x + offset_x

    sampling_positions_y = np.arange(mins_xyz[1], maxs_xyz[1], stride)
    offset_y = (maxs_xyz[1] - sampling_positions_y[-1]) / 2
    sampling_positions_y = sampling_positions_y + offset_y

    return sampling_positions_x, sampling_positions_y

def convert_face_matrix(face_num, fl):
    faf = -np.ones((1, face_num, 3)).astype("int32")
    counter = np.zeros(face_num, "int8")
    for f in fl:
        if counter[f[0]] < 3:
            faf[0, f[0], counter[f[0]]] = f[1]
            counter[f[0]] += 1
        if counter[f[1]] < 3:
            faf[0, f[1], counter[f[1]]] = f[0]
            counter[f[1]] += 1
    return faf

def recursive_split(min_bound, max_bound, F, V, labels, face_center, args, vis, split_dim=0, cf=None):
    contain_face = []
    contain_vertex = set()
    global data_suffix, label_suffix, output_path, mesh_counter, fcounter
    if cf is None:
        for idx, center in enumerate(face_center):
            # if vis[idx] == 1: continue
            min_check = (center >= min_bound).sum() == 3
            max_check = (center <= max_bound).sum() == 3
            if min_check and max_check:
                contain_face.append(idx)
                if args.save_mesh == 1:
                    for v in F[idx]:
                        contain_vertex.add(v)
        # vis[contain_face] = 1
    else:
        for fidx in cf:
            center = face_center[fidx]
            min_check = (center >= min_bound).sum() == 3
            max_check = (center <= max_bound).sum() == 3
            if min_check and max_check:
                contain_face.append(fidx)
                if args.save_mesh == 1:
                    for v in F[fidx]:
                        contain_vertex.add(v)
    # print(len(contain_face))
    if len(contain_face) > args.threshold:
        mid_bound_dim = (min_bound[split_dim] + max_bound[split_dim]) / 2
        nmax_bound = np.zeros(3)
        nmin_bound = np.zeros(3)
        nmax_bound[split_dim] = nmin_bound[split_dim] = mid_bound_dim
        nmax_bound[1 - split_dim] = max_bound[1 - split_dim]
        nmax_bound[2] = float("inf")
        nmin_bound[1 - split_dim] = min_bound[1 - split_dim] 
        nmin_bound[2] = float("-inf")
        # #print("rec")
        # print(min_bound, nmax_bound, nmin_bound, max_bound)
        recursive_split(min_bound, nmax_bound, F, V, labels, face_center, args, vis, 1-split_dim, contain_face)
        recursive_split(nmin_bound, max_bound, F, V, labels, face_center, args, vis, 1-split_dim, contain_face)
        # #print("erec")
        return
    if len(contain_face) == 0:
        return
    mesh_counter += 1
    if os.path.exists(os.path.join(output_path, f"{mesh_counter}_" + data_suffix)):
        return
    st = time.time()
    box_faces = F[contain_face]
    # box_labels = labels[contain_face]
    fcounter += len(contain_face)
    # if args.save_params == 1:
    #     # print("index time: ", time.time() - st)
    #     # get adjaceny
    #     st = time.time()
    #     af = trimesh.graph.face_adjacency(box_faces)
    #     faf = jt.array(convert_face_matrix(len(contain_face), af))
    #     # print("face num:", len(contain_face))
    #     # print("convert time: ", time.time() - st)
    #     # print(faf.shape)
    #     # get init faces
    #     extra = {}
    #     # st = time.time()
    #     init_faces = sample_faces_on_mesh(len(contain_face), args.patch_num[0], "ffs", centers=face_center[contain_face])
    #     #print("sample time: ", time.time() - st)
    #     # st = time.time()
    #     res_faces, _ = unoverlapping_bfs_patch_cpu(faf, len(contain_face), jt.array(init_faces).unsqueeze(0))
    #     find_unagg_face(face_center, res_faces, init_faces)
    #     #print("bfs time: ", time.time() - st)
    #     # st = time.time()
    #     NFAF = get_hierar_FAF_cpu(args.patch_num[0], faf, res_faces, args.patch_num[1])[0].numpy()
    #     #print("NFAF time: ", time.time() - st)
    #     res_dict = k_bfs_patch_cpu(faf.unsqueeze(0), len(contain_face), args.knn, False).numpy()[0]
    #     # jt.sync_all()
    #     extra['NFAF'] = NFAF
    #     extra['k_faces'] = res_dict
    #     extra['res_faces'] = res_faces.numpy()
    #     extra['faces'] = box_faces
    #     np.save(os.path.join(output_path, f"{mesh_counter}_" + data_suffix), extra)
    if args.save_mesh == 1:
        vertice_map = {}
        contain_vertex = list(contain_vertex)
        box_vertices = V[contain_vertex]
        for iv, v in enumerate(contain_vertex):
            vertice_map[v] = iv
        for f in box_faces:
            for fi in range(3):
                f[fi] = vertice_map[f[fi]]
        mesh = Trimesh(vertices=box_vertices, faces=box_faces)
        mesh.export(os.path.join(output_path, f"{mesh_counter}_" + data_suffix.replace(".npy", ".obj")))
    # jt.save(box_labels, os.path.join(output_path, f"{mesh_counter}_" + label_suffix))
    jt.save(contain_face, os.path.join(output_path, f"{mesh_counter}_mapping_" + label_suffix))

def load_mesh(path, label_path, split, args):
    import jittor as jt
    print("now: ", path)
    global data_suffix, label_suffix, output_path, mesh_counter, fcounter
    mesh_counter = 0
    fcounter = 0
    mesh = trimesh.load_mesh(path, process=False)
    Fs = mesh.faces.shape[0]
    # sim_mesh = mesh.simplify_quadratic_decimation(int(Fs * 0.5))
    # label_map = sim_mesh.as_open3d.face_mapping
    # print(label_map)
    data_suffix = path.split("/")[-1].replace(".obj", ".npy")
    label_suffix = path.split("/")[-1].replace(".obj", ".pkl")
    F = mesh.faces
    V = mesh.vertices
    if os.path.exists(os.path.join(output_path, "20_" + data_suffix)):
        return
    if split != "test":
        labels = jt.load(label_path)
    else:
        labels = None
    block_size = args.block_size
    stride = args.stride
    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    face_counter = np.zeros(Fs).astype("int32")
    sampling_positions_x, sampling_positions_y = get_sampling_positions(V, stride)
    block_counter = 0
    for x_pos in tqdm.tqdm(sampling_positions_x):
        for y_pos in sampling_positions_y:
            contain_face = []
            contain_vertex = set()
            vertice_map = {}
            # try:
            # find borders of current crop
            min_bound = np.array(
                [x_pos - block_size / 2, y_pos - block_size / 2, float("-inf")])
            max_bound = np.array(
                [x_pos + block_size / 2, y_pos + block_size / 2, float("inf")])

            recursive_split(min_bound, max_bound, F, V, labels, face_center, args, face_counter)
    print(Fs, fcounter)

def simplify_mesh(mp, lp, output_path):
    print(mp)
    mesh = open3d.io.read_triangle_mesh(mp)
    labels = jt.load(lp)[0]
    data_suffix = mp.split("/")[-1]
    # label_suffix = lp.split("/")[-1]
    Fs = np.asarray(mesh.triangles).shape[0]
    mesh = mesh.simplify_quadric_decimation(int(Fs*0.2))
    # simp_labels = labels[list(mesh.face_mapping)]
    open3d.io.write_triangle_mesh(os.path.join(output_path, data_suffix), mesh)
    # jt.save(simp_labels, os.path.join(output_path, label_suffix))
    return mesh

def check_crops(path1, path2=None):
    min_face = 10000
    count = 0
    for tname in os.listdir(path1):
        if tname.endswith(".obj"):
            mesh = jt.load(os.path.join(path1, tname.replace(".obj", ".pkl")))
            if mesh.shape[0] < 256:
                count += 1
            min_face = min(min_face, mesh.shape[0])
            if path2 is not None:
                mesh2 = trimesh.load(os.path.join(path2, tname))
                assert mesh2.faces.shape[0] == mesh.faces.shape[0]
    print(min_face, count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description="Create crops from graph hierarchies")
    parser.add_argument('--block_size', const=3., default=3., type=float, nargs='?',
                        help='block size in meter')
    parser.add_argument('--stride', const=1.5, default=1.5, type=float, nargs='?',
                        help='stride between center points')
    parser.add_argument('--threshold', default=10000, type=int)
    parser.add_argument('--patch_num', type=int, nargs='+')
    parser.add_argument('--knn', type=int, default=3)
    parser.add_argument('--simp', type=int, default=1)
    parser.add_argument('--save_params', type=int, default=1)
    parser.add_argument('--save_mesh', type=int, default=1)
    args = parser.parse_args()
    # check_crops("/data/penghy/processed_scannet/crop2_train")
    mesh_path = "/data/penghy/processed_scannet"
    split = ["test"]
    for sp in split:
        mp = os.path.join(mesh_path, "simp_" + sp) if args.simp == 0 else os.path.join(mesh_path, sp)
        output_path = os.path.join(mesh_path, "crop3_" + sp)
        for obj_p in os.listdir(mp):
            if obj_p.endswith(".obj"):
                label_p = obj_p.replace(".obj", ".pkl")
                if args.simp == 0:
                    load_mesh(os.path.join(mp, obj_p), os.path.join(mp, label_p), sp, args)
                else:
                    output_path = os.path.join(mesh_path, "simp_" + sp)
                    simplify_mesh(os.path.join(mp, obj_p), os.path.join(mp, label_p), output_path)
