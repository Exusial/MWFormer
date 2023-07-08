from sklearn.neighbors import NearestNeighbors
import trimesh
from trimesh.base import Trimesh
from trimesh.proximity import closest_point
import pymeshlab
import networkx as nx
import numpy as np
import jittor as jt
import time
from .utils import save_patches

def get_f2f_dis(mesh):
    fs = mesh.faces.shape[0]
    nf2f = np.ones((fs, fs)) * fs
    G = nx.Graph()
    G.add_edges_from(mesh.face_adjacency)
    f2f = nx.all_pairs_shortest_path_length(G)
    for s, d in f2f:
        for k, v in d.items():
            nf2f[s, k] = v
    return nf2f

def get_random_patches(num_face, patch_num):
    init_faces = set()
    while len(init_faces) < patch_num:
        init_faces.add(np.random.randint(0, num_face))
    return list(init_faces)

def get_hierar_FAF_cpu(patch_num, FAF, res_faces, next_size):
    face_num = FAF.shape[1]
    cpu_header='''
    #include<iostream>
    #include<cstring>
    using namespace std;
    '''
    cpu_src=f'''
    @alias(FAF, in0)
    @alias(res_faces, in1)
    @alias(NFAF, out0)
    int N = FAF_shape0;
    int patch_num = {patch_num};
    int F = FAF_shape1;
    int L = FAF_shape2;
    int next_size = {next_size};
    // int header_counter[{face_num}];
    // #pragma omp parallel for num_threads(3)
    for(int i=0;i<N;i++) {{
        for(int j=0;j<F;j++){{
            if(@res_faces(i, j) == -1) continue;
            for(int k=0;k<L;k++){{
                if(@FAF(i, j, k) == -1) break;
                if(@res_faces(i, @FAF(i, j, k)) == -1) continue;
                // std::cout << i << " " << j << " " << k << " " << " " << @res_faces(i, j) << " " << @FAF(i, j, k) << " " << @res_faces(i, @FAF(i, j, k)) << std::endl;
                // cout << i << " " << @res_faces(i, j)<< " " << @res_faces(i, @FAF(i, j, k)) << endl;;
                @NFAF(i, @res_faces(i, j), @res_faces(i, @FAF(i, j, k))) = 1;
            }}
        }}
        for(int j=0;j<patch_num;j++){{
            int face_header = 0;
            int k = 0;
            for(k=0;k<patch_num;k++){{
                if(@NFAF(i, j, k) != -1) {{
                    if(j != k)
                        @NFAF(i, j, face_header++) = k;
                }}
            }}
            for(;face_header<next_size;) {{
                @NFAF(i, j, face_header++) = -1;
            }}
        }}
    }}
    '''
    NFAF = -jt.ones((FAF.shape[0], patch_num, patch_num)).int32()
    NFAF = jt.code([FAF, res_faces], [NFAF], cpu_header=cpu_header, cpu_src=cpu_src)[0]
    return NFAF[:,:,:next_size]

def get_hierar_FAF_gpu(patch_num, FAF, res_faces, next_size):
    face_num = FAF.shape[1]
    cuda_header='''
    #include<cuda.h>
    __gloabl__ void get_hierar_FAF_gpu(int patch_num, int face_num, int L, int* FAF, int* res_faces, int* NFAF, int next_size) {
        int i = blockIdx.x;
        int f = threadIdx.x;
        int step = blockDim.x;
        if(f >= face_num) return;
        for(int j=f;j<face_num;j+=step){
            if(res_faces[i, j] == -1) continue;
            for(int k=0;k<L;k++){
                if(FAF[i, j, k] == -1) break;
                if(res_faces[i, FAF[i, j, k]] == -1) continue;
                NFAF[i, res_faces[i, j], res_faces[i, FAF[i, j, k]]] = 1;
            }
        }
    }
    __global__ void postprocess_NFAF(int patch_num, int face_num, int L, int* FAF, int* res_faces, int* NFAF, int next_size) {
        int i = blockIdx.x;
        int j = threadIdx.x;
        int step = blockDim.x;
        if(j >= patch_num) return;
        for(int j=0;j<patch_num;j+=step){
            int face_header = 0;
            int k = 0;
            for(k=0;k<patch_num;k++){
                if(NFAF[i, j, k] != -1) {
                    if(j != k)
                        NFAF[i, j, face_header++] = k;
                }
            }
            for(;face_header<next_size;) {
                NFAF[i, j, face_header++] = -1;
            }
    }
    '''
    cpu_src=f'''
    @alias(FAF, in0)
    @alias(res_faces, in1)
    @alias(NFAF, out0)
    int N = FAF_shape0;
    int patch_num = {patch_num};
    int F = FAF_shape1;
    int L = FAF_shape2;
    int next_size = {next_size};
    get_hierar_FAF_gpu<<<1, 128>>>(patch_num, face_num, L, FAF_p, res_faces_p, NFAF_p, next_size);
    get_hierar_FAF_gpu<<<1, 128>>>(patch_num, face_num, L, FAF_p, res_faces_p, NFAF_p, next_size);
    '''
    NFAF = -jt.ones((FAF.shape[0], patch_num, patch_num), jt.int32)
    NFAF = jt.code([FAF, res_faces], [NFAF], cuda_header=cpu_header, cuda_src=cpu_src)[0]
    return NFAF[:,:,:next_size]

def unoverlapping_bfs_patch_cpu(adjdict, face_num, sf, debug=False):
    N = adjdict.shape[0]
    patch_num = sf.shape[1]
    res_face = -1 * jt.ones((N, face_num)).int32()
    res_face_counter = jt.ones((N, patch_num, 1)).int32()
    # print(adjdict.shape, sf.shape)
    res_face, res_face_counter = jt.code([adjdict, sf], [res_face, res_face_counter], cpu_header=f'''
        #include<vector>
        #include<map>
        #include<cstring>
        #include<iostream>
        using namespace std;
    ''', cpu_src=f'''
        @alias(res_face, out0)
        @alias(res_face_counter, out1)
        @alias(adjdict, in0)
        @alias(sf, in1)
        int N = adjdict_shape0;
        int face_num = adjdict_shape1;
        int L = adjdict_shape2;
        int patch_num = sf_shape1;
        // int left_face = face_num;
        #pragma omp parallel for num_threads(3)
        for(int batch_idx=0;batch_idx<N;batch_idx++) {{
            vector<int> res_patch[{patch_num}];
            int res_header[{patch_num}];
            int face_ids[2561];
            int left_face = face_num - patch_num;
            memset(face_ids, 0 , sizeof(face_ids));
            memset(res_header, 0, sizeof(res_header));
            for(int i=0;i<patch_num;i++) {{
                res_patch[i].clear();
                res_patch[i].push_back(@sf(batch_idx, i));
                @res_face(batch_idx, @sf(batch_idx, i)) = i;
            }}
            // cout << "DASDASDASD " << @res_face(batch_idx, 0) << endl;
            int iter = 0;
            int iter_thre = 300;
            while(left_face > 0) {{
                // cout << left_face << endl;
                int ori_left_face = left_face;
                // if(iter > iter_thre)
                //    break;
                for(int i=0;i<patch_num;i++) {{
                    bool found = false;
                    while(res_header[i] < res_patch[i].size() && !found) {{
                        // std::cout << i << " " << res_header[i] << " " << res_patch[i][res_header[i]] << std::endl;
                        int nowf = res_patch[i].at(res_header[i]);
                        for(int& j=face_ids[nowf];j<L;j++) {{
                            if(@adjdict(batch_idx, nowf, j) == -1) {{
                                face_ids[nowf] = L;
                                break;
                            }}
                            if(@res_face(batch_idx, @adjdict(batch_idx, nowf, j)) == -1) {{
                                found = true;
                                @res_face(batch_idx, @adjdict(batch_idx, nowf, j)) = i;
                                res_patch[i].push_back(@adjdict(batch_idx, nowf, j));
                                j++;
                                @res_face_counter(batch_idx, i, 0) += 1;
                                left_face--;
                                break;
                            }}
                        }}
                        if(!found) {{
                            res_header[i] ++;
                        }}
                    }}
                }}
                // iter++;     
                if(ori_left_face == left_face)
                    break;
            }}
            // cout << "DASDASDASD " << @res_face(batch_idx, 0) << endl;
        }}
    ''')
    return res_face, res_face_counter

def find_unagg_face(face_center, res_faces, init_faces):
    face_index = jt.where(res_faces == -1)[1]
    ig_face_center = face_center[face_index]
    if ig_face_center.shape[0] == 0:
        return res_faces
    cf = face_center[init_faces]
    distance = ((ig_face_center[...,None,:] - cf[None,...])**2).sum(-1)
    patch_idx = jt.argmin(distance, -1)[0]
    res_faces[0, face_index] = patch_idx
    assert (res_faces.numpy() == -1).sum() == 0
    return res_faces

def k_bfs_patch_cpu(adjdict, face_num, k, debug=False):
    N = adjdict.shape[0]
    k_faces = -jt.ones((N, adjdict.shape[1], k)).int32()
    k_faces = jt.code([adjdict, jt.array([k]).int().clone()], [k_faces], cpu_header=f'''
        #include<vector>
        #include<map>
        #include<cstring>
        using namespace std;
        int counter[10000], tail[10000], vis[10000];
    ''', cpu_src=f'''
        @alias(adjdict, in0)
        @alias(k_faces, out0)
        int face_num = adjdict_shape1;
        int next_size = adjdict_shape2;
        int k = @in1(0); 
        // #pragma omp parallel for num_threads(3)
        for(int idx=0;idx<{N};idx++) {{
            memset(counter, 0xff, sizeof(counter));
            memset(tail, 0 , sizeof(tail));
            for(int i=0;i<face_num;i++) {{
                memset(vis, 0, sizeof(vis));
                vis[i] = 1;
                while(tail[i] < k) {{ // k>=3
                    if(counter[i] == -1) {{
                        for(int j=0;j<next_size;j++) {{
                            if(@adjdict(idx, i, j) == -1) break; 
                            @k_faces(idx, i, tail[i]++) = @adjdict(idx, i, j);
                            vis[@adjdict(idx, i, j)] = 1;
                        }}
                        counter[i] ++;
                    }} else {{
                        {'cout << i << " " << counter[i] << " " << tail[i] << endl;' if debug else ""}
                        if(@k_faces(idx, i, counter[i]) == -1) break;
                        for(int j=0;j<next_size;j++) {{
                            {'cout << i << " " << j << " " << @k_faces(idx, i, counter[i]) << " " << counter[i] << " " << tail[i] << endl;' if debug else ""}
                            if(@adjdict(idx, @k_faces(idx, i, counter[i]), j) == -1) break; 
                            if(vis[@adjdict(idx, @k_faces(idx, i, counter[i]), j)] != 0) continue;
                            @k_faces(idx, i, tail[i]++) = @adjdict(idx, @k_faces(idx, i, counter[i]), j);
                            vis[@adjdict(idx, @k_faces(idx, i, counter[i]), j)] = 1;
                            if(tail[i] == k) break;
                        }}
                        counter[i] ++;
                    }}
                }}
            }}  
        }}
    ''')[0]
    return k_faces # already remove the face itself.

def get_jt_kset(adjdict, res_face, face_num, k):
    N = adjdict.shape[0]
    kset = []
    adjlist = []
    for i in range(N):
        nadj = get_hie_adjacent_dict(adjdict[i].data, res_face[i].data, face_num)
        kset.append(k_bfs_patch(nadj, face_num, k))
        adjlist.append(nadj)
        res_faces = np.zeros((face_num, k))
        for K, v in res_dict.items():
            assert len(v) == k + 1
            res_faces[K] = np.array(v[1:])
        kset[i] = res_faces
    return jt.stack(adjlist, 0), jt.array(kset)

def convert_face_matrix(face_num, fl):
    faf = -np.ones((face_num, 3)).astype("int32")
    counter = np.zeros(face_num, "int8")
    for f in fl:
        if counter[f[0]] < 3:
            faf[f[0], counter[f[0]]] = f[1]
            counter[f[0]] += 1
        if counter[f[1]] < 3:
            faf[f[1], counter[f[1]]] = f[0]
            counter[f[1]] += 1
    return faf

def compute_face_adjacency_faces(faces, unmanifold=False):
    F = faces.shape[0]
    if F % 2 != 0 or unmanifold:
        af = trimesh.graph.face_adjacency(faces)
        return jt.array(convert_face_matrix(faces.shape[0], af))
    FAF = -jt.ones_like(faces).int()
    E = jt.concat([
        faces[:F, [1, 2]],
        faces[:F, [2, 0]],
        faces[:F, [0, 1]],
    ], dim=0)
    E_hash = E.min(dim=1).astype('int64') * E.max() + E.max(dim=1)
    S, values = jt.argsort(E_hash)
    S = S.reshape(-1, 2)
    FAF[S[:, 0] % F, S[:, 0] // F] = S[:, 1] % F
    FAF[S[:, 1] % F, S[:, 1] // F] = S[:, 0] % F
    return FAF

def sample_faces_on_mesh(Fs, sample_num, mode="random", ms=None, centers=None, st=None, f2f=None):
    assert sample_num <= Fs, "sample num of faces must no bigger than the total face num of a mesh."
    face_idx = []
    if mode == "random":
        face_idx = np.random.choices(Fs, sample_num, replace=False)
    elif mode == "ffs": # support MeshGraph
        st_dis = np.zeros(Fs)
        sel = np.zeros(Fs)
        if st is None:
            pos = np.random.randint(0, Fs)
        else:
            pos = st
        if f2f is None:
            st_dis = np.sqrt(np.sum((centers - centers[pos])**2, -1))
            face_idx.append(pos)
            sel[pos] = 1
            while len(face_idx) < sample_num:
                next_p = np.argmax(st_dis)
                n_dis = np.sqrt(np.sum((centers - centers[next_p])**2, -1))
                st_dis = np.minimum(n_dis, st_dis)
                face_idx.append(next_p)
        else:
            weight_dis = np.sqrt(np.mean(np.sum((centers[...,None,:] - centers[None,...])**2, -1))) / 2
            st_dis = np.sqrt(np.sum((centers - centers[pos])**2, -1))  + f2f[pos] * weight_dis
            face_idx.append(pos)
            sel[pos] = 1
            while len(face_idx) < sample_num:
                next_p = np.argmax(st_dis)
                n_dis = np.sqrt(np.sum((centers - centers[next_p])**2, -1)) + f2f[next_p] * weight_dis
                st_dis = np.minimum(n_dis, st_dis)
                face_idx.append(next_p)
    elif mode == "voronoi" or mode == "montecarlo" or mode == "poisson": # use pymeshlab and trimesh for sampling
        assert ms is not None, "Please input PyMeshLab's MeshSet for processing."
        if mode == "poisson":
            ms.poisson_disk_sampling(samplenum=patch_num+10, exactnumflag=True)
        elif mode == "montecarlo":
            ms.generate_sampling_montecarlo(samplenum=patch_num+10)
        else:
            ms.generate_sampling_voronoi(samplenum=patch_num+10,preprocessflag=True,refinefactor=2)
        point_mesh = ms.current_mesh()
        ms.set_current_mesh(0)
        origin_mesh = ms.current_mesh()
        Vs = origin_mesh.vertex_matrix()
        F = origin_mesh.face_matrix()
        mesh = Trimesh(vertices=Vs, faces=F)
        tris = [(idx, [Vs[face[0]], Vs[face[1]], Vs[face[2]]]) for idx, face in enumerate(F)]
        sample_vertex = point_mesh.vertex_matrix()
        _, _, init_faces = closest_point(mesh, sample_vertex)
        face_idx = set(list(init_faces)[:patch_num])
        while len(face_idx) < patch_num:
            face_idx.add(np.random.randint(0,F.shape[0]))
    return face_idx

def get_patches_ffs(centers, face_num, patch_num, st=None, Fs=None):
    N = centers.shape[0]
    if st is None:
        pos = jt.randint(0, face_num, (N,))
    else:
        pos = jt.array([st]).expand(N)
    # print(centers.shape)
    center_pos = centers.reindex((N, 1, 3), ("i0", "@e0(i0)", "i2"), extras=[pos])
    st_dis = jt.sqrt(jt.sum((centers - center_pos)**2, -1))
    # print(st_dis.shape)
    face_idx = []
    face_idx.append(pos)
    while len(face_idx) < patch_num:
        next_p, _ = jt.argmax(st_dis, -1)
        center_pos = centers.reindex((N, 1, 3), ("i0", "@e0(i0)", "i2"), extras=[next_p])
        n_dis = jt.sqrt(jt.sum((centers - center_pos)**2, -1))
        st_dis = jt.minimum(n_dis, st_dis)
        face_idx.append(next_p)
    return jt.stack(face_idx, -1)

def get_adj_size(args, stage_idx):
    if args.transformer_type == "vit":
        return args.knn
    if args.transformer_type == "pvt":
        return args.sr_ratio[stage_idx]
    if args.transformer_type == "swin":
        return args.window_size
    return int(args.patch_num[0] // 2)

def get_p2f(res_faces, patch_num, args):
    fs = res_faces.shape[0]
    ex_size = int(fs / patch_num * 1.5)
    p2f = np.ones((patch_num, ex_size)).astype("int32") * -1
    ct = np.zeros((patch_num)).astype("int32")
    for f in range(fs):
        p = res_faces[f]
        if ct[p] < ex_size - 1:
            p2f[p, ct[p]] = f
            ct[p] += 1
    return p2f

def process_bfs(faces, Fs, V, face_center, args, npz_data=None, mesh=None, init_faces=None):
    sample_num = patch_num = args.patch_num[0]
    Fs = faces.shape[0]
    extra = {}
    # extra["init_faces"] = init_faces
    if npz_data is None:
        if args.use_k_neig == 0:
            adjdict = compute_face_adjacency_faces(jt.array(faces), True)
            if args.knn > 3:
                res_dict = k_bfs_patch_cpu(adjdict.unsqueeze(0), Fs, args.knn)
                extra["k_faces"] = res_dict[0].numpy()
            else:
                extra["k_faces"] = adjdict.numpy()
            adjdict.sync()
        else:
            fc = jt.array(face_center)
            if face_center.shape[0] < args.knn + 1:
                adjdict = np.ones((face_center.shape[0], 3)).astype("int32") * -1
                for i in range(face_center.shape[0]):
                    count = 0
                    for j in range(face_center.shape[0]):
                        if i == j: continue
                        adjdict[i, count] = j
                        count += 1
            else:
                adjdict = knn_indices_func_cpu(fc, fc, args.knn, 1)
            extra["k_faces"] = adjdict
            adjdict = jt.array(adjdict[:,:3]) # still use 3 for bfs search.
    else:
        adjdict = jt.array(npz_data["neighbors"])
    # adjdict2 = general_get_adjacent_dict(faces)
    # adjdict = np.zeros((Fs, 3)).astype("int32")
    # for k, v in adjdict2.items():
    #     adjdict[k] = np.array(v)
    # adjdict = jt.array(adjdict)
    # adjdict = adjdict.unsqueeze(0)
    f2f = None
    # if init_faces is None:
    #     f2f = get_f2f_dis(mesh)
    if Fs <= args.patch_num[0]:
        init_faces = jt.arange(0, Fs, 1).int32()
        res_faces = jt.arange(0, Fs, 1).int32().unsqueeze(0)
        # print(Fs, init_faces, res_faces)
        res_face_counter = jt.concat([jt.ones(Fs), jt.zeros(args.patch_num[0]-Fs)], 0).int32().unsqueeze(0)
    else:
        if init_faces is None:
            st = None
            if args.sample_method == "montecarlo" or args.sample_method == "voronoi" or args.sample_method == "poisson":
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(V, faces))
                init_faces = sample_faces_on_mesh(Fs, sample_num, args.sample_method, ms, centers=None, st=None)
            elif args.sample_method == "ffs":
                init_faces = sample_faces_on_mesh(Fs, sample_num, args.sample_method, centers=face_center, st=None, f2f=f2f)
            elif args.sample_method == "random":
                init_faces = sample_faces_on_mesh(Fs, sample_num, args.sample_method)
            extra['init_faces'] = init_faces
        res_faces, res_face_counter = unoverlapping_bfs_patch_cpu(adjdict.unsqueeze(0), Fs, jt.array(init_faces).unsqueeze(0))
        res_faces.sync()
        # res_faces = find_unagg_face(face_center, res_faces, init_faces)
        # res_faces.sync()  
    # save_patches(V, faces, res_faces.numpy()[0], patch_num, "test_ffs.ply")
    extra["res_faces"] = res_faces[0].numpy()
    centers = face_center[init_faces]
    extra["res_face_counter"] = res_face_counter[0].numpy()
    centers = face_center[init_faces]
    face_center = jt.array(face_center)
    centers = face_center.reindex_reduce("add", [sample_num, 3], ['@e0(0, i0)', 'i1'], extras=[res_faces])
    centers = (centers / res_face_counter[0]).numpy()
    extra["centers"] = centers
    extra["p2f"] = get_p2f(extra["res_faces"], sample_num, args)
    if args.euc == 0:
        next_size = get_adj_size(args, 0) 
        if len(args.patch_num) > 1:
            next_size = min(next_size * 3, args.patch_num[1])
        res_faces.sync()
        extra["NFAF"] = get_hierar_FAF_cpu(patch_num, adjdict.unsqueeze(0), res_faces, next_size)[0]
        extra['NFAF'].sync()
        extra['NFAF'] = extra['NFAF'].numpy()
    else:
        # knn_indices_func_cpu(extra["centers"], face_center, 3, 1)
        extra["NFAF"] = knn_indices_func_cpu(extra["centers"], extra["centers"], 3, 1)
    # for test.
    # extra["adjdict"] = adjdict.numpy()
    # res_mat = k_bfs_patch_cpu(extra["NFAF"].unsqueeze(0), patch_num, args.knn)
    # extra["k_faces_2"] = res_mat[0]
    # init_faces = get_patches_ffs(jt.array(extra["centers"]).unsqueeze(0), patch_num, args.patch_num[1], 1)
    # # for i in range(N):
    # #     assert patch_num == len(list(init_faces[0].numpy()))
    # res_mat, res_mat_counter = unoverlapping_bfs_patch_cpu(extra["NFAF"].unsqueeze(0), patch_num, init_faces)
    # extra["res_faces_2"] = res_mat
    # print(extra["res_faces"][:10])
    # exit(0)
    return extra

def patch_aggregation(op, x, res_faces, patch_num, res_mat_counter=None):
    N = x.shape[0]
    if op == "add" or op == "mean" or op == "max":
        x = x.reindex_reduce(op=op,
            shape=[N, patch_num, x.shape[2]], # (N, F, C)
            indexes=[
                'i0',
                '@e0(i0,i1)',
                'i2'
            ],
            extras=[res_faces],
        )
        if op == "mean":
            x /= res_mat_counter
    elif op == "select":
        x = x.reindex(
            shape=[N, patch_num, x.shape[2]], # (N, F, C)
            indexes=[
                'i0',
                '@e0(i0,i1)',
                'i2'
            ],
            extras=[res_faces],
        )
    return x

def topk(input, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = -1
    if dim<0:
        dim+=input.ndim

    transpose_dims = [i for i in range(input.ndim)]
    transpose_dims[0] = dim
    transpose_dims[dim] = 0
    input = input.transpose(transpose_dims)
    index,values = jt.argsort(input,dim=0,descending=largest)
    indices = index[:k]
    values = values[:k]
    indices = indices.transpose(transpose_dims)
    values = values.transpose(transpose_dims)
    return [values,indices]

def knn_indices_func_cpu(rep_pts,  # (pts, dim)
                         pts,      # (x, dim)
                         K : int,
                         D : int):
    """
    CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    rep_pts = rep_pts.data
    pts = pts.data
    region_idx = []

    nbrs = NearestNeighbors(n_neighbors=D*K + 1, algorithm = "ball_tree").fit(pts)
    indices = nbrs.kneighbors(rep_pts)[1]

    return indices[:, 1::D]

def knn_indices_func_gpu(rep_pts,  # (N, pts, dim)
                         pts,      # (N, x, dim)
                         k : int, d : int ): # (N, pts, K)
    region_idx = []
    batch_size = rep_pts.shape[0]
    # print(rep_pts.shape)
    # print(pts.shape)
    for idx in range (batch_size):
        qry = rep_pts[idx]
        ref = pts[idx]
        n, d = ref.shape
        m, d = qry.shape
        mref = ref.view(1, n, d).repeat(m, 1, 1)
        mqry = qry.view(m, 1, d).repeat(1, n, 1)
        
        dist2 = jt.sum((mqry - mref)**2, 2) # pytorch has squeeze 
        _, inds = topk(dist2, k*d + 1, dim = 1, largest = False)
        
        region_idx.append(inds[:,1::d])

    region_idx = jt.stack(region_idx, dim = 0) 
    # print(region_idx.shape)
    return region_idx

def get_roll_feature(x, k_faces):
    B, F, C = x.shape
    cuda_header ='''
    __global__ void get_roll_feature(const float* x, const int* k_faces, float* output,
    int batch_size, int f, int k, int c) {
        int bidx = blockIdx.x;
        int fidx = threadIdx.x;
        if(bidx >= batch_size || threadIdx.x > f) return;
        int idx = bidx * f * k + fidx * k;
        int cidx = bidx * f * c + fidx * c;
        bool cont = false; 
        for(int i=0;cont;i++){
            int k2;
            if(i==k-1||k2 == -1) {
                if(i==0) return;
                cont = true;
                k2 = k_faces[idx];;
            } else k2 = k_faces[idx+i+1];
            int k1 = k_faces[idx + i]; 
            for(int j=0;j<c;j++)
                atomicAdd(&output[cidx + j], abs(x[bidx*f*c+k1*c+j]-x[bidx*f*c+k2*c+j]));
        }
    }
    '''
    cuda_src = '''
    @alias(x, in0)
    @alias(k_faces, in1)
    @alias(output, out0)
    int b = x_shape0, f = x_shape1, c=x_shape2, k = k_faces_shape1;
    get_roll_feature<<<32, 32>>>(x_p, k_faces_p, output_p, b, f, k, c);
    '''
    return jt.code((B, F, C, 1), jt.float32, [x, k_faces], cuda_header=cuda_header, cuda_src=cuda_src).permute(0,2,1,3)

if __name__ == "__main__":
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("/home/penghy/learn/mesh/shrec10/alien/train/T5.obj")
    # print(ms.current_mesh().face_matrix())
    adjdict = oget_adjacent_dict(ms.current_mesh().face_matrix())
    adjdict2 = get_adjacent_dict(ms.current_mesh().face_matrix())
    init_faces = list(set([283, 41, 349, 273, 322, 295, 432, 43, 43, 198, 383, 467, 363, 362, 141, 450, 313, 429, 183, 52, 382, 23, 477, 56, 377, 205, 370, 138, 447, 373, 328, 301, 137, 170, 204, 396, 0, 376, 324, 311, 98, 109, 374, 202, 384, 110, 158, 243, 221, 452, 445, 475, 338, 340, 81, 226, 439, 29, 187, 283, 448, 466, 430, 446, 365, 498, 34, 249, 230, 491, 51, 88, 241, 4, 268, 204, 80, 228, 145, 317, 261, 30, 305, 395, 143, 185, 171, 345, 274, 299, 409, 372, 74, 315, 280, 181, 235, 353, 355, 82, 29, 8, 437, 234, 465, 175, 202, 195, 78, 278, 271, 97, 106, 70, 182, 95, 55, 252, 100, 470, 360, 339, 498, 17, 423, 253, 434, 39, 366, 316, 176, 487, 347, 361, 282, 369, 381, 430, 376, 308, 464, 216, 457, 433, 286, 400, 217, 83, 13, 259, 248, 498, 58, 259, 71, 392, 237, 351, 344, 365, 341, 245, 127, 101, 452, 53, 425, 172, 399, 28, 38, 242, 472, 403, 371, 219, 379, 115, 260, 404, 332, 116, 399, 251, 15, 443, 290, 44, 27, 380, 168, 131, 245, 394, 164, 38, 179, 312, 441, 329, 189, 246, 155, 298, 203, 478, 269, 63, 485, 143, 186, 89, 301, 156, 46, 87, 256, 113, 297, 255, 375, 285, 144, 28, 397, 150, 488, 288, 39, 471, 456, 207, 153, 117, 13, 456, 184, 276, 206, 476, 2, 197, 161, 300, 274, 36, 310, 194, 450, 263, 172, 124, 90, 6, 150]))
    res_face = bfs_patch(adjdict, ms.current_mesh().face_matrix().shape[0], init_faces)
    res_face2 = bfs_patch_c(jt.array(adjdict2), ms.current_mesh().face_matrix().shape[0], jt.array(init_faces))
    # # check c version
    res_mat = np.zeros((ms.current_mesh().face_matrix().shape[0])).astype("int32")
    for idx, res_face in res_face.items():
        for face in res_face:
            res_mat[face] = idx   
    res_mat = jt.array(res_mat)
    adj = oget_hie_adjacent_dict(adjdict, res_mat, len(init_faces))
    adj2 = get_hie_adjacent_dict(adjdict2, res_face2.data, len(init_faces))
    st = time.time()
    for i in range(200):
        adj = oget_hie_adjacent_dict(adjdict, res_mat, len(init_faces))
    print(f"python time is {time.time() - st}")
    st = time.time()
    for i in range(200):
        adj2 = get_hie_adjacent_dict(adjdict2, res_face2.data, len(init_faces))
    print(f"c++ time is {time.time() - st}")
    # # print(init_faces)

    # for key, val in adjdict.items():
    #     for i in range(len(val)):
    #         if val[i] != adjdict2[key][i]:
    #             print("AAAA", val, res_faces2[key])
    #             break
    # np.set_printoptions(threshold=np.inf)
    # # print(res_faces2)

    # rr = {}
    # print(res_faces)
    # for idx in range(res_faces.shape[0]):
    #     ii = int(res_faces[idx].data)
    #     if ii not in rr:
    #         rr[ii] = []
    #     rr[ii].append(idx)
    # for key, val in rr.items():
    #     print(val)
    #     print(res_faces2[key])
    #     assert len(val) == len(res_faces2[key]), str(key)
    #     for i in range(len(val)):
    #         if val[i] != res_faces2[key][i]:
    #             print("AAAA", key, val, res_faces2[key])
    #             break
