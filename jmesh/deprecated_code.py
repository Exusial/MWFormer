def get_adjdict_from_hash(F, S, value):
#     adjdict = -jt.ones((F, 3)).int32()
#     cpu_header = f'''
#     #include<iostream>
#     #include<cstring>
#     using namespace std;
#     int face_header[{F}];
#     '''
#     cpu_src = '''
#     @alias(S, in0)
#     @alias(value, in1)
#     @alias(FAF, out0)
#     int idx = 0;
#     int F = FAF_shape0;
#     int L = FAF_shape1;
#     int l = S_shape0;
#     while(idx < l - 1) {{
#         if(@value(idx) == @value(idx+1)) {{
#             if(face_header[@S(idx) % F] < L)
#                 @FAF[@S(idx) % F, face_header[@S(idx) % F]++] = @S(idx + 1) % F;
#             if(face_header[@S(idx+1) % F] < L)
#                 @FAF[@S(idx + 1) % F, face_header[@S(idx+1) % F]++] = @S(idx) % F;
#             idx += 2;
#         }}
#         else 
#             idx += 1;
#     }}
#     '''
#     return jt.code([S, value], [F], cpu_header=cpu_header, cpu_src=cpu_src)[0]

def get_adjacent_dict(F):
    edgedict = {}
    adjdict = {}
    for idx, face in enumerate(F):
        for i in range(3):
            half_edge = ((min(face[i], face[(i+1) % 3]), max(face[i], face[(i+1)%3])))
            if half_edge in edgedict:
                edgedict[half_edge].append(idx)
            else:
                edgedict[half_edge] = [idx]
    for edge, faces in edgedict.items():
        for i in range(len(faces)):
            adjdict.setdefault(faces[i], [])
            for j in range(len(faces)):
                if i == j: continue
                adjdict[faces[i]].append(faces[j])
    return adjdict

def get_hie_adjacent_dict(oridict, res_faces, patch_num):
    adjdict = {}
    vis_mat = np.eye(patch_num)
    # print(res_faces)
    for face in oridict.keys():
        adjdict.setdefault(int(res_faces[face].data), [])
        for adjface in oridict[face]:
            if vis_mat[int(res_faces[face].data), int(res_faces[adjface].data)] != 1:
                vis_mat[int(res_faces[face].data), int(res_faces[adjface].data)] = 1
                adjdict[int(res_faces[face].data)].append(int(res_faces[adjface].data))
    return adjdict

def bfs_patch(adjdict, face_num, sf):
    patch_num = len(sf)
    patch_faces = {}
    res_faces = {}
    res_head = []
    for idx, face in enumerate(sf):
        patch_faces[face] = idx
        res_faces[idx] = [face] # bfs queue and result
        res_head.append(0)
    patch_size = len(sf)
    iter = 0
    while patch_size < face_num:
        # print(patch_size, face_num)
        if iter > 500: 
            print("warn: exceed iteration limits!")
            return res_faces
        try: # for abnormal mesh
            for i in range(patch_num):
                found = False
                # print(i, len(res_faces[i]), res_head[i])
                while res_head[i] < len(res_faces[i]) and not found:
                    cur_face = res_faces[i][res_head[i]]
                    for adjface in adjdict[cur_face]:
                        if not adjface in patch_faces:
                            found = True
                            patch_faces[adjface] = i
                            res_faces[i].append(adjface)
                            patch_size += 1
                            break
                    if not found:
                        res_head[i] += 1
        except KeyError:
            print("warn: KeyError occurred!")
            return res_faces
        iter += 1
    return res_faces

# def get_hie_adjacent_dict(oridict, res_faces, patch_num):
#     adjdict = -1 * np.ones((patch_num, 10)).astype("int32")
#     vis_mat = np.eye(patch_num)
#     counter = np.zeros((patch_num))
#     for face in range(res_faces.shape[0]):
#         for adjface in oridict[face]:
#             if counter[int(res_faces[face])] >= 10:
#                 continue
#             if adjface != -1 and vis_mat[int(res_faces[face]), int(res_faces[adjface])] != 1:
#                 vis_mat[int(res_faces[face]), int(res_faces[adjface])] = 1
#                 adjdict[int(res_faces[face])][int(counter[int(res_faces[face])])] = int(res_faces[adjface])
#                 counter[int(res_faces[face])] += 1
#     return jt.array(adjdict)

def oget_hie_adjacent_dict(oridict, res_faces, patch_num):
    adjdict = {}
    vis_mat = np.eye(patch_num)
    # print(res_faces)
    for face in oridict.keys():
        adjdict.setdefault(int(res_faces[face].data), [])
        for adjface in oridict[face]:
            if vis_mat[int(res_faces[face].data), int(res_faces[adjface].data)] != 1:
                vis_mat[int(res_faces[face].data), int(res_faces[adjface].data)] = 1
                adjdict[int(res_faces[face].data)].append(int(res_faces[adjface].data))
    return adjdict

def general_get_adjacent_dict(F):
    edgedict = {}
    Fs = F.shape[0]
    adjdict = -np.ones([Fs, 3]).astype(np.int32)
    face_counter = np.zeros([Fs]).astype(np.int32)
    for idx, face in enumerate(F):
        for i in range(3):
            half_edge = ((min(face[i], face[(i+1) % 3]), max(face[i], face[(i+1)%3])))
            if half_edge in edgedict:
                edgedict[half_edge].append(idx)
            else:
                edgedict[half_edge] = [idx]
    for edge, faces in edgedict.items():
        for i in range(len(faces)):
            for j in range(len(faces)):
                if face_counter[faces[i]] == 3:break
                if i == j: continue
                adjdict[faces[i], face_counter[faces[i]]] = faces[j]
                face_counter[faces[i]] += 1
    return adjdict

def process_bfs_ori(faces, Fs, V, face_center, args):
    extra = {}
    k = args.knn
    patch_num = args.patch_num[0]
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(V, faces))
    adjdict = get_adjacent_dict(ms.current_mesh().face_matrix())
    init_faces = get_patches_ffs_ori(face_center, Fs, patch_num, 0)
    extra["init_faces"] = init_faces
    # init_faces = get_patches(ms, patch_num)
    res_faces = bfs_patch(adjdict, ms.current_mesh().face_matrix().shape[0], init_faces)
    res_mat = np.zeros((faces.shape[0])).astype("int32")
    for idx, res_face in res_faces.items():
        for face in res_face:
            res_mat[face] = idx        
    extra["res_faces"] = res_mat
    extra["NFAF"] = adjdict
    res_dict = k_bfs_patch(adjdict, ms.current_mesh().face_matrix().shape[0], k)
    res_faces = np.zeros((ms.current_mesh().face_matrix().shape[0], k))
    for K, v in res_dict.items():
        assert len(v) == k + 1
        res_faces[K] = np.array(v[1:])
    extra["k_faces"] = res_faces
    extra["centers"] = face_center[init_faces]
    # for further test
    # N = 1 
    # res_mat = np.zeros((N, patch_num, args.knn))
    # adjlist = []
    # for i in range(N):
    #     adjdict = get_hie_adjacent_dict(adjdict, jt.array(extra["res_faces"]), patch_num)
    #     for ii in range(patch_num):
    #         adjdict[ii] = np.sort(adjdict[ii])
    #     extra["NFAF2"] = adjdict
    #     adjlist.append(adjdict)
    #     res_dict = k_bfs_patch(adjdict, patch_num, args.knn)
    #     for idx, res_face in res_dict.items():
    #             res_mat[i, idx] = np.array(res_face[1:])  
    # extra["k_faces_2"] = res_mat.squeeze(0).astype("int32")
    # res_mat = np.zeros((N, patch_num))
    # # adjlist = []
    # centers = []
    # for i in range(N):
    #     init_faces = get_patches_ffs_ori(extra["centers"], patch_num, args.patch_num[1], 1)
    #     # print(init_faces)
    #     new_res_faces = bfs_patch(adjlist[i], patch_num, init_faces)
    #     for idx, res_face in new_res_faces.items():
    #         for face in res_face:
    #             res_mat[i, face] = idx 
    # extra["res_faces_2"] = res_mat.squeeze(0).astype("int32")
    return extra
