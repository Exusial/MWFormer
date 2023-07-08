import os
import trimesh
import numpy as np
from trimesh.proximity import closest_point
import jittor as jt
from plyfile import PlyData

palette=np.array([
    (200, 200, 200),        # other
    (152, 223, 138),		# floor
    (174, 199, 232),		# wall
    (31, 119, 180), 		# cabinet
    (255, 187, 120),		# bed
    (188, 189, 34), 		# chair
    (140, 86, 75),  		# sofa
    (255, 152, 150),		# table
    (214, 39, 40),  		# door
    (197, 176, 213),		# window
    (148, 103, 189),		# bookshelf
    (196, 156, 148),		# picture
    (23, 190, 207), 		# counter
    (247, 182, 210),		# desk
    (219, 219, 141),		# curtain
    (255, 127, 14), 		# refrigerator
    (227, 119, 194),		# bathtub
    (158, 218, 229),		# shower curtain
    (44, 160, 44),  		# toilet
    (112, 128, 144),		# sink
    (82, 84, 163),          # otherfurn
])

def visual_scene(tname, preds, gt, acc):
    mesh = trimesh.load("/data/penghy/scannet/scans/" + tname +  f'/{tname}_vh_clean_2.ply', process=False)
    mesh.visual.vertex_colors[:, :3] = palette[preds.astype("int32")]
    mesh.export("/data/penghy/processed_scannet/preds/" + f'{tname}.ply')
    mesh.visual.vertex_colors[:, :3] = palette[gt.astype("int32")]
    mesh.export("/data/penghy/processed_scannet/preds/" + f'gt_{tname}.ply')

SCANNET_CLASS_REMAP = np.zeros(41).astype("int32")
SCANNET_CLASS_REMAP[1] = 1
SCANNET_CLASS_REMAP[2] = 2
SCANNET_CLASS_REMAP[3] = 3
SCANNET_CLASS_REMAP[4] = 4
SCANNET_CLASS_REMAP[5] = 5
SCANNET_CLASS_REMAP[6] = 6
SCANNET_CLASS_REMAP[7] = 7
SCANNET_CLASS_REMAP[8] = 8
SCANNET_CLASS_REMAP[9] = 9
SCANNET_CLASS_REMAP[10] = 10
SCANNET_CLASS_REMAP[11] = 11
SCANNET_CLASS_REMAP[12] = 12
SCANNET_CLASS_REMAP[13] = 14
SCANNET_CLASS_REMAP[14] = 16
SCANNET_CLASS_REMAP[15] = 24
SCANNET_CLASS_REMAP[16] = 28
SCANNET_CLASS_REMAP[17] = 33
SCANNET_CLASS_REMAP[18] = 34
SCANNET_CLASS_REMAP[19] = 36
SCANNET_CLASS_REMAP[20] = 39
SCANNET_CLASS_REMAP[0] = 40
acc = 0
def map_tname(tname, mode):
    if mode == "test":
        raw_mpath = os.path.join("/data/penghy/scannet/scans_test/", tname.replace(".obj", ""), tname.replace(".obj", "") + "_vh_clean_2.ply")
        # raw_mpath = os.path.join("/data/penghy/processed_scannet/test/", tname)
        simp_path = os.path.join("/data/penghy/processed_scannet/simp_test", tname)
        output_path = os.path.join("/data/penghy/processed_scannet/testmap", tname.replace(".obj", ".pkl"))
    else:
        # raw_mpath = os.path.join("/data/penghy/scannet/scans/", tname.replace(".obj", ""), tname.replace(".obj", "") + "_vh_clean_2.ply")
        raw_mpath = os.path.join("/data/penghy/processed_scannet/val", tname)
        simp_path = os.path.join(f"/data/penghy/processed_scannet/simp_{mode}", tname)
        output_path = os.path.join(f"/data/penghy/processed_scannet/valmap", tname.replace(".obj", ".pkl"))
    # origin_mesh = trimesh.load(raw_mpath)
    simp_mesh = trimesh.load(simp_path)
    raw_mesh = PlyData.read(raw_mpath)
    # vv = origin_mesh.vertices[origin_mesh.faces.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vv = np.stack([raw_mesh["vertex"]["x"], raw_mesh["vertex"]["y"], raw_mesh["vertex"]["z"]], -1)
    _, _, fl = closest_point(simp_mesh, vv)

    assert fl.shape[0] == vv.shape[0]
    jt.save(fl, output_path)

def check_vertices(tname):
    origin_mpath = os.path.join("/data/penghy/processed_scannet/test", tname)
    raw_mpath = os.path.join("/data/penghy/scannet/scans_test/", tname.replace(".obj", ""), tname.replace(".obj", "") + "_vh_clean_2.ply")
    origin_mesh = trimesh.load(origin_mpath, process=False)
    raw_mesh = PlyData.read(raw_mpath)
    print(raw_mpath)
    assert origin_mesh.vertices.shape[0] == len(raw_mesh['vertex']), str(origin_mesh.vertices.shape[0]) + " " + str(len(raw_mesh['vertex']))

def map_for_origin_label(tname, model, mode):
    global acc
    if mode == "test":
        prefix = ""
    else:
        prefix = mode + "_"
        
    # origin_mpath = os.path.join("/data/penghy/processed_scannet/test", tname)
    origin_path = os.path.join(f"/data/penghy/processed_scannet/test_preds_{model}", tname.replace(".obj", ".pkl"))
    # map_path = os.path.join(f"/data/penghy/processed_scannet/testmap", tname.replace(".obj", ".pkl"))
    map_path = os.path.join(f"/data/penghy/processed_scannet/map_val", tname.replace(".obj", ".pkl"))
    # pro_mesh_path = os.path.join(f"/data/penghy/processed_scannet/{prefix[:-1]}", tname.replace(".obj", ".pkl"))
    # origin_mesh = trimesh.load(origin_mpath, process=False)
    m = jt.load(map_path)
    # om = jt.load(pro_mesh_path)
    labels = jt.load(origin_path)
    origin_label = labels[m]
    need_draw = 1
    if mode == "val" and need_draw:
        raw_mpath = os.path.join("/data/penghy/scannet/", "val_gt_txt", tname.replace(".obj", ".txt"))
        raw_label = np.loadtxt(raw_mpath).astype("int32")
        print(raw_label.shape, origin_label.shape)
        SCANNET_CLASS_REMAP = np.zeros(41)
        SCANNET_CLASS_REMAP[1] = 1
        SCANNET_CLASS_REMAP[2] = 2
        SCANNET_CLASS_REMAP[3] = 3
        SCANNET_CLASS_REMAP[4] = 4
        SCANNET_CLASS_REMAP[5] = 5
        SCANNET_CLASS_REMAP[6] = 6
        SCANNET_CLASS_REMAP[7] = 7
        SCANNET_CLASS_REMAP[8] = 8
        SCANNET_CLASS_REMAP[9] = 9
        SCANNET_CLASS_REMAP[10] = 10
        SCANNET_CLASS_REMAP[11] = 11
        SCANNET_CLASS_REMAP[12] = 12
        SCANNET_CLASS_REMAP[14] = 13
        SCANNET_CLASS_REMAP[16] = 14
        SCANNET_CLASS_REMAP[24] = 15
        SCANNET_CLASS_REMAP[28] = 16
        SCANNET_CLASS_REMAP[33] = 17
        SCANNET_CLASS_REMAP[34] = 18
        SCANNET_CLASS_REMAP[36] = 19
        SCANNET_CLASS_REMAP[39] = 20
        raw_label = SCANNET_CLASS_REMAP[raw_label]
        index = raw_label != 0
        findex = raw_label == 0
        acc = (origin_label[index] == raw_label[index]).sum() / index.sum()
        if acc > 0.9:
            origin_label[findex] = 0
            visual_scene(tname.replace(".obj", ""), origin_label, raw_label, acc)
    else:
        vertices_label = SCANNET_CLASS_REMAP[origin_label]
        # acc += (origin_label == om[0]).sum() / om[0].shape[0]
        # print((origin_label == om[0]).sum() / om[0].shape[0])
        # assert m.shape[0] == origin_mesh.shape[0]
        # origin_mesh.visual.face_colors[:, :3] = palette[origin_label]
        # origin_mesh.export("/data/penghy/processed_scannet/corr/" + f'corr-{tname}.ply')
        # vertices_map = np.zeros((origin_mesh.vertices.shape[0], 21))
        # for idx, f in enumerate(origin_mesh.faces):
        #     for fi in f:
        #         vertices_map[fi][origin_label[idx]] += 1
        with open(os.path.join(f"/data/penghy/processed_scannet/{prefix}origin_{model}", tname.replace(".obj", ".txt")), "w", encoding='utf-8') as f:
            for i in range(vertices_label.shape[0]):
                suffix = "\n" if i != vertices_label.shape[0] - 1 else ""
                f.write(str(int(vertices_label[i])) + suffix)
            f.close()
        print(vertices_label.shape[0])
        # np.savetxt(os.path.join(f"/data/penghy/processed_scannet/origin_{model}", tname.replace(".obj", ".txt")), vertices_label, delimiter="\n", fmt="%d")

def generate_txt_gt(tname):
    raw_mpath = os.path.join("/data/penghy/scannet/scans/", tname.replace(".obj", ""), tname.replace(".obj", "") + "_vh_clean_2.labels.ply")
    vertex_labels = np.asarray(PlyData.read(raw_mpath)['vertex']['label'])
    print(vertex_labels.shape)
    with open(os.path.join(f"/data/penghy/scannet/val_gt_txt", tname.replace(".obj", ".txt")), "w", encoding='utf-8') as f:
        for i in range(vertex_labels.shape[0]):
            suffix = "\n" if i != vertex_labels.shape[0] - 1 else ""
            f.write(str(int(vertex_labels[i])) + suffix)
        f.close()

if __name__ == "__main__":
    mode = "val"
    for tname in os.listdir(f"/data/penghy/processed_scannet/simp_{mode}"):
        if tname.endswith(".obj"):
            # map_tname(tname, mode)
            map_for_origin_label(tname, "vit", mode)
            # map_for_origin_label(tname, "swin", mode)
            # map_for_origin_label(tname, "pvt", mode)
            # check_vertices(tname)
            # generate_txt_gt(tname)
    # print("acc: ", acc / len(os.listdir(f"/data/penghy/processed_scannet/simp_{mode}")) * 2)