import os
import argparse
import numpy as np
import trimesh
import jittor as jt

palette= np.array([
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
    (200, 200, 200),              # other
])

patch_color = np.array([
    (128,0,0),		# floor
    (220,20,60),		# wall
    (255,0,0), 		# cabinet
    (255,127,80),		# bed
    (255,160,122), 		# chair
    (255,215,0),  		# sofa
    (238,232,170),		# table
    (189,183,107),  		# door
    (255,255,0),		# window
    (154,205,50),		# bookshelf
    (127,255,0),		# picture
    (0,100,0), 		# counter
    (144,238,144),		# desk
    (32,178,170),		# curtain
    (0,255,255), 		# refrigerator
    (175,238,238),		# bathtub
    (70,130,180),		# shower curtain
    (135,206,235),  		# toilet
    (0,0,139),		# sink
    (65,105,225),          # otherfurn
    (138,43,226),              # other
    (123,104,238),
    (216,191,216),
    (238,130,238),
    (245,245,220),
    (210,105,30),
    (255,240,245),
    (105,105,105),
    (128,128,0),
    (0,0,128),
    (255,20,147),
    (188,143,143)
])

def visual_segmentation(mesh_path, preds, out_path):
    mesh = trimesh.load(mesh_path, process=False)
    face_color = np.zeros((mesh.faces.shape[0], 4))
    for face in range(mesh.faces.shape[0]):
        face_color[face, :3] = palette[int(preds[0, face])]
        face_color[face, 3] = 100
    mesh.visual.face_colors = face_color.astype("int32")
    mesh.export(out_path)

def save_seg_patches(mesh_path, res_faces, out_path):
    global patch_color
    if not out_path.endswith(".ply"):
        out_path += ".ply"
    mesh = trimesh.load(mesh_path, process=False)
    F = mesh.faces
    face_res_face = np.arange(F.shape[0]).astype(np.int32)
    face_color = np.zeros((F.shape[0], 4))
    for fmap in res_faces[1:]:
        for i in range(F.shape[0]):
            face_res_face[i] = fmap[0,face_res_face[i]]
    for face in range(F.shape[0]):
        face_color[face, :3] = patch_color[int(face_res_face[face])]
        face_color[face, 3] = 100
    mesh.visual.face_colors = face_color.astype("int32")
    mesh.export(out_path)

def visualize(args):
    print("visualizing...")
    data_path = "/data/penghy/processed_scannet/simp_val"
    scene_path = os.path.join(data_path,  args.scene_id + ".obj")
    label_path = os.path.join(data_path,  args.scene_id + ".pkl")
    vertex = []
    mesh = trimesh.load(scene_path)
    label = jt.load(label_path)
    # for i in range(preds.shape[0]):
    #     mesh_path = mesh_infos['mesh_paths'][i]
    #     mesh_name = Path(mesh_path).stem
    mesh.visual.face_colors[:, :3] = palette[label.astype("int32")]
    mesh.export("/data/penghy/processed_scannet/corr/" + f'corr-{args.scene_id}.ply')

def save_patches(args):
    print("visualizing...")
    data_path = "/data/penghy/processed_scannet/crop_train"
    scene_path = os.path.join(data_path, "1001_" + args.scene_id + ".obj")
    npy_path = "/data/penghy/processed_scannet/npy_train"
    npy = os.path.join(npy_path, "1001_" + args.scene_id + ".npy")
    ndata = np.load(npy, allow_pickle=True).item()
    res_faces = ndata["res_faces"]
    mesh = trimesh.load(scene_path)
    print(mesh.vertices)
    patch_color = np.random.uniform(0,254,(256, 3))
    face_color = np.zeros((mesh.faces.shape[0], 4))
    for face in range(mesh.faces.shape[0]):
        face_color[face, :3] = patch_color[int(res_faces[face])]
        face_color[face, 3] = 100
    mesh.visual.face_colors = face_color.astype("int32")
    mesh.export("/data/penghy/processed_scannet/corr/" + f'bfs-{args.scene_id}.ply')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, required=True)
    args = parser.parse_args()

    visualize(args)
    print("done!")