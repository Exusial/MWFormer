import jittor as jt
import os
import numpy as np
count = np.zeros((21)).astype("int32")

def get_weight():
    global count
    path = "/data/penghy/processed_scannet/simp_train"
    for fn in os.listdir(path):
        if fn.endswith(".obj"):
            label = jt.load(os.path.join(path, fn.replace(".obj", ".pkl")))
            # print(label.shape)
            tmp,_ = np.histogram(label,range(21 + 1))
            # print(tmp)
            count += tmp
            # for l in label:
            #     count[int(l)] += 1
            # print(count)
    count[0] = 0
    print(count)
    count = count / np.sum(count)
    count_weights = 1 / np.log(1.2+count[1:])
    print(count_weights)

get_weight()