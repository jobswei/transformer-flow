
import shutil
import os
import os.path as osp
import tqdm

root="/root/transformer-flow/work_dirs/mvtec_msAttnFlow_pool421_allChannel_noNeck_8blocks_reverse/mvtec"

for class_name in tqdm.tqdm(os.listdir(root)):
    dir=osp.join(root,class_name)
    if osp.isfile(dir):
        continue
    for file_name in os.listdir(dir):
        if file_name=="last.pt":
        # if file_name.endswith(".pt"):
            os.remove(osp.join(dir,file_name))
