
import shutil
import os
import os.path as osp


root="/home/xiaomi/transformer-flow/work_dirs2/attn_ffn_8blocks/mvtec"

for class_name in os.listdir(root):
    dir=osp.join(root,class_name)
    if osp.isfile(dir):
        continue
    for file_name in os.listdir(dir):
        if file_name.endswith(".pt"):
            os.remove(osp.join(dir,file_name))
