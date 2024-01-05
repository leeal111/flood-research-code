import shutil
import cv2
import os
import method.preprocess as preprocess
import method.stiv as method
import time
import pandas as pd
import math

root=''
for dir1 in os.listdir(root):
    for dir2 in os.listdir(os.path.join(root, dir1)):
        fileName=os.path.join(root, dir1,dir2)
        copyName=os.path.join(root, dir1,'cop',dir2)
        if dir2.startswith('STI_MOT'):
            if not  os.path.exists(os.path.join(root, dir1,'hwMot')):
                os.mkdir(os.path.join(root, dir1,'hwMot'))
            shutil.move(src=fileName,dst=os.path.join(root, dir1,'hwMot',dir2))
            continue
        if dir2.startswith('sti'):
            continue
        if not os.path.exists(os.path.join(root, dir1,'cop')):
            os.mkdir(os.path.join(root, dir1,'cop'))
        shutil.move(src=fileName,dst=copyName)
