import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

from show_segDataset_label_cls_id import get_label_cls_ids

def parse_horse_label_to0and1():
    # 遍历文件夹
    horse_lable_dir='segDataset/horse/Annotations'
    for _,_,files in os.walk(horse_lable_dir):
        t=tqdm(files)
        all_cls_ids=[]
        for f in t:
            img_path=os.path.join(horse_lable_dir, f)
            img=cv2.imread(img_path, 0)  # 灰度图读法 -- 单通道图片
            img_map=np.zeros_like(img)
            img_map[img!=0] = 1 # 在map中将图像中非0的位置对应的设置为1

            cls_ids = list(np.unique(img_map))
            all_cls_ids += cls_ids
            all_cls_ids = list(set(all_cls_ids))

            img=Image.fromarray(img_map)
            img.save(img_path.split('.')[0]+'.png')
            os.remove(img_path) # 删除源jpg文件
        print(all_cls_ids)


if __name__ == "__main__":

    parse_horse_label_to0and1()  # 预处理

    # 马为二分类分割，但是label中包含多个像数值，可以采用二值进行转换的预处理 -- 即非0为1，0为0
    # 0：表示马的分类
    # 1: 表示背景
    horse_label_path='segDataset/horse/Annotations'
    get_label_cls_ids(horse_label_path, dataset_name="horse")