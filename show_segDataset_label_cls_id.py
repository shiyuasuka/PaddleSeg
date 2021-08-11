import PIL.Image as Image
import numpy as np
import os
from tqdm import tqdm

# 读取label中的像素值，分析其类别大致情况
def get_label_cls_ids(label_path, dataset_name=""):
    all_cls_ids=[]
    for _, _, files in os.walk(label_path):
        t=tqdm(files)
        for f in t:
            img_path = os.path.join(label_path, f)
            img = Image.open(img_path)
            img = np.asarray(img).copy()
            cls_ids = list(np.unique(img))

            all_cls_ids += cls_ids
            all_cls_ids = list(set(all_cls_ids))
    print(dataset_name+"-cls_id: ", all_cls_ids)
    print(dataset_name+"为{0}分类".format(len(all_cls_ids)))



if __name__ == "__main__":

    # 马为二分类分割，但是label中包含多个像数值，可以采用二值进行转换的预处理 -- 即非0为1，0为0
    # 0：表示马的分类
    # 1: 表示背景
    horse_label_path='segDataset/horse/Annotations'
    get_label_cls_ids(horse_label_path, dataset_name="horse")
    print('horse'+"实际应转换为{0}分类(将非0像素转换为像素值为1)".format(2))
    print('\n')

    facade_label_path='segDataset/facade/Annotations'
    get_label_cls_ids(facade_label_path, dataset_name="facade")
    print('\n')

    fundusvessels_label_path='segDataset/FundusVessels/Annotations'
    get_label_cls_ids(fundusvessels_label_path, dataset_name="fundusvessels")
    print('\n')

    laneline_label_path='segDataset/laneline/Annotations'
    get_label_cls_ids(laneline_label_path, dataset_name="laneline")