import os
from tqdm import tqdm

def create_index_txt(dataset_dir=None):
    sub_dirs=['Annotations', 'Images']  # 子目录
    data_dirs=[
        os.path.join(dataset_dir, sub_dirs[0]),
        os.path.join(dataset_dir, sub_dirs[1])
    ]
    label_files=[]
    img_files=[]
    for _,_,anno_files in os.walk(data_dirs[0]):
        label_files=anno_files
        break
    for _,_,png_files in os.walk(data_dirs[1]):
        img_files=png_files
        break
    
    save_path=os.path.join(dataset_dir, 'train_list.txt')
    with open(save_path, 'w') as f:
        t=tqdm(label_files)
        for label_ in t:
            for img_ in img_files:
                if label_.split('.')[0] == img_.split('.')[0]:
                    label_path=os.path.join(sub_dirs[0], label_)
                    img_path=os.path.join(sub_dirs[1], img_)
                    f.write(img_path+' '+label_path+'\n')
                    break

if __name__ == "__main__":
    create_index_txt(dataset_dir='segDataset/horse')