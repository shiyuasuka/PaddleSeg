# [AI训练营]PaddleSeg实现语义分割Baseline

手把手教你基于PaddleSeg实现语义分割任务

------


# 一、作业任务

> 本次任务将基于PaddleSeg展开语义分割任务的学习与实践，baseline会提供PaddleSeg套件的基本使用，相关细节如有遗漏，可参考[10分钟上手PaddleSeg
](https://aistudio.baidu.com/aistudio/projectdetail/1672610?channelType=0&channel=0)

1. 选择提供的**五个数据集**中的一个数据集作为本次作业的训练数据，并**根据baseline跑通项目**

2. **可视化1-3张**预测图片与预测结果（本次提供的数据集没有验证与测试集，可将训练数据直接进行预测，展示训练数据的预测结果即可）


**加分项:**

3. 尝试**划分验证集**，再进行训练

4. 选择**其他网络**，调整训练参数进行更好的尝试


**PS**:PaddleSeg相关参考项目:

- [常规赛：PALM病理性近视病灶检测与分割基线方案](https://aistudio.baidu.com/aistudio/projectdetail/1941312)

- [PaddleSeg 2.0动态图：车道线图像分割任务简介](https://aistudio.baidu.com/aistudio/projectdetail/1752986?channelType=0&channel=0)

------

------

# 二、数据集说明

------

本项目使用的数据集是:[AI训练营]语义分割数据集合集，包含马分割，眼底血管分割，车道线分割，场景分割以及人像分割。

该数据集已加载到本环境中，位于:

**data/data103787/segDataset.zip**



```python
# unzip: 解压指令
# -o: 表示解压后进行输出
# -q: 表示静默模式，即不输出解压过程的日志
# -d: 解压到指定目录下，如果该文件夹不存在，会自动创建
!unzip -oq data/data103787/segDataset.zip -d segDataset
```

解压完成后，会在左侧文件目录多出一个**segDataset**的文件夹，该文件夹下有**5**个子文件夹：

- **horse -- 马分割数据**<二分类任务>

![](https://ai-studio-static-online.cdn.bcebos.com/2b12a7fab9ee409587a2aec332a70ba2bce0fcc4a10345a4aa38941db65e8d02)

- **fundusVessels -- 眼底血管分割数据**

> 灰度图，每个像素值对应相应的类别 -- 因此label不宜观察，但符合套件训练需要

![](https://ai-studio-static-online.cdn.bcebos.com/b515662defe548bdaa517b879722059ad53b5d87dd82441c8c4611124f6fdad0)

- **laneline -- 车道线分割数据**

![](https://ai-studio-static-online.cdn.bcebos.com/2aeccfe514e24cf98459df7c36421cddf78d9ddfc2cf41ffa4aafc10b13c8802)

- **facade -- 场景分割数据**

![](https://ai-studio-static-online.cdn.bcebos.com/57752d86fc5c4a10a3e4b91ae05a3e38d57d174419be4afeba22eb75b699112c)

- **cocome -- 人像分割数据**

> label非直接的图片，为json格式的标注文件，有需要的小伙伴可以看一看PaddleSeg的[PaddleSeg实战——人像分割](https://aistudio.baidu.com/aistudio/projectdetail/2177440?channelType=0&channel=0)


```python
# tree: 查看文件夹树状结构
# -L: 表示遍历深度
!tree segDataset -L 2
```

    segDataset
    ├── cocome
    │   ├── Annotations
    │   └── Images
    ├── facade
    │   ├── Annotations
    │   └── Images
    ├── FundusVessels
    │   ├── Annotations
    │   └── Images
    ├── horse
    │   ├── Annotations
    │   ├── Images
    │   └── train_list.txt
    └── laneline
        ├── Annotations
        └── Images
    
    15 directories, 1 file


> 查看数据label的像素分布，可从中得知分割任务的类别数： 脚本位于: **show_segDataset_label_cls_id.py**

> 关于人像分割数据分析，这里不做提示，可参考[PaddleSeg实战——人像分割](https://aistudio.baidu.com/aistudio/projectdetail/2177440?channelType=0&channel=0)


```python
# 查看label中像素分类情况
!python show_segDataset_label_cls_id.py
```

    100%|███████████████████████████████████████| 498/498 [00:00<00:00, 1016.79it/s]
    horse-cls_id:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
    horse为90分类
    horse实际应转换为2分类(将非0像素转换为像素值为1)
    
    
    100%|████████████████████████████████████████| 845/845 [00:04<00:00, 196.13it/s]
    facade-cls_id:  [0, 1, 2, 3, 4, 5, 6, 7, 8]
    facade为9分类
    
    
    100%|████████████████████████████████████████| 200/200 [00:01<00:00, 182.93it/s]
    fundusvessels-cls_id:  [0, 1]
    fundusvessels为2分类
    
    
    100%|███████████████████████████████████████| 4878/4878 [01:30<00:00, 53.72it/s]
    laneline-cls_id:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    laneline为20分类


# 三、数据预处理

> 这里就以horse数据作为示例

-----

- 首先，通过上边的像素值分析以及horse本身的标签表现，我们确定horse数据集为二分类任务

- 然而，实际label中，却包含多个像素值，因此需要将horse中的所有label进行一个预处理

- 预处理内容为: 0值不变，非0值变为1，然后再保存label

- **并且保存文件格式为png，单通道图片为Label图片，最好保存为png——否则可能出现异常像素**

**对应horse的预处理脚本，位于:**

parse_horse_label.py


```python
!python parse_horse_label.py
```

    100%|████████████████████████████████████████| 498/498 [00:01<00:00, 389.33it/s]
    [0, 1]
    100%|███████████████████████████████████████| 236/236 [00:00<00:00, 1111.00it/s]
    horse-cls_id:  [0, 1]
    horse为2分类


- 预处理完成后，配置训练的索引文件txt，方便后边套件读取数据

> txt创建脚本位于: **horse_create_train_list.py**

> 同时，生成的txt位于: **segDataset/horse/train_list.txt**


```python
# 创建训练的数据索引txt
# 格式如下
# line1: train_img1.jpg train_label1.png
# line2: train_img2.jpg train_label2.png
!python horse_create_train_list.py
```

    100%|██████████████████████████████████████| 236/236 [00:00<00:00, 14358.01it/s]


# 四、使用套件开始训练

- 1.解压套件: 已挂载到本项目, 位于:**data/data102250/PaddleSeg-release-2.1.zip**


```python
# 解压套件
!unzip -oq data/data102250/PaddleSeg-release-2.1.zip
# 通过mv指令实现文件夹重命名
!mv PaddleSeg-release-2.1 PaddleSeg
```

    mv: cannot move 'PaddleSeg-release-2.1' to 'PaddleSeg/PaddleSeg-release-2.1': Directory not empty


- 2.选择模型，baseline选择**bisenet**, 位于: **PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml**

- 3.配置模型文件

> 首先修改训练数据集加载的dataset类型:

![](https://ai-studio-static-online.cdn.bcebos.com/2f5363d71034490290f720ea8bb0d6873d7df2712d4b4e84ae41b0378aed8b89)

> 然后配置训练数据集如下:

![](https://ai-studio-static-online.cdn.bcebos.com/29547856db4b4bfc80aa3732e143f2788589f9316c694f369c9bd1da44b815dc)

> 类似的，配置验证数据集: -- **注意修改train_path为val_path**

![](https://ai-studio-static-online.cdn.bcebos.com/09713aaaed6b4611a525d25aae67e4f0538224f7ac0241eb941d97892bf6c4c1)

<font color="red" size=4>其它模型可能需要到: PaddleSeg/configs/$_base_$  中的数据集文件进行配置，但修改参数与bisenet中的数据集修改参数相同 </font>

![](https://ai-studio-static-online.cdn.bcebos.com/b154dcbf15e14f43aa13455c0ceeaaebe0489c9a09dd439f9d32e8b0a31355ec)


- 4.开始训练

使用PaddleSeg的train.py，传入模型文件即可开启训练


```python
!python PaddleSeg/train.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--batch_size 4\
--iters 2000\
--learning_rate 0.01\
--save_interval 200\
--save_dir PaddleSeg/output\
--seed 2021\
--log_iters 20\
--do_eval\
--use_vdl

# --batch_size 4\  # 批大小
# --iters 2000\    # 迭代次数 -- 根据数据大小，批大小估计迭代次数
# --learning_rate 0.01\ # 学习率
# --save_interval 200\ # 保存周期 -- 迭代次数计算周期
# --save_dir PaddleSeg/output\ # 输出路径
# --seed 2021\ # 训练中使用到的随机数种子
# --log_iters 20\ # 日志频率 -- 迭代次数计算周期
# --do_eval\ # 边训练边验证
# --use_vdl # 使用vdl可视化记录
# 用于断训==即中断后继续上一次状态进行训练
# --resume_model model_dir
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    2021-08-11 09:11:27 [INFO]	
    ------------Environment Information-------------
    platform: Linux-4.4.0-150-generic-x86_64-with-debian-stretch-sid
    Python: 3.7.4 (default, Aug 13 2019, 20:35:49) [GCC 7.3.0]
    Paddle compiled with cuda: True
    NVCC: Cuda compilation tools, release 10.1, V10.1.243
    cudnn: 7.6
    GPUs used: 1
    CUDA_VISIBLE_DEVICES: None
    GPU: ['GPU 0: Tesla V100-SXM2-32GB']
    GCC: gcc (Ubuntu 7.5.0-3ubuntu1~16.04) 7.5.0
    PaddlePaddle: 2.0.2
    OpenCV: 4.1.1
    ------------------------------------------------
    2021-08-11 09:11:27 [INFO]	
    ---------------Config Information---------------
    batch_size: 4
    iters: 2000
    loss:
      coef:
      - 1
      - 1
      - 1
      - 1
      - 1
      types:
      - ignore_index: 255
        type: CrossEntropyLoss
    lr_scheduler:
      end_lr: 0
      learning_rate: 0.01
      power: 0.9
      type: PolynomialDecay
    model:
      pretrained: null
      type: BiSeNetV2
    optimizer:
      momentum: 0.9
      type: sgd
      weight_decay: 4.0e-05
    train_dataset:
      dataset_root: segDataset/horse
      mode: train
      num_classes: 2
      train_path: segDataset/horse/train_list.txt
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: RandomHorizontalFlip
      - type: Normalize
      type: Dataset
    val_dataset:
      dataset_root: segDataset/horse
      mode: val
      num_classes: 2
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: Normalize
      type: Dataset
      val_path: segDataset/horse/train_list.txt
    ------------------------------------------------
    W0811 09:11:27.758955   384 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0811 09:11:27.759001   384 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    2021-08-11 09:11:35 [INFO]	[TRAIN] epoch: 1, iter: 20/2000, loss: 3.1863, lr: 0.009914, batch_cost: 0.1069, reader_cost: 0.01135, ips: 37.4259 samples/sec | ETA 00:03:31
    2021-08-11 09:11:36 [INFO]	[TRAIN] epoch: 1, iter: 40/2000, loss: 2.2679, lr: 0.009824, batch_cost: 0.0944, reader_cost: 0.00015, ips: 42.3835 samples/sec | ETA 00:03:04
    2021-08-11 09:11:38 [INFO]	[TRAIN] epoch: 2, iter: 60/2000, loss: 2.0758, lr: 0.009734, batch_cost: 0.0963, reader_cost: 0.00474, ips: 41.5578 samples/sec | ETA 00:03:06
    2021-08-11 09:11:40 [INFO]	[TRAIN] epoch: 2, iter: 80/2000, loss: 1.9930, lr: 0.009644, batch_cost: 0.0943, reader_cost: 0.00010, ips: 42.4325 samples/sec | ETA 00:03:00
    2021-08-11 09:11:42 [INFO]	[TRAIN] epoch: 2, iter: 100/2000, loss: 2.1444, lr: 0.009553, batch_cost: 0.0922, reader_cost: 0.00007, ips: 43.3753 samples/sec | ETA 00:02:55
    2021-08-11 09:11:44 [INFO]	[TRAIN] epoch: 3, iter: 120/2000, loss: 1.8337, lr: 0.009463, batch_cost: 0.0959, reader_cost: 0.00427, ips: 41.7089 samples/sec | ETA 00:03:00
    2021-08-11 09:11:46 [INFO]	[TRAIN] epoch: 3, iter: 140/2000, loss: 1.8322, lr: 0.009372, batch_cost: 0.0942, reader_cost: 0.00008, ips: 42.4602 samples/sec | ETA 00:02:55
    2021-08-11 09:11:48 [INFO]	[TRAIN] epoch: 3, iter: 160/2000, loss: 1.7678, lr: 0.009282, batch_cost: 0.0922, reader_cost: 0.00008, ips: 43.4015 samples/sec | ETA 00:02:49
    2021-08-11 09:11:50 [INFO]	[TRAIN] epoch: 4, iter: 180/2000, loss: 1.8116, lr: 0.009191, batch_cost: 0.0961, reader_cost: 0.00457, ips: 41.6161 samples/sec | ETA 00:02:54
    2021-08-11 09:11:52 [INFO]	[TRAIN] epoch: 4, iter: 200/2000, loss: 1.7562, lr: 0.009100, batch_cost: 0.0920, reader_cost: 0.00007, ips: 43.4865 samples/sec | ETA 00:02:45
    2021-08-11 09:11:52 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.INT32, but right dtype is VarType.BOOL, the right dtype will convert to VarType.INT32
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.INT64, but right dtype is VarType.BOOL, the right dtype will convert to VarType.INT64
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    236/236 [==============================] - 6s 25ms/step - batch_cost: 0.0244 - reader cost: 1.2406e-
    2021-08-11 09:11:57 [INFO]	[EVAL] #Images: 236 mIoU: 0.7025 Acc: 0.8551 Kappa: 0.6403 
    2021-08-11 09:11:57 [INFO]	[EVAL] Class IoU: 
    [0.8175 0.5876]
    2021-08-11 09:11:57 [INFO]	[EVAL] Class Acc: 
    [0.9209 0.6985]
    2021-08-11 09:11:58 [INFO]	[EVAL] The model with the best validation mIoU (0.7025) was saved at iter 200.
    2021-08-11 09:12:00 [INFO]	[TRAIN] epoch: 4, iter: 220/2000, loss: 1.8217, lr: 0.009009, batch_cost: 0.0924, reader_cost: 0.00007, ips: 43.2914 samples/sec | ETA 00:02:44
    2021-08-11 09:12:01 [INFO]	[TRAIN] epoch: 5, iter: 240/2000, loss: 1.5713, lr: 0.008918, batch_cost: 0.0951, reader_cost: 0.00369, ips: 42.0534 samples/sec | ETA 00:02:47
    2021-08-11 09:12:03 [INFO]	[TRAIN] epoch: 5, iter: 260/2000, loss: 1.6455, lr: 0.008827, batch_cost: 0.0938, reader_cost: 0.00010, ips: 42.6346 samples/sec | ETA 00:02:43
    2021-08-11 09:12:05 [INFO]	[TRAIN] epoch: 5, iter: 280/2000, loss: 1.6267, lr: 0.008735, batch_cost: 0.0920, reader_cost: 0.00006, ips: 43.4798 samples/sec | ETA 00:02:38
    2021-08-11 09:12:07 [INFO]	[TRAIN] epoch: 6, iter: 300/2000, loss: 1.7670, lr: 0.008644, batch_cost: 0.0964, reader_cost: 0.00483, ips: 41.4833 samples/sec | ETA 00:02:43
    2021-08-11 09:12:09 [INFO]	[TRAIN] epoch: 6, iter: 320/2000, loss: 1.6481, lr: 0.008552, batch_cost: 0.0938, reader_cost: 0.00009, ips: 42.6312 samples/sec | ETA 00:02:37
    2021-08-11 09:12:11 [INFO]	[TRAIN] epoch: 6, iter: 340/2000, loss: 1.5175, lr: 0.008461, batch_cost: 0.0926, reader_cost: 0.00008, ips: 43.2016 samples/sec | ETA 00:02:33
    2021-08-11 09:12:13 [INFO]	[TRAIN] epoch: 7, iter: 360/2000, loss: 1.4907, lr: 0.008369, batch_cost: 0.0962, reader_cost: 0.00376, ips: 41.5947 samples/sec | ETA 00:02:37
    2021-08-11 09:12:15 [INFO]	[TRAIN] epoch: 7, iter: 380/2000, loss: 1.6426, lr: 0.008277, batch_cost: 0.0943, reader_cost: 0.00008, ips: 42.4181 samples/sec | ETA 00:02:32
    2021-08-11 09:12:16 [INFO]	[TRAIN] epoch: 7, iter: 400/2000, loss: 1.4134, lr: 0.008185, batch_cost: 0.0927, reader_cost: 0.00007, ips: 43.1727 samples/sec | ETA 00:02:28
    2021-08-11 09:12:17 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    236/236 [==============================] - 6s 24ms/step - batch_cost: 0.0240 - reader cost: 1.1432e-
    2021-08-11 09:12:22 [INFO]	[EVAL] #Images: 236 mIoU: 0.7898 Acc: 0.9069 Kappa: 0.7592 
    2021-08-11 09:12:22 [INFO]	[EVAL] Class IoU: 
    [0.8814 0.6982]
    2021-08-11 09:12:22 [INFO]	[EVAL] Class Acc: 
    [0.9366 0.8232]
    2021-08-11 09:12:23 [INFO]	[EVAL] The model with the best validation mIoU (0.7898) was saved at iter 400.
    2021-08-11 09:12:24 [INFO]	[TRAIN] epoch: 8, iter: 420/2000, loss: 1.4602, lr: 0.008093, batch_cost: 0.0949, reader_cost: 0.00405, ips: 42.1533 samples/sec | ETA 00:02:29
    2021-08-11 09:12:26 [INFO]	[TRAIN] epoch: 8, iter: 440/2000, loss: 1.4779, lr: 0.008001, batch_cost: 0.0926, reader_cost: 0.00008, ips: 43.1948 samples/sec | ETA 00:02:24
    2021-08-11 09:12:28 [INFO]	[TRAIN] epoch: 8, iter: 460/2000, loss: 1.4824, lr: 0.007909, batch_cost: 0.0931, reader_cost: 0.00008, ips: 42.9788 samples/sec | ETA 00:02:23
    2021-08-11 09:12:30 [INFO]	[TRAIN] epoch: 9, iter: 480/2000, loss: 1.5104, lr: 0.007816, batch_cost: 0.0962, reader_cost: 0.00427, ips: 41.5861 samples/sec | ETA 00:02:26
    2021-08-11 09:12:32 [INFO]	[TRAIN] epoch: 9, iter: 500/2000, loss: 1.5064, lr: 0.007724, batch_cost: 0.0927, reader_cost: 0.00008, ips: 43.1648 samples/sec | ETA 00:02:19
    2021-08-11 09:12:34 [INFO]	[TRAIN] epoch: 9, iter: 520/2000, loss: 1.4446, lr: 0.007631, batch_cost: 0.0922, reader_cost: 0.00007, ips: 43.3854 samples/sec | ETA 00:02:16
    2021-08-11 09:12:36 [INFO]	[TRAIN] epoch: 10, iter: 540/2000, loss: 1.5191, lr: 0.007538, batch_cost: 0.0946, reader_cost: 0.00371, ips: 42.2914 samples/sec | ETA 00:02:18
    2021-08-11 09:12:38 [INFO]	[TRAIN] epoch: 10, iter: 560/2000, loss: 1.4190, lr: 0.007445, batch_cost: 0.0936, reader_cost: 0.00008, ips: 42.7240 samples/sec | ETA 00:02:14
    2021-08-11 09:12:39 [INFO]	[TRAIN] epoch: 10, iter: 580/2000, loss: 1.4746, lr: 0.007352, batch_cost: 0.0910, reader_cost: 0.00006, ips: 43.9525 samples/sec | ETA 00:02:09
    2021-08-11 09:12:41 [INFO]	[TRAIN] epoch: 11, iter: 600/2000, loss: 1.3897, lr: 0.007259, batch_cost: 0.0969, reader_cost: 0.00471, ips: 41.2704 samples/sec | ETA 00:02:15
    2021-08-11 09:12:41 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    236/236 [==============================] - 6s 25ms/step - batch_cost: 0.0247 - reader cost: 1.1859e-0
    2021-08-11 09:12:47 [INFO]	[EVAL] #Images: 236 mIoU: 0.7989 Acc: 0.9078 Kappa: 0.7724 
    2021-08-11 09:12:47 [INFO]	[EVAL] Class IoU: 
    [0.8795 0.7183]
    2021-08-11 09:12:47 [INFO]	[EVAL] Class Acc: 
    [0.9614 0.7829]
    2021-08-11 09:12:47 [INFO]	[EVAL] The model with the best validation mIoU (0.7989) was saved at iter 600.
    2021-08-11 09:12:49 [INFO]	[TRAIN] epoch: 11, iter: 620/2000, loss: 1.4339, lr: 0.007166, batch_cost: 0.0931, reader_cost: 0.00008, ips: 42.9628 samples/sec | ETA 00:02:08
    2021-08-11 09:12:51 [INFO]	[TRAIN] epoch: 11, iter: 640/2000, loss: 1.4065, lr: 0.007072, batch_cost: 0.0921, reader_cost: 0.00007, ips: 43.4098 samples/sec | ETA 00:02:05
    2021-08-11 09:12:53 [INFO]	[TRAIN] epoch: 12, iter: 660/2000, loss: 1.4765, lr: 0.006978, batch_cost: 0.0973, reader_cost: 0.00416, ips: 41.1123 samples/sec | ETA 00:02:10
    2021-08-11 09:12:55 [INFO]	[TRAIN] epoch: 12, iter: 680/2000, loss: 1.3897, lr: 0.006885, batch_cost: 0.0933, reader_cost: 0.00008, ips: 42.8693 samples/sec | ETA 00:02:03
    2021-08-11 09:12:57 [INFO]	[TRAIN] epoch: 12, iter: 700/2000, loss: 1.3388, lr: 0.006791, batch_cost: 0.0928, reader_cost: 0.00008, ips: 43.0962 samples/sec | ETA 00:02:00
    2021-08-11 09:12:59 [INFO]	[TRAIN] epoch: 13, iter: 720/2000, loss: 1.3674, lr: 0.006697, batch_cost: 0.0969, reader_cost: 0.00459, ips: 41.2993 samples/sec | ETA 00:02:03
    2021-08-11 09:13:01 [INFO]	[TRAIN] epoch: 13, iter: 740/2000, loss: 1.4520, lr: 0.006603, batch_cost: 0.0935, reader_cost: 0.00007, ips: 42.7756 samples/sec | ETA 00:01:57
    2021-08-11 09:13:02 [INFO]	[TRAIN] epoch: 13, iter: 760/2000, loss: 1.3462, lr: 0.006508, batch_cost: 0.0920, reader_cost: 0.00008, ips: 43.4813 samples/sec | ETA 00:01:54
    2021-08-11 09:13:04 [INFO]	[TRAIN] epoch: 14, iter: 780/2000, loss: 1.4062, lr: 0.006414, batch_cost: 0.0956, reader_cost: 0.00426, ips: 41.8235 samples/sec | ETA 00:01:56
    2021-08-11 09:13:06 [INFO]	[TRAIN] epoch: 14, iter: 800/2000, loss: 1.3690, lr: 0.006319, batch_cost: 0.0943, reader_cost: 0.00008, ips: 42.4022 samples/sec | ETA 00:01:53
    2021-08-11 09:13:06 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    236/236 [==============================] - 6s 25ms/step - batch_cost: 0.0246 - reader cost: 1.2305e-0
    2021-08-11 09:13:12 [INFO]	[EVAL] #Images: 236 mIoU: 0.8381 Acc: 0.9314 Kappa: 0.8204 
    2021-08-11 09:13:12 [INFO]	[EVAL] Class IoU: 
    [0.9117 0.7646]
    2021-08-11 09:13:12 [INFO]	[EVAL] Class Acc: 
    [0.9476 0.8832]
    2021-08-11 09:13:12 [INFO]	[EVAL] The model with the best validation mIoU (0.8381) was saved at iter 800.
    2021-08-11 09:13:14 [INFO]	[TRAIN] epoch: 14, iter: 820/2000, loss: 1.3551, lr: 0.006224, batch_cost: 0.0915, reader_cost: 0.00006, ips: 43.7261 samples/sec | ETA 00:01:47
    2021-08-11 09:13:16 [INFO]	[TRAIN] epoch: 15, iter: 840/2000, loss: 1.3531, lr: 0.006129, batch_cost: 0.0963, reader_cost: 0.00374, ips: 41.5196 samples/sec | ETA 00:01:51
    2021-08-11 09:13:18 [INFO]	[TRAIN] epoch: 15, iter: 860/2000, loss: 1.4461, lr: 0.006034, batch_cost: 0.0940, reader_cost: 0.00008, ips: 42.5352 samples/sec | ETA 00:01:47
    2021-08-11 09:13:20 [INFO]	[TRAIN] epoch: 15, iter: 880/2000, loss: 1.3417, lr: 0.005939, batch_cost: 0.0926, reader_cost: 0.00007, ips: 43.1866 samples/sec | ETA 00:01:43
    2021-08-11 09:13:22 [INFO]	[TRAIN] epoch: 16, iter: 900/2000, loss: 1.3524, lr: 0.005844, batch_cost: 0.0976, reader_cost: 0.00456, ips: 40.9940 samples/sec | ETA 00:01:47
    2021-08-11 09:13:24 [INFO]	[TRAIN] epoch: 16, iter: 920/2000, loss: 1.2673, lr: 0.005748, batch_cost: 0.0932, reader_cost: 0.00008, ips: 42.8984 samples/sec | ETA 00:01:40
    2021-08-11 09:13:26 [INFO]	[TRAIN] epoch: 16, iter: 940/2000, loss: 1.3250, lr: 0.005652, batch_cost: 0.0921, reader_cost: 0.00007, ips: 43.4280 samples/sec | ETA 00:01:37
    2021-08-11 09:13:28 [INFO]	[TRAIN] epoch: 17, iter: 960/2000, loss: 1.3780, lr: 0.005556, batch_cost: 0.0968, reader_cost: 0.00383, ips: 41.3099 samples/sec | ETA 00:01:40
    2021-08-11 09:13:29 [INFO]	[TRAIN] epoch: 17, iter: 980/2000, loss: 1.3727, lr: 0.005460, batch_cost: 0.0926, reader_cost: 0.00007, ips: 43.2106 samples/sec | ETA 00:01:34
    2021-08-11 09:13:31 [INFO]	[TRAIN] epoch: 17, iter: 1000/2000, loss: 1.3266, lr: 0.005364, batch_cost: 0.0918, reader_cost: 0.00007, ips: 43.5657 samples/sec | ETA 00:01:31
    2021-08-11 09:13:31 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    236/236 [==============================] - 6s 24ms/step - batch_cost: 0.0243 - reader cost: 1.2508e-
    2021-08-11 09:13:37 [INFO]	[EVAL] #Images: 236 mIoU: 0.8536 Acc: 0.9390 Kappa: 0.8392 
    2021-08-11 09:13:37 [INFO]	[EVAL] Class IoU: 
    [0.9215 0.7857]
    2021-08-11 09:13:37 [INFO]	[EVAL] Class Acc: 
    [0.9488 0.9089]
    2021-08-11 09:13:37 [INFO]	[EVAL] The model with the best validation mIoU (0.8536) was saved at iter 1000.
    2021-08-11 09:13:39 [INFO]	[TRAIN] epoch: 18, iter: 1020/2000, loss: 1.3707, lr: 0.005267, batch_cost: 0.0977, reader_cost: 0.00423, ips: 40.9570 samples/sec | ETA 00:01:35
    2021-08-11 09:13:41 [INFO]	[TRAIN] epoch: 18, iter: 1040/2000, loss: 1.2951, lr: 0.005170, batch_cost: 0.0926, reader_cost: 0.00007, ips: 43.2163 samples/sec | ETA 00:01:28
    2021-08-11 09:13:43 [INFO]	[TRAIN] epoch: 18, iter: 1060/2000, loss: 1.2424, lr: 0.005073, batch_cost: 0.0917, reader_cost: 0.00007, ips: 43.6244 samples/sec | ETA 00:01:26
    2021-08-11 09:13:45 [INFO]	[TRAIN] epoch: 19, iter: 1080/2000, loss: 1.3171, lr: 0.004976, batch_cost: 0.0978, reader_cost: 0.00558, ips: 40.9006 samples/sec | ETA 00:01:29
    2021-08-11 09:13:47 [INFO]	[TRAIN] epoch: 19, iter: 1100/2000, loss: 1.2490, lr: 0.004879, batch_cost: 0.0917, reader_cost: 0.00007, ips: 43.5989 samples/sec | ETA 00:01:22
    2021-08-11 09:13:49 [INFO]	[TRAIN] epoch: 19, iter: 1120/2000, loss: 1.2794, lr: 0.004781, batch_cost: 0.0904, reader_cost: 0.00007, ips: 44.2500 samples/sec | ETA 00:01:19
    2021-08-11 09:13:51 [INFO]	[TRAIN] epoch: 20, iter: 1140/2000, loss: 1.3012, lr: 0.004684, batch_cost: 0.0978, reader_cost: 0.00450, ips: 40.9195 samples/sec | ETA 00:01:24
    2021-08-11 09:13:52 [INFO]	[TRAIN] epoch: 20, iter: 1160/2000, loss: 1.2454, lr: 0.004586, batch_cost: 0.0920, reader_cost: 0.00007, ips: 43.4589 samples/sec | ETA 00:01:17
    2021-08-11 09:13:54 [INFO]	[TRAIN] epoch: 20, iter: 1180/2000, loss: 1.3367, lr: 0.004487, batch_cost: 0.0922, reader_cost: 0.00008, ips: 43.3931 samples/sec | ETA 00:01:15
    2021-08-11 09:13:56 [INFO]	[TRAIN] epoch: 21, iter: 1200/2000, loss: 1.2383, lr: 0.004389, batch_cost: 0.0958, reader_cost: 0.00424, ips: 41.7435 samples/sec | ETA 00:01:16
    2021-08-11 09:13:56 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    236/236 [==============================] - 6s 24ms/step - batch_cost: 0.0238 - reader cost: 1.1290e-
    2021-08-11 09:14:02 [INFO]	[EVAL] #Images: 236 mIoU: 0.8537 Acc: 0.9378 Kappa: 0.8394 
    2021-08-11 09:14:02 [INFO]	[EVAL] Class IoU: 
    [0.9191 0.7882]
    2021-08-11 09:14:02 [INFO]	[EVAL] Class Acc: 
    [0.9584 0.8802]
    2021-08-11 09:14:02 [INFO]	[EVAL] The model with the best validation mIoU (0.8537) was saved at iter 1200.
    2021-08-11 09:14:04 [INFO]	[TRAIN] epoch: 21, iter: 1220/2000, loss: 1.2637, lr: 0.004290, batch_cost: 0.0905, reader_cost: 0.00006, ips: 44.1944 samples/sec | ETA 00:01:10
    2021-08-11 09:14:06 [INFO]	[TRAIN] epoch: 22, iter: 1240/2000, loss: 1.3047, lr: 0.004191, batch_cost: 0.0934, reader_cost: 0.00384, ips: 42.8278 samples/sec | ETA 00:01:10
    2021-08-11 09:14:08 [INFO]	[TRAIN] epoch: 22, iter: 1260/2000, loss: 1.3020, lr: 0.004092, batch_cost: 0.0931, reader_cost: 0.00008, ips: 42.9710 samples/sec | ETA 00:01:08
    2021-08-11 09:14:10 [INFO]	[TRAIN] epoch: 22, iter: 1280/2000, loss: 1.2411, lr: 0.003992, batch_cost: 0.0926, reader_cost: 0.00007, ips: 43.2065 samples/sec | ETA 00:01:06
    2021-08-11 09:14:11 [INFO]	[TRAIN] epoch: 23, iter: 1300/2000, loss: 1.3079, lr: 0.003892, batch_cost: 0.0945, reader_cost: 0.00379, ips: 42.3070 samples/sec | ETA 00:01:06
    2021-08-11 09:14:13 [INFO]	[TRAIN] epoch: 23, iter: 1320/2000, loss: 1.2278, lr: 0.003792, batch_cost: 0.0928, reader_cost: 0.00008, ips: 43.0889 samples/sec | ETA 00:01:03
    2021-08-11 09:14:15 [INFO]	[TRAIN] epoch: 23, iter: 1340/2000, loss: 1.2506, lr: 0.003692, batch_cost: 0.0929, reader_cost: 0.00009, ips: 43.0404 samples/sec | ETA 00:01:01
    2021-08-11 09:14:17 [INFO]	[TRAIN] epoch: 24, iter: 1360/2000, loss: 1.2401, lr: 0.003591, batch_cost: 0.0947, reader_cost: 0.00405, ips: 42.2177 samples/sec | ETA 00:01:00
    2021-08-11 09:14:19 [INFO]	[TRAIN] epoch: 24, iter: 1380/2000, loss: 1.2395, lr: 0.003490, batch_cost: 0.0933, reader_cost: 0.00008, ips: 42.8629 samples/sec | ETA 00:00:57
    2021-08-11 09:14:21 [INFO]	[TRAIN] epoch: 24, iter: 1400/2000, loss: 1.2191, lr: 0.003389, batch_cost: 0.0921, reader_cost: 0.00007, ips: 43.4492 samples/sec | ETA 00:00:55
    2021-08-11 09:14:21 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    236/236 [==============================] - 6s 25ms/step - batch_cost: 0.0248 - reader cost: 1.1677e-
    2021-08-11 09:14:27 [INFO]	[EVAL] #Images: 236 mIoU: 0.8648 Acc: 0.9430 Kappa: 0.8527 
    2021-08-11 09:14:27 [INFO]	[EVAL] Class IoU: 
    [0.9256 0.8039]
    2021-08-11 09:14:27 [INFO]	[EVAL] Class Acc: 
    [0.9614 0.8913]
    2021-08-11 09:14:27 [INFO]	[EVAL] The model with the best validation mIoU (0.8648) was saved at iter 1400.
    2021-08-11 09:14:29 [INFO]	[TRAIN] epoch: 25, iter: 1420/2000, loss: 1.2588, lr: 0.003287, batch_cost: 0.0962, reader_cost: 0.00390, ips: 41.5634 samples/sec | ETA 00:00:55
    2021-08-11 09:14:31 [INFO]	[TRAIN] epoch: 25, iter: 1440/2000, loss: 1.2149, lr: 0.003185, batch_cost: 0.0955, reader_cost: 0.00007, ips: 41.8792 samples/sec | ETA 00:00:53
    2021-08-11 09:14:33 [INFO]	[TRAIN] epoch: 25, iter: 1460/2000, loss: 1.2589, lr: 0.003083, batch_cost: 0.0995, reader_cost: 0.00008, ips: 40.1990 samples/sec | ETA 00:00:53
    2021-08-11 09:14:35 [INFO]	[TRAIN] epoch: 26, iter: 1480/2000, loss: 1.1832, lr: 0.002980, batch_cost: 0.1021, reader_cost: 0.00441, ips: 39.1858 samples/sec | ETA 00:00:53
    2021-08-11 09:14:37 [INFO]	[TRAIN] epoch: 26, iter: 1500/2000, loss: 1.2638, lr: 0.002877, batch_cost: 0.0950, reader_cost: 0.00008, ips: 42.0985 samples/sec | ETA 00:00:47
    2021-08-11 09:14:39 [INFO]	[TRAIN] epoch: 26, iter: 1520/2000, loss: 1.2912, lr: 0.002773, batch_cost: 0.0937, reader_cost: 0.00007, ips: 42.7002 samples/sec | ETA 00:00:44
    2021-08-11 09:14:41 [INFO]	[TRAIN] epoch: 27, iter: 1540/2000, loss: 1.2144, lr: 0.002669, batch_cost: 0.0959, reader_cost: 0.00463, ips: 41.7015 samples/sec | ETA 00:00:44
    2021-08-11 09:14:42 [INFO]	[TRAIN] epoch: 27, iter: 1560/2000, loss: 1.2511, lr: 0.002565, batch_cost: 0.0921, reader_cost: 0.00006, ips: 43.4114 samples/sec | ETA 00:00:40
    2021-08-11 09:14:44 [INFO]	[TRAIN] epoch: 27, iter: 1580/2000, loss: 1.2007, lr: 0.002460, batch_cost: 0.0925, reader_cost: 0.00007, ips: 43.2318 samples/sec | ETA 00:00:38
    2021-08-11 09:14:46 [INFO]	[TRAIN] epoch: 28, iter: 1600/2000, loss: 1.2285, lr: 0.002355, batch_cost: 0.0956, reader_cost: 0.00426, ips: 41.8393 samples/sec | ETA 00:00:38
    2021-08-11 09:14:46 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    236/236 [==============================] - 6s 24ms/step - batch_cost: 0.0240 - reader cost: 1.1264e-
    2021-08-11 09:14:52 [INFO]	[EVAL] #Images: 236 mIoU: 0.8634 Acc: 0.9436 Kappa: 0.8509 
    2021-08-11 09:14:52 [INFO]	[EVAL] Class IoU: 
    [0.9273 0.7995]
    2021-08-11 09:14:52 [INFO]	[EVAL] Class Acc: 
    [0.9506 0.9219]
    2021-08-11 09:14:52 [INFO]	[EVAL] The model with the best validation mIoU (0.8648) was saved at iter 1400.
    2021-08-11 09:14:54 [INFO]	[TRAIN] epoch: 28, iter: 1620/2000, loss: 1.2271, lr: 0.002249, batch_cost: 0.0913, reader_cost: 0.00006, ips: 43.7932 samples/sec | ETA 00:00:34
    2021-08-11 09:14:56 [INFO]	[TRAIN] epoch: 28, iter: 1640/2000, loss: 1.1427, lr: 0.002142, batch_cost: 0.0916, reader_cost: 0.00007, ips: 43.6748 samples/sec | ETA 00:00:32
    2021-08-11 09:14:58 [INFO]	[TRAIN] epoch: 29, iter: 1660/2000, loss: 1.2257, lr: 0.002035, batch_cost: 0.0954, reader_cost: 0.00366, ips: 41.9280 samples/sec | ETA 00:00:32
    2021-08-11 09:15:00 [INFO]	[TRAIN] epoch: 29, iter: 1680/2000, loss: 1.2294, lr: 0.001927, batch_cost: 0.0937, reader_cost: 0.00009, ips: 42.6920 samples/sec | ETA 00:00:29
    2021-08-11 09:15:01 [INFO]	[TRAIN] epoch: 29, iter: 1700/2000, loss: 1.2177, lr: 0.001819, batch_cost: 0.0929, reader_cost: 0.00008, ips: 43.0802 samples/sec | ETA 00:00:27
    2021-08-11 09:15:03 [INFO]	[TRAIN] epoch: 30, iter: 1720/2000, loss: 1.1561, lr: 0.001710, batch_cost: 0.0963, reader_cost: 0.00360, ips: 41.5232 samples/sec | ETA 00:00:26
    2021-08-11 09:15:05 [INFO]	[TRAIN] epoch: 30, iter: 1740/2000, loss: 1.2160, lr: 0.001600, batch_cost: 0.0936, reader_cost: 0.00008, ips: 42.7557 samples/sec | ETA 00:00:24
    2021-08-11 09:15:07 [INFO]	[TRAIN] epoch: 30, iter: 1760/2000, loss: 1.2292, lr: 0.001489, batch_cost: 0.0924, reader_cost: 0.00007, ips: 43.3072 samples/sec | ETA 00:00:22
    2021-08-11 09:15:09 [INFO]	[TRAIN] epoch: 31, iter: 1780/2000, loss: 1.1478, lr: 0.001377, batch_cost: 0.0957, reader_cost: 0.00377, ips: 41.8133 samples/sec | ETA 00:00:21
    2021-08-11 09:15:11 [INFO]	[TRAIN] epoch: 31, iter: 1800/2000, loss: 1.1793, lr: 0.001265, batch_cost: 0.0943, reader_cost: 0.00008, ips: 42.4384 samples/sec | ETA 00:00:18
    2021-08-11 09:15:11 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    236/236 [==============================] - 6s 24ms/step - batch_cost: 0.0243 - reader cost: 1.1464e-
    2021-08-11 09:15:17 [INFO]	[EVAL] #Images: 236 mIoU: 0.8685 Acc: 0.9451 Kappa: 0.8570 
    2021-08-11 09:15:17 [INFO]	[EVAL] Class IoU: 
    [0.9286 0.8084]
    2021-08-11 09:15:17 [INFO]	[EVAL] Class Acc: 
    [0.9589 0.905 ]
    2021-08-11 09:15:17 [INFO]	[EVAL] The model with the best validation mIoU (0.8685) was saved at iter 1800.
    2021-08-11 09:15:19 [INFO]	[TRAIN] epoch: 31, iter: 1820/2000, loss: 1.2455, lr: 0.001151, batch_cost: 0.0932, reader_cost: 0.00008, ips: 42.9133 samples/sec | ETA 00:00:16
    2021-08-11 09:15:21 [INFO]	[TRAIN] epoch: 32, iter: 1840/2000, loss: 1.1621, lr: 0.001036, batch_cost: 0.0965, reader_cost: 0.00449, ips: 41.4583 samples/sec | ETA 00:00:15
    2021-08-11 09:15:23 [INFO]	[TRAIN] epoch: 32, iter: 1860/2000, loss: 1.1983, lr: 0.000919, batch_cost: 0.0937, reader_cost: 0.00007, ips: 42.6673 samples/sec | ETA 00:00:13
    2021-08-11 09:15:24 [INFO]	[TRAIN] epoch: 32, iter: 1880/2000, loss: 1.1833, lr: 0.000801, batch_cost: 0.0917, reader_cost: 0.00007, ips: 43.6235 samples/sec | ETA 00:00:11
    2021-08-11 09:15:26 [INFO]	[TRAIN] epoch: 33, iter: 1900/2000, loss: 1.2284, lr: 0.000681, batch_cost: 0.0970, reader_cost: 0.00508, ips: 41.2294 samples/sec | ETA 00:00:09
    2021-08-11 09:15:28 [INFO]	[TRAIN] epoch: 33, iter: 1920/2000, loss: 1.1886, lr: 0.000558, batch_cost: 0.0918, reader_cost: 0.00007, ips: 43.5951 samples/sec | ETA 00:00:07
    2021-08-11 09:15:30 [INFO]	[TRAIN] epoch: 33, iter: 1940/2000, loss: 1.1939, lr: 0.000432, batch_cost: 0.0919, reader_cost: 0.00006, ips: 43.5362 samples/sec | ETA 00:00:05
    2021-08-11 09:15:32 [INFO]	[TRAIN] epoch: 34, iter: 1960/2000, loss: 1.1295, lr: 0.000302, batch_cost: 0.0960, reader_cost: 0.00425, ips: 41.6742 samples/sec | ETA 00:00:03
    2021-08-11 09:15:34 [INFO]	[TRAIN] epoch: 34, iter: 1980/2000, loss: 1.1910, lr: 0.000166, batch_cost: 0.0916, reader_cost: 0.00006, ips: 43.6826 samples/sec | ETA 00:00:01
    2021-08-11 09:15:36 [INFO]	[TRAIN] epoch: 34, iter: 2000/2000, loss: 1.1710, lr: 0.000011, batch_cost: 0.0911, reader_cost: 0.00006, ips: 43.8900 samples/sec | ETA 00:00:00
    2021-08-11 09:15:36 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    236/236 [==============================] - 6s 24ms/step - batch_cost: 0.0242 - reader cost: 1.1701e-
    2021-08-11 09:15:41 [INFO]	[EVAL] #Images: 236 mIoU: 0.8691 Acc: 0.9457 Kappa: 0.8578 
    2021-08-11 09:15:41 [INFO]	[EVAL] Class IoU: 
    [0.9295 0.8088]
    2021-08-11 09:15:41 [INFO]	[EVAL] Class Acc: 
    [0.9568 0.9125]
    2021-08-11 09:15:42 [INFO]	[EVAL] The model with the best validation mIoU (0.8691) was saved at iter 2000.
    <class 'paddle.nn.layer.conv.Conv2D'>'s flops has been counted
    Customize Function has been applied to <class 'paddle.nn.layer.norm.SyncBatchNorm'>
    Cannot find suitable count function for <class 'paddle.nn.layer.pooling.MaxPool2D'>. Treat it as zero FLOPs.
    <class 'paddle.nn.layer.pooling.AdaptiveAvgPool2D'>'s flops has been counted
    <class 'paddle.nn.layer.pooling.AvgPool2D'>'s flops has been counted
    Cannot find suitable count function for <class 'paddle.nn.layer.activation.Sigmoid'>. Treat it as zero FLOPs.
    <class 'paddle.nn.layer.common.Dropout'>'s flops has been counted
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:143: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.FP32, but right dtype is VarType.INT32, the right dtype will convert to VarType.FP32
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    Total Flops: 8061050880     Total Params: 2328346



```python
# 单独进行评估 -- 上边do_eval就是这个工作
!python PaddleSeg/val.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--model_path PaddleSeg/output/best_model/model.pdparams
# model_path： 模型参数路径
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    2021-08-11 09:16:48 [INFO]	
    ---------------Config Information---------------
    batch_size: 4
    iters: 1000
    loss:
      coef:
      - 1
      - 1
      - 1
      - 1
      - 1
      types:
      - type: CrossEntropyLoss
    lr_scheduler:
      end_lr: 0
      learning_rate: 0.01
      power: 0.9
      type: PolynomialDecay
    model:
      pretrained: null
      type: BiSeNetV2
    optimizer:
      momentum: 0.9
      type: sgd
      weight_decay: 4.0e-05
    train_dataset:
      dataset_root: segDataset/horse
      mode: train
      num_classes: 2
      train_path: segDataset/horse/train_list.txt
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: RandomHorizontalFlip
      - type: Normalize
      type: Dataset
    val_dataset:
      dataset_root: segDataset/horse
      mode: val
      num_classes: 2
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: Normalize
      type: Dataset
      val_path: segDataset/horse/train_list.txt
    ------------------------------------------------
    W0811 09:16:48.046437   807 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0811 09:16:48.046486   807 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    2021-08-11 09:16:52 [INFO]	Loading pretrained model from PaddleSeg/output/best_model/model.pdparams
    2021-08-11 09:16:52 [INFO]	There are 356/356 variables loaded into BiSeNetV2.
    2021-08-11 09:16:52 [INFO]	Loaded trained params of model successfully
    2021-08-11 09:16:52 [INFO]	Start evaluating (total_samples: 236, total_iters: 236)...
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.INT32, but right dtype is VarType.BOOL, the right dtype will convert to VarType.INT32
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.INT64, but right dtype is VarType.BOOL, the right dtype will convert to VarType.INT64
      format(lhs_dtype, rhs_dtype, lhs_dtype))
    236/236 [==============================] - 6s 24ms/step - batch_cost: 0.0236 - reader cost: 6.9181e-
    2021-08-11 09:16:58 [INFO]	[EVAL] #Images: 236 mIoU: 0.8691 Acc: 0.9457 Kappa: 0.8578 
    2021-08-11 09:16:58 [INFO]	[EVAL] Class IoU: 
    [0.9295 0.8088]
    2021-08-11 09:16:58 [INFO]	[EVAL] Class Acc: 
    [0.9568 0.9125]


- 5.开始预测


```python
# 进行预测
!python PaddleSeg/predict.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--model_path PaddleSeg/output/best_model/model.pdparams\
--image_path segDataset/horse/Images\
--save_dir PaddleSeg/output/horse
# image_path: 预测图片路径/文件夹 -- 这里直接对训练数据进行预测，得到预测结果
# save_dir： 保存预测结果的路径 -- 保存的预测结果为图片
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    2021-08-11 09:17:09 [INFO]	
    ---------------Config Information---------------
    batch_size: 4
    iters: 1000
    loss:
      coef:
      - 1
      - 1
      - 1
      - 1
      - 1
      types:
      - type: CrossEntropyLoss
    lr_scheduler:
      end_lr: 0
      learning_rate: 0.01
      power: 0.9
      type: PolynomialDecay
    model:
      pretrained: null
      type: BiSeNetV2
    optimizer:
      momentum: 0.9
      type: sgd
      weight_decay: 4.0e-05
    train_dataset:
      dataset_root: segDataset/horse
      mode: train
      num_classes: 2
      train_path: segDataset/horse/train_list.txt
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: RandomHorizontalFlip
      - type: Normalize
      type: Dataset
    val_dataset:
      dataset_root: segDataset/horse
      mode: val
      num_classes: 2
      transforms:
      - target_size:
        - 512
        - 512
        type: Resize
      - type: Normalize
      type: Dataset
      val_path: segDataset/horse/train_list.txt
    ------------------------------------------------
    W0811 09:17:09.815398   901 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0811 09:17:09.815444   901 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    2021-08-11 09:17:14 [INFO]	Number of predict images = 328
    2021-08-11 09:17:14 [INFO]	Loading pretrained model from PaddleSeg/output/best_model/model.pdparams
    2021-08-11 09:17:14 [INFO]	There are 356/356 variables loaded into BiSeNetV2.
    2021-08-11 09:17:14 [INFO]	Start to predict...
    328/328 [==============================] - 16s 48ms/ste


# 五、可视化预测结果

通过PaddleSeg预测输出的结果为图片，对应位于:PaddleSeg/output/horse

其中包含两种结果：

- 一种为掩膜图像，即叠加预测结果与原始结果的图像 -- 位于: **PaddleSeg/output/horse/added_prediction**
- 另一种为预测结果的伪彩色图像，即预测的结果图像 -- 位于: **PaddleSeg/output/horse/pseudo_color_prediction**


```python
# 查看预测结果文件夹分布
!tree PaddleSeg/output/horse -L 1
```

    PaddleSeg/output/horse
    ├── added_prediction
    └── pseudo_color_prediction
    
    2 directories, 0 files


![](https://ai-studio-static-online.cdn.bcebos.com/0a6c081855804cada54a868aa5229e62ffbc6c0296d34e2fbfb3be7b8d88675a)
![](https://ai-studio-static-online.cdn.bcebos.com/2b6eb5f33d9a48c18df5c8c063df796519f4f339fb3947358a226dd04d8004c9)


分别展示两个文件夹中的预测结果(下载每个预测结果文件夹中的一两张图片到本地，然后上传notebook)

上传说明:

![](https://ai-studio-static-online.cdn.bcebos.com/29cb48e1263a4ea49557a8564f289be1690e1b23dab7412388420f9c244f366c)

> 以下为展示结果

<font color='red' size=5> ---数据集 horse 的预测结果展示--- </font>

<font color='black' size=5> 掩膜图像： </font>

![](https://ai-studio-static-online.cdn.bcebos.com/5fcc723d07be43059a347c2c1183ec113a033133ba924222886caf6b596876f8)

<font color='black' size=5> 伪彩色图像： </font>

![](https://ai-studio-static-online.cdn.bcebos.com/f206c673a96445928e35c87d325a6d0a31279f6196d342cfa4130ff6fc750747)


<font color='red' size=5>特别声明，使用horse数据集进行提交时，预测结果展示不允许使用horse242.jpg和horse242.png的预测结果，否则将可能认定为未进行本baseline作业的训练、预测过程 </font>

# 六、提交作业流程

1. 生成项目版本

![](https://ai-studio-static-online.cdn.bcebos.com/1c19ac6cfb314353b5377421c74bc5d777dcb5724fad47c1a096df758198f625)

2. (有能力的同学可以多捣鼓一下)根据需要可将notebook转为markdown，自行提交到github上

![](https://ai-studio-static-online.cdn.bcebos.com/0363ab3eb0da4242844cc8b918d38588bb17af73c2e14ecd92831b89db8e1c46)

3. (一定要先生成项目版本哦)公开项目

![](https://ai-studio-static-online.cdn.bcebos.com/8a6a2352f11c4f3e967bdd061f881efc5106827c55c445eabb060b882bf6c827)

