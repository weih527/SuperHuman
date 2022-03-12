# Superhuman accuracy on the SNEMI3D connectomics challenge



## Introduction

This repository is the **reproduced implementation** of the paper, "Superhuman accuracy on the SNEMI3D connectomics challenge".



## Installation

This code was tested with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4 and Ubuntu 16.04. It is worth mentioning that, besides some commonly used image processing packages, you also need to install some special post-processing packages for neuron segmentation, such as [waterz](https://github.com/funkey/waterz) and [elf](https://github.com/constantinpape/elf).

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows,

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/renwu527/auto-emseg:v5.4
```

or

```shell
docker pull renwu527/auto-emseg:v5.4
```



## Dataset

| Datasets                                                     | Training set       | Validation set     | Test set            | Download (Processed)                                         |
| ------------------------------------------------------------ | ------------------ | ------------------ | ------------------- | ------------------------------------------------------------ |
| [AC3/AC4](https://software.rc.fas.harvard.edu/lichtman/vast/<br/>AC3AC4Package.zip) | 1024x1024x80 (AC4) | 1024x1024x20 (AC4) | 1024x1024x100 (AC3) | [BaiduYun](https://pan.baidu.com/s/1rY6MlALpzvkYTgn04qghjQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1IsPmaBjDXkSyzPXKjB4GIwHb_5pVVXBe?usp=sharing) |

Download and unzip them in corresponding folders in './data'.



## Model Zoo

| Datasets                                                     | Models           | Download                                                     |
| ------------------------------------------------------------ | ---------------- | ------------------------------------------------------------ |
| [AC3/AC4](https://software.rc.fas.harvard.edu/lichtman/vast/<br/>AC3AC4Package.zip) | ac3ac4-test.ckpt | [BaiduYun](https://pan.baidu.com/s/1LB9xsVHjWDlSQ5nZ6kkzow) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1JmiHr6DVDG9RFc_jvSKF-uMTVA_bUg4y?usp=sharing) |



## Training and Inference

### 1. Training

```shell
python main.py -c=seg_3d
```

### 2. Inference

```shell
python inference.py -c=seg_3d -mn=ac3ac4 -id=ac3ac4-test -m=ac3
```

Output:

waterz: voi_split=1.095144, voi_merge=0.342404, voi_sum=1.437549, arand=0.168990

LMC: voi_split=1.144543, voi_merge=0.262998, voi_sum=1.407541, arand=0.122037



## Contact

If you have any problem with the released code, please do not hesitate to contact me by email (weih527@mail.ustc.edu.cn).

