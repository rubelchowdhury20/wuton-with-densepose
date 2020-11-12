## End-to-End Learning of Geometric Deformations of Feature Maps for Virtual Try-On

Pytorch implementation of **WUTON: a Warping U-net for a Virtual Try-On system** using an agnostic person representation along with it's densepose information.

[Paper](https://arxiv.org/pdf/1906.01347.pdf)

#### Output Results
![image 1](https://i.imgur.com/G7JUFa8.jpg)
![image_2](https://i.imgur.com/4ame6wW.jpg)
![image_3](https://i.imgur.com/SuEGces.jpg)

* The first one is the appaerl image cut out of the person image.
* The second one is the product image of the apparel after geometric transformation based on predicted TPS transformation parameters.
* The third one is the person image.
* The fourth one is the person image reconstructed from an agnostic image, given the apparel product image.
* The sixth one is of the same person wearing a differnt apparel altogether which is the fifth image.


#### Architecture of the Network
![archetecture](https://i.imgur.com/MpEX6ZH.png)

#### Prerequisites
* Linux
* Python3, PyTorch
* NVIDIA GPU (8G memory or larger) + CUDA cuDNN

#### Dataset
We have used Person Image along with it's apparel image scapred from zalando website. Along with Person-Apparel pair we have used the segmentation information and desnepose information of the person to train our model.
