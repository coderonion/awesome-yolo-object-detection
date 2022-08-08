# Awesome-YOLO-Object-Detection
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

🔥🔥🔥 YOLO is a great real-time one-stage object detection framework. This repository lists some awesome public YOLO object detection series projects.

## Contents
- [Awesome-YOLO-Object-Detection](#awesome-yolo-object-detection)
  - [Summary](#summary)
    - [Official YOLO](#official-yolo)
    - [Awesome List](#awesome-list)
    - [Paper and Code Overview](#paper-and-code-overview)
      - [Paper Review](#paper-review)
      - [Code Review](#code-review)
    - [Blogs](#blogs)
  - [Other Versions of YOLO](#other-versions-of-yolo)
    - [PyTorch Implementation](#pytorch-implementation)
    - [Tensorflow and Keras Implementation](#tensorflow-and-keras-implementation)
    - [PaddlePaddle Implementation](#paddlepaddle-implementation)
    - [Caffe Implementation](#caffe-implementation)
    - [MXNet Implementation](#mxnet-implementation)
    - [LibTorch Implementation](#libtorch-implementation)
    - [OpenCV Implementation](#opencv-implementation)
    - [ROS Implementation](#ros-implementation)
    - [Dotnet Implementation](#dotnet-implementation)
    - [Rust Implementation](#rust-implementation)
    - [Go Implementation](#go-implementation)
    - [Web Implementation](#web-implementation)
    - [Others](#others)
  - [Extensional Frameworks](#extensional-frameworks)
  - [Applications](#applications)
    - [Lighter and Faster](#lighter-and-faster)
      - [Lightweight Backbones and FPN (轻量级骨干网络和特征金字塔网络)](#lightweight-backbones-and-fpn)
      - [Pruning Knoweldge-Distillation Quantization (剪枝 知识蒸馏 量化)](#pruning-knoweldge-distillation-quantization)
      - [High-performance Inference Engine (高性能推理引擎)](#high-performance-inference-engine)
      - [FPGA TPU NPU Hardware Deployment (FPGA TPU NPU 硬件部署)](#fpga-tpu-npu-hardware-deployment)
    - [Video Object Detection (视频目标检测)](#video-object-detection)
    - [Object Tracking (目标跟踪)](#object-tracking)
      - [Multi-Object Tracking (多目标跟踪)](#multi-object-tracking)
    - [Deep Reinforcement Learning (深度强化学习)](#deep-reinforcement-learning)
    - [Multi-Modality Information Fusion (多模态信息融合)](#multi-modality-information-fusion)
    - [Motion Control Field (运动控制领域)](#motion-control-field)
    - [Super-Resolution Field (超分辨率领域)](#super-resolution-field)
    - [Spiking Neural Network (SNN, 脉冲神经网络)](#spiking-neural-network)
    - [Attention and Transformer (注意力机制)](#attention-and-transformer)
    - [Small Object Detection (小目标检测)](#small-object-detection)
    - [Few-shot Object Detection (少样本目标检测)](#few-shot-object-detection)
    - [Oriented Object Detection (旋转目标检测)](#oriented-object-detection)
    - [Face Detection and Recognition (人脸检测与识别)](#face-detection-and-recognition)
    - [Face Mask Detection (口罩检测)](#face-mask-detection)
    - [Social Distance Detection (社交距离检测)](#social-distance-detection)
    - [Intelligent Transportation Field Detection (智能交通领域检测)](#intelligent-transportation-field-detection)
      - [Vehicle Detection (车辆检测)](#vehicle-detection)
      - [License Plate Detection and Recognition (车牌检测)](#license-plate-detection-and-recognition)
      - [Lane Detection (车道线检测)](#lane-detection)
      - [Driving Behavior Detection (驾驶行为检测)](#driving-behavior-detection)
      - [Parking Slot Detection (停车位检测)](#parking-slot-detection)
      - [Traffic Light Detection (交通灯检测)](#traffic-light-detection)
      - [Traffic Sign Detection (交通标志检测)](#traffic-sign-detection)
      - [Crosswalk Detection (人行横道/斑马线检测)](#crosswalk-detection)
      - [Traffic Accidents Detection (交通事故检测)](#traffic-accidents-detection)
      - [Road Damage Detection (道路损伤检测)](#road-damage-detection)
    - [Helmet Detection (头盔/安全帽检测)](#helmet-detection)
    - [Hand Detection (手部检测)](#hand-detection)
    - [Gesture Recognition (手势/手语识别)](#gesture-recognition)
    - [Action Detection (行为检测)](#action-detection)
    - [Emotion Recognition (情感识别)](#emotion-recognition)
    - [Human Pose Estimation (人体姿态估计)](#human-pose-estimation)
    - [Distance Measurement (距离测量)](#distance-measurement)
    - [3D Object Detection (三维目标检测)](#3d-object-detection)
    - [SLAM Field Detection (SLAM领域检测)](#slam-field-detection)
    - [Industrial Defect Detection (工业缺陷检测)](#industrial-defect-detection)
    - [SAR Image Detection (合成孔径雷达图像检测)](#sar-image-detection)
    - [Safety Monitoring Field Detection (安防监控领域检测)](#safety-monitoring-field-detection)
    - [Medical Field Detection (医学领域检测)](#medical-field-detection)
    - [Chemistry Field Detection (化学领域检测)](#chemistry-field-detection)
    - [Agricultural Field Detection (农业领域检测)](#agricultural-field-detection)
    - [Adverse Weather Conditions (恶劣天气情况)](#adverse-weather-conditions)
    - [Adversarial Attack and Defense (对抗攻击与防御)](#adversarial-attack-and-defense)
    - [Instance and Semantic Segmentation (实例和语义分割)](#instance-and-semantic-segmentation)
    - [Game Field Detection (游戏领域检测)](#game-field-detection)
    - [Automatic Annotation Tool (自动标注工具)](#automatic-annotation-tool)
    - [Feature Map Visualization (特征图可视化)](#feature-map-visualization)
    - [Object Detection Evaluation Metrics (目标检测性能评价指标)](#object-detection-evaluation-metrics)
    - [GUI (图形用户界面)](#gui)
    - [Other Applications](#other-applications)






## Summary

  - ### Official YOLO

    - [YOLO](https://pjreddie.com/darknet/yolov1) ([Darknet](https://github.com/pjreddie/darknet) <img src="https://img.shields.io/github/stars/pjreddie/darknet?style=social"/>) : "You Only Look Once: Unified, Real-Time Object Detection". (**[CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)**)

    - [YOLOv2](https://pjreddie.com/darknet/yolov2) ([Darknet](https://github.com/pjreddie/darknet) <img src="https://img.shields.io/github/stars/pjreddie/darknet?style=social"/>) : "YOLO9000: Better, Faster, Stronger". (**[CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)**)

    - [YOLOv3](https://pjreddie.com/darknet/yolo) ([Darknet](https://github.com/pjreddie/darknet) <img src="https://img.shields.io/github/stars/pjreddie/darknet?style=social"/>) : "YOLOv3: An Incremental Improvement". (**[arXiv 2018](https://arxiv.org/abs/1804.02767)**)

    - [YOLOv4](https://github.com/AlexeyAB/darknet) <img src="https://img.shields.io/github/stars/AlexeyAB/darknet?style=social"/> ([WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4) <img src="https://img.shields.io/github/stars/WongKinYiu/PyTorch_YOLOv4?style=social"/>) : "YOLOv4: Optimal Speed and Accuracy of Object Detection". (**[arXiv 2020](https://arxiv.org/abs/2004.10934)**)

    - [Scaled-YOLOv4](https://github.com/AlexeyAB/darknet) <img src="https://img.shields.io/github/stars/AlexeyAB/darknet?style=social"/> ([WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4) <img src="https://img.shields.io/github/stars/WongKinYiu/ScaledYOLOv4?style=social"/>) : "Scaled-YOLOv4: Scaling Cross Stage Partial Network". (**[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html)**)

    - [YOLOv5](https://github.com/ultralytics/yolov5) <img src="https://img.shields.io/github/stars/ultralytics/yolov5?style=social"/> : YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite.

    - [YOLOv6](https://github.com/meituan/YOLOv6) <img src="https://img.shields.io/github/stars/meituan/YOLOv6?style=social"/> : YOLOv6: a single-stage object detection framework dedicated to industrial application. "微信公众号「美团技术团队」《[YOLOv6：又快又准的目标检测框架开源啦](https://mp.weixin.qq.com/s/RrQCP4pTSwpTmSgvly9evg)》"

    - [YOLOv7](https://github.com/WongKinYiu/yolov7) <img src="https://img.shields.io/github/stars/WongKinYiu/yolov7?style=social"/> : "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors". (**[arXiv 2022](https://arxiv.org/abs/2207.02696)**)


  - ### Awesome List

    - [awesome-yolo-object-detection](https://github.com/dotnet-py/awesome-yolo-object-detection) <img src="https://img.shields.io/github/stars/dotnet-py/awesome-yolo-object-detection?style=social"/> : 🔥🔥🔥 A collection of some awesome YOLO object detection series projects.  

    - [srebroa/awesome-yolo](https://github.com/srebroa/awesome-yolo) <img src="https://img.shields.io/github/stars/srebroa/awesome-yolo?style=social"/> : 🚀 ⭐ The list of the most popular YOLO algorithms - awesome YOLO. 


    - [Bubble-water/YOLO-Summary](https://github.com/Bubble-water/YOLO-Summary) <img src="https://img.shields.io/github/stars/Bubble-water/YOLO-Summary?style=social"/> : YOLO-Summary. 

    - [hoya012/deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection) <img src="https://img.shields.io/github/stars/hoya012/deep_learning_object_detection?style=social"/> : A paper list of object detection using deep learning.  

    - [WZMIAOMIAO/deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing) <img src="https://img.shields.io/github/stars/WZMIAOMIAO/deep-learning-for-image-processing?style=social"/> : deep learning for image processing including classification and object-detection etc. 

    - [amusi/awesome-object-detection](https://github.com/amusi/awesome-object-detection) <img src="https://img.shields.io/github/stars/amusi/awesome-object-detection?style=social"/> : Awesome Object Detection.
  

  - ### Paper and Code Overview

    - #### Paper Review

      - [GreenTeaHua/YOLO-Review](https://github.com/GreenTeaHua/YOLO-Review) <img src="https://img.shields.io/github/stars/GreenTeaHua/YOLO-Review?style=social"/> : "A Review of YOLO Object Detection Based on Deep Learning". "基于深度学习的YOLO目标检测综述". (**[Journal of Electronics & Information Technology 2022](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT210790)**)

      - "A Review of Yolo Algorithm Developments". (**[Procedia Computer Science 2022](https://www.sciencedirect.com/science/article/pii/S1877050922001363)**)


    - #### Code Review

      - [jizhishutong/YOLOU](https://github.com/jizhishutong/YOLOU) <img src="https://img.shields.io/github/stars/jizhishutong/YOLOU?style=social"/> : YOLOU：United, Study and easier to Deploy. The purpose of our creation of YOLOU is to better learn the algorithms of the YOLO series and pay tribute to our predecessors. YOLOv3、YOLOv4、YOLOv5、YOLOv5-Lite、YOLOv6、YOLOv7、YOLOX、YOLOX-Lite、TensorRT、NCNN、Tengine、OpenVINO. "微信公众号「集智书童」《[YOLOU开源 | 汇集YOLO系列所有算法，集算法学习、科研改进、落地于一身！](https://mp.weixin.qq.com/s/clupheQ8iHnhR4FJcTtB8A)》"


  - ### Blogs

    - [知乎「江大白」/ 微信公众号「江大白」](https://www.zhihu.com/people/nan-yang-8-13)
      - [深入浅出0基础入门AI及目标检测详细学习路径](https://zhuanlan.zhihu.com/p/463221190)
      - [深入浅出Yolo系列之Yolov3&Yolov4&Yolov5&Yolox核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/143747206)    
      - [深入浅出Yolo系列之Yolov5核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/172121380)   
      - [深入浅出Yolov5之自有数据集训练超详细教程](https://zhuanlan.zhihu.com/p/463176500)
      - [深入浅出Yolo系列之Yolox核心基础完整讲解](https://zhuanlan.zhihu.com/p/397993315)
      - [深入浅出Yolox之自有数据集训练超详细教程](https://zhuanlan.zhihu.com/p/397499216)
    - 微信公众号「所向披靡的张大刀」
      - [【小白入坑篇】目标检测的评价指标map](https://mp.weixin.qq.com/s/q308cHT0XliCK3NtIRjyqA)
      - [【yolov6系列】细节拆解网络框架](https://mp.weixin.qq.com/s/DFSROue8InARk-96I_Kptg)
      - [【yolov7系列】网络框架细节拆解](https://mp.weixin.qq.com/s/VEcUIaDrhc1ETIPr39l4rg)
      - [【yolov7系列三】实战从0构建训练自己的数据集](https://mp.weixin.qq.com/s/S80mMimu4YpHwClHIH07eA)
    - 微信公众号「集智书童」
      - [YOLOv7官方开源 | Alexey Bochkovskiy站台，精度速度超越所有YOLO，还得是AB](https://mp.weixin.qq.com/s/5SeD09vG6nv46-YuN_uU1w)
      - [YOLOU开源 | 汇集YOLO系列所有算法，集算法学习、科研改进、落地于一身！](https://mp.weixin.qq.com/s/clupheQ8iHnhR4FJcTtB8A)
    - 微信公众号「WeThinkln」
      - [【Make YOLO Great Again】YOLOv1-v7全系列大解析（Neck篇）](https://mp.weixin.qq.com/s/nEWL9ZAYuVngoejf-muFRw)


## Other Versions of YOLO

  - ### PyTorch Implementation

    - [MMDetection](https://github.com/open-mmlab/mmdetection) <img src="https://img.shields.io/github/stars/open-mmlab/mmdetection?style=social"/> : OpenMMLab Detection Toolbox and Benchmark. (**[arXiv 2019](https://arxiv.org/abs/1906.07155)**)

    - [ultralytics/yolov3](https://github.com/ultralytics/yolov3) <img src="https://img.shields.io/github/stars/ultralytics/yolov3?style=social"/> : YOLOv3 in PyTorch > ONNX > CoreML > TFLite.

    - [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) <img src="https://img.shields.io/github/stars/eriklindernoren/PyTorch-YOLOv3?style=social"/> : Minimal PyTorch implementation of YOLOv3.

    - [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) <img src="https://img.shields.io/github/stars/Tianxiaomo/pytorch-YOLOv4?style=social"/> : PyTorch ,ONNX and TensorRT implementation of YOLOv4.

    - [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) <img src="https://img.shields.io/github/stars/ayooshkathuria/pytorch-yolo-v3?style=social"/> : A PyTorch implementation of the YOLO v3 object detection algorithm.

    - [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4) <img src="https://img.shields.io/github/stars/WongKinYiu/PyTorch_YOLOv4?style=social"/> : PyTorch implementation of YOLOv4.

    - [argusswift/YOLOv4-pytorch](https://github.com/argusswift/YOLOv4-pytorch) <img src="https://img.shields.io/github/stars/argusswift/YOLOv4-pytorch?style=social"/> : This is a pytorch repository of YOLOv4, attentive YOLOv4 and mobilenet YOLOv4 with PASCAL VOC and COCO.

    - [longcw/yolo2-pytorch](https://github.com/longcw/yolo2-pytorch) <img src="https://img.shields.io/github/stars/longcw/yolo2-pytorch?style=social"/> : YOLOv2 in PyTorch.

    - [bubbliiiing/yolov5-v6.1-pytorch](https://github.com/bubbliiiing/yolov5-v6.1-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov5-v6.1-pytorch?style=social"/> : 这是一个yolov5-v6.1-pytorch的源码，可以用于训练自己的模型。 

    - [bubbliiiing/yolov5-pytorch](https://github.com/bubbliiiing/yolov5-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov5-pytorch?style=social"/> : 这是一个YoloV5-pytorch的源码，可以用于训练自己的模型。

    - [bubbliiiing/yolov4-pytorch](https://github.com/bubbliiiing/yolov4-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov4-pytorch?style=social"/> : 这是一个YoloV4-pytorch的源码，可以用于训练自己的模型。

    - [bubbliiiing/yolov4-tiny-pytorch](https://github.com/bubbliiiing/yolov4-tiny-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov4-tiny-pytorch?style=social"/> : 这是一个YoloV4-tiny-pytorch的源码，可以用于训练自己的模型。

    - [bubbliiiing/yolov3-pytorch](https://github.com/bubbliiiing/yolo3-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolo3-pytorch?style=social"/> : 这是一个yolo3-pytorch的源码，可以用于训练自己的模型。

    - [bubbliiiing/yolox-pytorch](https://github.com/bubbliiiing/yolox-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolox-pytorch?style=social"/> : 这是一个yolox-pytorch的源码，可以用于训练自己的模型。

    - [bubbliiiing/yolov7-pytorch](https://github.com/bubbliiiing/yolov7-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov7-pytorch?style=social"/> : 这是一个yolov7的库，可以用于训练自己的数据集。 

    - [BobLiu20/YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch) <img src="https://img.shields.io/github/stars/BobLiu20/YOLOv3_PyTorch?style=social"/> : Full implementation of YOLOv3 in PyTorch.

    - [ruiminshen/yolo2-pytorch](https://github.com/ruiminshen/yolo2-pytorch) <img src="https://img.shields.io/github/stars/ruiminshen/yolo2-pytorch?style=social"/> : PyTorch implementation of the YOLO (You Only Look Once) v2.

    - [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3) <img src="https://img.shields.io/github/stars/DeNA/PyTorch_YOLOv3?style=social"/> : Implementation of YOLOv3 in PyTorch.

    - [abeardear/pytorch-YOLO-v1](https://github.com/abeardear/pytorch-YOLO-v1) <img src="https://img.shields.io/github/stars/abeardear/pytorch-YOLO-v1?style=social"/> : an experiment for yolo-v1, including training and testing.

    - [wuzhihao7788/yolodet-pytorch](https://github.com/wuzhihao7788/yolodet-pytorch) <img src="https://img.shields.io/github/stars/wuzhihao7788/yolodet-pytorch?style=social"/> : reproduce the YOLO series of papers in pytorch, including YOLOv4, PP-YOLO, YOLOv5，YOLOv3, etc.

    - [uvipen/Yolo-v2-pytorch](https://github.com/uvipen/Yolo-v2-pytorch) <img src="https://img.shields.io/github/stars/uvipen/Yolo-v2-pytorch?style=social"/> : YOLO for object detection tasks.

    - [Peterisfar/YOLOV3](https://github.com/Peterisfar/YOLOV3) <img src="https://img.shields.io/github/stars/Peterisfar/YOLOV3?style=social"/> : yolov3 by pytorch.

    - [misads/easy_detection](https://github.com/misads/easy_detection) <img src="https://img.shields.io/github/stars/misads/easy_detection?style=social"/> : 一个简单方便的目标检测框架(PyTorch环境可直接运行，不需要cuda编译)，支持Faster_RCNN、Yolo系列(v2~v5)、EfficientDet、RetinaNet、Cascade-RCNN等经典网络。

    - [miemiedetection](https://github.com/miemie2013/miemiedetection) <img src="https://img.shields.io/github/stars/miemie2013/miemiedetection?style=social"/> : Pytorch and ncnn implementation of PPYOLOE、YOLOX、PPYOLO、PPYOLOv2、SOLOv2 an so on. 


  - ### Tensorflow and Keras Implementation

    - [YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3) <img src="https://img.shields.io/github/stars/YunYang1994/tensorflow-yolov3?style=social"/> : 🔥 TensorFlow Code for technical report: "YOLOv3: An Incremental Improvement".

    - [zzh8829/yolov3-tf2](https://github.com/zzh8829/yolov3-tf2) <img src="https://img.shields.io/github/stars/zzh8829/yolov3-tf2?style=social"/> : YoloV3 Implemented in Tensorflow 2.0.

    - [hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite) <img src="https://img.shields.io/github/stars/hunglc007/tensorflow-yolov4-tflite?style=social"/> : YOLOv4, YOLOv4-tiny, YOLOv3, YOLOv3-tiny Implemented in Tensorflow 2.0, Android. Convert YOLO v4 .weights tensorflow, tensorrt and tflite.

    - [gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow) <img src="https://img.shields.io/github/stars/gliese581gg/YOLO_tensorflow?style=social"/> : tensorflow implementation of 'YOLO : Real-Time Object Detection'.

    - [llSourcell/YOLO_Object_Detection](https://github.com/llSourcell/YOLO_Object_Detection) <img src="https://img.shields.io/github/stars/llSourcell/YOLO_Object_Detection?style=social"/> : This is the code for "YOLO Object Detection" by Siraj Raval on Youtube.

    - [wizyoung/YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow) <img src="https://img.shields.io/github/stars/wizyoung/YOLOv3_TensorFlow?style=social"/> : Complete YOLO v3 TensorFlow implementation. Support training on your own dataset.

    - [theAIGuysCode/yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort) <img src="https://img.shields.io/github/stars/theAIGuysCode/yolov4-deepsort?style=social"/> : Object tracking implemented with YOLOv4, DeepSort, and TensorFlow.

    - [mystic123/tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3) <img src="https://img.shields.io/github/stars/mystic123/tensorflow-yolo-v3?style=social"/> : Implementation of YOLO v3 object detector in Tensorflow (TF-Slim).

    - [hizhangp/yolo_tensorflow](https://github.com/hizhangp/yolo_tensorflow) <img src="https://img.shields.io/github/stars/hizhangp/yolo_tensorflow?style=social"/> : Tensorflow implementation of YOLO, including training and test phase.

    - [nilboy/tensorflow-yolo](https://github.com/nilboy/tensorflow-yolo) <img src="https://img.shields.io/github/stars/nilboy/tensorflow-yolo?style=social"/> : tensorflow implementation of 'YOLO : Real-Time Object Detection'(train and test).

    - [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3) <img src="https://img.shields.io/github/stars/qqwweee/keras-yolo3?style=social"/> : A Keras implementation of YOLOv3 (Tensorflow backend).

    - [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K) <img src="https://img.shields.io/github/stars/allanzelener/YAD2K?style=social"/> : YAD2K: Yet Another Darknet 2 Keras.

    - [experiencor/keras-yolo2](https://github.com/experiencor/keras-yolo2) <img src="https://img.shields.io/github/stars/experiencor/keras-yolo2?style=social"/> : YOLOv2 in Keras and Applications.

    - [experiencor/keras-yolo3](https://github.com/experiencor/keras-yolo3) <img src="https://img.shields.io/github/stars/experiencor/keras-yolo3?style=social"/> : Training and Detecting Objects with YOLO3.

    - [alternativebug/yolo-tf2](https://github.com/alternativebug/yolo-tf2) <img src="https://img.shields.io/github/stars/alternativebug/yolo-tf2?style=social"/> : yolo(all versions) implementation in keras and tensorflow 2.x.

    - [SpikeKing/keras-yolo3-detection](https://github.com/SpikeKing/keras-yolo3-detection) <img src="https://img.shields.io/github/stars/SpikeKing/keras-yolo3-detection?style=social"/> : YOLO v3 物体检测算法。

    - [xiaochus/YOLOv3](https://github.com/xiaochus/YOLOv3) <img src="https://img.shields.io/github/stars/xiaochus/YOLOv3?style=social"/> : Keras implementation of yolo v3 object detection.

    - [bubbliiiing/yolo3-keras](https://github.com/bubbliiiing/yolo3-keras) <img src="https://img.shields.io/github/stars/bubbliiiing/yolo3-keras?style=social"/> : 这是一个yolo3-keras的源码，可以用于训练自己的模型。

    - [bubbliiiing/yolov4-keras](https://github.com/bubbliiiing/yolov4-keras) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov4-keras?style=social"/> : 这是一个YoloV4-keras的源码，可以用于训练自己的模型。

    - [bubbliiiing/yolov4-tf2](https://github.com/bubbliiiing/yolov4-tf2) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov4-tf2?style=social"/> : 这是一个yolo4-tf2（tensorflow2）的源码，可以用于训练自己的模型。

    - [bubbliiiing/yolov4-tiny-tf2](https://github.com/bubbliiiing/yolov4-tiny-tf2) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov4-tiny-tf2?style=social"/> : 这是一个YoloV4-tiny-tf2的源码，可以用于训练自己的模型。

    - [pythonlessons/TensorFlow-2.x-YOLOv3](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3) <img src="https://img.shields.io/github/stars/pythonlessons/TensorFlow-2.x-YOLOv3?style=social"/> : YOLOv3 implementation in TensorFlow 2.3.1.

    - [miemie2013/Keras-YOLOv4](https://github.com/miemie2013/Keras-YOLOv4) <img src="https://img.shields.io/github/stars/miemie2013/Keras-YOLOv4?style=social"/> : PPYOLO AND YOLOv4.

    - [Ma-Dan/keras-yolo4](https://github.com/Ma-Dan/keras-yolo4) <img src="https://img.shields.io/github/stars/Ma-Dan/keras-yolo4?style=social"/> : A Keras implementation of YOLOv4 (Tensorflow backend).

    - [miranthajayatilake/YOLOw-Keras](https://github.com/miranthajayatilake/YOLOw-Keras) <img src="https://img.shields.io/github/stars/miranthajayatilake/YOLOw-Keras?style=social"/> : YOLOv2 Object Detection w/ Keras (in just 20 lines of code).

    - [maiminh1996/YOLOv3-tensorflow](https://github.com/maiminh1996/YOLOv3-tensorflow) <img src="https://img.shields.io/github/stars/maiminh1996/YOLOv3-tensorflow?style=social"/> : Re-implement YOLOv3 with TensorFlow.

    - [Stick-To/Object-Detection-Tensorflow](https://github.com/Stick-To/Object-Detection-Tensorflow) <img src="https://img.shields.io/github/stars/Stick-To/Object-Detection-Tensorflow?style=social"/> : Object Detection API Tensorflow.

    - [avBuffer/Yolov5_tf](https://github.com/avBuffer/Yolov5_tf) <img src="https://img.shields.io/github/stars/avBuffer/Yolov5_tf?style=social"/> : Yolov5/Yolov4/ Yolov3/ Yolo_tiny in tensorflow.

    - [ruiminshen/yolo-tf](https://github.com/ruiminshen/yolo-tf) <img src="https://img.shields.io/github/stars/ruiminshen/yolo-tf?style=social"/> : TensorFlow implementation of the YOLO (You Only Look Once).

    - [xiao9616/yolo4_tensorflow2](https://github.com/xiao9616/yolo4_tensorflow2) <img src="https://img.shields.io/github/stars/xiao9616/yolo4_tensorflow2?style=social"/> : yolo 4th edition implemented by tensorflow2.0.

    - [sicara/tf2-yolov4](https://github.com/sicara/tf2-yolov4) <img src="https://img.shields.io/github/stars/sicara/tf2-yolov4?style=social"/> : A TensorFlow 2.0 implementation of YOLOv4: Optimal Speed and Accuracy of Object Detection.

    - [LongxingTan/Yolov5](https://github.com/LongxingTan/Yolov5) <img src="https://img.shields.io/github/stars/LongxingTan/Yolov5?style=social"/> : Efficient implementation of YOLOV5 in TensorFlow2.

    - [geekjr/quickai](https://github.com/geekjr/quickai) <img src="https://img.shields.io/github/stars/geekjr/quickai?style=social"/> : QuickAI is a Python library that makes it extremely easy to experiment with state-of-the-art Machine Learning models.

    - [CV_Lab/yolov5_rt_tfjs](https://gitee.com/CV_Lab/yolov5_rt_tfjs) : 🚀 基于TensorFlow.js的YOLOv5实时目标检测项目。

    - [Burf/TFDetection](https://github.com/Burf/TFDetection) <img src="https://img.shields.io/github/stars/Burf/TFDetection?style=social"/> : A Detection Toolbox for Tensorflow2.


  - ### PaddlePaddle Implementation

    - [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) <img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleDetection?style=social"/> : Object Detection toolkit based on PaddlePaddle. "PP-YOLO: An Effective and Efficient Implementation of Object Detector". (**[arXiv 2020](https://arxiv.org/abs/2007.12099)**)

    - [miemie2013/Paddle-YOLOv4](https://github.com/miemie2013/Paddle-YOLOv4) <img src="https://img.shields.io/github/stars/miemie2013/Paddle-YOLOv4?style=social"/> : Paddle-YOLOv4.

    - [Sharpiless/PaddleDetection-Yolov5](https://github.com/Sharpiless/PaddleDetection-Yolov5) <img src="https://img.shields.io/github/stars/Sharpiless/PaddleDetection-Yolov5?style=social"/> : 基于Paddlepaddle复现yolov5，支持PaddleDetection接口。

    - [nemonameless/PaddleDetection_YOLOv5](https://github.com/nemonameless/PaddleDetection_YOLOv5) <img src="https://img.shields.io/github/stars/nemonameless/PaddleDetection_YOLOv5?style=social"/> : YOLOv5 of PaddleDetection, Paddle implementation of YOLOv5.

    - [nemonameless/PaddleDetection_YOLOX](https://github.com/nemonameless/PaddleDetection_YOLOX) <img src="https://img.shields.io/github/stars/nemonameless/PaddleDetection_YOLOX?style=social"/> : Paddle YOLOX, 51.8% on COCO val by YOLOX-x, 44.6% on YOLOX-ConvNeXt-s.

    - [nemonameless/PaddleDetection_YOLOset](https://github.com/nemonameless/PaddleDetection_YOLOset) <img src="https://img.shields.io/github/stars/nemonameless/PaddleDetection_YOLOset?style=social"/> : Paddle YOLO set: YOLOv3, PPYOLO, PPYOLOE, YOLOX, YOLOv5, YOLOv7 and so on.

    - [Nioolek/PPYOLOE_pytorch](https://github.com/Nioolek/PPYOLOE_pytorch) <img src="https://img.shields.io/github/stars/Nioolek/PPYOLOE_pytorch?style=social"/> : An unofficial implementation of Pytorch version PP-YOLOE,based on Megvii YOLOX training code. 


  - ### Caffe Implementation

    - [ChenYingpeng/caffe-yolov3](https://github.com/ChenYingpeng/caffe-yolov3) <img src="https://img.shields.io/github/stars/ChenYingpeng/caffe-yolov3?style=social"/> : A real-time object detection framework of Yolov3/v4 based on caffe.

    - [ChenYingpeng/darknet2caffe](https://github.com/ChenYingpeng/darknet2caffe) <img src="https://img.shields.io/github/stars/ChenYingpeng/darknet2caffe?style=social"/> : Convert darknet weights to caffemodel.

    - [eric612/Caffe-YOLOv3-Windows](https://github.com/eric612/Caffe-YOLOv3-Windows) <img src="https://img.shields.io/github/stars/eric612/Caffe-YOLOv3-Windows?style=social"/> : A windows caffe implementation of YOLO detection network.

    - [Harick1/caffe-yolo](https://github.com/Harick1/caffe-yolo) <img src="https://img.shields.io/github/stars/Harick1/caffe-yolo?style=social"/> : Caffe for YOLO.

    - [choasup/caffe-yolo9000](https://github.com/choasup/caffe-yolo9000) <img src="https://img.shields.io/github/stars/choasup/caffe-yolo9000?style=social"/> : Caffe for YOLOv2 & YOLO9000.

    - [gklz1982/caffe-yolov2](https://github.com/gklz1982/caffe-yolov2) <img src="https://img.shields.io/github/stars/gklz1982/caffe-yolov2?style=social"/> : caffe-yolov2.


  - ### MXNet Implementation

    - [Gluon CV Toolkit](https://github.com/dmlc/gluon-cv) <img src="https://img.shields.io/github/stars/dmlc/gluon-cv?style=social"/> : GluonCV provides implementations of the state-of-the-art (SOTA) deep learning models in computer vision.

    - [zhreshold/mxnet-yolo](https://github.com/zhreshold/mxnet-yolo) <img src="https://img.shields.io/github/stars/zhreshold/mxnet-yolo?style=social"/> : YOLO: You only look once real-time object detector.


  - ### LibTorch Implementation

    - [walktree/libtorch-yolov3](https://github.com/walktree/libtorch-yolov3) <img src="https://img.shields.io/github/stars/walktree/libtorch-yolov3?style=social"/> : A Libtorch implementation of the YOLO v3 object detection algorithm, written with pure C++.

    - [yasenh/libtorch-yolov5](https://github.com/yasenh/libtorch-yolov5) <img src="https://img.shields.io/github/stars/yasenh/libtorch-yolov5?style=social"/> : A LibTorch inference implementation of the yolov5.

    - [Nebula4869/YOLOv5-LibTorch](https://github.com/Nebula4869/YOLOv5-LibTorch) <img src="https://img.shields.io/github/stars/Nebula4869/YOLOv5-LibTorch?style=social"/> : Real time object detection with deployment of YOLOv5 through LibTorch C++ API.

    - [ncdhz/YoloV5-LibTorch](https://github.com/ncdhz/YoloV5-LibTorch) <img src="https://img.shields.io/github/stars/ncdhz/YoloV5-LibTorch?style=social"/> : 一个 C++ 版本的 YoloV5 封装库.


  - ### OpenCV Implementation

    - [stephanecharette/DarkHelp](https://github.com/stephanecharette/DarkHelp) <img src="https://img.shields.io/github/stars/stephanecharette/DarkHelp?style=social"/> : The DarkHelp C++ API is a wrapper to make it easier to use the Darknet neural network framework within a C++ application.

    - [UNeedCryDear/yolov5-opencv-dnn-cpp](https://github.com/UNeedCryDear/yolov5-opencv-dnn-cpp) <img src="https://img.shields.io/github/stars/UNeedCryDear/yolov5-opencv-dnn-cpp?style=social"/> : 使用opencv模块部署yolov5-6.0版本。

    - [hpc203/yolov5-dnn-cpp-python](https://github.com/hpc203/yolov5-dnn-cpp-python) <img src="https://img.shields.io/github/stars/hpc203/yolov5-dnn-cpp-python?style=social"/> : 用opencv的dnn模块做yolov5目标检测，包含C++和Python两个版本的程序。

    - [hpc203/yolox-opencv-dnn](https://github.com/hpc203/yolox-opencv-dnn) <img src="https://img.shields.io/github/stars/hpc203/yolox-opencv-dnn?style=social"/> : 使用OpenCV部署YOLOX，支持YOLOX-S、YOLOX-M、YOLOX-L、YOLOX-X、YOLOX-Darknet53五种结构，包含C++和Python两种版本的程序。

    - [hpc203/yolov7-opencv-onnxrun-cpp-py](https://github.com/hpc203/yolov7-opencv-onnxrun-cpp-py) <img src="https://img.shields.io/github/stars/hpc203/yolov7-opencv-onnxrun-cpp-py?style=social"/> : 分别使用OpenCV、ONNXRuntime部署YOLOV7目标检测，一共包含12个onnx模型，依然是包含C++和Python两个版本的程序。


  - ### ROS Implementation

    - [leggedrobotics/darknet_ros](https://github.com/leggedrobotics/darknet_ros) <img src="https://img.shields.io/github/stars/leggedrobotics/darknet_ros?style=social"/> : Real-Time Object Detection for ROS.

    - [engcang/ros-yolo-sort](https://github.com/engcang/ros-yolo-sort) <img src="https://img.shields.io/github/stars/engcang/ros-yolo-sort?style=social"/> : YOLO and SORT, and ROS versions of them.

    - [chrisgundling/YoloLight](https://github.com/chrisgundling/YoloLight) <img src="https://img.shields.io/github/stars/chrisgundling/YoloLight?style=social"/> : Tiny-YOLO-v2 ROS Node for Traffic Light Detection.

    - [Ar-Ray-code/YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS) <img src="https://img.shields.io/github/stars/Ar-Ray-code/YOLOX-ROS?style=social"/> : YOLOX + ROS2 object detection package.

    - [Ar-Ray-code/YOLOv5-ROS](https://github.com/Ar-Ray-code/YOLOv5-ROS) <img src="https://img.shields.io/github/stars/Ar-Ray-code/YOLOv5-ROS?style=social"/> : YOLOv5 + ROS2 object detection package.

    - [Tossy0423/yolov4-for-darknet_ros](https://github.com/Tossy0423/yolov4-for-darknet_ros) <img src="https://img.shields.io/github/stars/Tossy0423/yolov4-for-darknet_ros?style=social"/> : This is the environment in which YOLO V4 is ported to darknet_ros.

    - [qianmin/yolov5_ROS](https://github.com/qianmin/yolov5_ROS) <img src="https://img.shields.io/github/stars/qianmin/yolov5_ROS?style=social"/> : run YOLOv5 in ROS，ROS使用YOLOv5。

    - [ailllist/yolov5_ROS](https://github.com/ailllist/yolov5_ROS) <img src="https://img.shields.io/github/stars/ailllist/yolov5_ROS?style=social"/> : yolov5 for ros, not webcam.

    - [Shua-Kang/ros_pytorch_yolov5](https://github.com/Shua-Kang/ros_pytorch_yolov5) <img src="https://img.shields.io/github/stars/Shua-Kang/ros_pytorch_yolov5?style=social"/> : A ROS wrapper for yolov5. (master branch is v5.0 of yolov5; for v6.1, see branch v6.1).

    - [ziyan0302/Yolov5_DeepSort_Pytorch_ros](https://github.com/ziyan0302/Yolov5_DeepSort_Pytorch_ros) <img src="https://img.shields.io/github/stars/ziyan0302/Yolov5_DeepSort_Pytorch_ros?style=social"/> : Connect Yolov5 detection module and DeepSort tracking module via ROS.

    - [U07157135/ROS2-with-YOLOv5](https://github.com/U07157135/ROS2-with-YOLOv5) <img src="https://img.shields.io/github/stars/U07157135/ROS2-with-YOLOv5?style=social"/> : 在無人機上以ROS2技術實現YOLOv5物件偵測。

    - [lukazso/yolov6-ros](https://github.com/lukazso/yolov6-ros) <img src="https://img.shields.io/github/stars/lukazso/yolov6-ros?style=social"/> : ROS package for YOLOv6.

    - [qq44642754a/Yolov5_ros](https://github.com/qq44642754a/Yolov5_ros) <img src="https://img.shields.io/github/stars/qq44642754a/Yolov5_ros?style=social"/> : Real-time object detection with ROS, based on YOLOv5 and PyTorch (基于 YOLOv5的ROS实时对象检测).

    - [lukazso/yolov7-ros](https://github.com/lukazso/yolov7-ros) <img src="https://img.shields.io/github/stars/lukazso/yolov7-ros?style=social"/> : ROS package for official YOLOv7.

    - [ConfusionTechnologies/ros-yolov5-node](https://github.com/ConfusionTechnologies/ros-yolov5-node) <img src="https://img.shields.io/github/stars/ConfusionTechnologies/ros-yolov5-node?style=social"/> : For ROS2, uses ONNX GPU Runtime to inference YOLOv5.


  - ### Dotnet Implementation

    - [ML.NET](https://github.com/dotnet/machinelearning) <img src="https://img.shields.io/github/stars/dotnet/machinelearning?style=social"/> : ML.NET is an open source and cross-platform machine learning framework for .NET. 

    - [TorchSharp](https://github.com/dotnet/TorchSharp) <img src="https://img.shields.io/github/stars/dotnet/TorchSharp?style=social"/> : A .NET library that provides access to the library that powers PyTorch.    

    - [harujoh/KelpNet](https://github.com/harujoh/KelpNet) <img src="https://img.shields.io/github/stars/harujoh/KelpNet?style=social"/> : Pure C# machine learning framework.

    - [Alturos.Yolo](https://github.com/AlturosDestinations/Alturos.Yolo) <img src="https://img.shields.io/github/stars/AlturosDestinations/Alturos.Yolo?style=social"/> : C# Yolo Darknet Wrapper (real-time object detection).

    - [mentalstack/yolov5-net](https://github.com/mentalstack/yolov5-net) <img src="https://img.shields.io/github/stars/mentalstack/yolov5-net?style=social"/> : YOLOv5 object detection with C#, ML.NET, ONNX.

    - [keijiro/TinyYOLOv2Barracuda](https://github.com/keijiro/TinyYOLOv2Barracuda) <img src="https://img.shields.io/github/stars/keijiro/TinyYOLOv2Barracuda?style=social"/> : Tiny YOLOv2 on Unity Barracuda.

    - [derenlei/Unity_Detection2AR](https://github.com/derenlei/Unity_Detection2AR) <img src="https://img.shields.io/github/stars/derenlei/Unity_Detection2AR?style=social"/> : Localize 2D image object detection in 3D Scene with Yolo in Unity Barracuda and ARFoundation.

    - [died/YOLO3-With-OpenCvSharp4](https://github.com/died/YOLO3-With-OpenCvSharp4) <img src="https://img.shields.io/github/stars/died/YOLO3-With-OpenCvSharp4?style=social"/> : Demo of implement YOLO v3 with OpenCvSharp v4 on C#.

    - [mbaske/yolo-unity](https://github.com/mbaske/yolo-unity) <img src="https://img.shields.io/github/stars/mbaske/yolo-unity?style=social"/> : YOLO In-Game Object Detection for Unity (Windows).

    - [BobLd/YOLOv4MLNet](https://github.com/BobLd/YOLOv4MLNet) <img src="https://img.shields.io/github/stars/BobLd/YOLOv4MLNet?style=social"/> : Use the YOLO v4 and v5 (ONNX) models for object detection in C# using ML.Net.

    - [keijiro/YoloV4TinyBarracuda](https://github.com/keijiro/YoloV4TinyBarracuda) <img src="https://img.shields.io/github/stars/keijiro/YoloV4TinyBarracuda?style=social"/> : YoloV4TinyBarracuda is an implementation of the YOLOv4-tiny object detection model on the Unity Barracuda neural network inference library.

    - [zhang8043/YoloWrapper](https://github.com/zhang8043/YoloWrapper) <img src="https://img.shields.io/github/stars/zhang8043/YoloWrapper?style=social"/> : C#封装YOLOv4算法进行目标检测。

    - [maalik0786/FastYolo](https://github.com/maalik0786/FastYolo) <img src="https://img.shields.io/github/stars/maalik0786/FastYolo?style=social"/> : Fast Yolo for fast initializing, object detection and tracking.

    - [Uehwan/CSharp-Yolo-Video](https://github.com/Uehwan/CSharp-Yolo-Video) <img src="https://img.shields.io/github/stars/Uehwan/CSharp-Yolo-Video?style=social"/> : C# Yolo for Video.

    - [HTTP123-A/HumanDetection_Yolov5NET](https://github.com/https://github.com/HTTP123-A/HumanDetection_Yolov5NET) <img src="https://img.shields.io/github/stars/HTTP123-A/HumanDetection_Yolov5NET?style=social"/> : YOLOv5 object detection with ML.NET, ONNX.

    - [Celine-Hsieh/Hand_Gesture_Training--yolov4](https://github.com/Celine-Hsieh/Hand_Gesture_Training--yolov4) <img src="https://img.shields.io/github/stars/Celine-Hsieh/Hand_Gesture_Training--yolov4?style=social"/> : Recognize the gestures' features using the YOLOv4 algorithm.

    - [lin-tea/YOLOv5DetectionWithCSharp](https://github.com/lin-tea/YOLOv5DetectionWithCSharp) <img src="https://img.shields.io/github/stars/lin-tea/YOLOv5DetectionWithCSharp?style=social"/> : YOLOv5s inference In C# and Training In Python.

    - [MirCore/Unity-Object-Detection-and-Localization-with-VR](https://github.com/MirCore/Unity-Object-Detection-and-Localization-with-VR) <img src="https://img.shields.io/github/stars/MirCore/Unity-Object-Detection-and-Localization-with-VR?style=social"/> : Detect and localize objects from the front-facing camera image of a VR Headset in a 3D Scene in Unity using Yolo and Barracuda.

    - [CarlAreDHopen-eaton/YoloObjectDetection](https://github.com/CarlAreDHopen-eaton/YoloObjectDetection) <img src="https://img.shields.io/github/stars/CarlAreDHopen-eaton/YoloObjectDetection?style=social"/> : Yolo Object Detection Application for RTSP streams.

    - [TimothyMeadows/Yolo6.NetCore](https://github.com/TimothyMeadows/Yolo6.NetCore) <img src="https://img.shields.io/github/stars/TimothyMeadows/Yolo6.NetCore?style=social"/> : You Only Look Once (v6) for .NET Core LTS.

    - [mwetzko/EasyYoloDarknet](https://github.com/mwetzko/EasyYoloDarknet) <img src="https://img.shields.io/github/stars/mwetzko/EasyYoloDarknet?style=social"/> : EasyYoloDarknet.

    - [ivilson/Yolov7net](https://github.com/ivilson/Yolov7net) <img src="https://img.shields.io/github/stars/ivilson/Yolov7net?style=social"/> : Yolov7 Detector for .Net 6.

    - [mwetzko/EasyYoloDarknet](https://github.com/mwetzko/EasyYoloDarknet) <img src="https://img.shields.io/github/stars/mwetzko/EasyYoloDarknet?style=social"/> : Windows optimized Yolo / Darknet Compile, Train and Detect.


  - ### Rust Implementation

    - [LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs) <img src="https://img.shields.io/github/stars/LaurentMazare/tch-rs?style=social"/> : Rust bindings for the C++ api of PyTorch. 

    - [sonos/tract](https://github.com/sonos/tract) <img src="https://img.shields.io/github/stars/sonos/tract?style=social"/> : Sonos' Neural Network inference engine.

    - [webonnx/wonnx](https://github.com/webonnx/wonnx) <img src="https://img.shields.io/github/stars/webonnx/wonnx?style=social"/> : Wonnx is a GPU-accelerated ONNX inference run-time written 100% in Rust, ready for the web.

    - [ptaxom/pnn](https://github.com/ptaxom/pnn) <img src="https://img.shields.io/github/stars/ptaxom/pnn?style=social"/> : pnn is Darknet compatible neural nets inference engine implemented in Rust.

    - [12101111/yolo-rs](https://github.com/12101111/yolo-rs) <img src="https://img.shields.io/github/stars/12101111/yolo-rs?style=social"/> : Yolov3 & Yolov4 with TVM and rust.

    - [TKGgunter/yolov4_tiny_rs](https://github.com/TKGgunter/yolov4_tiny_rs) <img src="https://img.shields.io/github/stars/TKGgunter/yolov4_tiny_rs?style=social"/> : A rust implementation of yolov4_tiny algorithm.

    - [flixstn/You-Only-Look-Once](https://github.com/flixstn/You-Only-Look-Once) <img src="https://img.shields.io/github/stars/flixstn/You-Only-Look-Once?style=social"/> : A Rust implementation of Yolo for object detection and tracking.

    - [lenna-project/yolo-plugin](https://github.com/lenna-project/yolo-plugin) <img src="https://img.shields.io/github/stars/lenna-project/yolo-plugin?style=social"/> : Yolo Object Detection Plugin for Lenna.

    - [masc-it/yolov5-api-rust](https://github.com/masc-it/yolov5-api-rust) <img src="https://img.shields.io/github/stars/masc-it/yolov5-api-rust?style=social"/> :  yolov5-api-rust.



  - ### Go Implementation

    - [LdDl/go-darknet](https://github.com/LdDl/go-darknet) <img src="https://img.shields.io/github/stars/LdDl/go-darknet?style=social"/> : go-darknet: Go bindings for Darknet (Yolo V4, Yolo V7-tiny, Yolo V3).

    - [wimspaargaren/yolov3](https://github.com/wimspaargaren/yolov3) <img src="https://img.shields.io/github/stars/wimspaargaren/yolov3?style=social"/> : Go implementation of the yolo v3 object detection system.      

    - [wimspaargaren/yolov5](https://github.com/wimspaargaren/yolov5) <img src="https://img.shields.io/github/stars/wimspaargaren/yolov5?style=social"/> : Go implementation of the yolo v5 object detection system.    

    - [genert/real_time_object_detection_go](https://github.com/genert/real_time_object_detection_go) <img src="https://img.shields.io/github/stars/genert/real_time_object_detection_go?style=social"/> : Real Time Object Detection with OpenCV, Go, and Yolo v4.    


  - ### Web Implementation

    - [ModelDepot/tfjs-yolo-tiny](https://github.com/ModelDepot/tfjs-yolo-tiny) <img src="https://img.shields.io/github/stars/ModelDepot/tfjs-yolo-tiny?style=social"/> : In-Browser Object Detection using Tiny YOLO on Tensorflow.js.

    - [justadudewhohacks/tfjs-tiny-yolov2](https://github.com/justadudewhohacks/tfjs-tiny-yolov2) <img src="https://img.shields.io/github/stars/justadudewhohacks/tfjs-tiny-yolov2?style=social"/> : Tiny YOLO v2 object detection with tensorflow.js.

    - [reu2018DL/YOLO-LITE](https://github.com/reu2018DL/YOLO-LITE) <img src="https://img.shields.io/github/stars/reu2018DL/YOLO-LITE?style=social"/> : YOLO-LITE is a web implementation of YOLOv2-tiny.

    - [mobimeo/node-yolo](https://github.com/mobimeo/node-yolo) <img src="https://img.shields.io/github/stars/mobimeo/node-yolo?style=social"/> : Node bindings for YOLO/Darknet image recognition library.

    - [Sharpiless/Yolov5-Flask-VUE](https://github.com/Sharpiless/Yolov5-Flask-VUE) <img src="https://img.shields.io/github/stars/Sharpiless/Yolov5-Flask-VUE?style=social"/> : 基于Flask开发后端、VUE开发前端框架，在WEB端部署YOLOv5目标检测模型。

    - [shaqian/tfjs-yolo](https://github.com/shaqian/tfjs-yolo) <img src="https://img.shields.io/github/stars/shaqian/tfjs-yolo?style=social"/> : YOLO v3 and Tiny YOLO v1, v2, v3 with Tensorflow.js.

    - [zqingr/tfjs-yolov3](https://github.com/zqingr/tfjs-yolov3) <img src="https://img.shields.io/github/stars/zqingr/tfjs-yolov3?style=social"/> : A Tensorflow js implementation of YOLOv3 and YOLOv3-tiny.

    - [bennetthardwick/darknet.js](https://github.com/bennetthardwick/darknet.js) <img src="https://img.shields.io/github/stars/bennetthardwick/darknet.js?style=social"/> : A NodeJS wrapper of pjreddie's darknet / yolo.

    - [nihui/ncnn-webassembly-yolov5](https://github.com/nihui/ncnn-webassembly-yolov5) <img src="https://img.shields.io/github/stars/nihui/ncnn-webassembly-yolov5?style=social"/> : Deploy YOLOv5 in your web browser with ncnn and webassembly.

    - [muhk01/Yolov5-on-Flask](https://github.com/muhk01/Yolov5-on-Flask) <img src="https://img.shields.io/github/stars/muhk01/Yolov5-on-Flask?style=social"/> : Running YOLOv5 through web browser using Flask microframework.

    - [tcyfree/yolov5](https://github.com/tcyfree/yolov5) <img src="https://img.shields.io/github/stars/tcyfree/yolov5?style=social"/> : 基于Flask开发后端、VUE开发前端框架，在WEB端部署YOLOv5目标检测模型。

    - [siffyy/YOLOv5-Web-App-for-Vehicle-Detection](https://github.com/siffyy/YOLOv5-Web-App-for-Vehicle-Detection) <img src="https://img.shields.io/github/stars/siffyy/YOLOv5-Web-App-for-Vehicle-Detection?style=social"/> : Repo for Web Application for Vehicle detection from Satellite Imagery using YOLOv5 model.

    - [Devmawi/BlazorObjectDetection-Sample](https://github.com/Devmawi/BlazorObjectDetection-Sample) <img src="https://img.shields.io/github/stars/Devmawi/BlazorObjectDetection-Sample?style=social"/> : A sample for demonstrating online execution of an onnx model by a Blazor app.

    - [Hyuto/yolov5-onnxruntime-web](https://github.com/Hyuto/yolov5-onnxruntime-web) <img src="https://img.shields.io/github/stars/Hyuto/yolov5-onnxruntime-web?style=social"/> : YOLOv5 right in your browser with onnxruntime-web.


  - ### Others


    - [jinfagang/yolov7](https://github.com/jinfagang/yolov7) <img src="https://img.shields.io/github/stars/jinfagang/yolov7?style=social"/> :  🔥🔥🔥🔥 (Earlier YOLOv7 not official one) YOLO with Transformers and Instance Segmentation, with TensorRT acceleration! 🔥🔥🔥 

    - [shanglianlm0525/CvPytorch](https://github.com/shanglianlm0525/CvPytorch) <img src="https://img.shields.io/github/stars/shanglianlm0525/CvPytorch?style=social"/> : CvPytorch is an open source COMPUTER VISION toolbox based on PyTorch.

    - [Holocron](https://github.com/frgfm/Holocron) <img src="https://img.shields.io/github/stars/frgfm/Holocron?style=social"/> : PyTorch implementations of recent Computer Vision tricks (ReXNet, RepVGG, Unet3p, YOLOv4, CIoU loss, AdaBelief, PolyLoss). 

    - [DL-Practise/YoloAll](https://github.com/DL-Practise/YoloAll) <img src="https://img.shields.io/github/stars/DL-Practise/YoloAll?style=social"/> : YoloAll is a collection of yolo all versions. you you use YoloAll to test yolov3/yolov5/yolox/yolo_fastest.

    - [msnh2012/Msnhnet](https://github.com/msnh2012/Msnhnet) <img src="https://img.shields.io/github/stars/msnh2012/Msnhnet?style=social"/> : (yolov3 yolov4 yolov5 unet ...)A mini pytorch inference framework which inspired from darknet.

    - [1579093407/Yolov5_Magic](https://github.com/1579093407/Yolov5_Magic) <img src="https://img.shields.io/github/stars/1579093407/Yolov5_Magic?style=social"/> : Share some tricks of improving Yolov5 and experiment results. 分享一些关于改进Yolov5的tricks以及实验结果。

    - [xinghanliuying/yolov5-trick](https://github.com/xinghanliuying/yolov5-trick) <img src="https://img.shields.io/github/stars/xinghanliuying/yolov5-trick?style=social"/> : 基于yolov5的改进库。

    - [BMW-InnovationLab/BMW-YOLOv4-Training-Automation](https://github.com/BMW-InnovationLab/BMW-YOLOv4-Training-Automation) <img src="https://img.shields.io/github/stars/BMW-InnovationLab/BMW-YOLOv4-Training-Automation?style=social"/> : YOLOv4-v3 Training Automation API for Linux.

    - [AntonMu/TrainYourOwnYOLO](https://github.com/AntonMu/TrainYourOwnYOLO) <img src="https://img.shields.io/github/stars/AntonMu/TrainYourOwnYOLO?style=social"/> : Train a state-of-the-art yolov3 object detector from scratch!

    - [madhawav/YOLO3-4-Py](https://github.com/madhawav/YOLO3-4-Py) <img src="https://img.shields.io/github/stars/madhawav/YOLO3-4-Py?style=social"/> : A Python wrapper on Darknet. Compatible with YOLO V3.

    - [theAIGuysCode/yolov4-custom-functions](https://github.com/theAIGuysCode/yolov4-custom-functions) <img src="https://img.shields.io/github/stars/theAIGuysCode/yolov4-custom-functions?style=social"/> : A Wide Range of Custom Functions for YOLOv4, YOLOv4-tiny, YOLOv3, and YOLOv3-tiny Implemented in TensorFlow, TFLite, and TensorRT.

    - [fcakyon/yolov5-pip](https://github.com/fcakyon/yolov5-pip) <img src="https://img.shields.io/github/stars/fcakyon/yolov5-pip?style=social"/> : Packaged version of ultralytics/yolov5.

    - [Laughing-q/YOLO-Q](https://github.com/Laughing-q/YOLO-Q) <img src="https://img.shields.io/github/stars/Laughing-q/YOLO-Q?style=social"/> : 🔥🔥🔥A inference framework that support multi models of yolo5(torch and tensorrt), yolox(torch and tensorrt), nanodet(tensorrt), yolo-fastestV2(tensorrt) and yolov5-lite(tensorrt).

    - [tiquasar/FLAITER](https://github.com/tiquasar/FLAITER) <img src="https://img.shields.io/github/stars/tiquasar/FLAITER?style=social"/> : Machine Learning and AI Mobile Application.

    - [HuKai97/yolov5-5.x-annotations](https://github.com/HuKai97/yolov5-5.x-annotations) <img src="https://img.shields.io/github/stars/HuKai97/yolov5-5.x-annotations?style=social"/> : 一个基于yolov5-5.0的中文注释版本！ 

    - [kadirnar/Minimal-Yolov6](https://github.com/kadirnar/Minimal-Yolov6) <img src="https://img.shields.io/github/stars/kadirnar/Minimal-Yolov6?style=social"/> : Minimal-Yolov6. 

    - [DataXujing/YOLOv6](https://github.com/DataXujing/YOLOv6) <img src="https://img.shields.io/github/stars/DataXujing/YOLOv6?style=social"/> : 🌀 🌀 手摸手 美团 YOLOv6模型训练和TensorRT端到端部署方案教程。

    - [DataXujing/YOLOv7](https://github.com/DataXujing/YOLOv7) <img src="https://img.shields.io/github/stars/DataXujing/YOLOv7?style=social"/> : 🔥🔥🔥 Official YOLOv7训练自己的数据集并实现端到端的TensorRT模型加速推断。

    - [Code-keys/yolov5-darknet](https://github.com/Code-keys/yolov5-darknet) <img src="https://img.shields.io/github/stars/Code-keys/yolov5-darknet?style=social"/> : yolov5-darknet support yaml && cfg.

    - [Code-keys/yolo-darknet](https://github.com/Code-keys/yolo-darknet) <img src="https://img.shields.io/github/stars/Code-keys/yolo-darknet?style=social"/> : YOLO-family complemented by darknet. yolov5 yolov7 et al ... 

    - [pooya-mohammadi/deep_utils](https://github.com/pooya-mohammadi/deep_utils) <img src="https://img.shields.io/github/stars/pooya-mohammadi/deep_utils?style=social"/> : A toolkit full of handy functions including most used models and utilities for deep-learning practitioners!  

    - [yl-jiang/YOLOSeries](https://github.com/yl-jiang/YOLOSeries) <img src="https://img.shields.io/github/stars/yl-jiang/YOLOSeries?style=social"/> : YOLO Series.

    - [yjh0410/FreeYOLO](https://github.com/yjh0410/FreeYOLO) <img src="https://img.shields.io/github/stars/yjh0410/FreeYOLO?style=social"/> : Anchor-free YOLO detector.

    - [open-yolo/yolov7](https://github.com/open-yolo/yolov7) <img src="https://img.shields.io/github/stars/open-yolo/yolov7?style=social"/> : Improved and packaged version of WongKinYiu/yolov7.

    - [iloveai8086/YOLOC](https://github.com/iloveai8086/YOLOC) <img src="https://img.shields.io/github/stars/iloveai8086/YOLOC?style=social"/> : 🚀YOLOC is Combining different modules to build an different Object detection model.

    - [iscyy/yoloair](https://github.com/iscyy/yoloair) <img src="https://img.shields.io/github/stars/iscyy/yoloair?style=social"/> : 🔥🔥🔥YOLOAir：Including YOLOv5, YOLOv7, Transformer, YOLOX, YOLOR and other networks... Support to improve backbone, head, loss, IoU, NMS...The original version was created based on YOLOv5.


## Extensional Frameworks

  - [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) <img src="https://img.shields.io/github/stars/Megvii-BaseDetection/YOLOX?style=social"/> : "YOLOX: Exceeding YOLO Series in 2021". (**[arXiv 2021](https://arxiv.org/abs/2107.08430)**)

  - [YOLOR](https://github.com/WongKinYiu/yolor) <img src="https://img.shields.io/github/stars/WongKinYiu/yolor?style=social"/> : "You Only Learn One Representation: Unified Network for Multiple Tasks". (**[arXiv 2021](https://arxiv.org/abs/2105.04206)**)

  - [YOLOF](https://github.com/megvii-model/YOLOF) <img src="https://img.shields.io/github/stars/megvii-model/YOLOF?style=social"/> : "You Only Look One-level Feature". (**[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_You_Only_Look_One-Level_Feature_CVPR_2021_paper.html)**)

  - [YOLOS](https://github.com/hustvl/YOLOS) <img src="https://img.shields.io/github/stars/hustvl/YOLOS?style=social"/> : "You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection". (**[NeurIPS 2021](https://proceedings.neurips.cc//paper/2021/hash/dc912a253d1e9ba40e2c597ed2376640-Abstract.html)**)

  - [YOLACT & YOLACT++](https://github.com/dbolya/yolact) <img src="https://img.shields.io/github/stars/dbolya/yolact?style=social"/> : You Only Look At CoefficienTs. (**[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.html), [IEEE TPAMI 2020](https://ieeexplore.ieee.org/abstract/document/9159935)**)

  - [Alpha-IoU](https://github.com/Jacobi93/Alpha-IoU) <img src="https://img.shields.io/github/stars/Jacobi93/Alpha-IoU?style=social"/> : "Alpha-IoU: A Family of Power Intersection over Union Losses for Bounding Box Regression". (**[NeurIPS 2021](https://proceedings.neurips.cc//paper/2021/hash/a8f15eda80c50adb0e71943adc8015cf-Abstract.html)**)

  - [CIoU](https://github.com/Zzh-tju/CIoU) <img src="https://img.shields.io/github/stars/Zzh-tju/CIoU?style=social"/> : Complete-IoU (CIoU) Loss and Cluster-NMS for Object Detection and Instance Segmentation (YOLACT). (**[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6999), [IEEE TCYB 2021](https://ieeexplore.ieee.org/abstract/document/9523600)**)

  - [AIRDet](https://github.com/tinyvision/AIRDet) <img src="https://img.shields.io/github/stars/tinyvision/AIRDet?style=social"/> : Welcome to AIRDet! AIRDet is an efficiency-oriented anchor-free object detector, aims to enable robust object detection in various industry scene.

  - [Albumentations](https://github.com/albumentations-team/albumentations) <img src="https://img.shields.io/github/stars/albumentations-team/albumentations?style=social"/> : Albumentations is a Python library for image augmentation. Image augmentation is used in deep learning and computer vision tasks to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing data. "Albumentations: Fast and Flexible Image Augmentations". (**[Information 2020](https://www.mdpi.com/2078-2489/11/2/125)**)


## Applications

  - ### Lighter and Faster

    - #### Lightweight Backbones and FPN
      #### 轻量级骨干网络和特征金字塔网络

      - [murufeng/awesome_lightweight_networks](https://github.com/murufeng/awesome_lightweight_networks) <img src="https://img.shields.io/github/stars/murufeng/awesome_lightweight_networks?style=social"/> : The implementation of various lightweight networks by using PyTorch. such as：MobileNetV2，MobileNeXt，GhostNet，ParNet，MobileViT、AdderNet，ShuffleNetV1-V2，LCNet，ConvNeXt，etc. ⭐⭐⭐⭐⭐

      - [Bobo-y/flexible-yolov5](https://github.com/Bobo-y/flexible-yolov5) <img src="https://img.shields.io/github/stars/Bobo-y/flexible-yolov5?style=social"/> : More readable and flexible yolov5 with more backbone(resnet, shufflenet, moblienet, efficientnet, hrnet, swin-transformer) and (cbam，dcn and so on), and tensorrt.

      - [XingZeng307/YOLOv5_with_BiFPN](https://github.com/XingZeng307/YOLOv5_with_BiFPN) <img src="https://img.shields.io/github/stars/XingZeng307/YOLOv5_with_BiFPN?style=social"/> : This repo is mainly for replacing PANet with BiFPN in YOLOv5.

      - [dog-qiuqiu/MobileNet-Yolo](https://github.com/dog-qiuqiu/MobileNet-Yolo) <img src="https://img.shields.io/github/stars/dog-qiuqiu/MobileNet-Yolo?style=social"/> : MobileNetV2-YoloV3-Nano: 0.5BFlops 3MB HUAWEI P40: 6ms/img, YoloFace-500k:0.1Bflops 420KB🔥🔥🔥.

      - [eric612/MobileNet-YOLO](https://github.com/eric612/MobileNet-YOLO) <img src="https://img.shields.io/github/stars/eric612/MobileNet-YOLO?style=social"/> : A caffe implementation of MobileNet-YOLO detection network.

      - [eric612/Mobilenet-YOLO-Pytorch](https://github.com/eric612/Mobilenet-YOLO-Pytorch) <img src="https://img.shields.io/github/stars/eric612/Mobilenet-YOLO-Pytorch?style=social"/> : Include mobilenet series (v1,v2,v3...) and yolo series (yolov3,yolov4,...) .

      - [Adamdad/keras-YOLOv3-mobilenet](https://github.com/Adamdad/keras-YOLOv3-mobilenet) <img src="https://img.shields.io/github/stars/Adamdad/keras-YOLOv3-mobilenet?style=social"/> : A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

      - [fsx950223/mobilenetv2-yolov3](https://github.com/fsx950223/mobilenetv2-yolov3) <img src="https://img.shields.io/github/stars/fsx950223/mobilenetv2-yolov3?style=social"/> : yolov3 with mobilenetv2 and efficientnet.

      - [liux0614/yolo_nano](https://github.com/liux0614/yolo_nano) <img src="https://img.shields.io/github/stars/liux0614/yolo_nano?style=social"/> : Unofficial implementation of yolo nano.

      - [lingtengqiu/Yolo_Nano](https://github.com/lingtengqiu/Yolo_Nano) <img src="https://img.shields.io/github/stars/lingtengqiu/Yolo_Nano?style=social"/> : Pytorch implementation of yolo_Nano for pedestrian detection.

      - [bubbliiiing/mobilenet-yolov4-pytorch](https://github.com/bubbliiiing/mobilenet-yolov4-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/mobilenet-yolov4-pytorch?style=social"/> : 这是一个mobilenet-yolov4的库，把yolov4主干网络修改成了mobilenet，修改了Panet的卷积组成，使参数量大幅度缩小。

      - [bubbliiiing/efficientnet-yolo3-pytorch](https://github.com/bubbliiiing/efficientnet-yolo3-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/efficientnet-yolo3-pytorch?style=social"/> : 这是一个efficientnet-yolo3-pytorch的源码，将yolov3的主干特征提取网络修改成了efficientnet。

      - [HuKai97/YOLOv5-ShuffleNetv2](https://github.com/HuKai97/YOLOv5-ShuffleNetv2) <img src="https://img.shields.io/github/stars/HuKai97/YOLOv5-ShuffleNetv2?style=social"/> : YOLOv5的轻量化改进(蜂巢检测项目)。

      - [YOLO-ReT](https://github.com/guotao0628/yoloret) <img src="https://img.shields.io/github/stars/guotao0628/yoloret?style=social"/> : "YOLO-ReT: Towards High Accuracy Real-time Object Detection on Edge GPUs". (**[WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Ganesh_YOLO-ReT_Towards_High_Accuracy_Real-Time_Object_Detection_on_Edge_GPUs_WACV_2022_paper.html)**)


    - #### Pruning Knoweldge-Distillation Quantization
      #### 剪枝 知识蒸馏 量化

      - [SparseML](https://github.com/neuralmagic/sparseml) <img src="https://img.shields.io/github/stars/neuralmagic/sparseml?style=social"/> : Libraries for applying sparsification recipes to neural networks with a few lines of code, enabling faster and smaller models. "Inducing and Exploiting Activation Sparsity for Fast Inference on Deep Neural Networks". (**[PMLR 2020](http://proceedings.mlr.press/v119/kurtz20a.html)**). "Woodfisher: Efficient second-order approximation for neural network compression". (**[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d1ff1ec86b62cd5f3903ff19c3a326b2-Abstract.html)**)

      - [SparseZoo](https://github.com/neuralmagic/sparsezoo) <img src="https://img.shields.io/github/stars/neuralmagic/sparsezoo?style=social"/> : Neural network model repository for highly sparse and sparse-quantized models with matching sparsification recipes.

      - [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill) <img src="https://img.shields.io/github/stars/yoshitomo-matsubara/torchdistill?style=social"/> : torchdistill: A Modular, Configuration-Driven Framework for Knowledge Distillation. A coding-free framework built on PyTorch for reproducible deep learning studies. 🏆20 knowledge distillation methods presented at CVPR, ICLR, ECCV, NeurIPS, ICCV, etc are implemented so far. 🎁 Trained models, training logs and configurations are available for ensuring the reproducibiliy and benchmark. 

      - [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) <img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleSlim?style=social"/> : PaddleSlim is an open-source library for deep model compression and architecture search. PaddleSlim是一个专注于深度学习模型压缩的工具库，提供低比特量化、知识蒸馏、稀疏化和模型结构搜索等模型压缩策略，帮助用户快速实现模型的小型化。

      - [PPL量化工具](https://github.com/openppl-public/ppq) <img src="https://img.shields.io/github/stars/openppl-public/ppq?style=social"/> : PPL Quantization Tool (PPQ) is a powerful offline neural network quantization tool. PPL QuantTool 是一个高效的工业级神经网络量化工具。 

      - [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) <img src="https://img.shields.io/github/stars/PINTO0309/PINTO_model_zoo?style=social"/> : A repository for storing models that have been inter-converted between various frameworks. Supported frameworks are TensorFlow, PyTorch, ONNX, OpenVINO, TFJS, TFTRT, TensorFlowLite (Float32/16/INT8), EdgeTPU, CoreML. 

      - [ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite) <img src="https://img.shields.io/github/stars/ppogg/YOLOv5-Lite?style=social"/> : 🍅🍅🍅YOLOv5-Lite: lighter, faster and easier to deploy. Evolved from yolov5 and the size of model is only 930+kb (int8) and 1.7M (fp16). It can reach 10+ FPS on the Raspberry Pi 4B when the input size is 320×320~ 

      - [dog-qiuqiu/FastestDet](https://github.com/dog-qiuqiu/FastestDet) <img src="https://img.shields.io/github/stars/dog-qiuqiu/FastestDet?style=social"/> : ⚡ A newly designed ultra lightweight anchor free target detection algorithm， weight only 250K parameters， reduces the time consumption by 10% compared with yolo-fastest, and the post-processing is simpler. (**[知乎 2022](https://zhuanlan.zhihu.com/p/536500269)**)    

      - [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) <img src="https://img.shields.io/github/stars/dog-qiuqiu/Yolo-Fastest?style=social"/> : Yolo-Fastest：超超超快的开源ARM实时目标检测算法。 (**[Zenodo 2021](http://doi.org/10.5281/zenodo.5131532), [知乎 2020](https://zhuanlan.zhihu.com/p/234506503)**)

      - [dog-qiuqiu/Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2) <img src="https://img.shields.io/github/stars/dog-qiuqiu/Yolo-FastestV2?style=social"/> : Yolo-FastestV2:更快，更轻，移动端可达300FPS，参数量仅250k。 (**[知乎 2021](https://zhuanlan.zhihu.com/p/400474142)**)

      - [YOLObile](https://github.com/nightsnack/YOLObile) <img src="https://img.shields.io/github/stars/nightsnack/YOLObile?style=social"/> : "YOLObile: Real-Time Object Detection on Mobile Devices via Compression-Compilation Co-Design". (**[AAAI 2021](https://www.aaai.org/AAAI21Papers/AAAI-7561.CaiY.pdf)**)

      - [Gumpest/YOLOv5-Multibackbone-Compression](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression) <img src="https://img.shields.io/github/stars/Gumpest/YOLOv5-Multibackbone-Compression?style=social"/> : YOLOv5 Series Multi-backbone(TPH-YOLOv5, Ghostnet, ShuffleNetv2, Mobilenetv3Small, EfficientNetLite, PP-LCNet, SwinTransformer YOLO), Module(CBAM, DCN), Pruning (EagleEye, Network Slimming) and Quantization (MQBench) Compression Tool Box. 

      - [SlimYOLOv3](https://github.com/PengyiZhang/SlimYOLOv3) <img src="https://img.shields.io/github/stars/PengyiZhang/SlimYOLOv3?style=social"/> : "SlimYOLOv3: Narrower, Faster and Better for UAV Real-Time Applications". (**[arXiv 2019](https://arxiv.org/abs/1907.11093)**)

      - [uyzhang/yolov5_prune](https://github.com/uyzhang/yolov5_prune) <img src="https://img.shields.io/github/stars/uyzhang/yolov5_prune?style=social"/> : Yolov5 pruning on COCO Dataset.

      - [midasklr/yolov5prune](https://github.com/midasklr/yolov5prune) <img src="https://img.shields.io/github/stars/midasklr/yolov5prune?style=social"/> : yolov5模型剪枝。

      - [ZJU-lishuang/yolov5_prune](https://github.com/ZJU-lishuang/yolov5_prune) <img src="https://img.shields.io/github/stars/ZJU-lishuang/yolov5_prune?style=social"/> : yolov5 prune，Support V2, V3, V4 and V6 versions of yolov5.

      - [Syencil/mobile-yolov5-pruning-distillation](https://github.com/Syencil/mobile-yolov5-pruning-distillation) <img src="https://img.shields.io/github/stars/Syencil/mobile-yolov5-pruning-distillation?style=social"/> : mobilev2-yolov5s剪枝、蒸馏，支持ncnn，tensorRT部署。ultra-light but better performence！

      - [Lam1360/YOLOv3-model-pruning](https://github.com/Lam1360/YOLOv3-model-pruning) <img src="https://img.shields.io/github/stars/Lam1360/YOLOv3-model-pruning?style=social"/> : 在 oxford hand 数据集上对 YOLOv3 做模型剪枝（network slimming）。

      - [tanluren/yolov3-channel-and-layer-pruning](https://github.com/tanluren/yolov3-channel-and-layer-pruning) <img src="https://img.shields.io/github/stars/tanluren/yolov3-channel-and-layer-pruning?style=social"/> : yolov3 yolov4 channel and layer pruning, Knowledge Distillation 层剪枝，通道剪枝，知识蒸馏。

      - [coldlarry/YOLOv3-complete-pruning](https://github.com/coldlarry/YOLOv3-complete-pruning) <img src="https://img.shields.io/github/stars/coldlarry/YOLOv3-complete-pruning?style=social"/> : 提供对YOLOv3及Tiny的多种剪枝版本，以适应不同的需求。

      - [SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone](https://github.com/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone) <img src="https://img.shields.io/github/stars/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone?style=social"/> : YOLO ModelCompression MultidatasetTraining.

      - [talebolano/yolov3-network-slimming](https://github.com/talebolano/yolov3-network-slimming) <img src="https://img.shields.io/github/stars/talebolano/yolov3-network-slimming?style=social"/> : yolov3 network slimming剪枝的一种实现。

      - [AlexeyAB/yolo2_light](https://github.com/AlexeyAB/yolo2_light) <img src="https://img.shields.io/github/stars/AlexeyAB/yolo2_light?style=social"/> : Light version of convolutional neural network Yolo v3 & v2 for objects detection with a minimum of dependencies (INT8-inference, BIT1-XNOR-inference).

      - [j-marple-dev/AYolov2](https://github.com/j-marple-dev/AYolov2) <img src="https://img.shields.io/github/stars/j-marple-dev/AYolov2?style=social"/> : AYolov2.

      - [Wulingtian/yolov5_tensorrt_int8](https://github.com/Wulingtian/yolov5_tensorrt_int8) <img src="https://img.shields.io/github/stars/Wulingtian/yolov5_tensorrt_int8?style=social"/> : TensorRT int8 量化部署 yolov5s 模型，实测3.3ms一帧！ 

      - [wonbeomjang/yolov5-knowledge-distillation](https://github.com/wonbeomjang/yolov5-knowledge-distillation) <img src="https://img.shields.io/github/stars/wonbeomjang/yolov5-knowledge-distillation?style=social"/> : implementation of [Distilling Object Detectors with Fine-grained Feature Imitation](https://github.com/twangnh/Distilling-Object-Detectors) on yolov5. "Distilling Object Detectors with Fine-grained Feature Imitation". (**[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.html)**)

      - [Sharpiless/Yolov5-distillation-train-inference](https://github.com/Sharpiless/Yolov5-distillation-train-inference) <img src="https://img.shields.io/github/stars/Sharpiless/Yolov5-distillation-train-inference?style=social"/> : Yolov5 distillation training | Yolov5知识蒸馏训练，支持训练自己的数据。

      - [Sharpiless/yolov5-distillation-5.0](https://github.com/Sharpiless/yolov5-distillation-5.0) <img src="https://img.shields.io/github/stars/Sharpiless/yolov5-distillation-5.0?style=social"/> : yolov5 5.0 version distillation || yolov5 5.0版本知识蒸馏，yolov5l >> yolov5s。

      - [Sharpiless/yolov5-knowledge-distillation](https://github.com/Sharpiless/yolov5-knowledge-distillation) <img src="https://img.shields.io/github/stars/Sharpiless/yolov5-knowledge-distillation?style=social"/> : yolov5目标检测模型的知识蒸馏（基于响应的蒸馏）。

      - [chengpanghu/Knowledge-Distillation-yolov5](https://github.com/chengpanghu/Knowledge-Distillation-yolov5) <img src="https://img.shields.io/github/stars/chengpanghu/Knowledge-Distillation-yolov5?style=social"/> : Knowledge-Distillation-yolov5 基于yolov5的知识蒸馏。

      - [magicshuang/yolov5_distillation](https://github.com/magicshuang/yolov5_distillation) <img src="https://img.shields.io/github/stars/magicshuang/yolov5_distillation?style=social"/> : yolov5 知识蒸馏，yolov5-l模型压缩至yolov5-s 压缩算法是 [Distilling Object Detectors with Fine-grained Feature Imitation](https://github.com/twangnh/Distilling-Object-Detectors)。

      - [Sharpiless/Yolov3-MobileNet-Distillation](https://github.com/Sharpiless/Yolov3-MobileNet-Distillation) <img src="https://img.shields.io/github/stars/Sharpiless/Yolov3-MobileNet-Distillation?style=social"/> : 在Yolov3-MobileNet上进行模型蒸馏训练。

      - [Bigtuo/YOLOX-Lite](https://github.com/Bigtuo/YOLOX-Lite) <img src="https://img.shields.io/github/stars/Bigtuo/YOLOX-Lite?style=social"/> : 将YOLOv5-Lite代码中的head更换为YOLOX head。


    - #### High-performance Inference Engine
      #### 高性能推理引擎

      - [ONNX](https://github.com/onnx/onnx) <img src="https://img.shields.io/github/stars/onnx/onnx?style=social"/> : Open standard for machine learning interoperability.

      - [ONNX Runtime](https://github.com/microsoft/onnxruntime) <img src="https://img.shields.io/github/stars/microsoft/onnxruntime?style=social"/> : ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator.

      - [TVM](https://github.com/apache/tvm) <img src="https://img.shields.io/github/stars/apache/tvm?style=social"/> : Open deep learning compiler stack for cpu, gpu and specialized accelerators.

      - [TensorRT](https://github.com/NVIDIA/TensorRT) <img src="https://img.shields.io/github/stars/NVIDIA/TensorRT?style=social"/> : TensorRT is a C++ library for high performance inference on NVIDIA GPUs and deep learning accelerators.

      - [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) <img src="https://img.shields.io/github/stars/wang-xinyu/tensorrtx?style=social"/> : TensorRTx aims to implement popular deep learning networks with tensorrt network definition APIs. 

      - [ceccocats/tkDNN](https://github.com/ceccocats/tkDNN) <img src="https://img.shields.io/github/stars/ceccocats/tkDNN?style=social"/> : Deep neural network library and toolkit to do high performace inference on NVIDIA jetson platforms. "A Systematic Assessment of Embedded Neural Networks for Object Detection". (**[IEEE ETFA 2020](https://ieeexplore.ieee.org/document/9212130)**)

      - [OpenVINO](https://github.com/openvinotoolkit/openvino) <img src="https://img.shields.io/github/stars/openvinotoolkit/openvino?style=social"/> : This open source version includes several components: namely Model Optimizer, OpenVINO™ Runtime, Post-Training Optimization Tool, as well as CPU, GPU, MYRIAD, multi device and heterogeneous plugins to accelerate deep learning inferencing on Intel® CPUs and Intel® Processor Graphics.

      - [ncnn](https://github.com/Tencent/ncnn) <img src="https://img.shields.io/github/stars/Tencent/ncnn?style=social"/> : ncnn is a high-performance neural network inference framework optimized for the mobile platform.

      - [MNN](https://github.com/alibaba/MNN) <img src="https://img.shields.io/github/stars/alibaba/MNN?style=social"/> : MNN is a blazing fast, lightweight deep learning framework, battle-tested by business-critical use cases in Alibaba. (**[MLSys 2020](https://proceedings.mlsys.org/paper/2020/hash/8f14e45fceea167a5a36dedd4bea2543-Abstract.html)**)

      - [Tengine](https://github.com/OAID/Tengine) <img src="https://img.shields.io/github/stars/OAID/Tengine?style=social"/> : Tengine is a lite, high performance, modular inference engine for embedded device.

      - [Paddle Lite](https://github.com/paddlepaddle/paddle-lite) <img src="https://img.shields.io/github/stars/paddlepaddle/paddle-lite?style=social"/> : Multi-platform high performance deep learning inference engine (飞桨多端多平台高性能深度学习推理引擎）。

      - [DefTruth/lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) <img src="https://img.shields.io/github/stars/DefTruth/lite.ai.toolkit?style=social"/> : 🛠 A lite C++ toolkit of awesome AI models with ONNXRuntime, NCNN, MNN and TNN. YOLOX, YOLOP, YOLOv6, YOLOR, MODNet, YOLOX, SCRFD, YOLOX . MNN, NCNN, TNN, ONNXRuntime, CPU/GPU. “🛠Lite.Ai.ToolKit: 一个轻量级的C++ AI模型工具箱，用户友好（还行吧），开箱即用。已经包括 100+ 流行的开源模型。这是一个根据个人兴趣整理的C++工具箱，, 涵盖目标检测、人脸检测、人脸识别、语义分割、抠图等领域。”

      - [Baiyuetribe/ncnn-models](https://github.com/Baiyuetribe/ncnn-models) <img src="https://img.shields.io/github/stars/Baiyuetribe/ncnn-models?style=social"/> : awesome AI models with NCNN, and how they were converted ✨✨✨

      - [cmdbug/YOLOv5_NCNN](https://github.com/cmdbug/YOLOv5_NCNN) <img src="https://img.shields.io/github/stars/cmdbug/YOLOv5_NCNN?style=social"/> : 🍅 Deploy ncnn on mobile phones. Support Android and iOS. 移动端ncnn部署，支持Android与iOS。

      - [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) <img src="https://img.shields.io/github/stars/marcoslucianops/DeepStream-Yolo?style=social"/> : NVIDIA DeepStream SDK 6.0 configuration for YOLO models.

      - [DanaHan/Yolov5-in-Deepstream-5.0](https://github.com/DanaHan/Yolov5-in-Deepstream-5.0) <img src="https://img.shields.io/github/stars/DanaHan/Yolov5-in-Deepstream-5.0?style=social"/> : Describe how to use yolov5 in Deepstream 5.0.

      - [shouxieai/tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro) <img src="https://img.shields.io/github/stars/shouxieai/tensorRT_Pro?style=social"/> : C++ library based on tensorrt integration.

      - [zhiqwang/yolov5-rt-stack](https://github.com/zhiqwang/yolov5-rt-stack) <img src="https://img.shields.io/github/stars/zhiqwang/yolov5-rt-stack?style=social"/> : A runtime stack for object detection on specialized accelerators.

      - [enazoe/yolo-tensorrt](https://github.com/enazoe/yolo-tensorrt) <img src="https://img.shields.io/github/stars/enazoe/yolo-tensorrt?style=social"/> : TensorRT8.Support Yolov5n,s,m,l,x .darknet -> tensorrt. Yolov4 Yolov3 use raw darknet *.weights and *.cfg fils. If the wrapper is useful to you,please Star it.

      - [lewes6369/TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3) <img src="https://img.shields.io/github/stars/lewes6369/TensorRT-Yolov3?style=social"/> : TensorRT for Yolov3.

      - [CaoWGG/TensorRT-YOLOv4](https://github.com/CaoWGG/TensorRT-YOLOv4) <img src="https://img.shields.io/github/stars/CaoWGG/TensorRT-YOLOv4?style=social"/> :tensorrt5, yolov4, yolov3,yolov3-tniy,yolov3-tniy-prn.

      - [isarsoft/yolov4-triton-tensorrt](https://github.com/isarsoft/yolov4-triton-tensorrt) <img src="https://img.shields.io/github/stars/isarsoft/yolov4-triton-tensorrt?style=social"/> : YOLOv4 on Triton Inference Server with TensorRT.

      - [TrojanXu/yolov5-tensorrt](https://github.com/TrojanXu/yolov5-tensorrt) <img src="https://img.shields.io/github/stars/TrojanXu/yolov5-tensorrt?style=social"/> : A tensorrt implementation of yolov5.

      - [tjuskyzhang/Scaled-YOLOv4-TensorRT](https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT) <img src="https://img.shields.io/github/stars/tjuskyzhang/Scaled-YOLOv4-TensorRT?style=social"/> : Implement yolov4-tiny-tensorrt, yolov4-csp-tensorrt, yolov4-large-tensorrt(p5, p6, p7) layer by layer using TensorRT API.

      - [guojianyang/cv-detect-robot](https://github.com/guojianyang/cv-detect-robot) <img src="https://img.shields.io/github/stars/guojianyang/cv-detect-robot?style=social"/> : 🔥🔥🔥🔥🔥🔥docker+nvidia-docker2+yolov5+YOLOX+yolo+deepsort+tensorRT+ros+deepstream+jetson+nano+TX2+NX for High-performance deployment(高性能部署)。

      - [Syencil/tensorRT](https://github.com/Syencil/tensorRT) <img src="https://img.shields.io/github/stars/Syencil/tensorRT?style=social"/> : TensorRT-7 Network Lib 包括常用目标检测、关键点检测、人脸检测、OCR等 可训练自己数据。

      - [SeanAvery/yolov5-tensorrt](https://github.com/SeanAvery/yolov5-tensorrt) <img src="https://img.shields.io/github/stars/SeanAvery/yolov5-tensorrt?style=social"/> : YOLOv5 in TensorRT.

      - [PINTO0309/OpenVINO-YoloV3](https://github.com/PINTO0309/OpenVINO-YoloV3) <img src="https://img.shields.io/github/stars/PINTO0309/OpenVINO-YoloV3?style=social"/> : YoloV3/tiny-YoloV3 + RaspberryPi3/Ubuntu LaptopPC + NCS/NCS2 + USB Camera + Python + OpenVINO.

      - [TNTWEN/OpenVINO-YOLOV4](https://github.com/TNTWEN/OpenVINO-YOLOV4) <img src="https://img.shields.io/github/stars/TNTWEN/OpenVINO-YOLOV4?style=social"/> : This is implementation of YOLOv4,YOLOv4-relu,YOLOv4-tiny,YOLOv4-tiny-3l,Scaled-YOLOv4 and INT8 Quantization in OpenVINO2021.3.

      - [fb029ed/yolov5_cpp_openvino](https://github.com/fb029ed/yolov5_cpp_openvino) <img src="https://img.shields.io/github/stars/fb029ed/yolov5_cpp_openvino?style=social"/> : 用c++实现了yolov5使用openvino的部署。

      - [natanielruiz/android-yolo](https://github.com/natanielruiz/android-yolo) <img src="https://img.shields.io/github/stars/natanielruiz/android-yolo?style=social"/> : Real-time object detection on Android using the YOLO network with TensorFlow.

      - [nihui/ncnn-android-yolov5](https://github.com/nihui/ncnn-android-yolov5) <img src="https://img.shields.io/github/stars/nihui/ncnn-android-yolov5?style=social"/> : The YOLOv5 object detection android example.

      - [szaza/android-yolo-v2](https://github.com/szaza/android-yolo-v2) <img src="https://img.shields.io/github/stars/szaza/android-yolo-v2?style=social"/> : Android YOLO real time object detection sample application with Tensorflow mobile.

      - [FeiGeChuanShu/ncnn-android-yolox](https://github.com/FeiGeChuanShu/ncnn-android-yolox) <img src="https://img.shields.io/github/stars/FeiGeChuanShu/ncnn-android-yolox?style=social"/> : Real time yolox Android demo by ncnn.

      - [xiangweizeng/darknet2ncnn](https://github.com/xiangweizeng/darknet2ncnn) <img src="https://img.shields.io/github/stars/xiangweizeng/darknet2ncnn?style=social"/> : Darknet2ncnn converts the darknet model to the ncnn model.

      - [sunnyden/YOLOV5_NCNN_Android](https://github.com/sunnyden/YOLOV5_NCNN_Android) <img src="https://img.shields.io/github/stars/sunnyden/YOLOV5_NCNN_Android?style=social"/> : YOLOv5 C++ Implementation on Android using NCNN framework.

      - [duangenquan/YoloV2NCS](https://github.com/duangenquan/YoloV2NCS) <img src="https://img.shields.io/github/stars/duangenquan/YoloV2NCS?style=social"/> : This project shows how to run tiny yolo v2 with movidius stick.

      - [lp6m/yolov5s_android](https://github.com/lp6m/yolov5s_android) <img src="https://img.shields.io/github/stars/lp6m/yolov5s_android?style=social"/> : Run yolov5s on Android device!

      - [apxlwl/MNN-yolov3](https://github.com/apxlwl/MNN-yolov3) <img src="https://img.shields.io/github/stars/apxlwl/MNN-yolov3?style=social"/> : MNN demo of Strongeryolo, including channel pruning, android support...

      - [hollance/YOLO-CoreML-MPSNNGraph](https://github.com/hollance/YOLO-CoreML-MPSNNGraph) <img src="https://img.shields.io/github/stars/hollance/YOLO-CoreML-MPSNNGraph?style=social"/> : Tiny YOLO for iOS implemented using CoreML but also using the new MPS graph API.

      - [r4ghu/iOS-CoreML-Yolo](https://github.com/r4ghu/iOS-CoreML-Yolo) <img src="https://img.shields.io/github/stars/r4ghu/iOS-CoreML-Yolo?style=social"/> : This is the implementation of Object Detection using Tiny YOLO v1 model on Apple's CoreML Framework.

      - [KoheiKanagu/ncnn_yolox_flutter](https://github.com/KoheiKanagu/ncnn_yolox_flutter) <img src="https://img.shields.io/github/stars/KoheiKanagu/ncnn_yolox_flutter?style=social"/> : This is a plugin to run YOLOX on ncnn.

      - [SsisyphusTao/Object-Detection-Knowledge-Distillation](https://github.com/SsisyphusTao/Object-Detection-Knowledge-Distillation) <img src="https://img.shields.io/github/stars/SsisyphusTao/Object-Detection-Knowledge-Distillation?style=social"/> : An Object Detection Knowledge Distillation framework powered by pytorch, now having SSD and yolov5. 

      - [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo) <img src="https://img.shields.io/github/stars/airockchip/rknn_model_zoo?style=social"/> : Rockchip Neural Network(RKNN)是瑞芯微为了加速模型推理而基于自身NPU硬件架构定义的一套模型格式.使用该格式定义的模型在Rockchip NPU上可以获得远高于CPU/GPU的性能。

      - [Monday-Leo/YOLOv7_Tensorrt](https://github.com/Monday-Leo/YOLOv7_Tensorrt) <img src="https://img.shields.io/github/stars/Monday-Leo/YOLOv7_Tensorrt?style=social"/> : A simple implementation of Tensorrt YOLOv7. 

      - [ibaiGorordo/ONNX-YOLOv6-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv6-Object-Detection) <img src="https://img.shields.io/github/stars/ibaiGorordo/ONNX-YOLOv6-Object-Detection?style=social"/> : Python scripts performing object detection using the YOLOv6 model in ONNX. 

      - [ibaiGorordo/ONNX-YOLOv7-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection) <img src="https://img.shields.io/github/stars/ibaiGorordo/ONNX-YOLOv7-Object-Detection?style=social"/> : Python scripts performing object detection using the YOLOv7 model in ONNX. 

      - [triple-Mu/yolov7](https://github.com/triple-Mu/yolov7) <img src="https://img.shields.io/github/stars/triple-Mu/yolov7?style=social"/> : End2end TensorRT YOLOv7.

      - [Linaom1214/tensorrt-python](https://github.com/Linaom1214/tensorrt-python) <img src="https://img.shields.io/github/stars/Linaom1214/tensorrt-python?style=social"/> : YOLO Series TensorRT Python/C++. tensorrt for yolov7,yolov6,yolov5,yolox.

      - [cyrillkuettel/ncnn-android-yolov5](https://github.com/cyrillkuettel/ncnn-android-yolov5) <img src="https://img.shields.io/github/stars/cyrillkuettel/ncnn-android-yolov5?style=social"/> : This is a sample ncnn android project, it depends on ncnn library and opencv.

      - [hewen0901/yolov7_trt](https://github.com/hewen0901/yolov7_trt) <img src="https://img.shields.io/github/stars/hewen0901/yolov7_trt?style=social"/> : yolov7目标检测算法的c++ tensorrt部署代码。

      - [tsutof/tiny_yolov2_onnx_cam](https://github.com/tsutof/tiny_yolov2_onnx_cam) <img src="https://img.shields.io/github/stars/tsutof/tiny_yolov2_onnx_cam?style=social"/> : Tiny YOLO v2 Inference Application with NVIDIA TensorRT.

      - [DataXujing/ncnn_android_yolov6](https://github.com/DataXujing/ncnn_android_yolov6) <img src="https://img.shields.io/github/stars/DataXujing/ncnn_android_yolov6?style=social"/> : 手摸手实现基于QT和NCNN的安卓手机YOLOv6模型的部署！

      - [Qengineering/YoloV3-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/YoloV3-ncnn-Raspberry-Pi-4) <img src="https://img.shields.io/github/stars/Qengineering/YoloV3-ncnn-Raspberry-Pi-4?style=social"/> : YoloV3 Raspberry Pi 4.

      - [Qengineering/YoloV4-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/YoloV4-ncnn-Raspberry-Pi-4) <img src="https://img.shields.io/github/stars/Qengineering/YoloV4-ncnn-Raspberry-Pi-4?style=social"/> : YoloV4 on a bare Raspberry Pi 4 with ncnn framework.    

      - [Qengineering/YoloV5-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/YoloV5-ncnn-Raspberry-Pi-4) <img src="https://img.shields.io/github/stars/Qengineering/YoloV5-ncnn-Raspberry-Pi-4?style=social"/> : YoloV5 for a bare Raspberry Pi 4.

      - [Qengineering/YoloV6-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/YoloV6-ncnn-Raspberry-Pi-4) <img src="https://img.shields.io/github/stars/Qengineering/YoloV6-ncnn-Raspberry-Pi-4?style=social"/> : YoloV6 for a bare Raspberry Pi using ncnn.
      
      - [Qengineering/YoloV7-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/YoloV7-ncnn-Raspberry-Pi-4) <img src="https://img.shields.io/github/stars/Qengineering/YoloV7-ncnn-Raspberry-Pi-4?style=social"/> : YoloV7 for a bare Raspberry Pi using ncnn.       

      - [Monday-Leo/Yolov5_Tensorrt_Win10](https://github.com/Monday-Leo/Yolov5_Tensorrt_Win10) <img src="https://img.shields.io/github/stars/Monday-Leo/Yolov5_Tensorrt_Win10?style=social"/> : A simple implementation of tensorrt yolov5 python/c++🔥




    - #### FPGA TPU NPU Hardware Deployment
      #### FPGA TPU NPU 硬件部署

      - [Xilinx/Vitis-AI](https://github.com/Xilinx/Vitis-AI/tree/master/demo) <img src="https://img.shields.io/github/stars/Xilinx/Vitis-AI?style=social"/> : Vitis AI offers a unified set of high-level C++/Python programming APIs to run AI applications across edge-to-cloud platforms, including DPU for Alveo, and DPU for Zynq Ultrascale+ MPSoC and Zynq-7000. It brings the benefits to easily port AI applications from cloud to edge and vice versa. 10 samples in [VART Samples](https://github.com/Xilinx/Vitis-AI/tree/master/demo/VART) are available to help you get familiar with the unfied programming APIs. [Vitis-AI-Library](https://github.com/Xilinx/Vitis-AI/tree/master/demo/Vitis-AI-Library) provides an easy-to-use and unified interface by encapsulating many efficient and high-quality neural networks.

      - [tensil-ai/tensil](https://github.com/tensil-ai/tensil) <img src="https://img.shields.io/github/stars/tensil-ai/tensil?style=social"/> : Open source machine learning accelerators. [www.tensil.ai](https://www.tensil.ai/)

      - [19801201/SpinalHDL_CNN_Accelerator](https://github.com/19801201/SpinalHDL_CNN_Accelerator) <img src="https://img.shields.io/github/stars/19801201/SpinalHDL_CNN_Accelerator?style=social"/> : CNN accelerator implemented with Spinal HDL.

      - [dhm2013724/yolov2_xilinx_fpga](https://github.com/dhm2013724/yolov2_xilinx_fpga) <img src="https://img.shields.io/github/stars/dhm2013724/yolov2_xilinx_fpga?style=social"/> : YOLOv2 Accelerator in Xilinx's Zynq-7000 Soc(PYNQ-z2, Zedboard and ZCU102). (**[硕士论文 2019](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1019228234.nh&uid=WEEvREcwSlJHSldRa1FhdXNXaEhoOGhUTzA5T0tESzdFZ2pyR1NJR1ZBaz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!&v=MjE5NTN5dmdXN3JBVkYyNkY3RzZGdFBQcTVFYlBJUjhlWDFMdXhZUzdEaDFUM3FUcldNMUZyQ1VSTE9lWnVkdUY=), [电子技术应用 2019](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFDLAST2019&filename=DZJY201908009&uid=WEEvREcwSlJHSldRa1FhdXNXaEhoOGhUTzA5T0tESzdFZ2pyR1NJR1ZBaz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!&v=MDU0NDJDVVJMT2VadWR1Rnl2Z1c3ck1JVGZCZDdHNEg5ak1wNDlGYllSOGVYMUx1eFlTN0RoMVQzcVRyV00xRnI=), [计算机科学与探索 2019](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFDTEMP&filename=KXTS201910005&uid=WEEvREcwSlJHSldRa1FhdXNXaEhoOGhUTzA5T0tESzdFZ2pyR1NJR1ZBaz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!&v=MjkwNzdXTTFGckNVUkxPZVp1ZHVGeXZnVzdyT0xqWGZmYkc0SDlqTnI0OUZZWVI4ZVgxTHV4WVM3RGgxVDNxVHI=)**)

      - [Yu-Zhewen/Tiny_YOLO_v3_ZYNQ](https://github.com/Yu-Zhewen/Tiny_YOLO_v3_ZYNQ) <img src="https://img.shields.io/github/stars/Yu-Zhewen/Tiny_YOLO_v3_ZYNQ?style=social"/> : Implement Tiny YOLO v3 on ZYNQ. "A Parameterisable FPGA-Tailored Architecture for YOLOv3-Tiny". (**[ARC 2020](https://link.springer.com/chapter/10.1007/978-3-030-44534-8_25)**)

      - [HSqure/ultralytics-pt-yolov3-vitis-ai-edge](https://github.com/HSqure/ultralytics-pt-yolov3-vitis-ai-edge) <img src="https://img.shields.io/github/stars/HSqure/ultralytics-pt-yolov3-vitis-ai-edge?style=social"/> : This demo is only used for inference testing of Vitis AI v1.4 and quantitative compilation of DPU. It is compatible with the training results of [ultralytics/yolov3](https://github.com/ultralytics/yolov3) v9.5.0 (it needs to use the model saving method of Pytorch V1.4).

      - [mcedrdiego/Kria_yolov3_ppe](https://github.com/mcedrdiego/Kria_yolov3_ppe) <img src="https://img.shields.io/github/stars/mcedrdiego/Kria_yolov3_ppe?style=social"/> : Kria KV260 Real-Time Personal Protective Equipment Detection. "Deep Learning for Site Safety: Real-Time Detection of Personal Protective Equipment". (**[Automation in Construction 2020](https://www.sciencedirect.com/science/article/abs/pii/S0926580519308325)**)

      - [xlsjdjdk/Ship-Detection-based-on-YOLOv3-and-KV260](https://github.com/xlsjdjdk/Ship-Detection-based-on-YOLOv3-and-KV260) <img src="https://img.shields.io/github/stars/xlsjdjdk/Ship-Detection-based-on-YOLOv3-and-KV260?style=social"/> : This is the entry project of the Xilinx Adaptive Computing Challenge 2021. It uses YOLOv3 for ship target detection in optical remote sensing images, and deploys DPU on the KV260 platform to achieve hardware acceleration. 

      - [Pomiculture/YOLOv4-Vitis-AI](https://github.com/Pomiculture/YOLOv4-Vitis-AI) <img src="https://img.shields.io/github/stars/Pomiculture/YOLOv4-Vitis-AI?style=social"/> : Custom YOLOv4 for apple recognition (clean/damaged) on Alveo U280 accelerator card using Vitis AI framework. 

      - [mkshuvo2/ZCU104_YOLOv3_Post_Processing](https://github.com/mkshuvo2/ZCU104_YOLOv3_Post_Processing) <img src="https://img.shields.io/github/stars/mkshuvo2/ZCU104_YOLOv3_Post_Processing?style=social"/> : Tensor outputs form Vitis AI Runner Class for YOLOv3.
      
      - [puffdrum/v4tiny_pt_quant](https://github.com/puffdrum/v4tiny_pt_quant) <img src="https://img.shields.io/github/stars/puffdrum/v4tiny_pt_quant?style=social"/> : quantization for yolo with xilinx/vitis-ai-pytorch.     

      - [chanshann/LITE_YOLOV3_TINY_VITISAI](https://github.com/chanshann/LITE_YOLOV3_TINY_VITISAI) <img src="https://img.shields.io/github/stars/chanshann/LITE_YOLOV3_TINY_VITISAI?style=social"/> : LITE_YOLOV3_TINY_VITISAI. 

      - [LukiBa/zybo_yolo](https://github.com/LukiBa/zybo_yolo) <img src="https://img.shields.io/github/stars/LukiBa/zybo_yolo?style=social"/> : YOLO example implementation using Intuitus CNN accelerator on ZYBO ZYNQ-7000 FPGA board.

      - [matsuda-slab/YOLO_ZYNQ_MASTER](https://github.com/matsuda-slab/YOLO_ZYNQ_MASTER) <img src="https://img.shields.io/github/stars/matsuda-slab/YOLO_ZYNQ_MASTER?style=social"/> : Implementation of YOLOv3-tiny on FPGA.   

      - [AramisOposich/tiny_YOLO_Zedboard](https://github.com/AramisOposich/tiny_YOLO_Zedboard) <img src="https://img.shields.io/github/stars/AramisOposich/tiny_YOLO_Zedboard?style=social"/> : tiny_YOLO_Zedboard.
 
      - [FerberZhang/Yolov2-FPGA-CNN-](https://github.com/FerberZhang/Yolov2-FPGA-CNN-) <img src="https://img.shields.io/github/stars/FerberZhang/Yolov2-FPGA-CNN-?style=social"/> : A demo for accelerating YOLOv2 in xilinx's fpga PYNQ.

      - [Prithvi-Velicheti/FPGA-Accelerator-for-TinyYolov3](https://github.com/Prithvi-Velicheti/FPGA-Accelerator-for-TinyYolov3) <img src="https://img.shields.io/github/stars/Prithvi-Velicheti/FPGA-Accelerator-for-TinyYolov3?style=social"/> : An FPGA-Accelerator-for-TinyYolov3.

      - [ChainZeeLi/FPGA_DPU](https://github.com/ChainZeeLi/FPGA_DPU) <img src="https://img.shields.io/github/stars/ChainZeeLi/FPGA_DPU?style=social"/> : This project is to implement YOLO v3 on Xilinx FPGA with DPU.

      - [xbdxwyh/yolov3_fpga_project](https://github.com/xbdxwyh/yolov3_fpga_project) <img src="https://img.shields.io/github/stars/xbdxwyh/yolov3_fpga_project?style=social"/> : yolov3_fpga_project.

      - [ZLkanyo009/Yolo-compression-and-deployment-in-FPGA](https://github.com/ZLkanyo009/Yolo-compression-and-deployment-in-FPGA) <img src="https://img.shields.io/github/stars/ZLkanyo009/Yolo-compression-and-deployment-in-FPGA?style=social"/> : 基于FPGA量化的人脸口罩检测。

      - [xiying-boy/yolov3-AX7350](https://github.com/xiying-boy/yolov3-AX7350) <img src="https://img.shields.io/github/stars/xiying-boy/yolov3-AX7350?style=social"/> : 基于HLS_YOLOV3的驱动文件。

      - [himewel/yolowell](https://github.com/himewel/yolowell) <img src="https://img.shields.io/github/stars/himewel/yolowell?style=social"/> : A set of hardware architectures to build a co-design of convolutional neural networks inference at FPGA devices.

      - [embedeep/Free-TPU](https://github.com/embedeep/Free-TPU) <img src="https://img.shields.io/github/stars/embedeep/Free-TPU?style=social"/> : Free TPU for FPGA with Lenet, MobileNet, Squeezenet, Resnet, Inception V3, YOLO V3, and ICNet. Deep learning acceleration using Xilinx zynq (Zedboard or ZC702 ) or kintex-7 to solve image classification, detection, and segmentation problem.

      - [yarakigit/design_contest_yolo_change_ps_to_pl](https://github.com/yarakigit/design_contest_yolo_change_ps_to_pl) <img src="https://img.shields.io/github/stars/yarakigit/design_contest_yolo_change_ps_to_pl?style=social"/> : Converts pytorch yolo format weights to C header files for bare-metal (FPGA implementation).

      - [MasLiang/CNN-On-FPGA](https://github.com/MasLiang/CNN-On-FPGA) <img src="https://img.shields.io/github/stars/MasLiang/CNN-On-FPGA?style=social"/> : This is the code of the CNN on FPGA.But this can only be used for reference at present for some files are write coarsly using ISE.

      - [adamgallas/fpga_accelerator_yolov3tiny](https://github.com/adamgallas/fpga_accelerator_yolov3tiny) <img src="https://img.shields.io/github/stars/adamgallas/fpga_accelerator_yolov3tiny?style=social"/> : fpga_accelerator_yolov3tiny.

      - [zhen8838/K210_Yolo_framework](https://github.com/zhen8838/K210_Yolo_framework) <img src="https://img.shields.io/github/stars/zhen8838/K210_Yolo_framework?style=social"/> : Yolo v3 framework base on tensorflow, support multiple models, multiple datasets, any number of output layers, any number of anchors, model prune, and portable model to K210 !

      - [SEASKY-Master/SEASKY_K210](https://github.com/SEASKY-Master/SEASKY_K210) <img src="https://img.shields.io/github/stars/SEASKY-Master/SEASKY_K210?style=social"/> : K210 PCB YOLO.

      - [SEASKY-Master/Yolo-for-k210](https://github.com/SEASKY-Master/Yolo-for-k210) <img src="https://img.shields.io/github/stars/SEASKY-Master/Yolo-for-k210?style=social"/> : Yolo-for-k210.

      - [TonyZ1Min/yolo-for-k210](https://github.com/TonyZ1Min/yolo-for-k210) <img src="https://img.shields.io/github/stars/TonyZ1Min/yolo-for-k210?style=social"/> : keras-yolo-for-k210.

      - [guichristmann/edge-tpu-tiny-yolo](https://github.com/guichristmann/edge-tpu-tiny-yolo) <img src="https://img.shields.io/github/stars/guichristmann/edge-tpu-tiny-yolo?style=social"/> : Run Tiny YOLO-v3 on Google's Edge TPU USB Accelerator.

      - [Charlie839242/-Trash-Classification-Car](https://github.com/Charlie839242/-Trash-Classification-Car) <img src="https://img.shields.io/github/stars/Charlie839242/-Trash-Classification-Car?style=social"/> : 这是一个基于yolo-fastest模型的小车，主控是art-pi开发板，使用了rt thread操作系统。

      - [Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry](https://github.com/Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry) <img src="https://img.shields.io/github/stars/Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry?style=social"/> : This project deploys a yolo fastest model in the form of tflite on raspberry 3b+. 

      - [mahxn0/Hisi3559A_Yolov5](https://github.com/mahxn0/Hisi3559A_Yolov5) <img src="https://img.shields.io/github/stars/mahxn0/Hisi3559A_Yolov5?style=social"/> : 基于hisi3559a的yolov5训练部署全流程。

      - [ZhenxinYUAN/YOLO_hi3516Deploy](https://github.com/ZhenxinYUAN/YOLO_hi3516Deploy) <img src="https://img.shields.io/github/stars/ZhenxinYUAN/YOLO_hi3516Deploy?style=social"/> : Deploy Yolo series algorithms on Hisilicon platform hi3516, including yolov3, yolov5, yolox, etc.
      
      - [jveitchmichaelis/edgetpu-yolo](https://github.com/jveitchmichaelis/edgetpu-yolo) <img src="https://img.shields.io/github/stars/jveitchmichaelis/edgetpu-yolo?style=social"/> : Minimal-dependency Yolov5 export and inference demonstration for the Google Coral EdgeTPU.

      - [xiaqing10/Hisi_YoLoV5](https://github.com/xiaqing10/Hisi_YoLoV5) <img src="https://img.shields.io/github/stars/xiaqing10/Hisi_YoLoV5?style=social"/> : 海思nnie跑yolov5。

      - [BaronLeeLZP/hi3516dv300_nnie-yolov3-demo](https://github.com/BaronLeeLZP/hi3516dv300_nnie-yolov3-demo) <img src="https://img.shields.io/github/stars/BaronLeeLZP/hi3516dv300_nnie-yolov3-demo?style=social"/> : 在海思Hisilicon的Hi3516dv300芯片上，利用nnie和opencv库，简洁了官方yolov3用例中各种复杂的嵌套调用/复杂编译，提供了交叉编译后可成功上板部署运行的demo。

      - [Zhou-sx/yolov5_Deepsort_rknn](https://github.com/Zhou-sx/yolov5_Deepsort_rknn) <img src="https://img.shields.io/github/stars/Zhou-sx/yolov5_Deepsort_rknn?style=social"/> : Track vehicles and persons on rk3588 / rk3399pro. 

      - [OpenVINO-dev-contest/YOLOv7_OpenVINO](https://github.com/OpenVINO-dev-contest/YOLOv7_OpenVINO) <img src="https://img.shields.io/github/stars/OpenVINO-dev-contest/YOLOv7_OpenVINO?style=social"/> : This repository will demostrate how to deploy a offical YOLOv7 pre-trained model with OpenVINO runtime api. 



  - ### Video Object Detection
    #### 视频目标检测

      - [StreamYOLO](https://github.com/yancie-yjr/StreamYOLO) <img src="https://img.shields.io/github/stars/yancie-yjr/StreamYOLO?style=social"/> : "Real-time Object Detection for Streaming Perception". (**[CVPR 2022](https://arxiv.org/abs/2203.12338v1)**)

      - [NoScope](https://github.com/stanford-futuredata/noscope) <img src="https://img.shields.io/github/stars/stanford-futuredata/noscope?style=social"/> : "Noscope: optimizing neural network queries over video at scale". (**[arXiv 2017](https://arxiv.org/abs/1703.02529)**)


  - ### Object Tracking
    #### 目标跟踪

    - ####  Multi-Object Tracking
      #####  多目标跟踪

      - [mikel-brostrom/Yolov5_StrongSORT_OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet) <img src="https://img.shields.io/github/stars/mikel-brostrom/Yolov5_StrongSORT_OSNet?style=social"/> : Real-time multi-camera multi-object tracker using YOLOv5 and StrongSORT with OSNet.

      - [mikel-brostrom/Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch) <img src="https://img.shields.io/github/stars/mikel-brostrom/Yolov5_DeepSort_Pytorch?style=social"/> : Real-time multi-object tracker using YOLO v5 and deep sort.

      - [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch) <img src="https://img.shields.io/github/stars/ZQPei/deep_sort_pytorch?style=social"/> : MOT using deepsort and yolov3 with pytorch.

      - [Qidian213/deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3) <img src="https://img.shields.io/github/stars/Qidian213/deep_sort_yolov3?style=social"/> : Real-time Multi-person tracker using YOLO v3 and deep_sort with tensorflow.

      - [CSTrack](https://github.com/JudasDie/SOTS) <img src="https://img.shields.io/github/stars/JudasDie/SOTS?style=social"/> : "Rethinking the competition between detection and ReID in Multi-Object Tracking". (**[arXiv 2020](https://arxiv.org/abs/2010.12138)**)

      - [ROLO](https://github.com/Guanghan/ROLO) <img src="https://img.shields.io/github/stars/Guanghan/ROLO?style=social"/> : ROLO is short for Recurrent YOLO, aimed at simultaneous object detection and tracking.

      - [FastMOT](https://github.com/GeekAlexis/FastMOT) <img src="https://img.shields.io/github/stars/GeekAlexis/FastMOT?style=social"/> : "FastMOT: High-Performance Multiple Object Tracking Based on Deep SORT and KLT". (**[Zenodo 2020](https://doi.org/10.5281/zenodo.4294717)**)

      - [Sharpiless/Yolov5-deepsort-inference](https://github.com/Sharpiless/Yolov5-deepsort-inference) <img src="https://img.shields.io/github/stars/Sharpiless/Yolov5-deepsort-inference?style=social"/> : 使用YOLOv5+Deepsort实现车辆行人追踪和计数，代码封装成一个Detector类，更容易嵌入到自己的项目中。

      - [Sharpiless/Yolov5-Deepsort](https://github.com/Sharpiless/Yolov5-Deepsort) <img src="https://img.shields.io/github/stars/Sharpiless/Yolov5-Deepsort?style=social"/> : 最新版本yolov5+deepsort目标检测和追踪，能够显示目标类别，支持5.0版本可训练自己数据集。

      - [LeonLok/Multi-Camera-Live-Object-Tracking](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking) <img src="https://img.shields.io/github/stars/LeonLok/Multi-Camera-Live-Object-Tracking?style=social"/> : Multi-camera live traffic and object counting with YOLO v4, Deep SORT, and Flask.

      - [LeonLok/Deep-SORT-YOLOv4](https://github.com/LeonLok/Deep-SORT-YOLOv4) <img src="https://img.shields.io/github/stars/LeonLok/Deep-SORT-YOLOv4?style=social"/> : People detection and optional tracking with Tensorflow backend.

      - [obendidi/Tracking-with-darkflow](https://github.com/obendidi/Tracking-with-darkflow) <img src="https://img.shields.io/github/stars/obendidi/Tracking-with-darkflow?style=social"/> : Real-time people Multitracker using YOLO v2 and deep_sort with tensorflow.

      - [DrewNF/Tensorflow_Object_Tracking_Video](https://github.com/DrewNF/Tensorflow_Object_Tracking_Video) <img src="https://img.shields.io/github/stars/DrewNF/Tensorflow_Object_Tracking_Video?style=social"/> : Object Tracking in Tensorflow ( Localization Detection Classification ) developed to partecipate to ImageNET VID competition.

      - [dyh/unbox_yolov5_deepsort_counting](https://github.com/dyh/unbox_yolov5_deepsort_counting) <img src="https://img.shields.io/github/stars/dyh/unbox_yolov5_deepsort_counting?style=social"/> : yolov5 deepsort 行人 车辆 跟踪 检测 计数。

      - [theAIGuysCode/yolov3_deepsort](https://github.com/theAIGuysCode/yolov3_deepsort) <img src="https://img.shields.io/github/stars/theAIGuysCode/yolov3_deepsort?style=social"/> : Object tracking implemented with YOLOv3, Deep Sort and Tensorflow.

      - [weixu000/libtorch-yolov3-deepsort](https://github.com/weixu000/libtorch-yolov3-deepsort) <img src="https://img.shields.io/github/stars/weixu000/libtorch-yolov3-deepsort?style=social"/> : libtorch-yolov3-deepsort.

      - [pmj110119/YOLOX_deepsort_tracker](https://github.com/pmj110119/YOLOX_deepsort_tracker) <img src="https://img.shields.io/github/stars/pmj110119/YOLOX_deepsort_tracker?style=social"/> : using yolox+deepsort for object-tracking.

      - [abhyantrika/nanonets_object_tracking](https://github.com/abhyantrika/nanonets_object_tracking) <img src="https://img.shields.io/github/stars/abhyantrika/nanonets_object_tracking?style=social"/> : nanonets_object_tracking.

      - [mattzheng/keras-yolov3-KF-objectTracking](https://github.com/mattzheng/keras-yolov3-KF-objectTracking) <img src="https://img.shields.io/github/stars/mattzheng/keras-yolov3-KF-objectTracking?style=social"/> : 以kears-yolov3做detector，以Kalman-Filter算法做tracker，进行多人物目标追踪。

      - [rohanchandra30/TrackNPred](https://github.com/rohanchandra30/TrackNPred) <img src="https://img.shields.io/github/stars/rohanchandra30/TrackNPred?style=social"/> : A Software Framework for End-to-End Trajectory Prediction.

      - [RichardoMrMu/yolov5-deepsort-tensorrt](https://github.com/RichardoMrMu/yolov5-deepsort-tensorrt) <img src="https://img.shields.io/github/stars/RichardoMrMu/yolov5-deepsort-tensorrt?style=social"/> : A c++ implementation of yolov5 and deepsort.

      - [bamwani/car-counting-and-speed-estimation-yolo-sort-python](https://github.com/bamwani/car-counting-and-speed-estimation-yolo-sort-python) <img src="https://img.shields.io/github/stars/bamwani/car-counting-and-speed-estimation-yolo-sort-python?style=social"/> : This project imlements the following tasks in the project: 1. Vehicle counting, 2. Lane detection. 3.Lane change detection and 4.speed estimation.

      - [ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking) <img src="https://img.shields.io/github/stars/ArtLabss/tennis-tracking?style=social"/> : Open-source Monocular Python HawkEye for Tennis.

      - [CaptainEven/YOLOV4_MCMOT](https://github.com/CaptainEven/YOLOV4_MCMOT) <img src="https://img.shields.io/github/stars/CaptainEven/YOLOV4_MCMOT?style=social"/> : Using YOLOV4 as detector for MCMOT.

      - [opendatacam/node-moving-things-tracker](https://github.com/opendatacam/node-moving-things-tracker) <img src="https://img.shields.io/github/stars/opendatacam/node-moving-things-tracker?style=social"/> : javascript implementation of "tracker by detections" for realtime multiple object tracking (MOT).

      - [lanmengyiyu/yolov5-deepmar](https://github.com/lanmengyiyu/yolov5-deepmar) <img src="https://img.shields.io/github/stars/lanmengyiyu/yolov5-deepmar?style=social"/> : 行人轨迹和属性分析。

      - [zengwb-lx/Yolov5-Deepsort-Fastreid](https://github.com/zengwb-lx/Yolov5-Deepsort-Fastreid) <img src="https://img.shields.io/github/stars/zengwb-lx/Yolov5-Deepsort-Fastreid?style=social"/> : YoloV5 + deepsort + Fast-ReID 完整行人重识别系统。

      - [tensorturtle/classy-sort-yolov5](https://github.com/tensorturtle/classy-sort-yolov5) <img src="https://img.shields.io/github/stars/tensorturtle/classy-sort-yolov5?style=social"/> : Ready-to-use realtime multi-object tracker that works for any object category. YOLOv5 + SORT implementation.

      - [supperted825/FairMOT-X](https://github.com/supperted825/FairMOT-X) <img src="https://img.shields.io/github/stars/supperted825/FairMOT-X?style=social"/> : FairMOT for Multi-Class MOT using YOLOX as Detector.

      - [deyiwang89/pytorch-yolov7-deepsort](https://github.com/deyiwang89/pytorch-yolov7-deepsort) <img src="https://img.shields.io/github/stars/deyiwang89/pytorch-yolov7-deepsort?style=social"/> : an implentation of yolov7-deepsort based on pytorch.

      - [xuarehere/yolovx_deepsort_pytorch](https://github.com/xuarehere/yolovx_deepsort_pytorch) <img src="https://img.shields.io/github/stars/xuarehere/yolovx_deepsort_pytorch?style=social"/> : this project support the existing yolo detection model algorithm (YOLOv3, YOLOV4, YOLOV4Scaled, YOLOV5, YOLOV6, YOLOV7 ). 

      - [deshwalmahesh/yolov7-deepsort-tracking](https://github.com/deshwalmahesh/yolov7-deepsort-tracking) <img src="https://img.shields.io/github/stars/deshwalmahesh/yolov7-deepsort-tracking?style=social"/> : Modular and ready to deploy code to detect and track videos using YOLO-v7 and DeepSORT.

      - [BoT-SORT](https://github.com/NirAharon/BoT-SORT) <img src="https://img.shields.io/github/stars/NirAharon/BoT-SORT?style=social"/> : "BoT-SORT: Robust Associations Multi-Pedestrian Tracking". (**[arXiv 2022](https://arxiv.org/abs/2206.14651)**)

      - [bharath5673/StrongSORT-YOLO](https://github.com/bharath5673/StrongSORT-YOLO) <img src="https://img.shields.io/github/stars/bharath5673/StrongSORT-YOLO?style=social"/> : Real-time multi-camera multi-object tracker using (YOLOv5, YOLOv7) and StrongSORT with OSNet.


  - #### Deep Reinforcement Learning
    #### 深度强化学习

    - [uzkent/EfficientObjectDetection](https://github.com/uzkent/EfficientObjectDetection) <img src="https://img.shields.io/github/stars/uzkent/EfficientObjectDetection?style=social"/> : "Efficient Object Detection in Large Images with Deep Reinforcement Learning". (**[WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/html/Uzkent_Efficient_Object_Detection_in_Large_Images_Using_Deep_Reinforcement_Learning_WACV_2020_paper.html)**)


  - #### Multi-Modality Information Fusion
    #### 多模态信息融合

      - [mjoshi07/Visual-Sensor-Fusion](https://github.com/mjoshi07/Visual-Sensor-Fusion) <img src="https://img.shields.io/github/stars/mjoshi07/Visual-Sensor-Fusion?style=social"/> : LiDAR Fusion with Vision.

      - [DocF/multispectral-object-detection](https://github.com/DocF/multispectral-object-detection) <img src="https://img.shields.io/github/stars/DocF/multispectral-object-detection?style=social"/> : Multispectral Object Detection with Yolov5 and Transformer.

      - [MAli-Farooq/Thermal-YOLO](https://github.com/MAli-Farooq/Thermal-YOLO) <img src="https://img.shields.io/github/stars/sierprinsky/YoloV5_blood_cells?style=social"/> : This study is related to object detection in thermal infrared spectrum using YOLO-V5 framework for ADAS application.    

      - [Ye-zixiao/Double-YOLO-Kaist](https://github.com/Ye-zixiao/Double-YOLO-Kaist) <img src="https://img.shields.io/github/stars/Ye-zixiao/Double-YOLO-Kaist?style=social"/> : 一种基于YOLOv3/4的双流混合模态道路行人检测方法🌊💧💦。 

      - [eralso/yolov5_Visible_Infrared_Vehicle_Detection](https://github.com/eralso/yolov5_Visible_Infrared_Vehicle_Detection) <img src="https://img.shields.io/github/stars/eralso/yolov5_Visible_Infrared_Vehicle_Detection?style=social"/> : 基于可见光和红外图像的深度学习车辆目标检测。 


  - #### Motion Control Field
    #### 运动控制领域

    - [icns-distributed-cloud/adaptive-cruise-control](https://github.com/icns-distributed-cloud/adaptive-cruise-control) <img src="https://img.shields.io/github/stars/icns-distributed-cloud/adaptive-cruise-control?style=social"/> : YOLO-v5 기반 "단안 카메라"의 영상을 활용해 차간 거리를 일정하게 유지하며 주행하는 Adaptive Cruise Control 기능 구현.

    - [LeBronLiHD/ZJU2021_MotionControl_PID_YOLOv5](https://github.com/LeBronLiHD/ZJU2021_MotionControl_PID_YOLOv5) <img src="https://img.shields.io/github/stars/LeBronLiHD/ZJU2021_MotionControl_PID_YOLOv5?style=social"/> : ZJU2021_MotionControl_PID_YOLOv5.

    - [SananSuleymanov/PID_YOLOv5s_ROS_Diver_Tracking](https://github.com/SananSuleymanov/PID_YOLOv5s_ROS_Diver_Tracking) <img src="https://img.shields.io/github/stars/SananSuleymanov/PID_YOLOv5s_ROS_Diver_Tracking?style=social"/> : PID_YOLOv5s_ROS_Diver_Tracking.


  - #### Super-Resolution Field
    #### 超分辨率领域

    - [Fireboltz/Psychic-CCTV](https://github.com/Fireboltz/Psychic-CCTV) <img src="https://img.shields.io/github/stars/Fireboltz/Psychic-CCTV?style=social"/> : A video analysis tool built completely in python. 


  - #### Spiking Neural Network
    #### SNN, 脉冲神经网络

    - [Spiking-YOLOv3](https://github.com/cwq159/PyTorch-Spiking-YOLOv3) <img src="https://img.shields.io/github/stars/cwq159/PyTorch-Spiking-YOLOv3?style=social"/> : A PyTorch implementation of Spiking-YOLOv3. Two branches are provided, based on two common PyTorch implementation of YOLOv3([ultralytics/yolov3](https://github.com/ultralytics/yolov3) & [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)), with support for Spiking-YOLOv3-Tiny at present. (**[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6787)**)

    - [fjcu-ee-islab/Spiking_Converted_YOLOv4](https://github.com/fjcu-ee-islab/Spiking_Converted_YOLOv4) <img src="https://img.shields.io/github/stars/fjcu-ee-islab/Spiking_Converted_YOLOv4?style=social"/> : Object Detection Based on Dynamic Vision Sensor with Spiking Neural Network.

    - [Zaabon/spiking_yolo](https://github.com/Zaabon/spiking_yolo) <img src="https://img.shields.io/github/stars/Zaabon/spiking_yolo?style=social"/> : This project is a combined neural network utilizing an spiking CNN with backpropagation and YOLOv3 for object detection.

    - [Dignity-ghost/PyTorch-Spiking-YOLOv3](https://github.com/Dignity-ghost/PyTorch-Spiking-YOLOv3) <img src="https://img.shields.io/github/stars/Dignity-ghost/PyTorch-Spiking-YOLOv3?style=social"/> : A modified repository based on [Spiking-YOLOv3](https://github.com/cwq159/PyTorch-Spiking-YOLOv3) and [YOLOv3](https://pjreddie.com/darknet/yolo), which makes it suitable for VOC-dataset and YOLOv2.


  - #### Attention and Transformer
    #### 注意力机制

    - [xmu-xiaoma666/External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch) <img src="https://img.shields.io/github/stars/xmu-xiaoma666/External-Attention-pytorch?style=social"/> : 🍀 Pytorch implementation of various Attention Mechanisms, MLP, Re-parameter, Convolution, which is helpful to further understand papers.⭐⭐⭐.

    - [MenghaoGuo/Awesome-Vision-Attentions](https://github.com/MenghaoGuo/Awesome-Vision-Attentions) <img src="https://img.shields.io/github/stars/MenghaoGuo/Awesome-Vision-Attentions?style=social"/> : Summary of related papers on visual attention. Related code will be released based on Jittor gradually. "Attention Mechanisms in Computer Vision: A Survey". (**[arXiv 2021](https://arxiv.org/abs/2111.07624)**)

    - [pprp/awesome-attention-mechanism-in-cv](https://github.com/pprp/awesome-attention-mechanism-in-cv) <img src="https://img.shields.io/github/stars/pprp/awesome-attention-mechanism-in-cv?style=social"/> : 👊 CV中常用注意力模块;即插即用模块;ViT模型. PyTorch Implementation Collection of Attention Module and Plug&Play Module.

    - [positive666/yolov5_research](https://github.com/positive666/yolov5_research) <img src="https://img.shields.io/github/stars/positive666/yolov5_research?style=social"/> : add yolov7 core ,improvement research based on yolov5,SwintransformV2 and Attention Series. training skills, business customization, engineering deployment C. 🌟 基于yolov5&&yolov7的改进库。

    - [HaloTrouvaille/YOLO-Multi-Backbones-Attention](https://github.com/HaloTrouvaille/YOLO-Multi-Backbones-Attention) <img src="https://img.shields.io/github/stars/HaloTrouvaille/YOLO-Multi-Backbones-Attention?style=social"/> : This Repository includes YOLOv3 with some lightweight backbones (ShuffleNetV2, GhostNet, VoVNet), some computer vision attention mechanism (SE Block, CBAM Block, ECA Block), pruning,quantization and distillation for GhostNet.

    - [kay-cottage/CoordAttention_YOLOX_Pytorch](https://github.com/kay-cottage/CoordAttention_YOLOX_Pytorch) <img src="https://img.shields.io/github/stars/kay-cottage/CoordAttention_YOLOX_Pytorch?style=social"/> : CoordAttention_YOLOX(基于CoordAttention坐标注意力机制的改进版YOLOX目标检测平台）。 "Coordinate Attention for Efficient Mobile Network Design". (**[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.html), [ Andrew-Qibin/CoordAttention](https://github.com/Andrew-Qibin/CoordAttention)**)

    - [liangzhendong123/Attention-yolov5](https://github.com/liangzhendong123/Attention-yolov5) <img src="https://img.shields.io/github/stars/liangzhendong123/Attention-yolov5?style=social"/> : 基于注意力机制改进的yolov5模型。

    - [e96031413/AA-YOLO](https://github.com/e96031413/AA-YOLO) <img src="https://img.shields.io/github/stars/e96031413/AA-YOLO?style=social"/> : Attention ALL-CNN Twin Head YOLO (AA -YOLO). "Improving Tiny YOLO with Fewer Model Parameters". (**[IEEE BigMM 2021](https://ieeexplore.ieee.org/abstract/document/9643269/)**)

    - [anonymoussss/YOLOX-SwinTransformer](https://github.com/anonymoussss/YOLOX-SwinTransformer) <img src="https://img.shields.io/github/stars/anonymoussss/YOLOX-SwinTransformer?style=social"/> : YOLOX with Swin-Transformer backbone.



  - ### Small Object Detection
    #### 小目标检测

    - [kuanhungchen/awesome-tiny-object-detection](https://github.com/kuanhungchen/awesome-tiny-object-detection) <img src="https://img.shields.io/github/stars/kuanhungchen/awesome-tiny-object-detection?style=social"/> : 🕶 A curated list of Tiny Object Detection papers and related resources. 

    - [SAHI](https://github.com/obss/sahi) <img src="https://img.shields.io/github/stars/obss/sahi?style=social"/> : "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection". (**[arXiv 2022](https://arxiv.org/abs/2202.06934v2), [Zenodo 2021](https://doi.org/10.5281/zenodo.5718950)**). A lightweight vision library for performing large scale object detection/ instance segmentation. SAHI currently supports [YOLOv5 models](https://github.com/ultralytics/yolov5/releases), [MMDetection models](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/model_zoo.md), [Detectron2 models](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md), [HuggingFace models](https://huggingface.co/models?pipeline_tag=object-detection&sort=downloads) and [TorchVision models](https://pytorch.org/docs/stable/torchvision/models.html). 

    - [kadirnar/yolov5-sahi](https://github.com/kadirnar/yolov5-sahi) <img src="https://img.shields.io/github/stars/kadirnar/yolov5-sahi?style=social"/> : Yolov5 Modelini Kullanarak Özel Nesne Eğitimi ve SAHI Kullanımı.

    - [kadirnar/Yolov6-SAHI](https://github.com/kadirnar/Yolov6-SAHI) <img src="https://img.shields.io/github/stars/kadirnar/Yolov6-SAHI?style=social"/> : Yolov6-SAHI.

    - [YOLT](https://github.com/avanetten/yolt) <img src="https://img.shields.io/github/stars/avanetten/yolt?style=social"/> : "You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery". (**[arXiv 2018](https://arxiv.org/abs/1805.09512)**). "江大白：《[基于大尺寸图像的小目标检测竞赛经验总结](https://mp.weixin.qq.com/s/qbbd5FdyKKk7UI3mmGBt4Q)》"

    - [SIMRDWN](https://github.com/avanetten/simrdwn) <img src="https://img.shields.io/github/stars/avanetten/simrdwn?style=social"/> : "Satellite Imagery Multiscale Rapid Detection with Windowed Networks". (**[arXiv 2018](https://arxiv.org/abs/1809.09978), [WACV 2019](https://ieeexplore.ieee.org/abstract/document/8659155)**)

    - [YOLTv5](https://github.com/avanetten/yoltv5) <img src="https://img.shields.io/github/stars/avanetten/yoltv5?style=social"/> : YOLTv5 builds upon [YOLT](https://github.com/avanetten/yolt) and [SIMRDWN](https://github.com/avanetten/simrdwn), and updates these frameworks to use the [ultralytics/yolov5](https://github.com/ultralytics/yolov5) version of the YOLO object detection family.

    - [TPH-YOLOv5](https://github.com/cv516Buaa/tph-yolov5) <img src="https://img.shields.io/github/stars/cv516Buaa/tph-yolov5?style=social"/> : "TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-Captured Scenarios". (**[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/html/Zhu_TPH-YOLOv5_Improved_YOLOv5_Based_on_Transformer_Prediction_Head_for_Object_ICCVW_2021_paper.html)**)

    - [mwaseema/Drone-Detection](https://github.com/mwaseema/Drone-Detection) <img src="https://img.shields.io/github/stars/mwaseema/Drone-Detection?style=social"/> : "Dogfight: Detecting Drones from Drones Videos". (**[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Ashraf_Dogfight_Detecting_Drones_From_Drones_Videos_CVPR_2021_paper.html)**)

    - [KevinMuyaoGuo/yolov5s_for_satellite_imagery](https://github.com/KevinMuyaoGuo/yolov5s_for_satellite_imagery) <img src="https://img.shields.io/github/stars/KevinMuyaoGuo/yolov5s_for_satellite_imagery?style=social"/> : 基于YOLOv5的卫星图像目标检测demo | A demo for satellite imagery object detection based on YOLOv5。

    - [Hongyu-Yue/yoloV5_modify_smalltarget](https://github.com/Hongyu-Yue/yoloV5_modify_smalltarget) <img src="https://img.shields.io/github/stars/Hongyu-Yue/yoloV5_modify_smalltarget?style=social"/> : YOLOV5 小目标检测修改版。

    - [muyuuuu/Self-Supervise-Object-Detection](https://github.com/muyuuuu/Self-Supervise-Object-Detection) <img src="https://img.shields.io/github/stars/muyuuuu/Self-Supervise-Object-Detection?style=social"/> : Self-Supervised Object Detection. 水面漂浮垃圾目标检测，分析源码改善 yolox 检测小目标的缺陷，提出自监督算法预训练无标签数据，提升检测性能。

    - [swricci/small-boat-detector](https://github.com/swricci/small-boat-detector) <img src="https://img.shields.io/github/stars/swricci/small-boat-detector?style=social"/> : Trained yolo v3 model weights and configuration file to detect small boats in satellite imagery.

    - [Resham-Sundar/sahi-yolox](https://github.com/Resham-Sundar/sahi-yolox) <img src="https://img.shields.io/github/stars/Resham-Sundar/sahi-yolox?style=social"/> : YoloX with SAHI Implementation.

    - YOLO-Z : "YOLO-Z: Improving small object detection in YOLOv5 for autonomous vehicles". (**[arXiv 2021](https://arxiv.org/abs/2112.11798)**). "计算机视觉研究院：《[Yolo-Z：改进的YOLOv5用于小目标检测（附原论文下载）](https://mp.weixin.qq.com/s/ehkUapLOMdDghF2kAoAV4w)》".

    - [ultralytics/xview-yolov3](https://github.com/ultralytics/xview-yolov3) <img src="https://img.shields.io/github/stars/ultralytics/xview-yolov3?style=social"/> : xView 2018 Object Detection Challenge: YOLOv3 Training and Inference.

    - [inderpreet1390/yolov5-small-target](https://github.com/inderpreet1390/yolov5-small-target) <img src="https://img.shields.io/github/stars/inderpreet1390/yolov5-small-target?style=social"/> : Repository for improved yolov5 for small target detection.

    - [AllenSquirrel/YOLOv3_ReSAM](https://github.com/AllenSquirrel/YOLOv3_ReSAM) <img src="https://img.shields.io/github/stars/AllenSquirrel/YOLOv3_ReSAM?style=social"/> : YOLOv3_ReSAM:A Small Target Detection Method With Spatial Attention Module.

    - [shaunyuan22/SODA](https://github.com/shaunyuan22/SODA) <img src="https://img.shields.io/github/stars/shaunyuan22/SODA?style=social"/> : Official code library for SODA: A Large-scale Benchmark for Small Object Detection. "Towards Large-Scale Small Object Detection: Survey and Benchmarks". (**[arXiv 2022](https://arxiv.org/abs/2207.14096)**)


  - ### Few-shot Object Detection
    #### 少样本目标检测

    - [bingykang/Fewshot_Detection](https://github.com/bingykang/Fewshot_Detection) <img src="https://img.shields.io/github/stars/bingykang/Fewshot_Detection?style=social"/> : "Few-shot Object Detection via Feature Reweighting". (**[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Kang_Few-Shot_Object_Detection_via_Feature_Reweighting_ICCV_2019_paper.html)**). 

    - [SSDA-YOLO](https://github.com/hnuzhy/SSDA-YOLO) <img src="https://img.shields.io/github/stars/hnuzhy/SSDA-YOLO?style=social"/> : Codes for my paper "SSDA-YOLO: Semi-supervised Domain Adaptive YOLO for Cross-Domain Object Detection".


  - ### Oriented Object Detection
    #### 旋转目标检测

    - [AlphaRotate](https://github.com/yangxue0827/RotationDetection) <img src="https://img.shields.io/github/stars/yangxue0827/RotationDetection?style=social"/> : "AlphaRotate: A Rotation Detection Benchmark using TensorFlow". (**[arXiv 2021](https://arxiv.org/abs/2111.06677)**)

    - [hukaixuan19970627/yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb) <img src="https://img.shields.io/github/stars/hukaixuan19970627/yolov5_obb?style=social"/> : yolov5 + csl_label.(Oriented Object Detection)（Rotation Detection）（Rotated BBox）基于yolov5的旋转目标检测。

    - [BossZard/rotation-yolov5](https://github.com/BossZard/rotation-yolov5) <img src="https://img.shields.io/github/stars/BossZard/rotation-yolov5?style=social"/> : rotation detection based on yolov5.

    - [acai66/yolov5_rotation](https://github.com/acai66/yolov5_rotation) <img src="https://img.shields.io/github/stars/acai66/yolov5_rotation?style=social"/> : rotated bbox detection. inspired by [hukaixuan19970627/yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb), thanks hukaixuan19970627.

    - [ming71/rotate-yolov3](https://github.com/ming71/rotate-yolov3) <img src="https://img.shields.io/github/stars/ming71/rotate-yolov3?style=social"/> : Arbitrary oriented object detection implemented with yolov3 (attached with some tricks).

    - [ming71/yolov3-polygon](https://github.com/ming71/yolov3-polygon) <img src="https://img.shields.io/github/stars/ming71/yolov3-polygon?style=social"/> : Arbitrary-oriented object detection based on yolov3.

    - [kunnnnethan/R-YOLOv4](https://github.com/kunnnnethan/R-YOLOv4) <img src="https://img.shields.io/github/stars/kunnnnethan/R-YOLOv4?style=social"/> : This is a PyTorch-based R-YOLOv4 implementation which combines YOLOv4 model and loss function from R3Det for arbitrary oriented object detection.

    - [XinzeLee/PolygonObjectDetection](https://github.com/XinzeLee/PolygonObjectDetection) <img src="https://img.shields.io/github/stars/XinzeLee/PolygonObjectDetection?style=social"/> : This repository is based on Ultralytics/yolov5, with adjustments to enable polygon prediction boxes.

    - [hukaixuan19970627/DOTA_devkit_YOLO](https://github.com/hukaixuan19970627/DOTA_devkit_YOLO) <img src="https://img.shields.io/github/stars/hukaixuan19970627/DOTA_devkit_YOLO?style=social"/> : Trans DOTA OBB format(poly format) to YOLO format.

    - [hpc203/rotate-yolov5-opencv-onnxrun](https://github.com/hpc203/rotate-yolov5-opencv-onnxrun) <img src="https://img.shields.io/github/stars/hpc203/rotate-yolov5-opencv-onnxrun?style=social"/> : 分别使用OpenCV、ONNXRuntime部署yolov5旋转目标检测，包含C++和Python两个版本的程序。

    - [hpc203/rotateyolov5-opencv-onnxrun](https://github.com/hpc203/rotateyolov5-opencv-onnxrun) <img src="https://img.shields.io/github/stars/hpc203/rotateyolov5-opencv-onnxrun?style=social"/> : 分别使用OpenCV，ONNXRuntime部署yolov5旋转目标检测，包含C++和Python两个版本的程序。


  - ### Face Detection and Recognition
    #### 人脸检测与识别

    - [ChanChiChoi/awesome-Face_Recognition](https://github.com/ChanChiChoi/awesome-Face_Recognition) <img src="https://img.shields.io/github/stars/ChanChiChoi/awesome-Face_Recognition?style=social"/> : papers about Face Detection; Face Alignment; Face Recognition && Face Identification && Face Verification && Face Representation; Face Reconstruction; Face Tracking; Face Super-Resolution && Face Deblurring; Face Generation && Face Synthesis; Face Transfer; Face Anti-Spoofing; Face Retrieval. 

    - [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) <img src="https://img.shields.io/github/stars/ageitgey/face_recognition?style=social"/> : The world's simplest facial recognition api for Python and the command line. 

    - [takuya-takeuchi/FaceRecognitionDotNet](https://github.com/takuya-takeuchi/FaceRecognitionDotNet) <img src="https://img.shields.io/github/stars/takuya-takeuchi/FaceRecognitionDotNet?style=social"/> : The world's simplest facial recognition api for .NET on Windows, MacOS and Linux. 

    - [InsightFace](https://github.com/deepinsight/insightface) <img src="https://img.shields.io/github/stars/deepinsight/insightface?style=social"/> : State-of-the-art 2D and 3D Face Analysis Project. 

    - [serengil/deepface](https://github.com/serengil/deepface) <img src="https://img.shields.io/github/stars/serengil/deepface?style=social"/> : A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python.

    - [ZhaoJ9014/face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe) <img src="https://img.shields.io/github/stars/ZhaoJ9014/face.evoLVe?style=social"/> : 🔥🔥High-Performance Face Recognition Library on PaddlePaddle & PyTorch🔥🔥.

    - [OAID/TengineKit](https://github.com/OAID/TengineKit) <img src="https://img.shields.io/github/stars/OAID/TengineKit?style=social"/> : TengineKit - Free, Fast, Easy, Real-Time Face Detection & Face Landmarks & Face Attributes & Hand Detection & Hand Landmarks & Body Detection & Body Landmarks & Iris Landmarks & Yolov5 SDK On Mobile. 

    - [YOLO5Face](https://github.com/deepcam-cn/yolov5-face) <img src="https://img.shields.io/github/stars/deepcam-cn/yolov5-face?style=social"/> : "YOLO5Face: Why Reinventing a Face Detector". (**[arXiv 2021](https://arxiv.org/abs/2105.12931)**)

    - [xialuxi/yolov5_face_landmark](https://github.com/xialuxi/yolov5_face_landmark) <img src="https://img.shields.io/github/stars/xialuxi/yolov5_face_landmark?style=social"/> : 基于yolov5的人脸检测，带关键点检测。

    - [sthanhng/yoloface](https://github.com/sthanhng/yoloface) <img src="https://img.shields.io/github/stars/sthanhng/yoloface?style=social"/> : Deep learning-based Face detection using the YOLOv3 algorithm. 

    - [DayBreak-u/yolo-face-with-landmark](https://github.com/DayBreak-u/yolo-face-with-landmark) <img src="https://img.shields.io/github/stars/DayBreak-u/yolo-face-with-landmark?style=social"/> : yoloface大礼包 使用pytroch实现的基于yolov3的轻量级人脸检测（包含关键点）。

    - [abars/YoloKerasFaceDetection](https://github.com/abars/YoloKerasFaceDetection) <img src="https://img.shields.io/github/stars/abars/YoloKerasFaceDetection?style=social"/> : Face Detection and Gender and Age Classification using Keras.

    - [dannyblueliu/YOLO-Face-detection](https://github.com/dannyblueliu/YOLO-Face-detection) <img src="https://img.shields.io/github/stars/dannyblueliu/YOLO-Face-detection?style=social"/> : Face detection based on YOLO darknet.

    - [wmylxmj/YOLO-V3-IOU](https://github.com/wmylxmj/YOLO-V3-IOU) <img src="https://img.shields.io/github/stars/wmylxmj/YOLO-V3-IOU?style=social"/> : YOLO3 动漫人脸检测 (Based on keras and tensorflow) 2019-1-19.

    - [pranoyr/head-detection-using-yolo](https://github.com/pranoyr/head-detection-using-yolo) <img src="https://img.shields.io/github/stars/pranoyr/head-detection-using-yolo?style=social"/> : Detection of head using YOLO.

    - [grapeot/AnimeHeadDetector](https://github.com/grapeot/AnimeHeadDetector) <img src="https://img.shields.io/github/stars/grapeot/AnimeHeadDetector?style=social"/> : An object detector for character heads in animes, based on Yolo V3.

    - [hpc203/10kinds-light-face-detector-align-recognition](https://github.com/hpc203/10kinds-light-face-detector-align-recognition) <img src="https://img.shields.io/github/stars/hpc203/10kinds-light-face-detector-align-recognition?style=social"/> : 10种轻量级人脸检测算法的比拼。

    - [Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking](https://github.com/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking) <img src="https://img.shields.io/github/stars/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking?style=social"/> : This is a robot project for television live. System will tracking the host's face, making the face in the middle of the screen.

    - [zdfb/Yolov5_face](https://github.com/zdfb/Yolov5_face) <img src="https://img.shields.io/github/stars/zdfb/Yolov5_face?style=social"/> : 基于pytorch的Yolov5人脸检测。

    - [zhouyuchong/face-recognition-deepstream](https://github.com/zhouyuchong/face-recognition-deepstream) <img src="https://img.shields.io/github/stars/zhouyuchong/face-recognition-deepstream?style=social"/> : Deepstream app use YOLO, retinaface and arcface for face recognition. 


  - ### Face Mask Detection
    #### 口罩检测

    - [Bil369/MaskDetect-YOLOv4-PyTorch](https://github.com/Bil369/MaskDetect-YOLOv4-PyTorch) <img src="https://img.shields.io/github/stars/Bil369/MaskDetect-YOLOv4-PyTorch?style=social"/> : 基于PyTorch&YOLOv4实现的口罩佩戴检测 ⭐ 自建口罩数据集分享。

    - [adityap27/face-mask-detector](https://github.com/adityap27/face-mask-detector) <img src="https://img.shields.io/github/stars/adityap27/face-mask-detector?style=social"/> : 𝐑𝐞𝐚𝐥-𝐓𝐢𝐦𝐞 𝐅𝐚𝐜𝐞 𝐦𝐚𝐬𝐤 𝐝𝐞𝐭𝐞𝐜𝐭𝐢𝐨𝐧 𝐮𝐬𝐢𝐧𝐠 𝐝𝐞𝐞𝐩𝐥𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐀𝐥𝐞𝐫𝐭 𝐬𝐲𝐬𝐭𝐞𝐦 💻🔔.

    - [VictorLin000/YOLOv3_mask_detect](https://github.com/VictorLin000/YOLOv3_mask_detect) <img src="https://img.shields.io/github/stars/VictorLin000/YOLOv3_mask_detect?style=social"/> : Face mask detection using YOLOv3 on GoogleColab.

    - [amh28/IBM-Data-Science-Capstone-Alejandra-Marquez](https://github.com/amh28/IBM-Data-Science-Capstone-Alejandra-Marquez) <img src="https://img.shields.io/github/stars/amh28/IBM-Data-Science-Capstone-Alejandra-Marquez?style=social"/> : Homemade face mask detector fine-tuning a Yolo-v3 network.

    - [LorenRd/JetsonYolov4](https://github.com/LorenRd/JetsonYolov4) <img src="https://img.shields.io/github/stars/LorenRd/JetsonYolov4?style=social"/> : Face Mask Yolov4 detector - Nvidia Jetson Nano.

    - [Backl1ght/yolov4_face_mask_detection](https://github.com/Backl1ght/yolov4_face_mask_detection) <img src="https://img.shields.io/github/stars/Backl1ght/yolov4_face_mask_detection?style=social"/> : 基于yolov4实现口罩佩戴检测，在验证集上做到了0.954的mAP。

    - [pritul2/yolov5_FaceMask](https://github.com/pritul2/yolov5_FaceMask) <img src="https://img.shields.io/github/stars/pritul2/yolov5_FaceMask?style=social"/> : Detecting person with or without face mask. Trained using YOLOv5.

    - [NisargPethani/FACE-MASK-DETECTION-USING-YOLO-V3](https://github.com/NisargPethani/FACE-MASK-DETECTION-USING-YOLO-V3) <img src="https://img.shields.io/github/stars/NisargPethani/FACE-MASK-DETECTION-USING-YOLO-V3?style=social"/> : FACE-MASK DETECTION.

    - [waittim/mask-detector](https://github.com/waittim/mask-detector) <img src="https://img.shields.io/github/stars/waittim/mask-detector?style=social"/> : Real-time video streaming mask detection based on Python. Designed to defeat COVID-19.

    - [BogdanMarghescu/Face-Mask-Detection-Using-YOLOv4](https://github.com/BogdanMarghescu/Face-Mask-Detection-Using-YOLOv4) <img src="https://img.shields.io/github/stars/BogdanMarghescu/Face-Mask-Detection-Using-YOLOv4?style=social"/> : Face Mask Detector using YOLOv4.

    - [xinghanliuying/yolov5_bus](https://github.com/xinghanliuying/yolov5_bus) <img src="https://img.shields.io/github/stars/xinghanliuying/yolov5_bus?style=social"/> : 手把手教你使用YOLOV5训练自己的目标检测模型。

    - [song-laogou/yolov5-mask-42](https://gitee.com/song-laogou/yolov5-mask-42) : 基于YOLOV5的口罩检测系统-提供教学视频。


  - ### Social Distance Detection
    #### 社交距离检测

    - [Ank-Cha/Social-Distancing-Analyser-COVID-19](https://github.com/Ank-Cha/Social-Distancing-Analyser-COVID-19) <img src="https://img.shields.io/github/stars/Ank-Cha/Social-Distancing-Analyser-COVID-19?style=social"/> : Social Distancing Analyser to prevent COVID19. 

    - [abd-shoumik/Social-distance-detection](https://github.com/abd-shoumik/Social-distance-detection) <img src="https://img.shields.io/github/stars/abd-shoumik/Social-distance-detection?style=social"/> : Social distance detection, a deep learning computer vision project with yolo object detection.

    - [ChargedMonk/Social-Distancing-using-YOLOv5](https://github.com/ChargedMonk/Social-Distancing-using-YOLOv5) <img src="https://img.shields.io/github/stars/ChargedMonk/Social-Distancing-using-YOLOv5?style=social"/> : Classifying people as high risk and low risk based on their distance to other people.

    - [JohnBetaCode/Social-Distancing-Analyser](https://github.com/JohnBetaCode/Social-Distancing-Analyser) <img src="https://img.shields.io/github/stars/JohnBetaCode/Social-Distancing-Analyser?style=social"/> : Social Distancing Analyzer.

    - [Ashamaria/Safe-distance-tracker-using-YOLOv3-v3](https://github.com/Ashamaria/Safe-distance-tracker-using-YOLOv3-v3) <img src="https://img.shields.io/github/stars/Ashamaria/Safe-distance-tracker-using-YOLOv3-v3?style=social"/> : Safe Distance Tracker.


  - ### Intelligent Transportation Field Detection
    #### 智能交通领域检测

    - ####  Vehicle Detection
      #####  车辆检测

      - [Gaussian_YOLOv3](https://github.com/jwchoi384/Gaussian_YOLOv3) <img src="https://img.shields.io/github/stars/jwchoi384/Gaussian_YOLOv3?style=social"/> : "Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving". (**[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Choi_Gaussian_YOLOv3_An_Accurate_and_Fast_Object_Detector_Using_Localization_ICCV_2019_paper.html)**)

      - [streamlit/demo-self-driving](https://github.com/streamlit/demo-self-driving) <img src="https://img.shields.io/github/stars/streamlit/demo-self-driving?style=social"/> : Streamlit app demonstrating an image browser for the Udacity self-driving-car dataset with realtime object detection using YOLO.

      - [JunshengFu/vehicle-detection](https://github.com/JunshengFu/vehicle-detection) <img src="https://img.shields.io/github/stars/JunshengFu/vehicle-detection?style=social"/> : Created vehicle detection pipeline with two approaches: (1) deep neural networks (YOLO framework) and (2) support vector machines ( OpenCV + HOG).

      - [xslittlegrass/CarND-Vehicle-Detection](https://github.com/xslittlegrass/CarND-Vehicle-Detection) <img src="https://img.shields.io/github/stars/xslittlegrass/CarND-Vehicle-Detection?style=social"/> : Vehicle detection using YOLO in Keras runs at 21FPS.

      - [Kevinnan-teen/Intelligent-Traffic-Based-On-CV](https://github.com/Kevinnan-teen/Intelligent-Traffic-Based-On-CV) <img src="https://img.shields.io/github/stars/Kevinnan-teen/Intelligent-Traffic-Based-On-CV?style=social"/> : 基于计算机视觉的交通路口智能监控系统。

      - [subodh-malgonde/vehicle-detection](https://github.com/subodh-malgonde/vehicle-detection) <img src="https://img.shields.io/github/stars/subodh-malgonde/vehicle-detection?style=social"/> : Detect vehicles in a video.

      - [CaptainEven/Vehicle-Car-detection-and-multilabel-classification](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification) <img src="https://img.shields.io/github/stars/CaptainEven/Vehicle-Car-detection-and-multilabel-classification?style=social"/> : 使用YOLO_v3_tiny和B-CNN实现街头车辆的检测和车辆属性的多标签识别 Using yolo_v3_tiny to do vehicle or car detection and attribute's multilabel classification or recognize。

      - [kaylode/vehicle-counting](https://github.com/kaylode/vehicle-counting) <img src="https://img.shields.io/github/stars/kaylode/vehicle-counting?style=social"/> : Vehicle counting using Pytorch.

      - [MaryamBoneh/Vehicle-Detection](https://github.com/MaryamBoneh/Vehicle-Detection) <img src="https://img.shields.io/github/stars/MaryamBoneh/Vehicle-Detection?style=social"/> : Vehicle Detection Using Deep Learning and YOLO Algorithm.

      - [JeffWang0325/Image-Identification-for-Self-Driving-Cars](https://github.com/JeffWang0325/Image-Identification-for-Self-Driving-Cars) <img src="https://img.shields.io/github/stars/JeffWang0325/Image-Identification-for-Self-Driving-Cars?style=social"/> :  This project achieves some functions of image identification for Self-Driving Cars.

      - [AnarbekovAlt/Traffic-analysis](https://github.com/AnarbekovAlt/Traffic-analysis) <img src="https://img.shields.io/github/stars/AnarbekovAlt/Traffic-analysis?style=social"/> : A traffic analysis system is built on the basis of the YOLO network.

      - [ruhyadi/yolov5-nodeflux](https://github.com/ruhyadi/yolov5-nodeflux) <img src="https://img.shields.io/github/stars/ruhyadi/yolov5-nodeflux?style=social"/> : YOLOv5 Nodeflux Vehicle Detection.

      - [Daheer/Driving-Environment-Detector](https://github.com/Daheer/Driving-Environment-Detector) <img src="https://img.shields.io/github/stars/Daheer/Driving-Environment-Detector?style=social"/> : Detecting road objects using YOLO CNN Architecture.


    - ####  License Plate Detection and Recognition
      #####  车牌检测与识别

      - [zeusees/License-Plate-Detector](https://github.com/zeusees/License-Plate-Detector) <img src="https://img.shields.io/github/stars/zeusees/License-Plate-Detector?style=social"/> : License Plate Detection with Yolov5，基于Yolov5车牌检测。

      - [TheophileBuy/LicensePlateRecognition](https://github.com/TheophileBuy/LicensePlateRecognition) <img src="https://img.shields.io/github/stars/TheophileBuy/LicensePlateRecognition?style=social"/> : License Plate Recognition.

      - [alitourani/yolo-license-plate-detection](https://github.com/alitourani/yolo-license-plate-detection) <img src="https://img.shields.io/github/stars/alitourani/yolo-license-plate-detection?style=social"/> : A License-Plate detecttion application based on YOLO.

      - [HuKai97/YOLOv5-LPRNet-Licence-Recognition](https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition) <img src="https://img.shields.io/github/stars/HuKai97/YOLOv5-LPRNet-Licence-Recognition?style=social"/> : 使用YOLOv5和LPRNet进行车牌检测+识别（CCPD数据集）。 

      - [xialuxi/yolov5-car-plate](https://github.com/xialuxi/yolov5-car-plate) <img src="https://img.shields.io/github/stars/xialuxi/yolov5-car-plate?style=social"/> : 基于yolov5的车牌检测，包含车牌角点检测。 

      - [kyrielw24/License_Plate_Recognition](https://github.com/kyrielw24/License_Plate_Recognition) <img src="https://img.shields.io/github/stars/kyrielw24/License_Plate_Recognition?style=social"/> : 基于Yolo&CNN的车牌识别可视化项目。 


    - ####  Lane Detection
      #####  车道线检测
      
      - [YOLOP](https://github.com/hustvl/YOLOP) <img src="https://img.shields.io/github/stars/hustvl/YOLOP?style=social"/> : "YOLOP: You Only Look Once for Panoptic Driving Perception". (**[arXiv 2021](https://arxiv.org/abs/2108.11250)**). 

      - [visualbuffer/copilot](https://github.com/visualbuffer/copilot) <img src="https://img.shields.io/github/stars/visualbuffer/copilot?style=social"/> : Lane and obstacle detection for active assistance during driving.

      - [hpc203/YOLOP-opencv-dnn](https://github.com/hpc203/YOLOP-opencv-dnn) <img src="https://img.shields.io/github/stars/hpc203/YOLOP-opencv-dnn?style=social"/> : 使用OpenCV部署全景驾驶感知网络YOLOP，可同时处理交通目标检测、可驾驶区域分割、车道线检测，三项视觉感知任务。

      - [EdVince/YOLOP-NCNN](https://github.com/EdVince/YOLOP-NCNN) <img src="https://img.shields.io/github/stars/EdVince/YOLOP-NCNN?style=social"/> : YOLOP running in Android by ncnn.

    - ####  Driving Behavior Detection
      #####  驾驶行为检测
   
      - [JingyibySUTsoftware/Yolov5-deepsort-driverDistracted-driving-behavior-detection](https://github.com/JingyibySUTsoftware/Yolov5-deepsort-driverDistracted-driving-behavior-detection) <img src="https://img.shields.io/github/stars/JingyibySUTsoftware/Yolov5-deepsort-driverDistracted-driving-behavior-detection?style=social"/> : 基于深度学习的驾驶员分心驾驶行为（疲劳+危险行为）预警系统使用YOLOv5+Deepsort实现驾驶员的危险驾驶行为的预警监测。

    - ####  Parking Slot Detection
      #####  停车位检测

      - [visualbuffer/parkingslot](https://github.com/visualbuffer/parkingslot) <img src="https://img.shields.io/github/stars/visualbuffer/parkingslot?style=social"/> : Automated parking occupancy detection.

      - [anil2k/smart-car-parking-yolov5](https://github.com/anil2k/smart-car-parking-yolov5) <img src="https://img.shields.io/github/stars/anil2k/smart-car-parking-yolov5?style=social"/> : Detect free parking lot available for cars.

    - ####  Traffic Light Detection
      #####  交通灯检测

      - [berktepebag/Traffic-light-detection-with-YOLOv3-BOSCH-traffic-light-dataset](https://github.com/berktepebag/Traffic-light-detection-with-YOLOv3-BOSCH-traffic-light-dataset) <img src="https://img.shields.io/github/stars/berktepebag/Traffic-light-detection-with-YOLOv3-BOSCH-traffic-light-dataset?style=social"/> : Detecting Traffic Lights in Real-time with YOLOv3.

      - [mihir-m-gandhi/Adaptive-Traffic-Signal-Timer](https://github.com/mihir-m-gandhi/Adaptive-Traffic-Signal-Timer) <img src="https://img.shields.io/github/stars/mihir-m-gandhi/Adaptive-Traffic-Signal-Timer?style=social"/> : This Adaptive Traffic Signal Timer uses live images from the cameras at traffic junctions for real-time traffic density calculation using YOLO object detection and sets the signal timers accordingly.

      - [wade0125/Traffic_Light_Detection_Yolo](https://github.com/wade0125/Traffic_Light_Detection_Yolo) <img src="https://img.shields.io/github/stars/wade0125/Traffic_Light_Detection_Yolo?style=social"/> : Traffic Light Detection Yolo.


    - ####  Traffic Sign Detection
      #####  交通标志检测

      - [halftop/TT100K_YOLO_Label](https://github.com/halftop/TT100K_YOLO_Label) <img src="https://img.shields.io/github/stars/halftop/TT100K_YOLO_Label?style=social"/> : Tsinghua-Tencent 100K dataset XML and TXT Label.

      - [amazingcodeLYL/Traffic_signs_detection_darket](https://github.com/amazingcodeLYL/Traffic_signs_detection_darket) <img src="https://img.shields.io/github/stars/amazingcodeLYL/Traffic_signs_detection_darket?style=social"/> : darknet交通标志检测&TT100K数据集。

      - [TalkUHulk/yolov3-TT100k](https://github.com/TalkUHulk/yolov3-TT100k) <img src="https://img.shields.io/github/stars/TalkUHulk/yolov3-TT100k?style=social"/> : 使用yolov3训练的TT100k(交通标志)模型。     

      - [TalkUHulk/yolov4-TT100k](https://github.com/TalkUHulk/yolov4-TT100k) <img src="https://img.shields.io/github/stars/TalkUHulk/yolov4-TT100k?style=social"/> : 使用yolov4训练的TT100k(交通标志)模型。     

      - [sarah-antillia/YOLO_Realistic_USA_RoadSigns_160classes](https://github.com/sarah-antillia/YOLO_Realistic_USA_RoadSigns_160classes) <img src="https://img.shields.io/github/stars/sarah-antillia/YOLO_Realistic_USA_RoadSigns_160classes?style=social"/> : USA RoadSigns Dataset 160classes annotated by YOLO format.


    - ####  Crosswalk Detection
      #####  人行横道/斑马线检测

      - [CDNet](https://github.com/zhangzhengde0225/CDNet) <img src="https://img.shields.io/github/stars/zhangzhengde0225/CDNet?style=social"/> : "CDNet: a real-time and robust crosswalk detection network on Jetson nano based on YOLOv5". (**[Neural Computing and Applications 2022](https://link.springer.com/article/10.1007/s00521-022-07007-9)**). "CVer：《[上海交大提出CDNet：基于改进YOLOv5的斑马线和汽车过线行为检测](https://mp.weixin.qq.com/s/2F3WBtfN_7DkhERMOH8-QA)》"

      - [xN1ckuz/Crosswalks-Detection-using-YoloV5](https://github.com/xN1ckuz/Crosswalks-Detection-using-YoloV5) <img src="https://img.shields.io/github/stars/xN1ckuz/Crosswalks-Detection-using-YoloV5?style=social"/> : Crosswalks Detection using YoloV5. 


    - ####  Traffic Accidents Detection
      #####  交通事故检测
      - [khaledsabry97/Argus](https://github.com/khaledsabry97/Argus) <img src="https://img.shields.io/github/stars/khaledsabry97/Argus?style=social"/> : "Road Traffic Accidents Detection Based On Crash Estimation". (**[IEEE ICENCO 2021](https://ieeexplore.ieee.org/document/9698968)**)

    - ####  Road Damage Detection
      #####  道路损伤检测
      - [adnanmushtaq1996/Yolov4_Road_Damage_Detection](https://github.com/adnanmushtaq1996/Yolov4_Road_Damage_Detection) <img src="https://img.shields.io/github/stars/adnanmushtaq1996/Yolov4_Road_Damage_Detection?style=social"/> : A Repository to Train a Custom Yolov4 based object detector for road damage detection using the RDD2020 dataset. 

  - ### Helmet Detection
    #### 头盔/安全帽检测

    - [PeterH0323/Smart_Construction](https://github.com/PeterH0323/Smart_Construction) <img src="https://img.shields.io/github/stars/PeterH0323/Smart_Construction?style=social"/> : Head Person Helmet Detection on Construction Sites，基于目标检测工地安全帽和禁入危险区域识别系统。

    - [Byronnar/tensorflow-serving-yolov3](https://github.com/Byronnar/tensorflow-serving-yolov3) <img src="https://img.shields.io/github/stars/Byronnar/tensorflow-serving-yolov3?style=social"/> : 对原tensorflow-yolov3版本做了许多细节上的改进，增加了TensorFlow-Serving工程部署，训练了多个数据集，包括Visdrone2019, 安全帽等。

    - [gengyanlei/reflective-clothes-detect-yolov5](https://github.com/gengyanlei/reflective-clothes-detect-yolov5) <img src="https://img.shields.io/github/stars/gengyanlei/reflective-clothes-detect-yolov5?style=social"/> : reflective-clothes-detect-dataset、helemet detection yolov5、工作服(反光衣)检测数据集、安全帽检测、施工人员穿戴检测。

    - [DataXujing/YOLO-V3-Tensorflow](https://github.com/DataXujing/YOLO-V3-Tensorflow) <img src="https://img.shields.io/github/stars/DataXujing/YOLO-V3-Tensorflow?style=social"/> : 👷 👷👷 YOLO V3(Tensorflow 1.x) 安全帽 识别 | 提供数据集下载和与预训练模型。

    - [rafiuddinkhan/Yolo-Training-GoogleColab](https://github.com/rafiuddinkhan/Yolo-Training-GoogleColab) <img src="https://img.shields.io/github/stars/rafiuddinkhan/Yolo-Training-GoogleColab?style=social"/> : Helmet Detection using tiny-yolo-v3 by training using your own dataset and testing the results in the google colaboratory.

    - [BlcaKHat/yolov3-Helmet-Detection](https://github.com/BlcaKHat/yolov3-Helmet-Detection) <img src="https://img.shields.io/github/stars/BlcaKHat/yolov3-Helmet-Detection?style=social"/> : Training a YOLOv3 model to detect the presence of helmet for intrusion or traffic monitoring.

    - [yumulinfeng1/YOLOv4-Hat-detection](https://github.com/yumulinfeng1/YOLOv4-Hat-detection)) <img src="https://img.shields.io/github/stars/yumulinfeng1/YOLOv4-Hat-detection?style=social"/> : 基于YOLOv4的安全帽佩戴检测。

    - [FanDady/Helmet-Detection-YoloV5](https://github.com/FanDady/Helmet-Detection-YoloV5)) <img src="https://img.shields.io/github/stars/FanDady/Helmet-Detection-YoloV5?style=social"/> : Safety helmet wearing detection on construction site based on YoloV5s-V5.0 including helmet dataset（基于YoloV5-V5.0的工地安全帽检测并且包含开源的安全帽数据集）。

    - [RUI-LIU7/Helmet_Detection](https://github.com/RUI-LIU7/Helmet_Detection)) <img src="https://img.shields.io/github/stars/RUI-LIU7/Helmet_Detection?style=social"/> : 使用yolov5算法实现安全帽以及危险区域的监测，同时接入海康摄像头实现实时监测。


  - ### Hand Detection
    #### 手部检测

    - [cansik/yolo-hand-detection](https://github.com/cansik/yolo-hand-detection) <img src="https://img.shields.io/github/stars/cansik/yolo-hand-detection?style=social"/> : A pre-trained YOLO based hand detection network.


  - ### Gesture Recognition
    #### 手势/手语识别

    - [MahmudulAlam/Unified-Gesture-and-Fingertip-Detection](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection) <img src="https://img.shields.io/github/stars/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection?style=social"/> : "Unified learning approach for egocentric hand gesture recognition and fingertip detection". (**[Elsevier 2022](https://www.sciencedirect.com/science/article/abs/pii/S0031320321003824)**)

    - [insigh1/Interactive_ABCs_with_American_Sign_Language_using_Yolov5](https://github.com/insigh1/Interactive_ABCs_with_American_Sign_Language_using_Yolov5) <img src="https://img.shields.io/github/stars/insigh1/Interactive_ABCs_with_American_Sign_Language_using_Yolov5?style=social"/> : Interactive ABC's with American Sign Language.

    - [Dreaming-future/YOLO-Object-Detection](https://github.com/Dreaming-future/YOLO-Object-Detection) <img src="https://img.shields.io/github/stars/Dreaming-future/YOLO-Object-Detection?style=social"/> :  YOLO-Object-Detection 集成多种yolo模型，作为一个模板进行目标检测。

    
  - ### Action Detection
    #### 行为检测

    - [wufan-tb/yolo_slowfast](https://github.com/wufan-tb/yolo_slowfast) <img src="https://img.shields.io/github/stars/wufan-tb/yolo_slowfast?style=social"/> : A realtime action detection frame work based on PytorchVideo.


  - ### Emotion Recognition
    #### 情感识别

    - [Tandon-A/emotic](https://github.com/Tandon-A/emotic) <img src="https://img.shields.io/github/stars/Tandon-A/emotic?style=social"/> : "Context based emotion recognition using emotic dataset". (**[arXiv 2020](https://arxiv.org/abs/2003.13401)**)


  - ### Human Pose Estimation
    #### 人体姿态估计

    - [wmcnally/kapao](https://github.com/wmcnally/kapao) <img src="https://img.shields.io/github/stars/wmcnally/kapao?style=social"/> : KAPAO is a state-of-the-art single-stage human pose estimation model that detects keypoints and poses as objects and fuses the detections to predict human poses. "Rethinking Keypoint Representations: Modeling Keypoints and Poses as Objects for Multi-Person Human Pose Estimation". (**[arXiv 2021](https://arxiv.org/abs/2111.08557)**)

    - [TexasInstruments/edgeai-yolov5](https://github.com/TexasInstruments/edgeai-yolov5) <img src="https://img.shields.io/github/stars/TexasInstruments/edgeai-yolov5?style=social"/> : "YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss". (**[arXiv 2022](https://arxiv.org/abs/2204.06806)**)

    - [TexasInstruments/edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox) <img src="https://img.shields.io/github/stars/TexasInstruments/edgeai-yolox?style=social"/> : "YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss". (**[arXiv 2022](https://arxiv.org/abs/2204.06806)**)

    - [jinfagang/VIBE_yolov5](https://github.com/jinfagang/VIBE_yolov5) <img src="https://img.shields.io/github/stars/jinfagang/VIBE_yolov5?style=social"/> : Using YOLOv5 as detection on VIBE. "VIBE: Video Inference for Human Body Pose and Shape Estimation". (**[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.html)**)

    - [zhuoxiangpang/ism_person_openpose](https://github.com/zhuoxiangpang/ism_person_openpose) <img src="https://img.shields.io/github/stars/zhuoxiangpang/ism_person_openpose?style=social"/> : yolov5人体检测+openpose姿态检测 实现摔倒检测。

    - [pengyang1225/yolov5_person_pose](https://github.com/pengyang1225/yolov5_person_pose) <img src="https://img.shields.io/github/stars/pengyang1225/yolov5_person_pose?style=social"/> : 基于yolov5的person—pose。

    - [hpc203/yolov5_pose_opencv](https://github.com/hpc203/yolov5_pose_opencv) <img src="https://img.shields.io/github/stars/hpc203/yolov5_pose_opencv?style=social"/> : 使用OpenCV部署yolov5-pose目标检测+人体姿态估计，包含C++和Python两个版本的程序。支持yolov5s，yolov5m，yolov5l。


  - ### Distance Measurement
    #### 距离测量

    - [davidfrz/yolov5_distance_count](https://github.com/davidfrz/yolov5_distance_count) <img src="https://img.shields.io/github/stars/davidfrz/yolov5_distance_count?style=social"/> : 通过yolov5实现目标检测+双目摄像头实现距离测量。

    - [wenyishengkingkong/realsense-D455-YOLOV5](https://github.com/wenyishengkingkong/realsense-D455-YOLOV5) <img src="https://img.shields.io/github/stars/wenyishengkingkong/realsense-D455-YOLOV5?style=social"/> : 利用realsense深度相机实现yolov5目标检测的同时测出距离。

    - [Thinkin99/yolov5_d435i_detection](https://github.com/Thinkin99/yolov5_d435i_detection) <img src="https://img.shields.io/github/stars/Thinkin99/yolov5_d435i_detection?style=social"/> : 使用realsense d435i相机，基于pytorch实现yolov5目标检测，返回检测目标相机坐标系下的位置信息。 

    - [MUCHWAY/detect_distance_gazebo](https://github.com/MUCHWAY/detect_distance_gazebo) <img src="https://img.shields.io/github/stars/MUCHWAY/detect_distance_gazebo?style=social"/> : yolov5+camera_distance+gazebo.


  - ### 3D Object Detection
    #### 三维目标检测

    - [maudzung/YOLO3D-YOLOv4-PyTorch](https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch) <img src="https://img.shields.io/github/stars/maudzung/YOLO3D-YOLOv4-PyTorch?style=social"/> : The PyTorch Implementation based on YOLOv4 of the paper: "YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud". (**[ECCV 2018](https://openaccess.thecvf.com/content_eccv_2018_workshops/w18/html/Ali_YOLO3D_End-to-end_real-time_3D_Oriented_Object_Bounding_Box_Detection_from_ECCVW_2018_paper.html)**)

    - [maudzung/Complex-YOLOv4-Pytorch](https://github.com/maudzung/Complex-YOLOv4-Pytorch) <img src="https://img.shields.io/github/stars/maudzung/Complex-YOLOv4-Pytorch?style=social"/> : The PyTorch Implementation based on YOLOv4 of the paper: "Complex-YOLO: Real-time 3D Object Detection on Point Clouds". (**[arXiv 2018](https://arxiv.org/abs/1803.06199)**)

    - [AI-liu/Complex-YOLO](https://github.com/AI-liu/Complex-YOLO) <img src="https://img.shields.io/github/stars/AI-liu/Complex-YOLO?style=social"/> : This is an unofficial implementation of "Complex-YOLO: Real-time 3D Object Detection on Point Clouds in pytorch". (**[arXiv 2018](https://arxiv.org/abs/1803.06199)**)

    - [ghimiredhikura/Complex-YOLOv3](https://github.com/ghimiredhikura/Complex-YOLOv3) <img src="https://img.shields.io/github/stars/ghimiredhikura/Complex-YOLOv3?style=social"/> : Complete but Unofficial PyTorch Implementation of "Complex-YOLO: Real-time 3D Object Detection on Point Clouds with YoloV3". (**[arXiv 2018](https://arxiv.org/abs/1803.06199)**)

    - [ruhyadi/YOLO3D](https://github.com/ruhyadi/YOLO3D) <img src="https://img.shields.io/github/stars/ruhyadi/YOLO3D?style=social"/> : YOLO 3D Object Detection for Autonomous Driving Vehicle. Reference by [skhadem/3D-BoundingBox](https://github.com/skhadem/3D-BoundingBox), "3D Bounding Box Estimation Using Deep Learning and Geometry". (**[CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Mousavian_3D_Bounding_Box_CVPR_2017_paper.html)**)

    - [ruhyadi/yolo3d-lightning](https://github.com/ruhyadi/yolo3d-lightning) <img src="https://img.shields.io/github/stars/ruhyadi/YOLO3D?style=social"/> : YOLO for 3D Object Detection.

    - [Yuanchu/YOLO3D](https://github.com/Yuanchu/YOLO3D) <img src="https://img.shields.io/github/stars/Yuanchu/YOLO3D?style=social"/> : Implementation of a basic YOLO model for object detection in 3D.

    - [EmiyaNing/3D-YOLO](https://github.com/EmiyaNing/3D-YOLO) <img src="https://img.shields.io/github/stars/EmiyaNing/3D-YOLO?style=social"/> : YOLO v5 for Lidar-based 3D BEV Detection.


  - ### SLAM Field Detection
    #### SLAM领域检测

    - [bijustin/YOLO-DynaSLAM](https://github.com/bijustin/YOLO-DynaSLAM) <img src="https://img.shields.io/github/stars/bijustin/YOLO-DynaSLAM?style=social"/> : YOLO Dynamic ORB_SLAM is a visual SLAM system that is robust in dynamic scenarios for RGB-D configuration. 

    - [BzdTaisa/YoloPlanarSLAM](https://github.com/BzdTaisa/YoloPlanarSLAM) <img src="https://img.shields.io/github/stars/BzdTaisa/YoloPlanarSLAM?style=social"/> : YOLO-Planar-SLAM. 


  - ### Industrial Defect Detection
    #### 工业缺陷检测

    - [annsonic/Steel_defect](https://github.com/annsonic/Steel_defect) <img src="https://img.shields.io/github/stars/annsonic/Steel_defect?style=social"/> : Exercise: Use YOLO to detect hot-rolled steel strip surface defects (NEU-DET dataset).


  - ### SAR Image Detection
    #### 合成孔径雷达图像检测

    - [humblecoder612/SAR_yolov3](https://github.com/humblecoder612/SAR_yolov3) <img src="https://img.shields.io/github/stars/humblecoder612/SAR_yolov3?style=social"/> : Best Accruacy:speed ratio SAR Ship detection in the world.


  - ### Safety Monitoring Field Detection
    #### 安防监控领域检测

    - [gengyanlei/fire-smoke-detect-yolov4](https://github.com/gengyanlei/fire-smoke-detect-yolov4) <img src="https://img.shields.io/github/stars/gengyanlei/fire-smoke-detect-yolov4?style=social"/> : fire-smoke-detect-yolov4-yolov5 and fire-smoke-detection-dataset 火灾检测，烟雾检测。

    - [CVUsers/Smoke-Detect-by-YoloV5](https://github.com/CVUsers/Smoke-Detect-by-YoloV5) <img src="https://img.shields.io/github/stars/CVUsers/Smoke-Detect-by-YoloV5?style=social"/> : Yolov5 real time smoke detection system.

    - [CVUsers/Fire-Detect-by-YoloV5](https://github.com/CVUsers/Fire-Detect-by-YoloV5) <img src="https://img.shields.io/github/stars/CVUsers/Fire-Detect-by-YoloV5?style=social"/> : 火灾检测，浓烟检测，吸烟检测。

    - [spacewalk01/Yolov5-Fire-Detection](https://github.com/spacewalk01/Yolov5-Fire-Detection) <img src="https://img.shields.io/github/stars/spacewalk01/Yolov5-Fire-Detection?style=social"/> : Train yolov5 to detect fire in an image or video.

    - [roflcoopter/viseron](https://github.com/roflcoopter/viseron) <img src="https://img.shields.io/github/stars/roflcoopter/viseron?style=social"/> : Viseron - Self-hosted NVR with object detection.

    - [dcmartin/motion-ai](https://github.com/dcmartin/motion-ai) <img src="https://img.shields.io/github/stars/dcmartin/motion-ai?style=social"/> : AI assisted motion detection for Home Assistant.

    - [Nico31415/Drowning-Detector](https://github.com/Nico31415/Drowning-Detector) <img src="https://img.shields.io/github/stars/Nico31415/Drowning-Detector?style=social"/> : Using YOLO object detection, this program will detect if a person is drowning.

    - [mc-cat-tty/DoorbellCamDaemon](https://github.com/mc-cat-tty/DoorbellCamDaemon) <img src="https://img.shields.io/github/stars/mc-cat-tty/DoorbellCamDaemon?style=social"/> : Part of DoorbellCam project: daemon for people recognition with YOLO from a RTSP video stream. 

    - [Choe-Ji-Hwan/Fire_Detect_Custom_Yolov5](https://github.com/Choe-Ji-Hwan/Fire_Detect_Custom_Yolov5) <img src="https://img.shields.io/github/stars/Choe-Ji-Hwan/Fire_Detect_Custom_Yolov5?style=social"/> : 2022-1 Individual Research Assignment: Using YOLOv5 to simply recognize each type of fire. 

    - [bishal116/FireDetection](https://github.com/bishal116/FireDetection) <img src="https://img.shields.io/github/stars/bishal116/FireDetection?style=social"/> : This project builds fire detecton using YOLO v3 model.



  - ### Medical Field Detection
    #### 医学领域检测

    - [DataXujing/YOLO-v5](https://github.com/DataXujing/YOLO-v5) <img src="https://img.shields.io/github/stars/DataXujing/YOLO-v5?style=social"/> : YOLO v5在医疗领域中消化内镜目标检测的应用。

    - [Jafar-Abdollahi/Automated-detection-of-COVID-19-cases-using-deep-neural-networks-with-CTS-images](https://github.com/Jafar-Abdollahi/Automated-detection-of-COVID-19-cases-using-deep-neural-networks-with-CTS-images) <img src="https://img.shields.io/github/stars/Jafar-Abdollahi/Automated-detection-of-COVID-19-cases-using-deep-neural-networks-with-CTS-images?style=social"/> : In this project, a new model for automatic detection of covid-19 using raw chest X-ray images is presented. 

    - [fahriwps/breast-cancer-detection](https://github.com/fahriwps/breast-cancer-detection) <img src="https://img.shields.io/github/stars/fahriwps/breast-cancer-detection?style=social"/> : Breast cancer mass detection using YOLO object detection algorithm and GUI.

    - [niehusst/YOLO-Cancer-Detection](https://github.com/niehusst/YOLO-Cancer-Detection) <img src="https://img.shields.io/github/stars/niehusst/YOLO-Cancer-Detection?style=social"/> : An implementation of the YOLO algorithm trained to spot tumors in DICOM images.

    - [safakgunes/Blood-Cancer-Detection-YOLOV5](https://github.com/safakgunes/Blood-Cancer-Detection-YOLOV5) <img src="https://img.shields.io/github/stars/safakgunes/Blood-Cancer-Detection-YOLOV5?style=social"/> : Blood Cancer Detection with YOLOV5.

    - [shchiang0708/YOLOv2_skinCancer](https://github.com/shchiang0708/YOLOv2_skinCancer) <img src="https://img.shields.io/github/stars/shchiang0708/YOLOv2_skinCancer?style=social"/> : YOLOv2_skinCancer.

    - [avral1810/parkinsongait](https://github.com/avral1810/parkinsongait) <img src="https://img.shields.io/github/stars/avral1810/parkinsongait?style=social"/> : Parkinson’s Disease.

    - [sierprinsky/YoloV5_blood_cells](https://github.com/sierprinsky/YoloV5_blood_cells) <img src="https://img.shields.io/github/stars/sierprinsky/YoloV5_blood_cells?style=social"/> : The main idea of this project is to detect blood cells using YOLOV5 over a public roboflow dataset.

    - [LuozyCS/skin_disease_detection_yolov5](https://github.com/LuozyCS/skin_disease_detection_yolov5) <img src="https://img.shields.io/github/stars/LuozyCS/skin_disease_detection_yolov5?style=social"/> : skin_disease_detection_yolov5.

    - [Moqixis/object_detection_yolov5_deepsort](https://github.com/Moqixis/object_detection_yolov5_deepsort) <img src="https://img.shields.io/github/stars/Moqixis/object_detection_yolov5_deepsort?style=social"/> : 基于yolov5+deepsort的息肉目标检测。


  - ### Chemistry Field Detection
    #### 化学领域检测

    - [xuguodong1999/COCR](https://github.com/xuguodong1999/COCR) <img src="https://img.shields.io/github/stars/xuguodong1999/COCR?style=social"/> : COCR is designed to convert an image of hand-writing chemical structure to graph of that molecule.


  - ### Agricultural Field Detection
    #### 农业领域检测

    - [liao1fan/MGA-YOLO-for-apple-leaf-disease-detection](https://github.com/liao1fan/MGA-YOLO-for-apple-leaf-disease-detection) <img src="https://img.shields.io/github/stars/liao1fan/MGA-YOLO-for-apple-leaf-disease-detection?style=social"/> : MGA-YOLO: A Lightweight One-Stage Network for Apple Leaf Disease Detection. 

    - [tanmaypandey7/wheat-detection](https://github.com/tanmaypandey7/wheat-detection) <img src="https://img.shields.io/github/stars/tanmaypandey7/wheat-detection?style=social"/> : Detecting wheat heads using YOLOv5. 

    - [WoodratTradeCo/crop-rows-detection](https://github.com/WoodratTradeCo/crop-rows-detection) <img src="https://img.shields.io/github/stars/WoodratTradeCo/crop-rows-detection?style=social"/> : It is an real-time crop rows detection method using YOLOv5. 


  - ### Adverse Weather Conditions
    #### 恶劣天气情况

    - [LLVIP](https://github.com/bupt-ai-cz/LLVIP) <img src="https://img.shields.io/github/stars/bupt-ai-cz/LLVIP?style=social"/> : "LLVIP: A Visible-infrared Paired Dataset for Low-light Vision". (**[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021W/RLQ/html/Jia_LLVIP_A_Visible-Infrared_Paired_Dataset_for_Low-Light_Vision_ICCVW_2021_paper.html)**)

    - [Image-Adaptive YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO) <img src="https://img.shields.io/github/stars/wenyyu/Image-Adaptive-YOLO?style=social"/> : "Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions". (**[AAAI 2022](https://arxiv.org/abs/2112.08088)**). "计算机视觉研究院：《[图像自适应YOLO：模糊环境下的目标检测（附源代码）](https://mp.weixin.qq.com/s/QdM6Dx990VhN97MRIP74XA)》"


  - ### Adversarial Attack and Defense
    #### 对抗攻击与防御 

    - [EAVISE/adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo) : "Fooling automated surveillance cameras: adversarial patches to attack person detection". (**[CVPR 2019](https://openaccess.thecvf.com/content_CVPRW_2019/html/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.html)**)

    - [git-disl/TOG](https://github.com/git-disl/TOG) <img src="https://img.shields.io/github/stars/git-disl/TOG?style=social"/> : "Adversarial Objectness Gradient Attacks on Real-time Object Detection Systems". (**[IEEE TPS-ISA 2020](https://ieeexplore.ieee.org/abstract/document/9325397)**) | "Understanding Object Detection Through an Adversarial Lens". (**[ESORICS 2020](https://link.springer.com/chapter/10.1007/978-3-030-59013-0_23)**)

    - [VITA-Group/3D_Adversarial_Logo](https://github.com/VITA-Group/3D_Adversarial_Logo) <img src="https://img.shields.io/github/stars/VITA-Group/3D_Adversarial_Logo?style=social"/> : 3D adversarial logo attack on different3D object meshes to fool a YOLOV2 detector. "Can 3D Adversarial Logos Clock Humans?". (**[arXiv 2020](https://arxiv.org/abs/2006.14655)**)

    - [ASGuard-UCI/MSF-ADV](https://github.com/ASGuard-UCI/MSF-ADV) <img src="https://img.shields.io/github/stars/ASGuard-UCI/MSF-ADV?style=social"/> : MSF-ADV is a novel physical-world adversarial attack method, which can fool the Multi Sensor Fusion (MSF) based autonomous driving (AD) perception in the victim autonomous vehicle (AV) to fail in detecting a front obstacle and thus crash into it. "Invisible for both Camera and LiDAR: Security of Multi-Sensor Fusion based Perception in Autonomous Driving Under Physical-World Attacks". (**[IEEE S&P 2021](https://www.computer.org/csdl/proceedings-article/sp/2021/893400b302/1t0x9btzenu)**)

    - [veralauee/DPatch](https://github.com/veralauee/DPatch) <img src="https://img.shields.io/github/stars/veralauee/DPatch?style=social"/> : "DPatch: An Adversarial Patch Attack on Object Detectors". (**[arXiv 2018](https://arxiv.org/abs/1806.02299)**)

    - [Shudeng/GPAttack](https://github.com/Shudeng/GPAttack) <img src="https://img.shields.io/github/stars/Shudeng/GPAttack?style=social"/> : Grid Patch Attack for Object Detection. 

    - [Wu-Shudeng/DPAttack](https://github.com/Wu-Shudeng/DPAttack) <img src="https://img.shields.io/github/stars/Wu-Shudeng/DPAttack?style=social"/> : "DPAttack: Diffused Patch Attacks against Universal Object Detection". (**[arXiv 2020](https://arxiv.org/abs/2010.11679)**)

    - [FenHua/DetDak](https://github.com/FenHua/DetDak) <img src="https://img.shields.io/github/stars/FenHua/DetDak?style=social"/> : Patch adversarial attack; object detection; CIKM2020 安全AI挑战者计划第四期：通用目标检测的对抗攻击。 "Object Hider: Adversarial Patch Attack Against Object Detectors". (**[arXiv 2020](https://arxiv.org/abs/2010.14974)**)

    - [THUrssq/Tianchi04](https://github.com/THUrssq/Tianchi04) <img src="https://img.shields.io/github/stars/THUrssq/Tianchi04?style=social"/> : This is NO.4 solution for "CIKM-2020 Alibaba-Tsinghua Adversarial Challenge on Object Detection". "Sparse Adversarial Attack to Object Detection". (**[arXiv 2020](https://arxiv.org/abs/2012.13692)**)

    - [mesunhlf/UPC-tf](https://github.com/mesunhlf/UPC-tf) <img src="https://img.shields.io/github/stars/mesunhlf/UPC-tf?style=social"/> : "Universal Physical Camouflage Attacks on Object Detectors". (**[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_Universal_Physical_Camouflage_Attacks_on_Object_Detectors_CVPR_2020_paper.html)**)

    - [alex96295/YOLOv3_adversarial_defense](https://github.com/alex96295/YOLOv3_adversarial_defense) <img src="https://img.shields.io/github/stars/alex96295/YOLOv3_adversarial_defense?style=social"/> : YOLOv3_adversarial_defense. 

    - [alex96295/YOLO_adversarial_attacks](https://github.com/alex96295/YOLO_adversarial_attacks) <img src="https://img.shields.io/github/stars/alex96295/YOLO_adversarial_attacks?style=social"/> : YOLO_adversarial_attacks. 

    - [alex96295/Adversarial-Patch-Attacks-TRAINING-YOLO-SSD-Pytorch](https://github.com/alex96295/Adversarial-Patch-Attacks-TRAINING-YOLO-SSD-Pytorch) <img src="https://img.shields.io/github/stars/alex96295/Adversarial-Patch-Attacks-TRAINING-YOLO-SSD-Pytorch?style=social"/> : This repository has the code needed to train 'Adversarial Patch Attacks' on YOLO and SSD models for object detection in Pytorch.

    - [FranBesq/attack-yolo](https://github.com/FranBesq/attack-yolo) <img src="https://img.shields.io/github/stars/FranBesq/attack-yolo?style=social"/> : Developing adversarial attacks on YOLO algorithm for computer vision.

    - [Rushi314/GPR-Object-Detection](https://github.com/Rushi314/GPR-Object-Detection) <img src="https://img.shields.io/github/stars/Rushi314/GPR-Object-Detection?style=social"/> : Detecting Objects in Ground Penetrating Radars Scans.

    - [realtxy/pso-adversarial-yolo_v3](https://github.com/realtxy/pso-adversarial-yolo_v3) <img src="https://img.shields.io/github/stars/realtxy/pso-adversarial-yolo_v3?style=social"/> : pso-adversarial-yolo_v3.

    - [sowgali/ObjCAM](https://github.com/sowgali/ObjCAM) <img src="https://img.shields.io/github/stars/sowgali/ObjCAM?style=social"/> : Visualizations for adversarial attacks in object detectors like YOLO.

    - [andrewpatrickdu/adversarial-yolov3-cowc](https://github.com/andrewpatrickdu/adversarial-yolov3-cowc) <img src="https://img.shields.io/github/stars/andrewpatrickdu/adversarial-yolov3-cowc?style=social"/> : "Physical Adversarial Attacks on an Aerial Imagery Object Detector".  (**[WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Du_Physical_Adversarial_Attacks_on_an_Aerial_Imagery_Object_Detector_WACV_2022_paper.html)**)

    - [IQTLabs/camolo](https://github.com/IQTLabs/camolo) <img src="https://img.shields.io/github/stars/IQTLabs/camolo?style=social"/> : Camouflage YOLO - (CAMOLO) trains adversarial patches to confuse the YOLO family of object detectors.

    - [tsm55555/adversarial-yolov5](https://github.com/tsm55555/adversarial-yolov5) <img src="https://img.shields.io/github/stars/tsm55555/adversarial-yolov5?style=social"/> : The code is modified from [Adversarial YOLO](https://gitlab.com/EAVISE/adversarial-yolo)   

    - [AdvTexture](https://github.com/WhoTHU/Adversarial_Texture) <img src="https://img.shields.io/github/stars/WhoTHU/Adversarial_Texture?style=social"/> : "Adversarial Texture for Fooling Person Detectors in the Physical World". (**[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Adversarial_Texture_for_Fooling_Person_Detectors_in_the_Physical_World_CVPR_2022_paper.html)**). CVPR2022 Oral 物理对抗样本 如何做一件“隐形衣”。  (**[知乎 2022](https://zhuanlan.zhihu.com/p/499854846)**)


  - ### Instance and Semantic Segmentation
    #### 实例和语义分割

    - [Laughing-q/yolov5-q](https://github.com/Laughing-q/yolov5-q) <img src="https://img.shields.io/github/stars/Laughing-q/yolov5-q?style=social"/> : This repo is plan for instance segmentation based on yolov5-6.0 and yolact. 

    - [TomMao23/multiyolov5](https://github.com/TomMao23/multiyolov5) <img src="https://img.shields.io/github/stars/TomMao23/multiyolov5?style=social"/> : Multi YOLO V5——Detection and Semantic Segmentation.

    - [ArtyZe/yolo_segmentation](https://github.com/ArtyZe/yolo_segmentation) <img src="https://img.shields.io/github/stars/ArtyZe/yolo_segmentation?style=social"/> : image (semantic segmentation) instance segmentation by darknet or yolo.

    - [midasklr/yolov5ds](https://github.com/midasklr/yolov5ds) <img src="https://img.shields.io/github/stars/midasklr/yolov5ds?style=social"/> : multi-task yolov5 with detection and segmentation.


  - ### Game Field Detection
    #### 游戏领域检测

    - [petercunha/Pine](https://github.com/petercunha/Pine) <img src="https://img.shields.io/github/stars/petercunha/Pine?style=social"/> : 🌲 Aimbot powered by real-time object detection with neural networks, GPU accelerated with Nvidia. Optimized for use with CS:GO.

    - [chaoyu1999/FPSAutomaticAiming](https://github.com/chaoyu1999/FPSAutomaticAiming) <img src="https://img.shields.io/github/stars/chaoyu1999/FPSAutomaticAiming?style=social"/> : 基于yolov5的FPS类游戏AI自瞄AI。

    - [Lu-tju/CSGO_AI](https://github.com/Lu-tju/CSGO_AI) <img src="https://img.shields.io/github/stars/Lu-tju/CSGO_AI?style=social"/> : 基于YOLOv3的csgo自瞄。

    - [kir486680/csgo_aim](https://github.com/kir486680/csgo_aim) <img src="https://img.shields.io/github/stars/kir486680/csgo_aim?style=social"/> : Aim assist for CSGO with python and yolo.

    - [c925777075/yolov5-dnf](https://github.com/c925777075/yolov5-dnf) <img src="https://img.shields.io/github/stars/c925777075/yolov5-dnf?style=social"/> : yolov5-DNF.

    - [davidhoung2/APEX-yolov5-aim-assist](https://github.com/davidhoung2/APEX-yolov5-aim-assist) <img src="https://img.shields.io/github/stars/davidhoung2/APEX-yolov5-aim-assist?style=social"/> : using yolov5 to help you aim enemies.

     - [Brednan/CSGO-Aimbot](https://github.com/Brednan/CSGO-Aimbot) <img src="https://img.shields.io/github/stars/Brednan/CSGO-Aimbot?style=social"/> : Aimbot for the FPS game CSGO. It uses YOLOv5 to detect enemy players on my screen, then moves my cursor to the location. 

     - [2319590263/yolov5-csgo](https://github.com/2319590263/yolov5-csgo) <img src="https://img.shields.io/github/stars/2319590263/yolov5-csgo?style=social"/> : 基于yolov5实现的csgo自瞄。

     - [SCRN-VRC/YOLOv4-Tiny-in-UnityCG-HLSL](https://github.com/SCRN-VRC/YOLOv4-Tiny-in-UnityCG-HLSL) <img src="https://img.shields.io/github/stars/SCRN-VRC/YOLOv4-Tiny-in-UnityCG-HLSL?style=social"/> : A modern object detector inside fragment shaders.

     - [qcjxs-hn/yolov5-csgo](https://github.com/qcjxs-hn/yolov5-csgo) <img src="https://img.shields.io/github/stars/qcjxs-hn/yolov5-csgo?style=social"/> : 这是一个根据教程写的csgo-ai和我自己训练的模型，还有数据集。

     - [Sequoia](https://github.com/IgaoGuru/Sequoia) <img src="https://img.shields.io/github/stars/IgaoGuru/Sequoia?style=social"/> : A neural network for CounterStrike:GlobalOffensive character detection and classification. Built on a custom-made dataset (csgo-data-collector).

     - [ItGarbager/aimcf_yolov5](https://github.com/ItGarbager/aimcf_yolov5) <img src="https://img.shields.io/github/stars/ItGarbager/aimcf_yolov5?style=social"/> : 使用yolov5算法实现cf角色头部预测。

     - [jiaran-takeme/Target-Detection-for-CSGO-by-YOLOv5](https://github.com/jiaran-takeme/Target-Detection-for-CSGO-by-YOLOv5) <img src="https://img.shields.io/github/stars/jiaran-takeme/Target-Detection-for-CSGO-by-YOLOv5?style=social"/> : Target Detection for CSGO by YOLOv5.

     - [Lucid1ty/Yolov5ForCSGO](https://github.com/Lucid1ty/Yolov5ForCSGO) <img src="https://img.shields.io/github/stars/Lucid1ty/Yolov5ForCSGO?style=social"/> : CSGO character detection and auto aim.

     - [leo4048111/Yolov5-LabelMaker-For-CSGO](https://github.com/leo4048111/Yolov5-LabelMaker-For-CSGO) <img src="https://img.shields.io/github/stars/leo4048111/Yolov5-LabelMaker-For-CSGO?style=social"/> : A simple tool for making CSGO dataset in YOLO format.

     - [soloist-v/AutoStrike](https://github.com/soloist-v/AutoStrike) <img src="https://img.shields.io/github/stars/soloist-v/AutoStrike?style=social"/> : 使用yolov5自动瞄准，支持fps游戏 鼠标移动控制需要自行调整。


  - ### Automatic Annotation Tool
    #### 自动标注工具

    - [LabelImg](https://github.com/heartexlabs/labelImg) <img src="https://img.shields.io/github/stars/heartexlabs/labelImg?style=social"/> : 🖍️ LabelImg is a graphical image annotation tool and label object bounding boxes in images.

    - [Label Studio](https://github.com/heartexlabs/label-studio) <img src="https://img.shields.io/github/stars/heartexlabs/label-studio?style=social"/> : Label Studio is a multi-type data labeling and annotation tool with standardized output format.

    - [labelme](https://github.com/wkentaro/labelme) <img src="https://img.shields.io/github/stars/wkentaro/labelme?style=social"/> : Image Polygonal Annotation with Python (polygon, rectangle, circle, line, point and image-level flag annotation).  

    - [AlexeyAB/Yolo_mark](https://github.com/AlexeyAB/Yolo_mark) <img src="https://img.shields.io/github/stars/AlexeyAB/Yolo_mark?style=social"/> : GUI for marking bounded boxes of objects in images for training neural network Yolo v3 and v2.

    - [Cartucho/OpenLabeling](https://github.com/Cartucho/OpenLabeling) <img src="https://img.shields.io/github/stars/Cartucho/OpenLabeling?style=social"/> : Label images and video for Computer Vision applications. 

    - [CVAT](https://github.com/cvat-ai/cvat) <img src="https://img.shields.io/github/stars/cvat-ai/cvat?style=social"/> : Computer Vision Annotation Tool (CVAT). Annotate better with CVAT, the industry-leading data engine for machine learning. Used and trusted by teams at any scale, for data of any scale. 

    - [cnyvfang/labelGo-Yolov5AutoLabelImg](https://github.com/cnyvfang/labelGo-Yolov5AutoLabelImg) <img src="https://img.shields.io/github/stars/cnyvfang/labelGo-Yolov5AutoLabelImg?style=social"/> : 💕YOLOV5 semi-automatic annotation tool (Based on labelImg)💕一个基于labelImg及YOLOV5的图形化半自动标注工具。

    - [CVUsers/Auto_maker](https://github.com/CVUsers/Auto_maker) <img src="https://img.shields.io/github/stars/CVUsers/Auto_maker?style=social"/> : 深度学习数据自动标注器开源 目标检测和图像分类（高精度高效率）。

    - [DarkLabel](https://github.com/darkpgmr/DarkLabel) <img src="https://img.shields.io/github/stars/darkpgmr/DarkLabel?style=social"/> : Video/Image Labeling and Annotation Tool.

    - [wufan-tb/AutoLabelImg](https://github.com/wufan-tb/AutoLabelImg) <img src="https://img.shields.io/github/stars/wufan-tb/AutoLabelImg?style=social"/> : auto-labelimg based on yolov5, with many other useful tools. AutoLabelImg 多功能自动标注工具。

    - [MrZander/YoloMarkNet](https://github.com/MrZander/YoloMarkNet) <img src="https://img.shields.io/github/stars/MrZander/YoloMarkNet?style=social"/> : Darknet YOLOv2/3 annotation tool written in C#/WPF.

    - [mahxn0/Yolov3_ForTextLabel](https://github.com/mahxn0/Yolov3_ForTextLabel) <img src="https://img.shields.io/github/stars/mahxn0/Yolov3_ForTextLabel?style=social"/> : 基于yolov3的目标/自然场景文字自动标注工具。

    - [WangRongsheng/KDAT](https://github.com/WangRongsheng/KDAT) <img src="https://img.shields.io/github/stars/WangRongsheng/KDAT?style=social"/> : 一个专为视觉方向目标检测全流程的标注工具集，全称：Kill Object Detection Annotation Tools。 

    - [MNConnor/YoloV5-AI-Label](https://github.com/MNConnor/YoloV5-AI-Label) <img src="https://img.shields.io/github/stars/MNConnor/YoloV5-AI-Label?style=social"/> : YoloV5 AI Assisted Labeling. 

    - [LILINOpenGitHub/Labeling-Tool](https://github.com/LILINOpenGitHub/Labeling-Tool) <img src="https://img.shields.io/github/stars/LILINOpenGitHub/Labeling-Tool?style=social"/> : Free YOLO AI labeling tool. YOLO AI labeling tool is a Windows app for labeling YOLO dataset. 

    - [whs0523003/YOLOv5_6.1_autolabel](https://github.com/whs0523003/YOLOv5_6.1_autolabel) <img src="https://img.shields.io/github/stars/whs0523003/YOLOv5_6.1_autolabel?style=social"/> : YOLOv5_6.1 自动标记目标框。 

    - [2vin/PyYAT](https://github.com/2vin/PyYAT) <img src="https://img.shields.io/github/stars/2vin/PyYAT?style=social"/> : Semi-Automatic Yolo Annotation Tool In Python. 




  - ### Feature Map Visualization
    #### 特征图可视化
    
    - [pooya-mohammadi/yolov5-gradcam](https://github.com/pooya-mohammadi/yolov5-gradcam) <img src="https://img.shields.io/github/stars/pooya-mohammadi/yolov5-gradcam?style=social"/> : Visualizing Yolov5's layers using GradCam.

    - [TorchCAM](https://github.com/frgfm/torch-cam) <img src="https://img.shields.io/github/stars/frgfm/torch-cam?style=social"/> : Class activation maps for your PyTorch models (CAM, Grad-CAM, Grad-CAM++, Smooth Grad-CAM++, Score-CAM, SS-CAM, IS-CAM, XGrad-CAM, Layer-CAM).

    - [Him-wen/OD_Heatmap](https://github.com/Him-wen/OD_Heatmap) <img src="https://img.shields.io/github/stars/Him-wen/OD_Heatmap?style=social"/> : Heatmap visualization of the YOLO model using the Grad-CAM heatmap visualization method can Intuitively show which regions in the image contribute the most to the category classification.



  - ### Object Detection Evaluation Metrics
    #### 目标检测性能评价指标
    
    - [rafaelpadilla/review_object_detection_metrics](https://github.com/rafaelpadilla/review_object_detection_metrics) <img src="https://img.shields.io/github/stars/rafaelpadilla/review_object_detection_metrics?style=social"/> : Object Detection Metrics. 14 object detection metrics: mean Average Precision (mAP), Average Recall (AR), Spatio-Temporal Tube Average Precision (STT-AP). This project supports different bounding box formats as in COCO, PASCAL, Imagenet, etc. "A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit".  (**[Electronics 2021](https://www.mdpi.com/2079-9292/10/3/279)**)

    - [rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics) <img src="https://img.shields.io/github/stars/rafaelpadilla/Object-Detection-Metrics?style=social"/> : Most popular metrics used to evaluate object detection algorithms. "A Survey on Performance Metrics for Object-Detection Algorithms". (**[IWSSIP 2020](https://ieeexplore.ieee.org/abstract/document/9145130)**)

    - [Cartucho/mAP](https://github.com/Cartucho/mAP) <img src="https://img.shields.io/github/stars/Cartucho/mAP?style=social"/> : mean Average Precision - This code evaluates the performance of your neural net for object recognition. 

    - [Lightning-AI/metrics](https://github.com/Lightning-AI/metrics) <img src="https://img.shields.io/github/stars/Lightning-AI/metrics?style=social"/> : Machine learning metrics for distributed, scalable PyTorch applications. 

    - [laclouis5/ObjectDetectionEval](https://github.com/laclouis5/ObjectDetectionEval) <img src="https://img.shields.io/github/stars/laclouis5/ObjectDetectionEval?style=social"/> : Object Detection Evaluation Library. Unified framework to parse, create and evaluate object detections from many frameworks (COCO, YOLO, PascalVOC, ImageNet, LabelMe, ...). 


  - ### GUI
    #### 图形用户界面

    - [Javacr/PyQt5-YOLOv5](https://github.com/Javacr/PyQt5-YOLOv5) <img src="https://img.shields.io/github/stars/Javacr/PyQt5-YOLOv5?style=social"/> : YOLOv5检测界面-PyQt5实现。

    - [scutlrr/Yolov4-QtGUI](https://github.com/scutlrr/Yolov4-QtGUI) <img src="https://img.shields.io/github/stars/scutlrr/Yolov4-QtGUI?style=social"/> : Yolov4-QtGUI是基于[QtGuiDemo](https://github.com/jmu201521121021/QtGuiDemo)项目开发的可视化目标检测界面，可以简便选择本地图片、摄像头来展示图像处理算法的结果。

    - [xugaoxiang/yolov5-pyqt5](https://github.com/xugaoxiang/yolov5-pyqt5) <img src="https://img.shields.io/github/stars/xugaoxiang/yolov5-pyqt5?style=social"/> : 给yolov5加个gui界面，使用pyqt5，yolov5是5.0版本。

    - [mxy493/YOLOv5-Qt](https://github.com/mxy493/YOLOv5-Qt) <img src="https://img.shields.io/github/stars/mxy493/YOLOv5-Qt?style=social"/> : 基于YOLOv5的GUI程序，支持选择要使用的权重文件，设置是否使用GPU，设置置信度阈值等参数。

    - [BonesCat/YoloV5_PyQt5](https://github.com/BonesCat/YoloV5_PyQt5) <img src="https://img.shields.io/github/stars/BonesCat/YoloV5_PyQt5?style=social"/> : Add gui for YoloV5 using PyQt5.

    - [LuckyBoy1798/yolov5-pyqt](https://github.com/LuckyBoy1798/yolov5-pyqt) <img src="https://img.shields.io/github/stars/LuckyBoy1798/yolov5-pyqt?style=social"/> : 基于yolov5+pyqt的甲骨文图形化检测工具。

    - [PySimpleGUI/PySimpleGUI-YOLO](https://github.com/PySimpleGUI/PySimpleGUI-YOLO) <img src="https://img.shields.io/github/stars/PySimpleGUI/PySimpleGUI-YOLO?style=social"/> : A YOLO Artificial Intelligence algorithm demonstration using PySimpleGUI.

    - [prabindh/qt5-opencv3-darknet](https://github.com/prabindh/qt5-opencv3-darknet) <img src="https://img.shields.io/github/stars/prabindh/qt5-opencv3-darknet?style=social"/> : Qt5 + Darknet/Yolo + OpenCV3.

    - [GinkgoX/YOLOv3GUI_Pytorch_PyQt5](https://github.com/GinkgoX/YOLOv3GUI_Pytorch_PyQt5) <img src="https://img.shields.io/github/stars/GinkgoX/YOLOv3GUI_Pytorch_PyQt5?style=social"/> : This is a GUI project for Deep Learning Object Detection based on YOLOv3 model.

    - [xietx1995/YOLO-QT-Camera-Tool](https://github.com/xietx1995/YOLO-QT-Camera-Tool) <img src="https://img.shields.io/github/stars/xietx1995/YOLO-QT-Camera-Tool?style=social"/> : Detecting objects from camera or local video files vi qt and yolo.

    - [FatemeZamanian/Yolov5-Fruit-Detector](https://github.com/FatemeZamanian/Yolov5-Fruit-Detector) <img src="https://img.shields.io/github/stars/FatemeZamanian/Yolov5-Fruit-Detector?style=social"/> : A program to recognize fruits on pictures or videos using yolov5.

    - [BioMeasure/PyQt5_YoLoV5_DeepSort](https://github.com/BioMeasure/PyQt5_YoLoV5_DeepSort) <img src="https://img.shields.io/github/stars/BioMeasure/PyQt5_YoLoV5_DeepSort?style=social"/> : This is a PyQt5 GUI program, which is based on YoloV5 and DeepSort to track person.

    - [DongLizhong/YOLO_SORT_QT](https://github.com/DongLizhong/YOLO_SORT_QT) <img src="https://img.shields.io/github/stars/DongLizhong/YOLO_SORT_QT?style=social"/> : This code uses the opencv dnn module to load the darknet model for detection and add SORT for multi-object tracking(MOT).

    - [Whu-wxy/yolov5_deepsort_ncnn_qt](https://github.com/Whu-wxy/yolov5_deepsort_ncnn_qt) <img src="https://img.shields.io/github/stars/Whu-wxy/yolov5_deepsort_ncnn_qt?style=social"/> : 用ncnn调用yolov5和deep sort模型，opencv读取视频。

    - [jeswanthgalla/PyQt4_GUI_darknet_yolov4](https://github.com/jeswanthgalla/PyQt4_GUI_darknet_yolov4) <img src="https://img.shields.io/github/stars/jeswanthgalla/PyQt4_GUI_darknet_yolov4?style=social"/> : GUI App using PyQt4. Multithreading to process multiple camera streams and using darknet yolov4 model for object detection. 

    - [barleo01/yoloobjectdetector](https://github.com/barleo01/yoloobjectdetector) <img src="https://img.shields.io/github/stars/barleo01/yoloobjectdetector?style=social"/> : The pupose of this application is to capture video from a camera, apply a YOLO Object detector and display it on a simple Qt Gui.

    - [Eagle104fred/PyQt5-Yolov5](https://github.com/Eagle104fred/PyQt5-Yolov5) <img src="https://img.shields.io/github/stars/Eagle104fred/PyQt5-Yolov5?style=social"/> : 把YOLOv5的视频显示到pyqt5ui上。

    - [cnyvfang/YOLOv5-GUI](https://github.com/cnyvfang/YOLOv5-GUI) <img src="https://img.shields.io/github/stars/Eagle104fred/PyQt5-Yolov5?style=social"/> : Qt-GUI implementation of the YOLOv5 algorithm (ver.6 and ver.5). YOLOv5算法(ver.6及ver.5)的Qt-GUI实现。

    - [WeNN-Artificial-Intelligence/PyQT-Object-Detection-App](https://github.com/WeNN-Artificial-Intelligence/PyQT-Object-Detection-App) <img src="https://img.shields.io/github/stars/WeNN-Artificial-Intelligence/PyQT-Object-Detection-App?style=social"/> : Real-time object detection app with Python and PyQt framework. 

    - [RaminAbbaszadi/YoloWrapper-WPF](https://github.com/RaminAbbaszadi/YoloWrapper-WPF) <img src="https://img.shields.io/github/stars/RaminAbbaszadi/YoloWrapper-WPF?style=social"/> : WPF (C#) Yolo Darknet Wrapper.

    - [fengyhack/YoloWpf](https://github.com/fengyhack/YoloWpf) <img src="https://img.shields.io/github/stars/fengyhack/YoloWpf?style=social"/> : GUI demo for Object Detection with YOLO and OpenCVSharp.

    - [hanzhuang111/Yolov5Wpf](https://github.com/hanzhuang111/Yolov5Wpf) <img src="https://img.shields.io/github/stars/hanzhuang111/Yolov5Wpf?style=social"/> : 使用ML.NET部署YOLOV5 的ONNX模型。

    - [MaikoKingma/yolo-winforms-test](https://github.com/MaikoKingma/yolo-winforms-test) <img src="https://img.shields.io/github/stars/MaikoKingma/yolo-winforms-test?style=social"/> : A Windows forms application that can execute pre-trained object detection models via ML.NET. In this instance the You Only Look Once version 4 (yolov4) is used.

    - [Soju06/yolov5-annotation-viewer](https://github.com/Soju06/yolov5-annotation-viewer) <img src="https://img.shields.io/github/stars/Soju06/yolov5-annotation-viewer?style=social"/> : yolov5 annotation viewer.

    - [Powercube7/YOLOv5-GUI](https://github.com/Powercube7/YOLOv5-GUI) <img src="https://img.shields.io/github/stars/Powercube7/YOLOv5-GUI?style=social"/> : A simple GUI made for creating jobs in YOLOv5.

    - [cdmstrong/yolov5-pyqt-moke](https://github.com/cdmstrong/yolov5-pyqt-moke) <img src="https://img.shields.io/github/stars/cdmstrong/yolov5-pyqt-moke?style=social"/> : 利用yolov5和pyqt做可视化检测。


  - ### Other Applications
    #### 其它应用

    - [penny4860/Yolo-digit-detector](https://github.com/penny4860/Yolo-digit-detector) <img src="https://img.shields.io/github/stars/penny4860/Yolo-digit-detector?style=social"/> : Implemented digit detector in natural scene using resnet50 and Yolo-v2. I used SVHN as the training set, and implemented it using tensorflow and keras.

    - [chineseocr/table-detect](https://github.com/chineseocr/table-detect) <img src="https://img.shields.io/github/stars/chineseocr/table-detect?style=social"/> : table detect(yolo) , table line(unet) （表格检测/表格单元格定位）。

    - [thisiszhou/SexyYolo](https://github.com/thisiszhou/SexyYolo) <img src="https://img.shields.io/github/stars/thisiszhou/SexyYolo?style=social"/> : An implementation of Yolov3 with Tensorflow1.x, which could detect COCO and sexy or porn person simultaneously.

    - [javirk/Person_remover](https://github.com/javirk/Person_remover) <img src="https://img.shields.io/github/stars/javirk/Person_remover?style=social"/> : People removal in images using Pix2Pix and YOLO. 

    - [foschmitz/yolo-python-rtsp](https://github.com/foschmitz/yolo-python-rtsp) <img src="https://img.shields.io/github/stars/foschmitz/yolo-python-rtsp?style=social"/> : Object detection using deep learning with Yolo, OpenCV and Python via Real Time Streaming Protocol (RTSP).

    - [ismail-mebsout/Parsing-PDFs-using-YOLOV3](https://github.com/ismail-mebsout/Parsing-PDFs-using-YOLOV3) <img src="https://img.shields.io/github/stars/ismail-mebsout/Parsing-PDFs-using-YOLOV3?style=social"/> : Parsing pdf tables using YOLOV3.

    - [008karan/PAN_OCR](https://github.com/008karan/PAN_OCR) <img src="https://img.shields.io/github/stars/008karan/PAN_OCR?style=social"/> : Building OCR using YOLO and Tesseract.

    - [stephanecharette/DarkMark](https://github.com/stephanecharette/DarkMark) <img src="https://img.shields.io/github/stars/stephanecharette/DarkMark?style=social"/> : Marking up images for use with Darknet.

    - [zeyad-mansour/lunar](https://github.com/zeyad-mansour/lunar) <img src="https://img.shields.io/github/stars/zeyad-mansour/lunar?style=social"/> : Lunar is a neural network aimbot that uses real-time object detection accelerated with CUDA on Nvidia GPUs.

    - [lannguyen0910/food-detection-yolov5](https://github.com/lannguyen0910/food-detection-yolov5) <img src="https://img.shields.io/github/stars/lannguyen0910/food-detection-yolov5?style=social"/> : YOLOv5 meal analysis.

    - [killnice/yolov5-D435i](https://github.com/killnice/yolov5-D435i) <img src="https://img.shields.io/github/stars/killnice/yolov5-D435i?style=social"/> : using yolov5 and realsense D435i.

    - [SahilChachra/Video-Analytics-Dashboard](https://github.com/SahilChachra/Video-Analytics-Dashboard) <img src="https://img.shields.io/github/stars/SahilChachra/Video-Analytics-Dashboard?style=social"/> : Video Analytics dashboard built using YoloV5 and Streamlit.

    - [isLinXu/YOLOv5_Efficient](https://github.com/isLinXu/YOLOv5_Efficient) <img src="https://img.shields.io/github/stars/isLinXu/YOLOv5_Efficient?style=social"/> : Use yolov5 efficiently(高效地使用Yolo v5).

    - [HRan2004/Yolo-ArbV2](https://github.com/HRan2004/Yolo-ArbV2) <img src="https://img.shields.io/github/stars/HRan2004/Yolo-ArbV2?style=social"/> : Yolo-ArbV2 在完全保持YOLOv5功能情况下，实现可选多边形信息输出。

    - [Badw0lf613/wmreading_system](https://github.com/Badw0lf613/wmreading_system) <img src="https://img.shields.io/github/stars/Badw0lf613/wmreading_system?style=social"/> : 基于YOLOv5的水表读数系统。

    - [zgcr/SimpleAICV-pytorch-ImageNet-COCO-training](https://github.com/zgcr/SimpleAICV-pytorch-ImageNet-COCO-training) <img src="https://img.shields.io/github/stars/zgcr/SimpleAICV-pytorch-ImageNet-COCO-training?style=social"/> : SimpleAICV:pytorch training example on ImageNet(ILSVRC2012)/COCO2017/VOC2007+2012 datasets.Include ResNet/DarkNet/RetinaNet/FCOS/CenterNet/TTFNet/YOLOv3/YOLOv4/YOLOv5/YOLOX. 

    - [ErenKaymakci/Real-Time-QR-Detection-and-Decoding](https://github.com/ErenKaymakci/Real-Time-QR-Detection-and-Decoding) <img src="https://img.shields.io/github/stars/ErenKaymakci/Real-Time-QR-Detection-and-Decoding?style=social"/> : This repo explain how qr codes works, qr detection and decoding. 

    - [LUMAIS/AntDet_YOLOv5](https://github.com/LUMAIS/AntDet_YOLOv5) <img src="https://img.shields.io/github/stars/LUMAIS/AntDet_YOLOv5?style=social"/> : Ants and their Activiteis (Trophallaxis) Detection using YOLOv5 based on PyTorch. 

    - [buxihuo/OW-YOLO](https://github.com/buxihuo/OW-YOLO) <img src="https://img.shields.io/github/stars/buxihuo/OW-YOLO?style=social"/> : Detect known and unknown objects in the open world（具有区分已知与未知能力的全新检测器））.

    - [Jiseong-Ok/OCR-Yolov5-SwinIR-SVTR](https://github.com/Jiseong-Ok/OCR-Yolov5-SwinIR-SVTR) <img src="https://img.shields.io/github/stars/Jiseong-Ok/OCR-Yolov5-SwinIR-SVTR?style=social"/> : OCR(Korean).

    - [QIN2DIM/hcaptcha-challenger](https://github.com/QIN2DIM/hcaptcha-challenger) <img src="https://img.shields.io/github/stars/QIN2DIM/hcaptcha-challenger?style=social"/> : 🥂 Gracefully face hCaptcha challenge with YOLOv6(ONNX) embedded solution. 

    - [bobjiangps/vision](https://github.com/bobjiangps/vision) <img src="https://img.shields.io/github/stars/bobjiangps/vision?style=social"/> : UI auto test framework based on YOLO to recognize elements, less code, less maintenance, cross platform, cross project / 基于YOLO的UI层自动化测试框架, 可识别控件类型，减少代码和维护，一定程度上跨平台跨项目。








