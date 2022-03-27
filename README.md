# Awesome-YOLO
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

🔥🔥🔥 YOLO is a great real-time one-stage object detection framework. This repo lists some awesome open-source YOLO series object detection projects.

## Contents
- [Awesome-YOLO](#awesome-yolo)
    - [YOLO Family](#yolo-family)
        - [Master Versions](#master-versions)
        - [Other Versions](#Other-Versions)
          - [PyTorch Implementation](#pytorch-implementation)
          - [Tensorflow and Keras Implementation](#tensorflow-and-keras-implementation)
          - [PaddlePaddle Implementation](#paddlepaddle-implementation)
          - [Caffe Implementation](#caffe-implementation)
          - [MXNet Implementation](#mxnet-implementation)
          - [LibTorch Implementation](#libtorch-implementation)
          - [OpenCV Implementation](#opencv-implementation)
          - [ROS Implementation](#ros-implementation)
          - [CSharp Implementation](#csharp-implementation)
          - [Rust Implementation](#rust-implementation)
          - [Web Implementation](#web-implementation)
          - [Others](#others)
    - [Extensional Frameworks](#extensional-frameworks)
    - [Applications](#applications)
        - [Lighter and Faster](#lighter-and-faster)
          - [Lightweight Backbones (轻量级骨干网络)](#lightweight-backbones)
          - [Pruning Distillation Quantization (剪枝 蒸馏 量化)](#pruning-distillation-quantization)
          - [High-performance Inference Engine (高性能推理引擎)](#high-performance-inference-engine)
          - [FPGA TPU RISC-V MCU Hardware Deployment (FPGA TPU RISC-V MCU硬件部署)](#fpga-tpu-risc-v-mcu-hardware-deployment)
        - [Object Tracking (目标跟踪)](#object-tracking)
        - [Reinforcement Learning (强化学习)](#reinforcement-learning)
        - [Motion Control (运动控制)](#motion-control)
        - [Spiking Neural Network (SNN, 脉冲神经网络)](#spiking-neural-network)
        - [Attention and Transformer (注意力机制)](#attention-and-transformer)
        - [Small Object Detection (小目标检测)](#small-object-detection)
        - [Oriented Object Detection (旋转目标检测)](#oriented-object-detection)
        - [Face Detection (人脸检测)](#face-detection)
        - [Face Mask Detection (口罩检测)](#face-mask-detection)
        - [Social Distance Detection (社交距离检测)](#social-distance-detection)
        - [Intelligent Transportation Field Detection (智能交通领域检测)](#intelligent-transportation-field-detection)
          - [Vehicle Detection (车辆检测)](#vehicle-detection)
          - [License Plate Detection (车牌检测)](#license-plate-detection)
          - [Lane Detection (车道线检测)](#lane-detection)
          - [Driving Behavior Detection (驾驶行为检测)](#driving-behavior-detection)
          - [Parking Slot Detection (停车位检测)](#parking-slot-detection)
          - [Traffic Light Detection (交通灯检测)](#traffic-light-detection)
          - [Crosswalk Detection (人行横道/斑马线检测)](#crosswalk-detection)
          - [Traffic Accidents Detection (交通事故检测)](#traffic-accidents-detection)
          - [Road Damage Detection (道路损伤检测)](#road-damage-detection)
        - [Helmet Detection (头盔/安全帽检测)](#helmet-detection)
        - [Hand Detection (手部检测)](#hand-detection)
        - [Gesture Recognition (手势/手语识别)](#gesture-recognition)
        - [Action Detection (动作检测)](#action-detection)
        - [Human Pose Estimation (人体姿态估计)](#human-pose-estimation)
        - [3D Object Detection (三维目标检测)](#3d-object-detection)
        - [Safety Monitoring Field Detection (安防监控领域检测)](#safety-monitoring-field-detection)
        - [Industrial Defect Detection (工业缺陷检测)](#industrial-defect-detection)
        - [Medical Field Detection (医学领域检测)](#medical-field-detection)
        - [Adverse Weather Conditions (恶劣天气环境)](#adverse-weather-conditions)
        - [Adversarial Attack and Defense (对抗攻击与防御)](#adversarial-attack-and-defense)
        - [Semantic Segmentation (语义分割)](#semantic-segmentation)
        - [Game Field Detection (游戏领域检测)](#game-field-detection)
        - [Automatic Annotation Tool (自动标注工具)](#automatic-annotation-tool)
        - [GUI (图形界面)](#gui)
        - [Other Applications](#other-applications)


## YOLO Family

  - ### Master Versions

    - [YOLO](https://pjreddie.com/darknet/yolov1) ([Darknet](https://github.com/pjreddie/darknet) <img src="https://img.shields.io/github/stars/pjreddie/darknet?style=social"/>) : "You Only Look Once: Unified, Real-Time Object Detection". (**[CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)**)

    - [YOLOv2](https://pjreddie.com/darknet/yolov2) ([Darknet](https://github.com/pjreddie/darknet) <img src="https://img.shields.io/github/stars/pjreddie/darknet?style=social"/>) : "YOLO9000: Better, Faster, Stronger". (**[CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)**)

    - [YOLOv3](https://pjreddie.com/darknet/yolo) ([Darknet](https://github.com/pjreddie/darknet) <img src="https://img.shields.io/github/stars/pjreddie/darknet?style=social"/>) : "YOLOv3: An Incremental Improvement". (**[arXiv 2018](https://arxiv.org/abs/1804.02767)**)

    - [YOLOv4](https://github.com/AlexeyAB/darknet) <img src="https://img.shields.io/github/stars/AlexeyAB/darknet?style=social"/> : "YOLOv4: Optimal Speed and Accuracy of Object Detection". (**[arXiv 2020](https://arxiv.org/abs/2004.10934)**)

    - [Scaled-YOLOv4](https://github.com/AlexeyAB/darknet) <img src="https://img.shields.io/github/stars/AlexeyAB/darknet?style=social"/> ([PyTorch version](https://github.com/WongKinYiu/ScaledYOLOv4) <img src="https://img.shields.io/github/stars/WongKinYiu/ScaledYOLOv4?style=social"/>) : "Scaled-YOLOv4: Scaling Cross Stage Partial Network". (**[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html)**)

    - [ultralytics/yolov5](https://github.com/ultralytics/yolov5) <img src="https://img.shields.io/github/stars/ultralytics/yolov5?style=social"/> : YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite.

  - ### Other Versions

    - #### PyTorch Implementation

      - [MMDetection](https://github.com/open-mmlab/mmdetection) <img src="https://img.shields.io/github/stars/open-mmlab/mmdetection?style=social"/> : OpenMMLab Detection Toolbox and Benchmark. (**[arXiv 2019](https://arxiv.org/abs/1906.07155)**)

      - [ultralytics/yolov3](https://github.com/ultralytics/yolov3) <img src="https://img.shields.io/github/stars/ultralytics/yolov3?style=social"/> : YOLOv3 in PyTorch > ONNX > CoreML > TFLite.

      - [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) <img src="https://img.shields.io/github/stars/eriklindernoren/PyTorch-YOLOv3?style=social"/> : Minimal PyTorch implementation of YOLOv3.

      - [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) <img src="https://img.shields.io/github/stars/Tianxiaomo/pytorch-YOLOv4?style=social"/> : PyTorch ,ONNX and TensorRT implementation of YOLOv4.

      - [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) <img src="https://img.shields.io/github/stars/ayooshkathuria/pytorch-yolo-v3?style=social"/> : A PyTorch implementation of the YOLO v3 object detection algorithm.

      - [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4) <img src="https://img.shields.io/github/stars/WongKinYiu/PyTorch_YOLOv4?style=social"/> : PyTorch implementation of YOLOv4.

      - [argusswift/YOLOv4-pytorch](https://github.com/argusswift/YOLOv4-pytorch) <img src="https://img.shields.io/github/stars/argusswift/YOLOv4-pytorch?style=social"/> : This is a pytorch repository of YOLOv4, attentive YOLOv4 and mobilenet YOLOv4 with PASCAL VOC and COCO.

      - [longcw/yolo2-pytorch](https://github.com/longcw/yolo2-pytorch) <img src="https://img.shields.io/github/stars/longcw/yolo2-pytorch?style=social"/> : YOLOv2 in PyTorch.

      - [bubbliiiing/yolov5-pytorch](https://github.com/bubbliiiing/yolov5-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov5-pytorch?style=social"/> : 这是一个YoloV5-pytorch的源码，可以用于训练自己的模型。

      - [bubbliiiing/yolov4-pytorch](https://github.com/bubbliiiing/yolov4-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov4-pytorch?style=social"/> : 这是一个YoloV4-pytorch的源码，可以用于训练自己的模型。

      - [bubbliiiing/yolov4-tiny-pytorch](https://github.com/bubbliiiing/yolov4-tiny-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolov4-tiny-pytorch?style=social"/> : 这是一个YoloV4-tiny-pytorch的源码，可以用于训练自己的模型。

      - [bubbliiiing/yolov3-pytorch](https://github.com/bubbliiiing/yolo3-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolo3-pytorch?style=social"/> : 这是一个yolo3-pytorch的源码，可以用于训练自己的模型。

      - [bubbliiiing/yolox-pytorch](https://github.com/bubbliiiing/yolox-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/yolox-pytorch?style=social"/> : 这是一个yolox-pytorch的源码，可以用于训练自己的模型。

      - [BobLiu20/YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch) <img src="https://img.shields.io/github/stars/BobLiu20/YOLOv3_PyTorch?style=social"/> : Full implementation of YOLOv3 in PyTorch.

      - [ruiminshen/yolo2-pytorch](https://github.com/ruiminshen/yolo2-pytorch) <img src="https://img.shields.io/github/stars/ruiminshen/yolo2-pytorch?style=social"/> : PyTorch implementation of the YOLO (You Only Look Once) v2.

      - [DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3) <img src="https://img.shields.io/github/stars/DeNA/PyTorch_YOLOv3?style=social"/> : Implementation of YOLOv3 in PyTorch.

      - [abeardear/pytorch-YOLO-v1](https://github.com/abeardear/pytorch-YOLO-v1) <img src="https://img.shields.io/github/stars/abeardear/pytorch-YOLO-v1?style=social"/> : an experiment for yolo-v1, including training and testing.

      - [wuzhihao7788/yolodet-pytorch](https://github.com/wuzhihao7788/yolodet-pytorch) <img src="https://img.shields.io/github/stars/wuzhihao7788/yolodet-pytorch?style=social"/> : reproduce the YOLO series of papers in pytorch, including YOLOv4, PP-YOLO, YOLOv5，YOLOv3, etc.

      - [uvipen/Yolo-v2-pytorch](https://github.com/uvipen/Yolo-v2-pytorch) <img src="https://img.shields.io/github/stars/uvipen/Yolo-v2-pytorch?style=social"/> : YOLO for object detection tasks.

      - [Peterisfar/YOLOV3](https://github.com/Peterisfar/YOLOV3) <img src="https://img.shields.io/github/stars/Peterisfar/YOLOV3?style=social"/> : yolov3 by pytorch.

      - [misads/easy_detection](https://github.com/misads/easy_detection) <img src="https://img.shields.io/github/stars/misads/easy_detection?style=social"/> : 一个简单方便的目标检测框架(PyTorch环境可直接运行，不需要cuda编译)，支持Faster_RCNN、Yolo系列(v2~v5)、EfficientDet、RetinaNet、Cascade-RCNN等经典网络。

      - [miemiedetection](https://github.com/miemie2013/miemiedetection) <img src="https://img.shields.io/github/stars/miemie2013/miemiedetection?style=social"/> : Pytorch implementation of YOLOX、PPYOLO、PPYOLOv2、FCOS an so on.

    - #### Tensorflow and Keras Implementation

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

    - #### PaddlePaddle Implementation
 
      - [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) <img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleDetection?style=social"/> : Object Detection toolkit based on PaddlePaddle. "PP-YOLO: An Effective and Efficient Implementation of Object Detector". (**[arXiv 2020](https://arxiv.org/abs/2007.12099)**)

      - [miemie2013/Paddle-YOLOv4](https://github.com/miemie2013/Paddle-YOLOv4) <img src="https://img.shields.io/github/stars/miemie2013/Paddle-YOLOv4?style=social"/> : Paddle-YOLOv4.

      - [Sharpiless/PaddleDetection-Yolov5](https://github.com/Sharpiless/PaddleDetection-Yolov5) <img src="https://img.shields.io/github/stars/Sharpiless/PaddleDetection-Yolov5?style=social"/> : 基于Paddlepaddle复现yolov5，支持PaddleDetection接口。

    - #### Caffe Implementation

      - [ChenYingpeng/caffe-yolov3](https://github.com/ChenYingpeng/caffe-yolov3) <img src="https://img.shields.io/github/stars/ChenYingpeng/caffe-yolov3?style=social"/> : A real-time object detection framework of Yolov3/v4 based on caffe.

      - [ChenYingpeng/darknet2caffe](https://github.com/ChenYingpeng/darknet2caffe) <img src="https://img.shields.io/github/stars/ChenYingpeng/darknet2caffe?style=social"/> : Convert darknet weights to caffemodel.

      - [eric612/Caffe-YOLOv3-Windows](https://github.com/eric612/Caffe-YOLOv3-Windows) <img src="https://img.shields.io/github/stars/eric612/Caffe-YOLOv3-Windows?style=social"/> : A windows caffe implementation of YOLO detection network.

      - [Harick1/caffe-yolo](https://github.com/Harick1/caffe-yolo) <img src="https://img.shields.io/github/stars/Harick1/caffe-yolo?style=social"/> : Caffe for YOLO.

      - [choasup/caffe-yolo9000](https://github.com/choasup/caffe-yolo9000) <img src="https://img.shields.io/github/stars/choasup/caffe-yolo9000?style=social"/> : Caffe for YOLOv2 & YOLO9000.

      - [gklz1982/caffe-yolov2](https://github.com/gklz1982/caffe-yolov2) <img src="https://img.shields.io/github/stars/gklz1982/caffe-yolov2?style=social"/> : caffe-yolov2.

    - #### MXNet Implementation
      - [Gluon CV Toolkit](https://github.com/dmlc/gluon-cv) <img src="https://img.shields.io/github/stars/dmlc/gluon-cv?style=social"/> : GluonCV provides implementations of the state-of-the-art (SOTA) deep learning models in computer vision.

      - [zhreshold/mxnet-yolo](https://github.com/zhreshold/mxnet-yolo) <img src="https://img.shields.io/github/stars/zhreshold/mxnet-yolo?style=social"/> : YOLO: You only look once real-time object detector.

    - #### LibTorch Implementation

      - [walktree/libtorch-yolov3](https://github.com/walktree/libtorch-yolov3) <img src="https://img.shields.io/github/stars/walktree/libtorch-yolov3?style=social"/> : A Libtorch implementation of the YOLO v3 object detection algorithm, written with pure C++.

      - [yasenh/libtorch-yolov5](https://github.com/yasenh/libtorch-yolov5) <img src="https://img.shields.io/github/stars/yasenh/libtorch-yolov5?style=social"/> : A LibTorch inference implementation of the yolov5.

      - [Nebula4869/YOLOv5-LibTorch](https://github.com/Nebula4869/YOLOv5-LibTorch) <img src="https://img.shields.io/github/stars/Nebula4869/YOLOv5-LibTorch?style=social"/> : Real time object detection with deployment of YOLOv5 through LibTorch C++ API.

    - #### OpenCV Implementation

      - [hpc203/yolov5-dnn-cpp-python](https://github.com/hpc203/yolov5-dnn-cpp-python) <img src="https://img.shields.io/github/stars/hpc203/yolov5-dnn-cpp-python?style=social"/> : 用opencv的dnn模块做yolov5目标检测，包含C++和Python两个版本的程序。

      - [hpc203/yolox-opencv-dnn](https://github.com/hpc203/yolox-opencv-dnn) <img src="https://img.shields.io/github/stars/hpc203/yolox-opencv-dnn?style=social"/> : 使用OpenCV部署YOLOX，支持YOLOX-S、YOLOX-M、YOLOX-L、YOLOX-X、YOLOX-Darknet53五种结构，包含C++和Python两种版本的程序。

      - [UNeedCryDear/yolov5-opencv-dnn-cpp](https://github.com/UNeedCryDear/yolov5-opencv-dnn-cpp) <img src="https://img.shields.io/github/stars/UNeedCryDear/yolov5-opencv-dnn-cpp?style=social"/> : 使用opencv模块部署yolov5-6.0版本。

    - #### ROS Implementation

      - [leggedrobotics/darknet_ros](https://github.com/leggedrobotics/darknet_ros) <img src="https://img.shields.io/github/stars/leggedrobotics/darknet_ros?style=social"/> : Real-Time Object Detection for ROS.

      - [engcang/ros-yolo-sort](https://github.com/engcang/ros-yolo-sort) <img src="https://img.shields.io/github/stars/engcang/ros-yolo-sort?style=social"/> : YOLO and SORT, and ROS versions of them.

      - [chrisgundling/YoloLight](https://github.com/chrisgundling/YoloLight) <img src="https://img.shields.io/github/stars/chrisgundling/YoloLight?style=social"/> : Tiny-YOLO-v2 ROS Node for Traffic Light Detection.

      - [Ar-Ray-code/YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS) <img src="https://img.shields.io/github/stars/Ar-Ray-code/YOLOX-ROS?style=social"/> : YOLOX + ROS2 object detection package.

      - [Ar-Ray-code/YOLOv5-ROS](https://github.com/Ar-Ray-code/YOLOv5-ROS) <img src="https://img.shields.io/github/stars/Ar-Ray-code/YOLOv5-ROS?style=social"/> : YOLOv5 + ROS2 object detection package.

      - [Tossy0423/yolov4-for-darknet_ros](https://github.com/Tossy0423/yolov4-for-darknet_ros) <img src="https://img.shields.io/github/stars/Tossy0423/yolov4-for-darknet_ros?style=social"/> : This is the environment in which YOLO V4 is ported to darknet_ros.

      - [qianmin/yolov5_ROS](https://github.com/qianmin/yolov5_ROS) <img src="https://img.shields.io/github/stars/qianmin/yolov5_ROS?style=social"/> : run YOLOv5 in ROS，ROS使用YOLOv5。

      - [ailllist/yolov5_ROS](https://github.com/ailllist/yolov5_ROS) <img src="https://img.shields.io/github/stars/ailllist/yolov5_ROS?style=social"/> : yolov5 for ros, not webcam.

    - #### CSharp Implementation

      - [Alturos.Yolo](https://github.com/AlturosDestinations/Alturos.Yolo) <img src="https://img.shields.io/github/stars/AlturosDestinations/Alturos.Yolo?style=social"/> : C# Yolo Darknet Wrapper (real-time object detection).

      - [keijiro/TinyYOLOv2Barracuda](https://github.com/keijiro/TinyYOLOv2Barracuda) <img src="https://img.shields.io/github/stars/keijiro/TinyYOLOv2Barracuda?style=social"/> : Tiny YOLOv2 on Unity Barracuda.

      - [derenlei/Unity_Detection2AR](https://github.com/derenlei/Unity_Detection2AR) <img src="https://img.shields.io/github/stars/derenlei/Unity_Detection2AR?style=social"/> : Localize 2D image object detection in 3D Scene with Yolo in Unity Barracuda and ARFoundation.

      - [mentalstack/yolov5-net](https://github.com/mentalstack/yolov5-net) <img src="https://img.shields.io/github/stars/mentalstack/yolov5-net?style=social"/> : YOLOv5 object detection with C#, ML.NET, ONNX.

      - [died/YOLO3-With-OpenCvSharp4](https://github.com/died/YOLO3-With-OpenCvSharp4) <img src="https://img.shields.io/github/stars/died/YOLO3-With-OpenCvSharp4?style=social"/> : Demo of implement YOLO v3 with OpenCvSharp v4 on C#.

      - [mbaske/yolo-unity](https://github.com/mbaske/yolo-unity) <img src="https://img.shields.io/github/stars/mbaske/yolo-unity?style=social"/> : YOLO In-Game Object Detection for Unity (Windows).

      - [BobLd/YOLOv4MLNet](https://github.com/BobLd/YOLOv4MLNet) <img src="https://img.shields.io/github/stars/BobLd/YOLOv4MLNet?style=social"/> : Use the YOLO v4 and v5 (ONNX) models for object detection in C# using ML.Net.

      - [keijiro/YoloV4TinyBarracuda](https://github.com/keijiro/YoloV4TinyBarracuda) <img src="https://img.shields.io/github/stars/keijiro/YoloV4TinyBarracuda?style=social"/> : oloV4TinyBarracuda is an implementation of the YOLOv4-tiny object detection model on the Unity Barracuda neural network inference library.

      - [zhang8043/YoloWrapper](https://github.com/zhang8043/YoloWrapper) <img src="https://img.shields.io/github/stars/zhang8043/YoloWrapper?style=social"/> : C#封装YOLOv4算法进行目标检测。

      - [maalik0786/FastYolo](https://github.com/maalik0786/FastYolo) <img src="https://img.shields.io/github/stars/maalik0786/FastYolo?style=social"/> : Fast Yolo for fast initializing, object detection and tracking.

      - [Uehwan/CSharp-Yolo-Video](https://github.com/Uehwan/CSharp-Yolo-Video) <img src="https://img.shields.io/github/stars/Uehwan/CSharp-Yolo-Video?style=social"/> : C# Yolo for Video.

      - [HTTP123-A/HumanDetection_Yolov5NET](https://github.com/https://github.com/HTTP123-A/HumanDetection_Yolov5NET) <img src="https://img.shields.io/github/stars/HTTP123-A/HumanDetection_Yolov5NET?style=social"/> : YOLOv5 object detection with ML.NET, ONNX.


    - #### Rust Implementation

      - [ptaxom/pnn](https://github.com/ptaxom/pnn) <img src="https://img.shields.io/github/stars/ptaxom/pnn?style=social"/> : pnn is Darknet compatible neural nets inference engine implemented in Rust.

      - [12101111/yolo-rs](https://github.com/12101111/yolo-rs) <img src="https://img.shields.io/github/stars/12101111/yolo-rs?style=social"/> : Yolov3 & Yolov4 with TVM and rust.

      - [TKGgunter/yolov4_tiny_rs](https://github.com/TKGgunter/yolov4_tiny_rs) <img src="https://img.shields.io/github/stars/TKGgunter/yolov4_tiny_rs?style=social"/> : A rust implementation of yolov4_tiny algorithm.

      - [flixstn/You-Only-Look-Once](https://github.com/flixstn/You-Only-Look-Once) <img src="https://img.shields.io/github/stars/flixstn/You-Only-Look-Once?style=social"/> : A Rust implementation of Yolo for object detection and tracking.

      - [lenna-project/yolo-plugin](https://github.com/lenna-project/yolo-plugin) <img src="https://img.shields.io/github/stars/lenna-project/yolo-plugin?style=social"/> : Yolo Object Detection Plugin for Lenna.

    - #### Web Implementation

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

    - #### Others

      - [BMW-InnovationLab/BMW-YOLOv4-Training-Automation](https://github.com/BMW-InnovationLab/BMW-YOLOv4-Training-Automation) <img src="https://img.shields.io/github/stars/BMW-InnovationLab/BMW-YOLOv4-Training-Automation?style=social"/> : YOLOv4-v3 Training Automation API for Linux.

      - [AntonMu/TrainYourOwnYOLO](https://github.com/AntonMu/TrainYourOwnYOLO) <img src="https://img.shields.io/github/stars/AntonMu/TrainYourOwnYOLO?style=social"/> : Train a state-of-the-art yolov3 object detector from scratch!

      - [madhawav/YOLO3-4-Py](https://github.com/madhawav/YOLO3-4-Py) <img src="https://img.shields.io/github/stars/madhawav/YOLO3-4-Py?style=social"/> : A Python wrapper on Darknet. Compatible with YOLO V3.

      - [theAIGuysCode/yolov4-custom-functions](https://github.com/theAIGuysCode/yolov4-custom-functions) <img src="https://img.shields.io/github/stars/theAIGuysCode/yolov4-custom-functions?style=social"/> : A Wide Range of Custom Functions for YOLOv4, YOLOv4-tiny, YOLOv3, and YOLOv3-tiny Implemented in TensorFlow, TFLite, and TensorRT.

      - [DL-Practise/YoloAll](https://github.com/DL-Practise/YoloAll) <img src="https://img.shields.io/github/stars/DL-Practise/YoloAll?style=social"/> : YoloAll is a collection of yolo all versions. you you use YoloAll to test yolov3/yolov5/yolox/yolo_fastest.

      - [msnh2012/Msnhnet](https://github.com/msnh2012/Msnhnet) <img src="https://img.shields.io/github/stars/msnh2012/Msnhnet?style=social"/> : (yolov3 yolov4 yolov5 unet ...)A mini pytorch inference framework which inspired from darknet.

      - [fcakyon/yolov5-pip](https://github.com/fcakyon/yolov5-pip) <img src="https://img.shields.io/github/stars/fcakyon/yolov5-pip?style=social"/> : Packaged version of ultralytics/yolov5.

      - [Laughing-q/YOLO-Q](https://github.com/Laughing-q/YOLO-Q) <img src="https://img.shields.io/github/stars/Laughing-q/YOLO-Q?style=social"/> : A inference framework that support multi models of yolo5(torch and tensorrt), yolox(torch and tensorrt), nanodet(tensorrt), yolo-fastestV2(tensorrt) and yolov5-lite(tensorrt).


## Extensional Frameworks

  - [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) <img src="https://img.shields.io/github/stars/Megvii-BaseDetection/YOLOX?style=social"/> : "YOLOX: Exceeding YOLO Series in 2021". (**[arXiv 2021](https://arxiv.org/abs/2107.08430)**)

  - [YOLOR](https://github.com/WongKinYiu/yolor) <img src="https://img.shields.io/github/stars/WongKinYiu/yolor?style=social"/> : "You Only Learn One Representation: Unified Network for Multiple Tasks". (**[arXiv 2021](https://arxiv.org/abs/2105.04206)**)

  - [YOLOF](https://github.com/megvii-model/YOLOF) <img src="https://img.shields.io/github/stars/megvii-model/YOLOF?style=social"/> : "You Only Look One-level Feature". (**[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_You_Only_Look_One-Level_Feature_CVPR_2021_paper.html)**)

  - [YOLOS](https://github.com/hustvl/YOLOS) <img src="https://img.shields.io/github/stars/hustvl/YOLOS?style=social"/> : "You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection". (**[NeurIPS 2021](https://proceedings.neurips.cc//paper/2021/hash/dc912a253d1e9ba40e2c597ed2376640-Abstract.html)**)

  - [YOLACT & YOLACT++](https://github.com/dbolya/yolact) <img src="https://img.shields.io/github/stars/dbolya/yolact?style=social"/> : You Only Look At CoefficienTs. (**[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.html), [IEEE TPAMI 2020](https://ieeexplore.ieee.org/abstract/document/9159935)**)

  - [jinfagang/yolov7](https://github.com/jinfagang/yolov7) <img src="https://img.shields.io/github/stars/jinfagang/yolov7?style=social"/> : 🔥🔥🔥🔥 YOLO with Transformers and Instance Segmentation, with TensorRT acceleration! 🔥🔥🔥

  - [Alpha-IoU](https://github.com/Jacobi93/Alpha-IoU) <img src="https://img.shields.io/github/stars/Jacobi93/Alpha-IoU?style=social"/> : "Alpha-IoU: A Family of Power Intersection over Union Losses for Bounding Box Regression". (**[NeurIPS 2021](https://proceedings.neurips.cc//paper/2021/hash/a8f15eda80c50adb0e71943adc8015cf-Abstract.html)**)

  - [CIoU](https://github.com/Zzh-tju/CIoU) <img src="https://img.shields.io/github/stars/Zzh-tju/CIoU?style=social"/> : Complete-IoU (CIoU) Loss and Cluster-NMS for Object Detection and Instance Segmentation (YOLACT). (**[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6999), [IEEE TCYB 2021](https://ieeexplore.ieee.org/abstract/document/9523600)**)

## Applications

  - ### Lighter and Faster

    - #### Lightweight Backbones
      #### 轻量级骨干网络

      - [dog-qiuqiu/MobileNet-Yolo](https://github.com/dog-qiuqiu/MobileNet-Yolo) <img src="https://img.shields.io/github/stars/dog-qiuqiu/MobileNet-Yolo?style=social"/> : MobileNetV2-YoloV3-Nano: 0.5BFlops 3MB HUAWEI P40: 6ms/img, YoloFace-500k:0.1Bflops 420KB🔥🔥🔥.

      - [eric612/MobileNet-YOLO](https://github.com/eric612/MobileNet-YOLO) <img src="https://img.shields.io/github/stars/eric612/MobileNet-YOLO?style=social"/> : A caffe implementation of MobileNet-YOLO detection network.

      - [eric612/Mobilenet-YOLO-Pytorch](https://github.com/eric612/Mobilenet-YOLO-Pytorch) <img src="https://img.shields.io/github/stars/eric612/Mobilenet-YOLO-Pytorch?style=social"/> : Include mobilenet series (v1,v2,v3...) and yolo series (yolov3,yolov4,...) .

      - [Adamdad/keras-YOLOv3-mobilenet](https://github.com/Adamdad/keras-YOLOv3-mobilenet) <img src="https://img.shields.io/github/stars/Adamdad/keras-YOLOv3-mobilenet?style=social"/> : A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

      - [fsx950223/mobilenetv2-yolov3](https://github.com/fsx950223/mobilenetv2-yolov3) <img src="https://img.shields.io/github/stars/fsx950223/mobilenetv2-yolov3?style=social"/> : yolov3 with mobilenetv2 and efficientnet.

      - [yl305237731/flexible-yolov5](https://github.com/yl305237731/flexible-yolov5) <img src="https://img.shields.io/github/stars/yl305237731/flexible-yolov5?style=social"/> : More readable and flexible yolov5 with more backbone(resnet, shufflenet, moblienet, efficientnet, hrnet, swin-transformer) and (cbam，dcn and so on), and tensorrt.

      - [liux0614/yolo_nano](https://github.com/liux0614/yolo_nano) <img src="https://img.shields.io/github/stars/liux0614/yolo_nano?style=social"/> : Unofficial implementation of yolo nano.

      - [lingtengqiu/Yolo_Nano](https://github.com/lingtengqiu/Yolo_Nano) <img src="https://img.shields.io/github/stars/lingtengqiu/Yolo_Nano?style=social"/> : Pytorch implementation of yolo_Nano for pedestrian detection.

      - [bubbliiiing/mobilenet-yolov4-pytorch](https://github.com/bubbliiiing/mobilenet-yolov4-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/mobilenet-yolov4-pytorch?style=social"/> : 这是一个mobilenet-yolov4的库，把yolov4主干网络修改成了mobilenet，修改了Panet的卷积组成，使参数量大幅度缩小。

      - [bubbliiiing/efficientnet-yolo3-pytorch](https://github.com/bubbliiiing/efficientnet-yolo3-pytorch) <img src="https://img.shields.io/github/stars/bubbliiiing/efficientnet-yolo3-pytorch?style=social"/> : 这是一个efficientnet-yolo3-pytorch的源码，将yolov3的主干特征提取网络修改成了efficientnet。


    - #### Pruning Distillation Quantization
      #### 剪枝 蒸馏 量化

      - [ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite) <img src="https://img.shields.io/github/stars/ppogg/YOLOv5-Lite?style=social"/> : YOLOv5-Lite：Lighter, faster and easier to deploy.

      - [Bigtuo/YOLOX-Lite](https://github.com/Bigtuo/YOLOX-Lite) <img src="https://img.shields.io/github/stars/Bigtuo/YOLOX-Lite?style=social"/> : 将YOLOv5-Lite代码中的head更换为YOLOX head。

      - [SlimYOLOv3](https://github.com/PengyiZhang/SlimYOLOv3) <img src="https://img.shields.io/github/stars/PengyiZhang/SlimYOLOv3?style=social"/> : "SlimYOLOv3: Narrower, Faster and Better for UAV Real-Time Applications". (**[arXiv 2019](https://arxiv.org/abs/1907.11093)**)

      - [YOLObile](https://github.com/nightsnack/YOLObile) <img src="https://img.shields.io/github/stars/nightsnack/YOLObile?style=social"/> : "YOLObile: Real-Time Object Detection on Mobile Devices via Compression-Compilation Co-Design". (**[AAAI 2021](https://www.aaai.org/AAAI21Papers/AAAI-7561.CaiY.pdf)**)

      - [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) <img src="https://img.shields.io/github/stars/dog-qiuqiu/Yolo-Fastest?style=social"/> : Yolo-Fastest：超超超快的开源ARM实时目标检测算法。 (**[Zenodo 2021](http://doi.org/10.5281/zenodo.5131532), [知乎 2020](https://zhuanlan.zhihu.com/p/234506503)**)

      - [dog-qiuqiu/Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2) <img src="https://img.shields.io/github/stars/dog-qiuqiu/Yolo-FastestV2?style=social"/> : Yolo-FastestV2:更快，更轻，移动端可达300FPS，参数量仅250k。  (**[知乎 2021](https://zhuanlan.zhihu.com/p/400474142)**)

      - [Lam1360/YOLOv3-model-pruning](https://github.com/Lam1360/YOLOv3-model-pruning) <img src="https://img.shields.io/github/stars/Lam1360/YOLOv3-model-pruning?style=social"/> : 在 oxford hand 数据集上对 YOLOv3 做模型剪枝（network slimming）。

      - [tanluren/yolov3-channel-and-layer-pruning](https://github.com/tanluren/yolov3-channel-and-layer-pruning) <img src="https://img.shields.io/github/stars/tanluren/yolov3-channel-and-layer-pruning?style=social"/> : yolov3 yolov4 channel and layer pruning, Knowledge Distillation 层剪枝，通道剪枝，知识蒸馏。

      - [coldlarry/YOLOv3-complete-pruning](https://github.com/coldlarry/YOLOv3-complete-pruning) <img src="https://img.shields.io/github/stars/coldlarry/YOLOv3-complete-pruning?style=social"/> : 提供对YOLOv3及Tiny的多种剪枝版本，以适应不同的需求。

      - [Syencil/mobile-yolov5-pruning-distillation](https://github.com/Syencil/mobile-yolov5-pruning-distillation) <img src="https://img.shields.io/github/stars/Syencil/mobile-yolov5-pruning-distillation?style=social"/> : mobilev2-yolov5s剪枝、蒸馏，支持ncnn，tensorRT部署。ultra-light but better performence！

      - [SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone](https://github.com/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone) <img src="https://img.shields.io/github/stars/SpursLipu/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone?style=social"/> : YOLO ModelCompression MultidatasetTraining.

      - [talebolano/yolov3-network-slimming](https://github.com/talebolano/yolov3-network-slimming) <img src="https://img.shields.io/github/stars/talebolano/yolov3-network-slimming?style=social"/> : yolov3 network slimming剪枝的一种实现。

      - [AlexeyAB/yolo2_light](https://github.com/AlexeyAB/yolo2_light) <img src="https://img.shields.io/github/stars/AlexeyAB/yolo2_light?style=social"/> : Light version of convolutional neural network Yolo v3 & v2 for objects detection with a minimum of dependencies (INT8-inference, BIT1-XNOR-inference).

      - [ZJU-lishuang/yolov5_prune](https://github.com/ZJU-lishuang/yolov5_prune) <img src="https://img.shields.io/github/stars/ZJU-lishuang/yolov5_prune?style=social"/> : yolov5 prune，Support V2, V3, V4 and V6 versions of yolov5.

      - [Gumpest/YOLOv5-Multibackbone-Compression](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression) <img src="https://img.shields.io/github/stars/Gumpest/YOLOv5-Multibackbone-Compression?style=social"/> : YOLOv5 Series Multi-backbone(TPH-YOLOv5, Ghostnet, ShuffleNetv2, Mobilenetv3Small, EfficientNetLite, PP-LCNet, SwinTransformer YOLO), Module(CBAM, DCN), Pruning (EagleEye, Network Slimming) and Quantization (MQBench) Compression Tool Box.

      - [midasklr/yolov5prune](https://github.com/midasklr/yolov5prune) <img src="https://img.shields.io/github/stars/midasklr/yolov5prune?style=social"/> : yolov5模型剪枝。

      - [j-marple-dev/AYolov2](https://github.com/j-marple-dev/AYolov2) <img src="https://img.shields.io/github/stars/j-marple-dev/AYolov2?style=social"/> : AYolov2.

      - [Sharpiless/Yolov5-distillation-train-inference](https://github.com/Sharpiless/Yolov5-distillation-train-inference) <img src="https://img.shields.io/github/stars/Sharpiless/Yolov5-distillation-train-inference?style=social"/> : Yolov5 distillation training | Yolov5知识蒸馏训练，支持训练自己的数据。


    - #### High-performance Inference Engine
      #### 高性能推理引擎

      - [ONNX Runtime](https://github.com/microsoft/onnxruntime) <img src="https://img.shields.io/github/stars/microsoft/onnxruntime?style=social"/> : ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator.

      - [TVM](https://github.com/apache/tvm) <img src="https://img.shields.io/github/stars/apache/tvm?style=social"/> : Open deep learning compiler stack for cpu, gpu and specialized accelerators.

      - [TensorRT](https://github.com/NVIDIA/TensorRT) <img src="https://img.shields.io/github/stars/NVIDIA/TensorRT?style=social"/> : TensorRT is a C++ library for high performance inference on NVIDIA GPUs and deep learning accelerators.

      - [ceccocats/tkDNN](https://github.com/ceccocats/tkDNN) <img src="https://img.shields.io/github/stars/ceccocats/tkDNN?style=social"/> : Deep neural network library and toolkit to do high performace inference on NVIDIA jetson platforms. "A Systematic Assessment of Embedded Neural Networks for Object Detection". (**[IEEE ETFA 2020](https://ieeexplore.ieee.org/document/9212130)**)

      - [OpenVINO](https://github.com/openvinotoolkit/openvino) <img src="https://img.shields.io/github/stars/openvinotoolkit/openvino?style=social"/> : This open source version includes several components: namely Model Optimizer, OpenVINO™ Runtime, Post-Training Optimization Tool, as well as CPU, GPU, MYRIAD, multi device and heterogeneous plugins to accelerate deep learning inferencing on Intel® CPUs and Intel® Processor Graphics.

      - [ncnn](https://github.com/Tencent/ncnn) <img src="https://img.shields.io/github/stars/Tencent/ncnn?style=social"/> : ncnn is a high-performance neural network inference framework optimized for the mobile platform.

      - [MNN](https://github.com/alibaba/MNN) <img src="https://img.shields.io/github/stars/alibaba/MNN?style=social"/> : MNN is a blazing fast, lightweight deep learning framework, battle-tested by business-critical use cases in Alibaba. (**[MLSys 2020](https://proceedings.mlsys.org/paper/2020/hash/8f14e45fceea167a5a36dedd4bea2543-Abstract.html)**)

      - [DefTruth/lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) <img src="https://img.shields.io/github/stars/DefTruth/lite.ai.toolkit?style=social"/> : 🛠 A lite C++ toolkit of awesome AI models with ONNXRuntime, NCNN, MNN and TNN. YOLOX, YOLOP, YOLOv5, YOLOR, NanoDet, YOLOX, SCRFD, YOLOX . MNN, NCNN, TNN, ONNXRuntime, CPU/GPU.  

      - [Paddle Lite](https://github.com/paddlepaddle/paddle-lite) <img src="https://img.shields.io/github/stars/paddlepaddle/paddle-lite?style=social"/> : Multi-platform high performance deep learning inference engine (飞桨多端多平台高性能深度学习推理引擎）。

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

      - [cmdbug/YOLOv5_NCNN](https://github.com/cmdbug/YOLOv5_NCNN) <img src="https://img.shields.io/github/stars/cmdbug/YOLOv5_NCNN?style=social"/> : 🍅 Deploy ncnn on mobile phones. Support Android and iOS. 移动端ncnn部署，支持Android与iOS。

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


    - #### FPGA TPU RISC-V MCU Hardware Deployment
      #### FPGA TPU RISC-V MCU 硬件部署

      - [Xilinx/Vitis-AI](https://github.com/Xilinx/Vitis-AI/tree/master/demo) <img src="https://img.shields.io/github/stars/Xilinx/Vitis-AI?style=social"/> : Vitis AI offers a unified set of high-level C++/Python programming APIs to run AI applications across edge-to-cloud platforms, including DPU for Alveo, and DPU for Zynq Ultrascale+ MPSoC and Zynq-7000. It brings the benefits to easily port AI applications from cloud to edge and vice versa. 10 samples in [VART Samples](https://github.com/Xilinx/Vitis-AI/tree/master/demo/VART) are available to help you get familiar with the unfied programming APIs. [Vitis-AI-Library](https://github.com/Xilinx/Vitis-AI/tree/master/demo/Vitis-AI-Library) provides an easy-to-use and unified interface by encapsulating many efficient and high-quality neural networks.

      - [HSqure/ultralytics-pt-yolov3-vitis-ai-edge](https://github.com/HSqure/ultralytics-pt-yolov3-vitis-ai-edge) <img src="https://img.shields.io/github/stars/HSqure/ultralytics-pt-yolov3-vitis-ai-edge?style=social"/> : This demo is only used for inference testing of Vitis AI v1.4 and quantitative compilation of DPU. It is compatible with the training results of [ultralytics/yolov3](https://github.com/ultralytics/yolov3) v9.5.0 (it needs to use the model saving method of Pytorch V1.4).

      - [mcedrdiego/Kria_yolov3_ppe](https://github.com/mcedrdiego/Kria_yolov3_ppe) <img src="https://img.shields.io/github/stars/mcedrdiego/Kria_yolov3_ppe?style=social"/> : Kria KV260 Real-Time Personal Protective Equipment Detection. "Deep Learning for Site Safety: Real-Time Detection of Personal Protective Equipment". (**[Automation in Construction 2020](https://www.sciencedirect.com/science/article/abs/pii/S0926580519308325)**)

      - [Pomiculture/YOLOv4-Vitis-AI](https://github.com/Pomiculture/YOLOv4-Vitis-AI) <img src="https://img.shields.io/github/stars/Pomiculture/YOLOv4-Vitis-AI?style=social"/> : Custom YOLOv4 for apple recognition (clean/damaged) on Alveo U280 accelerator card using Vitis AI framework. 

      - [mkshuvo2/ZCU104_YOLOv3_Post_Processing](https://github.com/mkshuvo2/ZCU104_YOLOv3_Post_Processing) <img src="https://img.shields.io/github/stars/mkshuvo2/ZCU104_YOLOv3_Post_Processing?style=social"/> : Tensor outputs form Vitis AI Runner Class for YOLOv3.
      
      - [puffdrum/v4tiny_pt_quant](https://github.com/puffdrum/v4tiny_pt_quant) <img src="https://img.shields.io/github/stars/puffdrum/v4tiny_pt_quant?style=social"/> : quantization for yolo with xilinx/vitis-ai-pytorch.     

      - [chanshann/LITE_YOLOV3_TINY_VITISAI](https://github.com/chanshann/LITE_YOLOV3_TINY_VITISAI) <img src="https://img.shields.io/github/stars/chanshann/LITE_YOLOV3_TINY_VITISAI?style=social"/> : LITE_YOLOV3_TINY_VITISAI. 

      - [dhm2013724/yolov2_xilinx_fpga](https://github.com/dhm2013724/yolov2_xilinx_fpga) <img src="https://img.shields.io/github/stars/dhm2013724/yolov2_xilinx_fpga?style=social"/> : YOLOv2 Accelerator in Xilinx's Zynq-7000 Soc(PYNQ-z2, Zedboard and ZCU102). (**[硕士论文 2019](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1019228234.nh&uid=WEEvREcwSlJHSldRa1FhdXNXaEhoOGhUTzA5T0tESzdFZ2pyR1NJR1ZBaz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!&v=MjE5NTN5dmdXN3JBVkYyNkY3RzZGdFBQcTVFYlBJUjhlWDFMdXhZUzdEaDFUM3FUcldNMUZyQ1VSTE9lWnVkdUY=), [电子技术应用 2019](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFDLAST2019&filename=DZJY201908009&uid=WEEvREcwSlJHSldRa1FhdXNXaEhoOGhUTzA5T0tESzdFZ2pyR1NJR1ZBaz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!&v=MDU0NDJDVVJMT2VadWR1Rnl2Z1c3ck1JVGZCZDdHNEg5ak1wNDlGYllSOGVYMUx1eFlTN0RoMVQzcVRyV00xRnI=), [计算机科学与探索 2019](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFDTEMP&filename=KXTS201910005&uid=WEEvREcwSlJHSldRa1FhdXNXaEhoOGhUTzA5T0tESzdFZ2pyR1NJR1ZBaz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!&v=MjkwNzdXTTFGckNVUkxPZVp1ZHVGeXZnVzdyT0xqWGZmYkc0SDlqTnI0OUZZWVI4ZVgxTHV4WVM3RGgxVDNxVHI=)**)

      - [19801201/SpinalHDL_CNN_Accelerator](https://github.com/19801201/SpinalHDL_CNN_Accelerator) <img src="https://img.shields.io/github/stars/19801201/SpinalHDL_CNN_Accelerator?style=social"/> : CNN accelerator implemented with Spinal HDL.

      - [Yu-Zhewen/Tiny_YOLO_v3_ZYNQ](https://github.com/Yu-Zhewen/Tiny_YOLO_v3_ZYNQ) <img src="https://img.shields.io/github/stars/Yu-Zhewen/Tiny_YOLO_v3_ZYNQ?style=social"/> : Implement Tiny YOLO v3 on ZYNQ.

      - [LukiBa/zybo_yolo](https://github.com/LukiBa/zybo_yolo) <img src="https://img.shields.io/github/stars/LukiBa/zybo_yolo?style=social"/> : YOLO example implementation using Intuitus CNN accelerator on ZYBO ZYNQ-7000 FPGA board.

      - [matsuda-slab/YOLO_ZYNQ_MASTER](https://github.com/matsuda-slab/YOLO_ZYNQ_MASTER) <img src="https://img.shields.io/github/stars/matsuda-slab/YOLO_ZYNQ_MASTER?style=social"/> : Implementation of YOLOv3-tiny on FPGA.   

      - [AramisOposich/tiny_YOLO_Zedboard](https://github.com/AramisOposich/tiny_YOLO_Zedboard) <img src="https://img.shields.io/github/stars/AramisOposich/tiny_YOLO_Zedboard?style=social"/> : tiny_YOLO_Zedboard.
 
      - [FerberZhang/Yolov2-FPGA-CNN-](https://github.com/FerberZhang/Yolov2-FPGA-CNN-) <img src="https://img.shields.io/github/stars/FerberZhang/Yolov2-FPGA-CNN-?style=social"/> : A demo for accelerating YOLOv2 in xilinx's fpga PYNQ.

      - [Prithvi-Velicheti/FPGA-Accelerator-for-TinyYolov3](https://github.com/Prithvi-Velicheti/FPGA-Accelerator-for-TinyYolov3) <img src="https://img.shields.io/github/stars/Prithvi-Velicheti/FPGA-Accelerator-for-TinyYolov3?style=social"/> : StreamYOLO: An FPGA-Accelerator-for-TinyYolov3.

      - [ChainZeeLi/FPGA_DPU](https://github.com/ChainZeeLi/FPGA_DPU) <img src="https://img.shields.io/github/stars/ChainZeeLi/FPGA_DPU?style=social"/> : This project is to implement YOLO v3 on Xilinx FPGA with DPU.

      - [xbdxwyh/yolov3_fpga_project](https://github.com/xbdxwyh/yolov3_fpga_project) <img src="https://img.shields.io/github/stars/xbdxwyh/yolov3_fpga_project?style=social"/> : yolov3_fpga_project.

      - [ZLkanyo009/Yolo-compression-and-deployment-in-FPGA](https://github.com/ZLkanyo009/Yolo-compression-and-deployment-in-FPGA) <img src="https://img.shields.io/github/stars/ZLkanyo009/Yolo-compression-and-deployment-in-FPGA?style=social"/> : 基于FPGA量化的人脸口罩检测。

      - [himewel/yolowell](https://github.com/himewel/yolowell) <img src="https://img.shields.io/github/stars/himewel/yolowell?style=social"/> : A set of hardware architectures to build a co-design of convolutional neural networks inference at FPGA devices.

      - [embedeep/Free-TPU](https://github.com/embedeep/Free-TPU) <img src="https://img.shields.io/github/stars/embedeep/Free-TPU?style=social"/> : Free TPU for FPGA with Lenet, MobileNet, Squeezenet, Resnet, Inception V3, YOLO V3, and ICNet. Deep learning acceleration using Xilinx zynq (Zedboard or ZC702 ) or kintex-7 to solve image classification, detection, and segmentation problem.

      - [zhen8838/K210_Yolo_framework](https://github.com/zhen8838/K210_Yolo_framework) <img src="https://img.shields.io/github/stars/zhen8838/K210_Yolo_framework?style=social"/> : Yolo v3 framework base on tensorflow, support multiple models, multiple datasets, any number of output layers, any number of anchors, model prune, and portable model to K210 !

      - [SEASKY-Master/SEASKY_K210](https://github.com/SEASKY-Master/SEASKY_K210) <img src="https://img.shields.io/github/stars/SEASKY-Master/SEASKY_K210?style=social"/> : K210 PCB YOLO.

      - [SEASKY-Master/Yolo-for-k210](https://github.com/SEASKY-Master/Yolo-for-k210) <img src="https://img.shields.io/github/stars/SEASKY-Master/Yolo-for-k210?style=social"/> : Yolo-for-k210.

      - [TonyZ1Min/yolo-for-k210](https://github.com/TonyZ1Min/yolo-for-k210) <img src="https://img.shields.io/github/stars/TonyZ1Min/yolo-for-k210?style=social"/> : keras-yolo-for-k210.

      - [guichristmann/edge-tpu-tiny-yolo](https://github.com/guichristmann/edge-tpu-tiny-yolo) <img src="https://img.shields.io/github/stars/guichristmann/edge-tpu-tiny-yolo?style=social"/> : Run Tiny YOLO-v3 on Google's Edge TPU USB Accelerator.

      - [Charlie839242/-Trash-Classification-Car](https://github.com/Charlie839242/-Trash-Classification-Car) <img src="https://img.shields.io/github/stars/Charlie839242/-Trash-Classification-Car?style=social"/> : 这是一个基于yolo-fastest模型的小车，主控是art-pi开发板，使用了rt thread操作系统。

      - [Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry](https://github.com/Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry) <img src="https://img.shields.io/github/stars/Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry?style=social"/> : This project deploys a yolo fastest model in the form of tflite on raspberry 3b+. 

      - [mahxn0/Hisi3559A_Yolov5](https://github.com/mahxn0/Hisi3559A_Yolov5) <img src="https://img.shields.io/github/stars/mahxn0/Hisi3559A_Yolov5?style=social"/> : 基于hisi3559a的yolov5训练部署全流程。



  - ### Object Tracking
    #### 目标跟踪

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


  - #### Reinforcement Learning
    #### 强化学习

    - [uzkent/EfficientObjectDetection](https://github.com/uzkent/EfficientObjectDetection) <img src="https://img.shields.io/github/stars/uzkent/EfficientObjectDetection?style=social"/> : "Efficient Object Detection in Large Images with Deep Reinforcement Learning". (**[WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/html/Uzkent_Efficient_Object_Detection_in_Large_Images_Using_Deep_Reinforcement_Learning_WACV_2020_paper.html)**)


  - #### Motion Control
    #### 运动控制

    - [icns-distributed-cloud/adaptive-cruise-control](https://github.com/icns-distributed-cloud/adaptive-cruise-control) <img src="https://img.shields.io/github/stars/icns-distributed-cloud/adaptive-cruise-control?style=social"/> : YOLO-v5 기반 "단안 카메라"의 영상을 활용해 차간 거리를 일정하게 유지하며 주행하는 Adaptive Cruise Control 기능 구현.

    - [LeBronLiHD/ZJU2021_MotionControl_PID_YOLOv5](https://github.com/LeBronLiHD/ZJU2021_MotionControl_PID_YOLOv5) <img src="https://img.shields.io/github/stars/LeBronLiHD/ZJU2021_MotionControl_PID_YOLOv5?style=social"/> : ZJU2021_MotionControl_PID_YOLOv5.

    - [SananSuleymanov/PID_YOLOv5s_ROS_Diver_Tracking](https://github.com/SananSuleymanov/PID_YOLOv5s_ROS_Diver_Tracking) <img src="https://img.shields.io/github/stars/SananSuleymanov/PID_YOLOv5s_ROS_Diver_Tracking?style=social"/> : PID_YOLOv5s_ROS_Diver_Tracking.


  - #### Spiking Neural Network
    #### SNN, 脉冲神经网络

    - [Spiking-YOLOv3](https://github.com/cwq159/PyTorch-Spiking-YOLOv3) <img src="https://img.shields.io/github/stars/cwq159/PyTorch-Spiking-YOLOv3?style=social"/> : A PyTorch implementation of Spiking-YOLOv3. Two branches are provided, based on two common PyTorch implementation of YOLOv3([ultralytics/yolov3](https://github.com/ultralytics/yolov3) & [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)), with support for Spiking-YOLOv3-Tiny at present. (**[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6787)**)

    - [fjcu-ee-islab/Spiking_Converted_YOLOv4](https://github.com/fjcu-ee-islab/Spiking_Converted_YOLOv4) <img src="https://img.shields.io/github/stars/fjcu-ee-islab/Spiking_Converted_YOLOv4?style=social"/> : Object Detection Based on Dynamic Vision Sensor with Spiking Neural Network.

    - [Zaabon/spiking_yolo](https://github.com/Zaabon/spiking_yolo) <img src="https://img.shields.io/github/stars/Zaabon/spiking_yolo?style=social"/> : This project is a combined neural network utilizing an spiking CNN with backpropagation and YOLOv3 for object detection.

    - [Dignity-ghost/PyTorch-Spiking-YOLOv3](https://github.com/Dignity-ghost/PyTorch-Spiking-YOLOv3) <img src="https://img.shields.io/github/stars/Dignity-ghost/PyTorch-Spiking-YOLOv3?style=social"/> : A modified repository based on [Spiking-YOLOv3](https://github.com/cwq159/PyTorch-Spiking-YOLOv3) and [YOLOv3](https://pjreddie.com/darknet/yolo), which makes it suitable for VOC-dataset and YOLOv2.


  - #### Attention and Transformer
    #### 注意力机制

    - [xmu-xiaoma666/External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch) <img src="https://img.shields.io/github/stars/xmu-xiaoma666/External-Attention-pytorch?style=social"/> : 🍀 Pytorch implementation of various Attention Mechanisms, MLP, Re-parameter, Convolution, which is helpful to further understand papers.⭐⭐⭐.

    - [HaloTrouvaille/YOLO-Multi-Backbones-Attention](https://github.com/HaloTrouvaille/YOLO-Multi-Backbones-Attention) <img src="https://img.shields.io/github/stars/HaloTrouvaille/YOLO-Multi-Backbones-Attention?style=social"/> : This Repository includes YOLOv3 with some lightweight backbones (ShuffleNetV2, GhostNet, VoVNet), some computer vision attention mechanism (SE Block, CBAM Block, ECA Block), pruning,quantization and distillation for GhostNet.

    - [kay-cottage/CoordAttention_YOLOX_Pytorch](https://github.com/kay-cottage/CoordAttention_YOLOX_Pytorch) <img src="https://img.shields.io/github/stars/kay-cottage/CoordAttention_YOLOX_Pytorch?style=social"/> : CoordAttention_YOLOX(基于CoordAttention坐标注意力机制的改进版YOLOX目标检测平台）。 "Coordinate Attention for Efficient Mobile Network Design". (**[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.html), [ Andrew-Qibin/CoordAttention](https://github.com/Andrew-Qibin/CoordAttention)**)

    - [liangzhendong123/Attention-yolov5](https://github.com/liangzhendong123/Attention-yolov5) <img src="https://img.shields.io/github/stars/liangzhendong123/Attention-yolov5?style=social"/> : 基于注意力机制改进的yolov5模型。

    - [e96031413/AA-YOLO](https://github.com/e96031413/AA-YOLO) <img src="https://img.shields.io/github/stars/e96031413/AA-YOLO?style=social"/> : Attention ALL-CNN Twin Head YOLO (AA -YOLO). "Improving Tiny YOLO with Fewer Model Parameters". (**[IEEE BigMM 2021](https://ieeexplore.ieee.org/abstract/document/9643269/)**)

    - [anonymoussss/YOLOX-SwinTransformer](https://github.com/anonymoussss/YOLOX-SwinTransformer) <img src="https://img.shields.io/github/stars/anonymoussss/YOLOX-SwinTransformer?style=social"/> : YOLOX with Swin-Transformer backbone.


  - ### Small Object Detection
    #### 小目标检测

    - [TPH-YOLOv5](https://github.com/cv516Buaa/tph-yolov5) <img src="https://img.shields.io/github/stars/cv516Buaa/tph-yolov5?style=social"/> : "TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-Captured Scenarios". (**[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/html/Zhu_TPH-YOLOv5_Improved_YOLOv5_Based_on_Transformer_Prediction_Head_for_Object_ICCVW_2021_paper.html)**)

    - [SAHI](https://github.com/obss/sahi) <img src="https://img.shields.io/github/stars/obss/sahi?style=social"/> : A lightweight vision library for performing large scale object detection/ instance segmentation. "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection". (**[arXiv 2022](https://arxiv.org/abs/2202.06934v2), [Zenodo 2021](https://doi.org/10.5281/zenodo.5718950)**). "计算机视觉研究院：《[小目标Trick | Detectron2、MMDetection、YOLOv5都通用的小目标检测解决方案](https://mp.weixin.qq.com/s/MKtvEg0DQgAw3LAvfn3FdA)》"

    - [YOLT](https://github.com/avanetten/yolt) <img src="https://img.shields.io/github/stars/avanetten/yolt?style=social"/> : "You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery". (**[arXiv 2018](https://arxiv.org/abs/1805.09512)**). "江大白：《[基于大尺寸图像的小目标检测竞赛经验总结](https://mp.weixin.qq.com/s/qbbd5FdyKKk7UI3mmGBt4Q)》"

    - [SIMRDWN](https://github.com/avanetten/simrdwn) <img src="https://img.shields.io/github/stars/avanetten/simrdwn?style=social"/> : "Satellite Imagery Multiscale Rapid Detection with Windowed Networks". (**[arXiv 2018](https://arxiv.org/abs/1809.09978), [WACV 2019](https://ieeexplore.ieee.org/abstract/document/8659155)**)

    - [YOLTv5](https://github.com/avanetten/yoltv5) <img src="https://img.shields.io/github/stars/avanetten/yoltv5?style=social"/> : YOLTv5 builds upon [YOLT](https://github.com/avanetten/yolt) and [SIMRDWN](https://github.com/avanetten/simrdwn), and updates these frameworks to use the [ultralytics/yolov5](https://github.com/ultralytics/yolov5) version of the YOLO object detection family.

    - [mwaseema/Drone-Detection](https://github.com/mwaseema/Drone-Detection) <img src="https://img.shields.io/github/stars/mwaseema/Drone-Detection?style=social"/> : "Dogfight: Detecting Drones from Drones Videos". (**[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Ashraf_Dogfight_Detecting_Drones_From_Drones_Videos_CVPR_2021_paper.html)**)

    - [KevinMuyaoGuo/yolov5s_for_satellite_imagery](https://github.com/KevinMuyaoGuo/yolov5s_for_satellite_imagery) <img src="https://img.shields.io/github/stars/KevinMuyaoGuo/yolov5s_for_satellite_imagery?style=social"/> : 基于YOLOv5的卫星图像目标检测demo | A demo for satellite imagery object detection based on YOLOv5。

    - [Hongyu-Yue/yoloV5_modify_smalltarget](https://github.com/Hongyu-Yue/yoloV5_modify_smalltarget) <img src="https://img.shields.io/github/stars/Hongyu-Yue/yoloV5_modify_smalltarget?style=social"/> : YOLOV5 小目标检测修改版。

    - [muyuuuu/Self-Supervise-Object-Detection](https://github.com/muyuuuu/Self-Supervise-Object-Detection) <img src="https://img.shields.io/github/stars/muyuuuu/Self-Supervise-Object-Detection?style=social"/> : Self-Supervised Object Detection. 水面漂浮垃圾目标检测，分析源码改善 yolox 检测小目标的缺陷，提出自监督算法预训练无标签数据，提升检测性能。

    - [swricci/small-boat-detector](https://github.com/swricci/small-boat-detector) <img src="https://img.shields.io/github/stars/swricci/small-boat-detector?style=social"/> : Trained yolo v3 model weights and configuration file to detect small boats in satellite imagery.

    - [Resham-Sundar/sahi-yolox](https://github.com/Resham-Sundar/sahi-yolox) <img src="https://img.shields.io/github/stars/Resham-Sundar/sahi-yolox?style=social"/> : YoloX with SAHI Implementation.


  - ### Oriented Object Detection
    #### 旋转目标检测

    - [AlphaRotate](https://github.com/yangxue0827/RotationDetection) <img src="https://img.shields.io/github/stars/yangxue0827/RotationDetection?style=social"/> : "AlphaRotate: A Rotation Detection Benchmark using TensorFlow". (**[arXiv 2021](https://arxiv.org/abs/2111.06677)**)

    - [hukaixuan19970627/yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb) <img src="https://img.shields.io/github/stars/hukaixuan19970627/yolov5_obb?style=social"/> : yolov5 + csl_label.(Oriented Object Detection)（Rotation Detection）（Rotated BBox）基于yolov5的旋转目标检测。

    - [BossZard/rotation-yolov5](https://github.com/BossZard/rotation-yolov5) <img src="https://img.shields.io/github/stars/BossZard/rotation-yolov5?style=social"/> : rotation detection based on yolov5.

    - [ming71/rotate-yolov3](https://github.com/ming71/rotate-yolov3) <img src="https://img.shields.io/github/stars/ming71/rotate-yolov3?style=social"/> : Arbitrary oriented object detection implemented with yolov3 (attached with some tricks).

    - [ming71/yolov3-polygon](https://github.com/ming71/yolov3-polygon) <img src="https://img.shields.io/github/stars/ming71/yolov3-polygon?style=social"/> : Arbitrary-oriented object detection based on yolov3.

    - [kunnnnethan/R-YOLOv4](https://github.com/kunnnnethan/R-YOLOv4) <img src="https://img.shields.io/github/stars/kunnnnethan/R-YOLOv4?style=social"/> : This is a PyTorch-based R-YOLOv4 implementation which combines YOLOv4 model and loss function from R3Det for arbitrary oriented object detection.

    - [XinzeLee/PolygonObjectDetection](https://github.com/XinzeLee/PolygonObjectDetection) <img src="https://img.shields.io/github/stars/XinzeLee/PolygonObjectDetection?style=social"/> : This repository is based on Ultralytics/yolov5, with adjustments to enable polygon prediction boxes.

  - ### Face Detection
    #### 人脸检测

    - [OAID/TengineKit](https://github.com/OAID/TengineKit) <img src="https://img.shields.io/github/stars/OAID/TengineKit?style=social"/> : TengineKit - Free, Fast, Easy, Real-Time Face Detection & Face Landmarks & Face Attributes & Hand Detection & Hand Landmarks & Body Detection & Body Landmarks & Iris Landmarks & Yolov5 SDK On Mobile. 

    - [YOLO5Face](https://github.com/deepcam-cn/yolov5-face) <img src="https://img.shields.io/github/stars/deepcam-cn/yolov5-face?style=social"/> : "YOLO5Face: Why Reinventing a Face Detector". (**[arXiv 2021](https://arxiv.org/abs/2105.12931)**)

    - [sthanhng/yoloface](https://github.com/sthanhng/yoloface) <img src="https://img.shields.io/github/stars/sthanhng/yoloface?style=social"/> : Deep learning-based Face detection using the YOLOv3 algorithm. 

    - [DayBreak-u/yolo-face-with-landmark](https://github.com/DayBreak-u/yolo-face-with-landmark) <img src="https://img.shields.io/github/stars/DayBreak-u/yolo-face-with-landmark?style=social"/> : yoloface大礼包 使用pytroch实现的基于yolov3的轻量级人脸检测（包含关键点）。

    - [abars/YoloKerasFaceDetection](https://github.com/abars/YoloKerasFaceDetection) <img src="https://img.shields.io/github/stars/abars/YoloKerasFaceDetection?style=social"/> : Face Detection and Gender and Age Classification using Keras.

    - [dannyblueliu/YOLO-Face-detection](https://github.com/dannyblueliu/YOLO-Face-detection) <img src="https://img.shields.io/github/stars/dannyblueliu/YOLO-Face-detection?style=social"/> : Face detection based on YOLO darknet.

    - [wmylxmj/YOLO-V3-IOU](https://github.com/wmylxmj/YOLO-V3-IOU) <img src="https://img.shields.io/github/stars/wmylxmj/YOLO-V3-IOU?style=social"/> : YOLO3 动漫人脸检测 (Based on keras and tensorflow) 2019-1-19.

    - [xialuxi/yolov5_face_landmark](https://github.com/xialuxi/yolov5_face_landmark) <img src="https://img.shields.io/github/stars/xialuxi/yolov5_face_landmark?style=social"/> : 基于yolov5的人脸检测，带关键点检测。

    - [pranoyr/head-detection-using-yolo](https://github.com/pranoyr/head-detection-using-yolo) <img src="https://img.shields.io/github/stars/pranoyr/head-detection-using-yolo?style=social"/> : Detection of head using YOLO.

    - [grapeot/AnimeHeadDetector](https://github.com/grapeot/AnimeHeadDetector) <img src="https://img.shields.io/github/stars/grapeot/AnimeHeadDetector?style=social"/> : An object detector for character heads in animes, based on Yolo V3.

    - [hpc203/10kinds-light-face-detector-align-recognition](https://github.com/hpc203/10kinds-light-face-detector-align-recognition) <img src="https://img.shields.io/github/stars/hpc203/10kinds-light-face-detector-align-recognition?style=social"/> : 10种轻量级人脸检测算法的比拼。

    - [Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking](https://github.com/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking) <img src="https://img.shields.io/github/stars/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking?style=social"/> : This is a robot project for television live. System will tracking the host's face, making the face in the middle of the screen.

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

  - ### Social Distance Detection
    #### 社交距离检测

    - [Ank-Cha/Social-Distancing-Analyser-COVID-19](https://github.com/Ank-Cha/Social-Distancing-Analyser-COVID-19) <img src="https://img.shields.io/github/stars/Ank-Cha/Social-Distancing-Analyser-COVID-19?style=social"/> : Social Distancing Analyser to prevent COVID19. 

    - [abd-shoumik/Social-distance-detection](https://github.com/abd-shoumik/Social-distance-detection) <img src="https://img.shields.io/github/stars/abd-shoumik/Social-distance-detection?style=social"/> : Social distance detection, a deep learning computer vision project with yolo object detection.

    - [ChargedMonk/Social-Distancing-using-YOLOv5](https://github.com/ChargedMonk/Social-Distancing-using-YOLOv5) <img src="https://img.shields.io/github/stars/ChargedMonk/Social-Distancing-using-YOLOv5?style=social"/> : Classifying people as high risk and low risk based on their distance to other people.

    - [JohnBetaCode/Social-Distancing-Analyser](https://github.com/JohnBetaCode/Social-Distancing-Analyser) <img src="https://img.shields.io/github/stars/JohnBetaCode/Social-Distancing-Analyser?style=social"/> : Social Distancing Analyzer.

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

    - ####  License Plate Detection
      #####  车牌检测

      - [zeusees/License-Plate-Detector](https://github.com/zeusees/License-Plate-Detector) <img src="https://img.shields.io/github/stars/zeusees/License-Plate-Detector?style=social"/> : License Plate Detection with Yolov5，基于Yolov5车牌检测。

      - [TheophileBuy/LicensePlateRecognition](https://github.com/TheophileBuy/LicensePlateRecognition) <img src="https://img.shields.io/github/stars/TheophileBuy/LicensePlateRecognition?style=social"/> : License Plate Recognition.

      - [alitourani/yolo-license-plate-detection](https://github.com/alitourani/yolo-license-plate-detection) <img src="https://img.shields.io/github/stars/alitourani/yolo-license-plate-detection?style=social"/> : A License-Plate detecttion application based on YOLO.

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

    - ####  Crosswalk Detection
      #####  人行横道/斑马线检测

      - [CDNet](https://github.com/zhangzhengde0225/CDNet) <img src="https://img.shields.io/github/stars/zhangzhengde0225/CDNet?style=social"/> : "CDNet: a real-time and robust crosswalk detection network on Jetson nano based on YOLOv5". (**[Neural Computing and Applications 2022](https://link.springer.com/article/10.1007/s00521-022-07007-9)**). "CVer：《[上海交大提出CDNet：基于改进YOLOv5的斑马线和汽车过线行为检测](https://mp.weixin.qq.com/s/2F3WBtfN_7DkhERMOH8-QA)》"

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

    - [insigh1/Interactive_ABCs_with_American_Sign_Language_using_Yolov5](https://github.com/insigh1/Interactive_ABCs_with_American_Sign_Language_using_Yolov5) Interactive ABC's with American Sign Language.

  - ### Action Detection
    #### 动作检测

    - [wufan-tb/yolo_slowfast](https://github.com/wufan-tb/yolo_slowfast) <img src="https://img.shields.io/github/stars/wufan-tb/yolo_slowfast?style=social"/> : A realtime action detection frame work based on PytorchVideo.

  - ### Human Pose Estimation
    #### 人体姿态估计

    - [wmcnally/kapao](https://github.com/wmcnally/kapao) <img src="https://img.shields.io/github/stars/wmcnally/kapao?style=social"/> : KAPAO is a state-of-the-art single-stage human pose estimation model that detects keypoints and poses as objects and fuses the detections to predict human poses. "Rethinking Keypoint Representations: Modeling Keypoints and Poses as Objects for Multi-Person Human Pose Estimation". (**[arXiv 2021](https://arxiv.org/abs/2111.08557)**)

    - [jinfagang/VIBE_yolov5](https://github.com/jinfagang/VIBE_yolov5) <img src="https://img.shields.io/github/stars/jinfagang/VIBE_yolov5?style=social"/> : Using YOLOv5 as detection on VIBE. "VIBE: Video Inference for Human Body Pose and Shape Estimation". (**[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.html)**)


  - ### 3D Object Detection
    #### 三维目标检测

    - [maudzung/YOLO3D-YOLOv4-PyTorch](https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch) <img src="https://img.shields.io/github/stars/maudzung/YOLO3D-YOLOv4-PyTorch?style=social"/> : The PyTorch Implementation based on YOLOv4 of the paper: YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud. (**[ECCV 2018](https://openaccess.thecvf.com/content_eccv_2018_workshops/w18/html/Ali_YOLO3D_End-to-end_real-time_3D_Oriented_Object_Bounding_Box_Detection_from_ECCVW_2018_paper.html)**)

    - [maudzung/Complex-YOLOv4-Pytorch](https://github.com/maudzung/Complex-YOLOv4-Pytorch) <img src="https://img.shields.io/github/stars/maudzung/Complex-YOLOv4-Pytorch?style=social"/> : The PyTorch Implementation based on YOLOv4 of the paper: "Complex-YOLO: Real-time 3D Object Detection on Point Clouds". (**[arXiv 2018](https://arxiv.org/abs/1803.06199)**)

    - [AI-liu/Complex-YOLO](https://github.com/AI-liu/Complex-YOLO) <img src="https://img.shields.io/github/stars/AI-liu/Complex-YOLO?style=social"/> : This is an unofficial implementation of Complex-YOLO: Real-time 3D Object Detection on Point Clouds in pytorch.

    - [ghimiredhikura/Complex-YOLOv3](https://github.com/ghimiredhikura/Complex-YOLOv3) <img src="https://img.shields.io/github/stars/ghimiredhikura/Complex-YOLOv3?style=social"/> : Complete but Unofficial PyTorch Implementation of Complex-YOLO: Real-time 3D Object Detection on Point Clouds with YoloV3.

    - [Yuanchu/YOLO3D](https://github.com/Yuanchu/YOLO3D) <img src="https://img.shields.io/github/stars/Yuanchu/YOLO3D?style=social"/> : Implementation of a basic YOLO model for object detection in 3D.

    - [ruhyadi/YOLO3D](https://github.com/ruhyadi/YOLO3D) <img src="https://img.shields.io/github/stars/ruhyadi/YOLO3D?style=social"/> : YOLO 3D Object Detection for Autonomous Driving Vehicle.

  - ### Safety Monitoring Field Detection
    #### 安防监控领域检测

    - [gengyanlei/fire-smoke-detect-yolov4](https://github.com/gengyanlei/fire-smoke-detect-yolov4) <img src="https://img.shields.io/github/stars/gengyanlei/fire-smoke-detect-yolov4?style=social"/> : fire-smoke-detect-yolov4-yolov5 and fire-smoke-detection-dataset 火灾检测，烟雾检测。

    - [CVUsers/Smoke-Detect-by-YoloV5](https://github.com/CVUsers/Smoke-Detect-by-YoloV5) <img src="https://img.shields.io/github/stars/CVUsers/Smoke-Detect-by-YoloV5?style=social"/> : Yolov5 real time smoke detection system.

    - [CVUsers/Fire-Detect-by-YoloV5](https://github.com/CVUsers/Fire-Detect-by-YoloV5) <img src="https://img.shields.io/github/stars/CVUsers/Fire-Detect-by-YoloV5?style=social"/> : 火灾检测，浓烟检测，吸烟检测。

    - [spacewalk01/Yolov5-Fire-Detection](https://github.com/spacewalk01/Yolov5-Fire-Detection) <img src="https://img.shields.io/github/stars/spacewalk01/Yolov5-Fire-Detection?style=social"/> : Train yolov5 to detect fire in an image or video.

    - [roflcoopter/viseron](https://github.com/roflcoopter/viseron) <img src="https://img.shields.io/github/stars/roflcoopter/viseron?style=social"/> : Viseron - Self-hosted NVR with object detection.

    - [dcmartin/motion-ai](https://github.com/dcmartin/motion-ai) <img src="https://img.shields.io/github/stars/dcmartin/motion-ai?style=social"/> : AI assisted motion detection for Home Assistant.

    - [Nico31415/Drowning-Detector](https://github.com/Nico31415/Drowning-Detector) <img src="https://img.shields.io/github/stars/Nico31415/Drowning-Detector?style=social"/> : Using YOLO object detection, this program will detect if a person is drowning.

  - ### Industrial Defect Detection
    #### 工业缺陷检测

    - [annsonic/Steel_defect](https://github.com/annsonic/Steel_defect) <img src="https://img.shields.io/github/stars/annsonic/Steel_defect?style=social"/> : Exercise: Use YOLO to detect hot-rolled steel strip surface defects (NEU-DET dataset).

  - ### Medical Field Detection
    #### 医学领域检测
    - [DataXujing/YOLO-v5](https://github.com/DataXujing/YOLO-v5) <img src="https://img.shields.io/github/stars/DataXujing/YOLO-v5?style=social"/> : YOLO v5在医疗领域中消化内镜目标检测的应用。

    - [Jafar-Abdollahi/Automated-detection-of-COVID-19-cases-using-deep-neural-networks-with-CTS-images](https://github.com/Jafar-Abdollahi/Automated-detection-of-COVID-19-cases-using-deep-neural-networks-with-CTS-images) <img src="https://img.shields.io/github/stars/Jafar-Abdollahi/Automated-detection-of-COVID-19-cases-using-deep-neural-networks-with-CTS-images?style=social"/> : In this project, a new model for automatic detection of covid-19 using raw chest X-ray images is presented. 

    - [fahriwps/breast-cancer-detection](https://github.com/fahriwps/breast-cancer-detection) <img src="https://img.shields.io/github/stars/fahriwps/breast-cancer-detection?style=social"/> : Breast cancer mass detection using YOLO object detection algorithm and GUI.

    - [niehusst/YOLO-Cancer-Detection](https://github.com/niehusst/YOLO-Cancer-Detection) <img src="https://img.shields.io/github/stars/niehusst/YOLO-Cancer-Detection?style=social"/> : An implementation of the YOLO algorithm trained to spot tumors in DICOM images.

    - [safakgunes/Blood-Cancer-Detection-YOLOV5](https://github.com/safakgunes/Blood-Cancer-Detection-YOLOV5) <img src="https://img.shields.io/github/stars/safakgunes/Blood-Cancer-Detection-YOLOV5?style=social"/> : Blood Cancer Detection with YOLOV5.

    - [shchiang0708/YOLOv2_skinCancer](https://github.com/shchiang0708/YOLOv2_skinCancer) <img src="https://img.shields.io/github/stars/shchiang0708/YOLOv2_skinCancer?style=social"/> : YOLOv2_skinCancer.

    - [avral1810/parkinsongait](https://github.com/avral1810/parkinsongait) <img src="https://img.shields.io/github/stars/avral1810/parkinsongait?style=social"/> : Parkinson’s Disease.


  - ### Adverse Weather Conditions
    #### 恶劣天气环境

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


  - ### Semantic Segmentation
    #### 语义分割

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

  - ### Automatic Annotation Tool
    #### 自动标注工具

    - [cnyvfang/labelGo-Yolov5AutoLabelImg](https://github.com/cnyvfang/labelGo-Yolov5AutoLabelImg) <img src="https://img.shields.io/github/stars/cnyvfang/labelGo-Yolov5AutoLabelImg?style=social"/> : A graphical Semi-automatic annotation tool based on labelImg and YOLOv5，一个基于labelImg及YOLOV5的图形化半自动标注工具。

    - [CVUsers/Auto_maker](https://github.com/CVUsers/Auto_maker) <img src="https://img.shields.io/github/stars/CVUsers/Auto_maker?style=social"/> : 深度学习数据自动标注器开源 目标检测和图像分类（高精度高效率）。

    - [AlexeyAB/Yolo_mark](https://github.com/AlexeyAB/Yolo_mark) <img src="https://img.shields.io/github/stars/AlexeyAB/Yolo_mark?style=social"/> : GUI for marking bounded boxes of objects in images for training neural network Yolo v3 and v2.

    - [MrZander/YoloMarkNet](https://github.com/MrZander/YoloMarkNet) <img src="https://img.shields.io/github/stars/MrZander/YoloMarkNet?style=social"/> : Darknet YOLOv2/3 annotation tool written in C#/WPF.

    - [mahxn0/Yolov3_ForTextLabel](https://github.com/mahxn0/Yolov3_ForTextLabel) <img src="https://img.shields.io/github/stars/mahxn0/Yolov3_ForTextLabel?style=social"/> : 基于yolov3的目标/自然场景文字自动标注工具.


  - ### GUI
    #### 图形界面

    - [Javacr/PyQt5-YOLOv5](https://github.com/Javacr/PyQt5-YOLOv5) <img src="https://img.shields.io/github/stars/Javacr/PyQt5-YOLOv5?style=social"/> : YOLOv5检测界面-PyQt5实现。

    - [scutlrr/Yolov4-QtGUI](https://github.com/scutlrr/Yolov4-QtGUI) <img src="https://img.shields.io/github/stars/scutlrr/Yolov4-QtGUI?style=social"/> : Yolov4-QtGUI是基于[QtGuiDemo](https://github.com/jmu201521121021/QtGuiDemo)项目开发的可视化目标检测界面，可以简便选择本地图片、摄像头来展示图像处理算法的结果。

    - [xugaoxiang/yolov5-pyqt5](https://github.com/xugaoxiang/yolov5-pyqt5) <img src="https://img.shields.io/github/stars/xugaoxiang/yolov5-pyqt5?style=social"/> : 给yolov5加个gui界面，使用pyqt5，yolov5是5.0版本。

    - [mxy493/YOLOv5-Qt](https://github.com/mxy493/YOLOv5-Qt) <img src="https://img.shields.io/github/stars/mxy493/YOLOv5-Qt?style=social"/> : 基于YOLOv5的GUI程序，支持选择要使用的权重文件，设置是否使用GPU，设置置信度阈值等参数。

    - [BonesCat/YoloV5_PyQt5](https://github.com/BonesCat/YoloV5_PyQt5) <img src="https://img.shields.io/github/stars/BonesCat/YoloV5_PyQt5?style=social"/> : Add gui for YoloV5 using PyQt5.

    - [PySimpleGUI/PySimpleGUI-YOLO](https://github.com/PySimpleGUI/PySimpleGUI-YOLO) <img src="https://img.shields.io/github/stars/PySimpleGUI/PySimpleGUI-YOLO?style=social"/> : A YOLO Artificial Intelligence algorithm demonstration using PySimpleGUI.

    - [prabindh/qt5-opencv3-darknet](https://github.com/prabindh/qt5-opencv3-darknet) <img src="https://img.shields.io/github/stars/prabindh/qt5-opencv3-darknet?style=social"/> : Qt5 + Darknet/Yolo + OpenCV3.

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

  - ### Other Applications
    #### 其它应用

    - [penny4860/Yolo-digit-detector](https://github.com/penny4860/Yolo-digit-detector) <img src="https://img.shields.io/github/stars/penny4860/Yolo-digit-detector?style=social"/> : Implemented digit detector in natural scene using resnet50 and Yolo-v2. I used SVHN as the training set, and implemented it using tensorflow and keras.

    - [chineseocr/table-detect](https://github.com/chineseocr/table-detect) <img src="https://img.shields.io/github/stars/chineseocr/table-detect?style=social"/> : table detect(yolo) , table line(unet) （表格检测/表格单元格定位）。

    - [davidfrz/yolov5_distance_count](https://github.com/davidfrz/yolov5_distance_count) <img src="https://img.shields.io/github/stars/davidfrz/yolov5_distance_count?style=social"/> : 通过yolov5实现目标检测+双目摄像头实现距离测量。

    - [wenyishengkingkong/realsense-D455-YOLOV5](https://github.com/wenyishengkingkong/realsense-D455-YOLOV5) <img src="https://img.shields.io/github/stars/wenyishengkingkong/realsense-D455-YOLOV5?style=social"/> : 利用realsense深度相机实现yolov5目标检测的同时测出距离。

    - [thisiszhou/SexyYolo](https://github.com/thisiszhou/SexyYolo) <img src="https://img.shields.io/github/stars/thisiszhou/SexyYolo?style=social"/> : An implementation of Yolov3 with Tensorflow1.x, which could detect COCO and sexy or porn person simultaneously.

    - [javirk/Person_remover](https://github.com/javirk/Person_remover) <img src="https://img.shields.io/github/stars/javirk/Person_remover?style=social"/> : People removal in images using Pix2Pix and YOLO. 

    - [bijustin/YOLO-DynaSLAM](https://github.com/bijustin/YOLO-DynaSLAM) <img src="https://img.shields.io/github/stars/bijustin/YOLO-DynaSLAM?style=social"/> : YOLO Dynamic ORB_SLAM is a visual SLAM system that is robust in dynamic scenarios for RGB-D configuration. 

    - [BzdTaisa/YoloPlanarSLAM](https://github.com/BzdTaisa/YoloPlanarSLAM) <img src="https://img.shields.io/github/stars/BzdTaisa/YoloPlanarSLAM?style=social"/> : YOLO-Planar-SLAM. 

    - [foschmitz/yolo-python-rtsp](https://github.com/foschmitz/yolo-python-rtsp) <img src="https://img.shields.io/github/stars/foschmitz/yolo-python-rtsp?style=social"/> : Object detection using deep learning with Yolo, OpenCV and Python via Real Time Streaming Protocol (RTSP).

    - [ismail-mebsout/Parsing-PDFs-using-YOLOV3](https://github.com/ismail-mebsout/Parsing-PDFs-using-YOLOV3) <img src="https://img.shields.io/github/stars/ismail-mebsout/Parsing-PDFs-using-YOLOV3?style=social"/> : Parsing pdf tables using YOLOV3.

    - [008karan/PAN_OCR](https://github.com/008karan/PAN_OCR) <img src="https://img.shields.io/github/stars/008karan/PAN_OCR?style=social"/> : Building OCR using YOLO and Tesseract.

    - [stephanecharette/DarkMark](https://github.com/stephanecharette/DarkMark) <img src="https://img.shields.io/github/stars/stephanecharette/DarkMark?style=social"/> : Marking up images for use with Darknet.

    - [zeyad-mansour/lunar](https://github.com/zeyad-mansour/lunar) <img src="https://img.shields.io/github/stars/zeyad-mansour/lunar?style=social"/> : Lunar is a neural network aimbot that uses real-time object detection accelerated with CUDA on Nvidia GPUs.

    - [lannguyen0910/food-detection-yolov5](https://github.com/lannguyen0910/food-detection-yolov5) <img src="https://img.shields.io/github/stars/lannguyen0910/food-detection-yolov5?style=social"/> : YOLOv5 meal analysis.

