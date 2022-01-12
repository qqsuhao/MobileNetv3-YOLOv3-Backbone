# MobileNetv3-YOLOv3

该项目只基于MobileNetv3-small模型实现了YOLOv3类型的骨干网络。

希望能对相关研究者有帮助。



代码中在主函数中给出了模型基本使用。

模型只包含骨干网络前向推理，并返回两个YOLO层的输出，

空间尺寸分别是输入图像的1/32和1/16，

通道数分别为128和96。



主函数的示例中支持多张gpu加载模型，并且支持使用MobileNetv3的权重文件进行骨干网络初始化。



MobileNetv3的代码以及权重文件参考自

[[chufei1995/mobilenetv3.pytorch: 73.2% MobileNetV3-Large and 67.1% MobileNetV3-Small model on ImageNet (github.com)](https://github.com/chufei1995/mobilenetv3.pytorch)](https://github.com/chufei1995/mobilenetv3.pytorch)

[[d-li14/mobilenetv3.pytorch: 74.3% MobileNetV3-Large and 67.2% MobileNetV3-Small model on ImageNet (github.com)](https://github.com/d-li14/mobilenetv3.pytorch)](https://github.com/d-li14/mobilenetv3.pytorch)

