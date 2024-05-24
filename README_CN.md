# Emotion-recogniton-pytorch
该项目聚焦于利用深度学习技术从面部表情中识别人的情感。我们使用 PyTorch 实现了这一模型，并在情感识别领域广泛使用的 FER-2013 数据集上进行训练，模型达到了 81.3% 的准确率。此外，我们使用 Ascendcl 将模型部署到搭载华为 Ascend 310B NPU 的 Orange Pi AI Pro 上，实现了移动设备上的实时表情检测。
![Image text](http://www.orangepi.cn/img/aipro/aipro-18.png)

# 安装
```bash
git clone https://github.com/lyz678/Emotion-recogniton-pytorch.git  # clone
cd Emotion-recogniton-pytorch
pip install -r requirements.txt  # install
```

# FER2013 Dataset
- 数据集来自 [https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data](https://www.kaggle.com/datasets/deadskull7/fer2013)
图像属性：48 x 48像素（2304字节）
标签：0=愤怒，1=厌恶，2=恐惧，3=快乐，4=悲伤，5=惊讶，6=中性
训练集包含28,709个示例。公共测试集包含3,589个示例。私人测试集包含另外3,589个示例。
- 下载数据集（fer2013.csv）然后将其放入 "fer2013" 文件夹中
  
```bash
cd fer2013
pip install kaggle
kaggle datasets download -d deadskull7/fer2013
unzip fer2013
```

# 训练和评估模型
- python train_emotion_classifier.py --model MiniXception --bs 128 --lr 0.01


# .pth to .onnx
```bash
python pt2onnx.py
```

# 实时视频
```bash
python run_on_cpu.py
```
![Image text](https://github.com/lyz678/Emotion-recogniton-pytorch/blob/main/result/demo1.jpg)
![Image text](https://github.com/lyz678/Emotion-recogniton-pytorch/blob/main/result/demo2.jpg)


# 绘制混淆矩阵
- python plot_fer2013_confusion_matrix.py --model MiniXception --bs 128
![Image text](https://github.com/lyz678/Emotion-recogniton-pytorch/blob/main/result/ConfusionMatrix.jpg)

# FER2013 准确率     
- 模型：    miniXception ;        测试准确率：  65% <Br/>
- 模型：   Resnet18 ;      测试准确率：  82%

# 在 Orange Pi AI Pro (Ascend310B NPU) 上运行

- 将目录中的 run_on_Ascend310B 文件下载到 Orange Pi AI Pro

bash


  
```bash
cd run_on_Ascend310B
atc --model=miniXception.sim.onnx --framework=5 --output=miniXception.sim --input_format=NCHW --input_shape="input.1:1,1,48,48" --log=error --soc_version=Ascend310B1 #.onnx to .om
python run_om.py
```
![Image text](https://github.com/lyz678/Emotion-recogniton-pytorch/blob/main/result/demo3.JPG)



