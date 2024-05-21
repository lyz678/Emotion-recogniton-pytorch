# Emotion-recogniton-pytorch
This project focuses on recognizing human emotions from facial expressions using deep learning techniques implemented in PyTorch. The model is trained on the FER-2013 dataset, a widely-used dataset in the field of facial expression recognition, achieving 81.3% accuracy.Moreover, we deploy the model on Orange Pi AI Pro using Ascendcl.

# Quick Start

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```



## Demos ##
![Image text](https://github.com/lyz678/Emotion-recogniton-pytorch/blob/main/demo1.jpg)
![Image text](https://github.com/lyz678/Emotion-recogniton-pytorch/blob/main/demo2.jpg)


## Dependencies ##
- Python 3.9
- Pytorch 2.3
- opencv-python 4.9.0.80 
- onnx 1.12  <Br/>
......
  
## FER2013 Dataset ##
- Dataset from [https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data](https://www.kaggle.com/datasets/deadskull7/fer2013)
Image Properties: 48 x 48 pixels (2304 bytes)
labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.
- download the dataset(fer2013.csv) then put it in the "fer2013" folder

### Train and Eval model ###
- python train_emotion_classifier.py.py --model MiniXception --bs 128 --lr 0.01

### plot confusion matrix ###
- python plot_fer2013_confusion_matrix.py --model MiniXception --bs 128
![Image text](https://github.com/lyz678/Emotion-recogniton-pytorch/blob/main/ConfusionMatrix.jpg)

###              fer2013 Accurary             ###

- Model：    VGG19 ;        test accuracy：  65% <Br/>
- Model：   Resnet18 ;      test accuracy：  82%   

