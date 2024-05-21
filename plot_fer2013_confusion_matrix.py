"""
plot confusion_matrix 
"""

import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import confusion_matrix
from models.cnn import *
from load_and_process import load_fer2013, FER2013Dataset
from sklearn.model_selection import train_test_split

num_workers = 8
num_classes = 7
test_split = 0.2
input_shape = (1, 48, 48)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Testing')
parser.add_argument('--model', type=str, default='ResNet18', help='maybe MiniXception, BigXception, ResNet18')
parser.add_argument('--bs', default=128, type=int, help='batch size')
opt = parser.parse_args()


# Loading dataset
faces, emotions = load_fer2013()
_, xtest, _, ytest = train_test_split(faces, emotions, test_size=test_split, shuffle=True)


test_dataset = FER2013Dataset(xtest, ytest, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]))

test_loader = DataLoader(test_dataset, batch_size=opt.bs, shuffle=False, num_workers=num_workers)

# Model, loss, optimizer
if opt.model == 'MiniXception':
    model = MiniXception(input_shape, num_classes)
elif opt.model == 'ResNet18':
    model = ResNet18(input_shape, num_classes)

model.load_state_dict(torch.load('models/best_model.pth'))  
model.to(device)



print(f'test : {opt.model}')
all_preds = []
all_targets = []
test_correct = 0
test_total = 0

for faces, emotions in test_loader:
    faces = faces.to(device)
    emotions = emotions.to(device)
    outputs = model(faces)
    _, predicted = torch.max(outputs, 1)
    all_preds.append(predicted.cpu().numpy())
    all_targets.append(emotions.argmax(dim=1).cpu().numpy())
    test_total += emotions.size(0)
    test_correct += (predicted == torch.argmax(emotions, dim=1)).sum().item()


test_accuracy = test_correct / test_total



# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Compute confusion matrix
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)
matrix = confusion_matrix(all_targets, all_preds)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=True, title='Normalized Confusion Matrix')
plt.savefig('ConfusionMatrix.jpg')
plt.close()

print(f'test_accuracy : {test_accuracy}')
