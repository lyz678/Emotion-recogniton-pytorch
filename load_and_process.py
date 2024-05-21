import torch
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

dataset_path = 'fer2013/fer2013.csv'
image_size = (48, 48)

def load_fer2013():
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).values.astype('uint8')
    return faces, emotions



class FER2013Dataset(Dataset):
    def __init__(self, faces, emotions, transform=None):
        self.faces = faces
        self.emotions = emotions
        self.transform = transform

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        face = self.faces[idx]
        face = face.squeeze().astype(np.uint8)  # 去除维度为 1 的维度，并转换数据类型为 uint8

        try:
            # 确保face是一个2D或3D的numpy数组
            face = Image.fromarray(face)  # Convert numpy array to PIL Image
        except Exception as e:
            print(f"Error converting face to image: {e}, index: {idx}")
            raise
        emotion = self.emotions[idx]
        if self.transform:
            try:
                face = self.transform(face)
            except Exception as e:
                print(f"Error applying transform: {e}, index: {idx}")
                raise
        return face, torch.tensor(emotion, dtype=torch.float32)
