import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import argparse
import logging
from load_and_process import load_fer2013, FER2013Dataset
from models.cnn import *
from tqdm import tqdm 

# parameters
num_epochs = 200
input_shape = (1, 48, 48)
validation_split = 0.2
num_classes = 7
num_workers = 8
patience = 50
base_path = 'models/'
log_path = os.path.join(base_path, 'logging.txt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure logging
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='ResNet18', help='maybe MiniXception, ResNet18')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--pretrained', default=False, type=bool, help='True or False')
opt = parser.parse_args()

# loading dataset
faces, emotions = load_fer2013()
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)

train_dataset = FER2013Dataset(xtrain, ytrain, transform=transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]))

val_dataset = FER2013Dataset(xtest, ytest, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]))

train_loader = DataLoader(train_dataset, batch_size=opt.bs, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=opt.bs, shuffle=False, num_workers=num_workers)

# model, loss, optimizer
if opt.model == 'MiniXception':
    model = MiniXception(input_shape, num_classes)
elif opt.model == 'ResNet18':
    model = ResNet18(input_shape, num_classes)

best_model_path = os.path.join(base_path, 'best_model.pth')
if opt.pretrained:
    model.load_state_dict(torch.load(best_model_path))  # Load the pre-trained model
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=int(patience / 4))

# training and validation loop
best_val_loss = float('inf')
early_stop_counter = 0

logger.info(f'Training started with model: {opt.model}')

for epoch in range(num_epochs):
    #train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    for faces, emotions in train_progress:
        faces = faces.to(device)
        emotions = emotions.to(device)
        optimizer.zero_grad()
        outputs = model(faces)
        loss = criterion(outputs, emotions)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * faces.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += emotions.size(0)
        train_correct += (predicted == torch.argmax(emotions, dim=1)).sum().item()
        train_progress.set_postfix({
            'loss': loss.item(),
            'train_accuracy': train_correct / train_total
        })
    train_loss /= train_total
    train_accuracy = train_correct / train_total

    #val
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_progress = tqdm(val_loader, desc=f'Validation {epoch+1}/{num_epochs}', unit='batch')
    with torch.no_grad():
        for faces, emotions in val_progress:
            faces = faces.to(device)
            emotions = emotions.to(device)

            outputs = model(faces)
            loss = criterion(outputs, torch.argmax(emotions, dim=1))

            val_loss += loss.item() * faces.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += emotions.size(0)
            val_correct += (predicted == torch.argmax(emotions, dim=1)).sum().item()
            val_progress.set_postfix({
                'loss': loss.item(),
                'val_accuracy': val_correct / val_total
            })

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    scheduler.step(val_loss)

    log_message = (f'Epoch {epoch + 1}/{num_epochs}, '
                   f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, '  # 获取当前学习率
                   f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                   f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    print(log_message)
    logger.info(log_message)

    # checkpointing
    model_names = f'{base_path}{opt.model}.{epoch:02d}-accuracy{train_accuracy:.2f}.pth'
    if epoch % 10 == 0 and val_loss < best_val_loss:
        torch.save(model.state_dict(), model_names)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print("Early stopping")
        logger.info("Early stopping")
        break

logger.info("Training complete. Best model saved to: " + best_model_path)
print("Training complete. Best model saved to:", best_model_path)
