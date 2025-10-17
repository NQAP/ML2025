# -*- coding: utf-8 -*-
import os
import random
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from argparse import Namespace
from tqdm import tqdm

"""#### Hyperparameters and setting"""

# TODO: modify the hyperparameters
# ⚠️ 建議降低 LR，以應對移除 BatchNorm 後的不穩定性
config = Namespace(
    random_seed = 42,
    BATCH = 128,
    n_epoch = 50,  # 增加 epoch 數以應對更慢的收斂
    lr = 1.0e-3,   # 建議降低學習率
    weight_decay = 1e-5,
    ckpt_path = 'model_no_bn.pth', # 更改 checkpoint 名稱
)

TRA_PATH = 'data/train/'
TST_PATH = 'data/test/'
LABEL_PATH = 'data/train.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(config.random_seed)
torch.cuda.manual_seed_all(config.random_seed)
random.seed(config.random_seed)
np.random.seed(config.random_seed)

"""#### Dataset and Dataloader (保持不變)"""

class FaceExpressionDataset(Dataset):
    def __init__(self, img_path, label_path=None, tfm=T.ToTensor()):
        n_samples = len(os.listdir(img_path))
        if label_path is not None:
            self.images = [f'{img_path}/{i+7000}.jpg' for i in range(n_samples)]
            self.labels = pd.read_csv(label_path)['label'].values.tolist()
        else:
            self.images = [f'{img_path}/{i}.jpg' for i in range(n_samples)]
            self.labels = None
        self.tfm = tfm

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = self.tfm(img)
        if self.labels is not None:
            lab = torch.tensor(self.labels[idx]).long()
            return img, lab
        else:
            return img

    def __len__(self):
        return len(self.images)

# 數據統計與 DataLoader 設置 (為簡潔起見，省略計算過程，只保留結果和 Dataloader 設置)
# 假設之前計算的 MEAN/STD 依然有效
calc_tfm = T.Compose([T.Resize((64, 64)), T.Grayscale(num_output_channels=1), T.ToTensor()])
full_train_dataset = FaceExpressionDataset(TRA_PATH, LABEL_PATH, tfm=calc_tfm)
calc_loader = DataLoader(full_train_dataset, batch_size=256, shuffle=False)

def calculate_mean_std(loader):
    channels = 1 
    sum_val = torch.zeros(channels)
    sum_sq_val = torch.zeros(channels)
    total_pixels = 0
    for images, _ in tqdm(loader, desc="Calculating Mean and Std"):
        pixels = images.view(-1, channels) 
        sum_val += pixels.sum(dim=0).cpu()
        sum_sq_val += (pixels ** 2).sum(dim=0).cpu()
        total_pixels += pixels.size(0)
    mean = sum_val / total_pixels
    variance = (sum_sq_val / total_pixels) - (mean ** 2)
    std = torch.sqrt(variance)
    return mean.item(), std.item()

dataset_mean, dataset_std = calculate_mean_std(calc_loader)
MEAN = [dataset_mean] 
STD = [dataset_std]

train_tfm = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((64, 64)),
    T.RandomHorizontalFlip(p=0.5), 
    T.RandomRotation(15),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)), 
    T.ToTensor(),
    # T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)), # 保持註釋
    T.Normalize(mean=MEAN, std=STD),
])

eval_tfm = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

train_dataset = FaceExpressionDataset(TRA_PATH, LABEL_PATH, tfm=train_tfm)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
test_dataset = FaceExpressionDataset(TST_PATH, tfm=eval_tfm)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH, shuffle=False)


"""#### Model (移除所有 BatchNorm)"""

class FaceExpressionNet_NoBN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.relu = nn.LeakyReLU(0.1)

        # === Conv Block 1: 移除 BatchNorm2d ===
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), self.relu, # 移除 BatchNorm2d(64)
            nn.Conv2d(64, 64, kernel_size=3, padding=1), self.relu, # 移除 BatchNorm2d(64)
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        # === Conv Block 2: 移除 BatchNorm2d ===
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), self.relu, # 移除 BatchNorm2d(128)
            nn.Conv2d(128, 128, kernel_size=3, padding=1), self.relu, # 移除 BatchNorm2d(128)
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        # === Conv Block 3: 移除 BatchNorm2d ===
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), self.relu, # 移除 BatchNorm2d(256)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), self.relu, # 移除 BatchNorm2d(256)
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        INPUT_SIZE = 256 * 8 * 8  # 16384
        
        # === FC Layers: 移除 BatchNorm1d ===
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3), 
            nn.Linear(INPUT_SIZE, 256), 
            # ❗ 移除 nn.BatchNorm1d(256)
            self.relu,
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1) # 尺寸: BATCH x 16384
        
        x = self.fc_layers(x)
        return x

"""#### training loop (調整 scheduler 參數)"""

def train(model, train_loader, valid_loader, config):
    model.to(device)
    criteria = nn.CrossEntropyLoss()
    # 由於沒有 BatchNorm，可以考慮調整 weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # 調整為更積極的衰減策略，幫助模型突破平台期
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.2,   # 每次降 LR 80%
        patience=3    # 連續 3 個 epoch 沒進步就降 LR
    )
    
    best_acc = 0
    train_losses, valid_losses = [], []
    for epoch in range(config.n_epoch):
        # train
        model.train()
        train_loss, train_acc = 0, 0
        for img, lab in tqdm(train_loader):
            img, lab = img.to(device), lab.to(device)
            output = model(img)
            optimizer.zero_grad()
            loss = criteria(output, lab)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (torch.argmax(output, dim=-1) == lab).float().mean().item()
        train_loss, train_acc = train_loss/len(train_loader), train_acc/len(train_loader)
        train_losses.append(train_loss)
        print(f'Epoch: {epoch+1}/{config.n_epoch}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')

        # valid
        model.eval()
        valid_loss, valid_acc = 0, 0
        with torch.no_grad():
            for img, lab in valid_loader:
                img, lab = img.to(device), lab.to(device)
                output = model(img)
                loss = criteria(output, lab)
                valid_loss += loss.item()
                valid_acc += (torch.argmax(output, dim=-1) == lab).float().mean().item()
        valid_loss, valid_acc = valid_loss/len(valid_loader), valid_acc/len(valid_loader)
        valid_losses.append(valid_loss)
        print(f'Epoch: {epoch+1}/{config.n_epoch}, valid loss: {valid_loss:.4f}, valid acc: {valid_acc:.4f}')

        # update
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), config.ckpt_path)
            print(f'== best valid acc: {best_acc:.4f} ==')
        scheduler.step(valid_acc)
    model.load_state_dict(torch.load(config.ckpt_path))

    # plot the training/validation loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(config.n_epoch), train_losses, label='Training Loss')
    plt.plot(range(config.n_epoch), valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

"""### Training"""

# 實例化新的模型類別
model = FaceExpressionNet_NoBN() 
train(model, train_loader, valid_loader, config)

# ... (draw_confusion_matrix 和 test 函式保持不變，但請注意它們需要載入 'model_no_bn.pth')

def draw_confusion_matrix(model, valid_loader):
    predictions, labels = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for img, lab in tqdm(valid_loader):
            img = img.to(device)
            output = model(img)
            predictions += torch.argmax(output, dim=-1).tolist()
            labels += lab.tolist()
    # TODO draw the confusion matrix
    # 將列表轉換為 NumPy 陣列
    y_true = np.array(labels)
    y_pred = np.array(predictions)

    # 2. 手動計算混淆矩陣 (代替 sklearn.metrics.confusion_matrix)
    num_classes = 7 # 您的情緒類別數量
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        # cm[i, j] 是真實標籤 i 被預測為 j 的次數
        cm[true_label, pred_label] += 1
        
    # 假設情緒類別 (0-6) 對應的標籤名稱
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # 3. 繪製混淆矩陣
    plt.figure(figsize=(10, 8))
    
    # 使用 imshow 繪製顏色熱力圖
    # cmap=plt.cm.Blues 是 Matplotlib 內建的藍色色階
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) 
    plt.title('Confusion Matrix for Facial Expression Recognition')
    plt.colorbar() # 添加顏色條

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, emotion_labels, rotation=45) # 設置 X 軸標籤
    plt.yticks(tick_marks, emotion_labels)             # 設置 Y 軸標籤

    # 4. 在每個格子上添加數值標註 (Annotation)
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            # 根據背景顏色，調整文字顏色 (深色背景用白色字，淺色背景用黑色字)
            color = "white" if cm[i, j] > thresh else "black"
            
            # 使用 plt.text 函式，將數值放置在格子中央
            plt.text(j, i, format(cm[i, j], 'd'), # 'd' 表示整數格式
                     horizontalalignment="center",
                     color=color)

    plt.tight_layout() # 自動調整佈局，避免標籤重疊
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 確保在訓練後呼叫
model.load_state_dict(torch.load(config.ckpt_path)) # 確保載入最佳模型
draw_confusion_matrix(model, valid_loader)

"""### Testing"""

def test(model, test_loader):
    predictions = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for img in tqdm(test_loader):
            img = img.to(device)
            output = model(img)
            predictions += torch.argmax(output, dim=-1).tolist()
    with open('predict.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for id, r in enumerate(predictions):
            writer.writerow([id, r])

model.load_state_dict(torch.load(config.ckpt_path))
test(model, test_loader)