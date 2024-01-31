import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from sklearn.model_selection import train_test_split

# 定义模型
class ImageModel(nn.Module):
    def __init__(self, num_labels):
        super(ImageModel, self).__init__()
        # 图像模型
        self.image_model = models.resnet50(pretrained=True)
        num_features = self.image_model.fc.in_features
        self.image_model.fc = nn.Linear(num_features, num_labels)

    def forward(self, image):
        output = self.image_model(image)
        return output

# 定义数据集
class ImageDataset(Dataset):
    def __init__(self, label_file, image_folder, transform=None):
        self.labels = pd.read_csv(label_file)
        self.image_folder = image_folder
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_map = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': -1}
        guid = self.labels.iloc[idx]['guid']
        tag = self.labels.iloc[idx]['tag']
        if pd.isnull(tag):  # 检查标签是否为NaN
            label = -1
        else:
            label = label_map[tag]
        
        # 加载图像
        image = Image.open(f'{self.image_folder}/{int(guid)}.jpg')
        image = self.resize(image)
        image = self.to_tensor(image)
        if self.transform:
            image = self.transform(image)
        image = image.to(device)  # 移动图像数据到设备

        return image, label

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
dataset = ImageDataset('train.txt', 'data')

# 划分数据集
train_idx, valid_idx = train_test_split(np.arange(len(dataset)), test_size=0.2)
train_dataset = torch.utils.data.Subset(dataset, train_idx)
valid_dataset = torch.utils.data.Subset(dataset, valid_idx)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 初始化模型和优化器
model = ImageModel(num_labels=3)
model = model.to(device)  # 移动模型到设备
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(15):
    print(f'Epoch {epoch+1}')
    
    # 训练阶段
    model.train()
    total_loss = 0  # 重置总损失
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)  # 移动数据到设备
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        total_loss += loss.item()  # 累加损失
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)  # 计算平均损失
    print(f'Training loss: {avg_loss:.4f}')  # 打印平均损失

    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in valid_loader:
            image, label = image.to(device), label.to(device)  # 移动数据到设备
            output = model(image)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    print(f'Validation accuracy: {correct / total:.2f}')


# 加载测试数据
test_dataset = ImageDataset('test_without_label.txt', 'data')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 预测
model.eval()
predictions = []
with torch.no_grad():
    for image, _ in test_loader:
        image = image.to(device)  # 移动数据到设备
        output = model(image)
        _, predicted = torch.max(output, 1)
        predictions.extend(predicted.tolist())

label_map = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': -1}
# 将预测结果写入文件
with open('predictions_image.txt', 'w') as f:
    f.write('guid,tag\n')
    for guid, prediction in zip(test_dataset.labels['guid'], predictions):
        # 查找字典中与预测值对应的键
        prediction_key = list(label_map.keys())[list(label_map.values()).index(prediction)]
        f.write(f'{guid},{prediction_key}\n')
