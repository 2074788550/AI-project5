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
class TextModel(nn.Module):
    def __init__(self, text_vocab_size, text_embedding_dim, num_labels):
        super(TextModel, self).__init__()
        # 文本模型
        self.text_embedding = nn.Embedding(text_vocab_size, text_embedding_dim)
        self.text_rnn = nn.GRU(text_embedding_dim, 128, batch_first=True)
        self.text_fc = nn.Linear(128, num_labels)

    def forward(self, text):
        text_embed = self.text_embedding(text)
        _, text_hidden = self.text_rnn(text_embed)
        output = self.text_fc(text_hidden.squeeze(0))
        return output

# 定义数据集
class TextDataset(Dataset):
    def __init__(self, label_file, text_folder):
        self.labels = pd.read_csv(label_file)
        self.text_folder = text_folder

        # 创建分词器和词汇表
        self.tokenizer = get_tokenizer('basic_english')
        counter = Counter()
        for idx in range(self.__len__()):
            text_file = f'{self.text_folder}/{int(self.labels.iloc[idx]["guid"])}.txt'
            try:
                with open(text_file, encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(text_file, encoding='ANSI') as f:
                    text = f.read()
            counter.update(self.tokenizer(text))
        self.vocab = Vocab(counter)

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
        
        # 加载文本
        try:
            with open(f'{self.text_folder}/{int(guid)}.txt', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(f'{self.text_folder}/{int(guid)}.txt', encoding='ANSI') as f:
                text = f.read()
        text = self.text_pipeline(text)
        text = text.to(device)  # 移动文本数据到设备

        return text, label

    def text_pipeline(self, text):
        text = [self.vocab[token] for token in self.tokenizer(text)]
        if len(text) > 100:
            text = text[:100]  # 如果文本长度大于100，则进行截断
        else:
            text += [0] * (100 - len(text))  # 如果文本长度小于100，则进行填充
        return torch.tensor(text)


# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
dataset = TextDataset('train.txt', 'data')

# 划分数据集
train_idx, valid_idx = train_test_split(np.arange(len(dataset)), test_size=0.2)
train_dataset = torch.utils.data.Subset(dataset, train_idx)
valid_dataset = torch.utils.data.Subset(dataset, valid_idx)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 初始化模型和优化器
model = TextModel(text_vocab_size=10000, text_embedding_dim=300, num_labels=3)
model = model.to(device)  # 移动模型到设备
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()



# 训练模型
for epoch in range(15):
    print(f'Epoch {epoch+1}')
    
    # 训练阶段
    model.train()
    total_loss = 0  # 重置总损失
    for text, label in train_loader:
        text, label = text.to(device), label.to(device)  # 移动数据到设备
        optimizer.zero_grad()
        output = model(text)
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
        for text, label in valid_loader:
            text, label = text.to(device), label.to(device)  # 移动数据到设备
            output = model(text)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    print(f'Validation accuracy: {correct / total:.2f}')


# 加载测试数据
test_dataset = TextDataset('test_without_label.txt', 'data')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 预测
model.eval()
predictions = []
with torch.no_grad():
    for text, _ in test_loader:
        text = text.to(device)  # 移动数据到设备
        output = model(text)
        _, predicted = torch.max(output, 1)
        predictions.extend(predicted.tolist())

label_map = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': -1}
# 将预测结果写入文件
with open('predictions_text.txt', 'w') as f:
    f.write('guid,tag\n')
    for guid, prediction in zip(test_dataset.labels['guid'], predictions):
        # 查找字典中与预测值对应的键
        prediction_key = list(label_map.keys())[list(label_map.values()).index(prediction)]
        f.write(f'{guid},{prediction_key}\n')
