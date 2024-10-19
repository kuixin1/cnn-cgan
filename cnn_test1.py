import timm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms

data_dir = 'D:\\最终考核\\traffic_Data\\DATA'
test_dir = "D:\\最终考核\\traffic_Data\\TEST.1"
# 加载预训练的 InceptionResNetV2 模型
model = timm.create_model('inception_resnet_v2', pretrained=True)

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 冻结模型的权重
for param in model.parameters():
    param.requires_grad = False

transform = transforms.Compose([
    transforms.Resize((224, 224)),#适应inception_resnet_v2
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class CustomTrafficSignDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        self.images = []
        self.labels = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

dataset = CustomTrafficSignDataset(root_dir=data_dir, transform=transforms)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split1 = int(0.8 * dataset_size)  # 训练集占80%

# 随机打乱数据集
torch.manual_seed(42)
torch.cuda.manual_seed(42)
indices = torch.randperm(dataset_size).tolist()

# 划分数据集
train_indices = indices[:split1]
val_indices = indices[split1:]

# 创建 Subset 对象
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

test_dataset = CustomTrafficSignDataset(root_dir=test_dir,transform=transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 添加新的层
new_layers = torch.nn.Sequential(
    torch.nn.AdaptiveAvgPool2d((1, 1)),
    torch.nn.Flatten(),
    torch.nn.Linear(1536, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 58),
)

# 将新层添加到模型
model.fc = new_layers

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
model.train()

for epoch in range(10):
    # 训练步骤
    model.train()
    for image, label in train_loader:
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 验证步骤
    model.eval()
    with torch.no_grad():
        for image, label in val_loader:
            # 前向传播
            outputs = model(image)
            # 计算损失
            loss = criterion(outputs, label)
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = label.size(0)
            correct = (predicted == label).sum().item()
            accuracy = correct / total
            # 打印损失和准确率
            print(f"验证损失：{loss.item()}，验证准确率：{accuracy}")
        # 测试步骤
    model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            # 前向传播
            outputs = model(image)
            # 计算损失
            loss = criterion(outputs, label)
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = label.size(0)
            correct = (predicted == label).sum().item()
            accuracy = correct / total
            # 打印损失和准确率
            print(f"测试损失：{loss.item()}，测试准确率：{accuracy}")
