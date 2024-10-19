import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm # 进度条
# 设置中文显示
import matplotlib.font_manager as font_manager
from torchvision.datasets import DatasetFolder
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore")
from torchvision import datasets, transforms


data_dir = 'D:\\最终考核\\traffic_Data\\DATA'
test_dir = "D:\\最终考核\\traffic_Data\\TEST"
csv_dir = "D:\\最终考核\\labels.csv"
df = pd.read_csv(csv_dir)#导入csv

dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
)

# 划分数据集为训练集、验证集
dataset_size = len(dataset)
indices = list(range(dataset_size))
split1 = int(0.6 * dataset_size)  # 训练集占60%
split2 = int(0.8 * dataset_size)  # 验证集占20%

# 随机打乱数据集
torch.manual_seed(42)
torch.cuda.manual_seed(42)
indices = torch.randperm(dataset_size).tolist()

# 划分数据集
train_indices = indices[:split1]
val_indices = indices[split1:split2]

# 创建 Subset 对象
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# 创建 DataLoader 加载数据,同时增加数据量
train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

#构建卷积神经网络 一般就是卷积层+relu层+池化层
class TrafficSignClassifier(nn.Module):
    def __init__(self):
        super(TrafficSignClassifier, self).__init__()
        self.conv1 = nn.Sequential(  # 输入大小
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.fc1 = nn.Linear(128*16*16,512)
        self.fc2 = nn.Linear(512, 58)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv3(out)
        out = self.pool(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
def validate(cnn, val_loader, criterion):
    cnn.eval()  # 将模型切换到评估模式
    correct = 0
    total = 0
    validation_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.float().to(device)
            labels = labels.to(device)

            outputs = cnn(images)
            loss = criterion(outputs, labels)

            validation_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100.0 * correct / total
    validation_loss /= len(val_loader.dataset)

    return validation_loss, validation_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载保存好的数据，这个数据只是保存前面我们运行的权重，偏置
model = torch.load("traffic_sign_model.pth")
model = TrafficSignClassifier()
model = model.eval().to(device)

#标签对应
class_names = dataset.classes
# 从 DataFrame 中选择 'ClassId' 和 'Name' 两列
class_df = df[['ClassId', 'Name']]
# 将 'ClassId' 和 'Name' 列转换成字典
idx_to_labels = class_df.set_index('ClassId')['Name'].to_dict()

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize((128,128)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])


# 处理帧函数
def process_frame(img):
    # 记录该帧开始处理的时间
    start_time = time.time()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
    img_pil = Image.fromarray(img_rgb)  # array 转 PIL

    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    top_n = torch.topk(pred_softmax, 5)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析预测类别
    confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析置信度

    # 使用PIL绘制中文
    draw = ImageDraw.Draw(img_pil)
    # 在图像上写字
    for i in range(len(confs)):
        pred_class = idx_to_labels[pred_ids[i]]
        text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
        # 文字坐标，字体，bgra颜色
        draw.text((50, 100 + 50 * i), text, fill=(255, 0, 0, 1))
    img = np.array(img_pil)  # PIL 转 array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB转BGR

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)
    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
    img = cv2.putText(img, 'FPS  ' + str(int(FPS)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4,
                      cv2.LINE_AA)
    return img


# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0)

# 打开cap
cap.open(0)

# 无限循环，直到break被触发
while cap.isOpened():
    # 获取画面
    success, frame = cap.read()
    if not success:
        print('Error')
        break

    ## !!!处理帧函数
    frame = process_frame(frame)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break

# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()