'''
训练模型，然后保存下来，这里的epoch只有1000，而且训练的是cgan模型
'''

import os
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

ngpu = 1
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
data_dir = '..\\traffic_Data\\DATA'
data_transforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))#此变换将图像像素值归一化为范围 [-1, 1]
])
#定义数据集类
class CustomTrafficSignDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        #遍历标签和图像
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

train_dataset = CustomTrafficSignDataset(root_dir=data_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


#权重应当从均值为0，标准差为0.02的正态分布中随机初始化，防止梯度爆炸或消失
def weights_init(m):
    classname = m.__class__.__name__#获取模块的类名称
    if classname.find('Conv') != -1:#查看是否有conv，卷积层
        nn.init.normal_(m.weight.data, 0.0, 0.02)#有的话初始化卷积层的权重参数
    elif classname.find('BatchNorm') != -1:#查看是否有归一化层
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)#有的话将偏差初始化为0

# 生成器
#输出的维度是 (64, 3, 32, 32)，输入的维度分别是 (64, 100, 1, 1) 和 (64, 58, 1, 1)
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.image = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.label = nn.Sequential(  #标签处理
            nn.ConvTranspose2d(58, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),#对卷积层进行归一化，防止梯度爆炸或消失
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()  #使输出图像像素为（-1，1），最终是3*32*32
        )

    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        incat = torch.cat((image, label), dim=1)
        incat = self.main(incat)
        return incat

#生成器实例化
netG = Generator(ngpu).to(device)

if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)#权重改变

#判别器
#输出维度(64, 1)，输入的维度分别是 (64, 3, 32, 32) 和 (64, 58, 1, 1)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.image = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.label = nn.Sequential(
            nn.Conv2d(58, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.main = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        incat = torch.cat((image, label), dim=1)
        return self.main(incat)

#判别器的实例化
netD = Discriminator(ngpu).to(device)

if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)#初始化参数

#定义优化器和损失函数
criterion = nn.BCELoss()#损失函数

real_label_num = 1.
fake_label_num = 0.
#parameters获取训练时的参数，同时可以更新
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

#设置one_hot，其实就是先创建一个58*58的对角矩阵
label_1hots = torch.zeros(58, 58)
for i in range(58):
    label_1hots[i, i] = 1
label_1hots = label_1hots.view(58, 58, 1, 1).to(device)

#从生成器中获取，该张量可以用于分割任务，其中模型学习将每个像素分配给相应的类别。
label_fills = torch.zeros(58, 58, 32, 32)
ones = torch.ones(32, 32)
for i in range(58):
    label_fills[i][i] = ones
label_fills = label_fills.to(device)

#设置噪音向量，标签
fixed_noise = torch.randn(10, 100, 1, 1).to(device)
fixed_label = label_1hots[torch.arange(58).repeat(10).sort().values]

epoch = 1000
img_list = []
G_losses = []
D_losses = []
D_x_list = []
D_z_list = []
loss_tep = 10

print("开始")
for epoch in range(epoch):
    for i, data in enumerate(train_loader):
        netD.zero_grad()#梯度清零

        real_image = data[0].to(device)#真实图像的复制
        b_size = real_image.size(0)
        #创建真实的标签为1，虚假标签为0
        real_label = torch.full((b_size,), real_label_num).to(device)
        fake_label = torch.full((b_size,), fake_label_num).to(device)
        #创建独热编码和填充编码
        G_label = label_1hots[data[1]]
        D_label = label_fills[data[1]]

        output = netD(real_image, D_label).view(-1)#将数据传给判别器
        errD_real = criterion(output, real_label)#计算损失函数
        errD_real.backward()#前向传播
        D_x = output.mean().item()#计算判别器输出的均值

        noise = torch.randn(b_size, 100, 1, 1).to(device)#生成噪声向量
        fake = netG(noise, G_label)#将噪音向量和标签传给生成器
        output = netD(fake.detach(), D_label).view(-1)#分离判别器的图像，同时将标签传给判别器
        errD_fake = criterion(output, fake_label)#计算生成图像判别器的损失
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()#使用优化器更新判别器网络的权重

        netG.zero_grad()
        output = netD(fake, D_label).view(-1)
        errG = criterion(output, real_label)
        errG.backward()#反向传播
        D_G_z2 = output.mean().item()#表示判别器更有信心将生成图像分类为真实图像，1的时候最大
        optimizerG.step()
        print(
            f'Epoch: [{epoch + 1:0>{len(str(epoch))}}/{epoch}]',
            f'Loss-D: {errD.item():.4f}',
            f'Loss-G: {errG.item():.4f}',
            f'D(x): {D_x:.4f}',
            f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
            end='\r'
        )

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        D_x_list.append(D_x)
        D_z_list.append(D_G_z2)

        if errG < loss_tep:
            # 保存模型
            checkpoint = {
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict()
            }
            torch.save(checkpoint, 'model.pt')
            loss_tep = errG
    print()

print("完成")