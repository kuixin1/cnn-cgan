import os
import torchvision.utils as vutils
import torch
import torch.nn as nn

ngpu = 1
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
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

def weights_init(m):
    classname = m.__class__.__name__#获取模块的类名称
    if classname.find('Conv') != -1:#查看是否有conv，卷积层
        nn.init.normal_(m.weight.data, 0.0, 0.02)#有的话初始化卷积层的权重参数
    elif classname.find('BatchNorm') != -1:#查看是否有归一化层
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)#有的话将偏差初始化为0

checkpoint = torch.load('model.pt')
netG = Generator(ngpu).to(device)
netG.apply(weights_init)
netG.load_state_dict(checkpoint['netG_state_dict'])

#设置one_hot，其实就是先创建一个58*58的对角矩阵
label_1hots = torch.zeros(58, 58)
for i in range(58):
    label_1hots[i, i] = 1
label_1hots = label_1hots.view(58, 58, 1, 1).to(device)

#我们要生成58类
label_fills = torch.zeros(58, 58, 32, 32)
ones = torch.ones(32, 32)
for i in range(58):
    label_fills[i][i] = ones
label_fills = label_fills.to(device)

#设置噪音向量
fixed_noise = torch.randn(58, 100, 1, 1).to(device)
fixed_label = label_1hots[torch.arange(58).sort().values]

# 将连接后的向量传递给生成器
fake_images = netG(fixed_noise,fixed_label)

data_dir = "D:\\最终考核\\traffic_Data\\DATA.1"
for i in range(57):  # 循环保存 57 张图像
     vutils.save_image(fake_images[i], os.path.join(data_dir, f'image_{i}.png'))