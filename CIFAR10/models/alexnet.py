'''AlexNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from torch.cuda import synchronize
NUM_CLASSES = 10
Normal_Flag = 0
VDNN_Flag = 2
ZVC_Flag = 3
ZFP_Flag = 3



class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(256 * 2 * 2, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)

    def layer_maker(self, layer, input, type):
        pre = input
        torch.cswappre(pre,type)
        out = layer(pre)
        torch.cswappost(pre,out,type)
        return out

    def forward(self, out):
        synchronize()
        t1 = time.time()
        torch.cswapinit()
        out = self.layer_maker(self.conv1,out,Normal_Flag)
        out = self.layer_maker(self.relu,out,Normal_Flag)
        out = self.layer_maker(self.maxpool,out,Normal_Flag)
        out = self.layer_maker(self.conv2,out,VDNN_Flag)
        out = self.layer_maker(self.relu,out,Normal_Flag)
        out = self.layer_maker(self.maxpool,out,Normal_Flag)
        out = self.layer_maker(self.conv3,out,VDNN_Flag)
        out = self.layer_maker(self.relu,out,Normal_Flag)
        out = self.layer_maker(self.conv4,out,VDNN_Flag)
        out = self.layer_maker(self.relu,out,Normal_Flag)
        out = self.layer_maker(self.conv5,out,VDNN_Flag)
        out = self.layer_maker(self.relu,out,Normal_Flag)
        out = self.layer_maker(self.maxpool,out,Normal_Flag)
        pre = out
        torch.cswappre(pre,0)
        out = pre.view(pre.size(0), 256 * 2 * 2)
        torch.cswappost(pre,out,0)
        out = self.layer_maker(self.linear1,out,Normal_Flag)
        out = self.layer_maker(self.relu,out,Normal_Flag)
        out = self.layer_maker(self.linear2,out,Normal_Flag)
        out = self.layer_maker(self.relu,out,Normal_Flag)
        out = self.layer_maker(self.linear3,out,Normal_Flag)
        synchronize()
        # print(time.time() - t1)
        # x = self.features(x)
        # x = x.view(x.size(0), 256 * 2 * 2)
        # x = self.classifier(x)
        return out