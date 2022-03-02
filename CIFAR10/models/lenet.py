'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def layer_maker(self, layer, input, type):
        pre = input
        torch.cswappre(pre,type)
        out = layer(pre)
        torch.cswappost(pre,out,type)
        return out


    def forward(self, out):
        torch.cswapinit()
        pre = out
        #torch.cswappre(pre,3)
        out = self.conv1(pre)
        #torch.cswappost(pre,out,3)
        pre = out
        torch.cswappre(pre,2)
        out = F.relu(pre)
        torch.cswappost(pre,out,2)
        pre = out
        torch.cswappre(pre,2)
        out = F.max_pool2d(pre, 2)
        torch.cswappost(pre,out,2)

        # pre = out
        # torch.cswappre(pre,2)
        # out = self.conv2(pre)
        # torch.cswappost(pre,out,2)

        out = self.layer_maker(self.conv2,out,2)

        pre = out
        torch.cswappre(pre,2)
        out = F.relu(pre)
        torch.cswappost(pre,out,2)
        pre = out
        torch.cswappre(pre,2)
        out = F.max_pool2d(pre, 2)
        torch.cswappost(pre,out,2)
        pre = out
        torch.cswappre(pre,2)
        out = pre.view(pre.size(0), -1)
        torch.cswappost(pre,out,2)
        pre = out
        torch.cswappre(pre,2)
        out = self.fc1(pre)
        torch.cswappost(pre,out,2)
        pre = out
        torch.cswappre(pre,0)
        out = F.relu(pre)
        torch.cswappost(pre,out,0)
        pre = out
        torch.cswappre(pre,0)
        out = self.fc2(pre)
        torch.cswappost(pre,out,0)
        pre = out
        torch.cswappre(pre,0)
        out = F.relu(pre)
        torch.cswappost(pre,out,0)
        pre = out
        torch.cswappre(pre,0)
        out = self.fc3(pre)
        torch.cswappost(pre,out,0)
        return out
