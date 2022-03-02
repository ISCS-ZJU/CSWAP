'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import time
from torch.cuda import synchronize

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
    # @w_test
    def forward(self, out):
        torch.cswapinit()
        synchronize()
        t1 = time.time()
        for i in self.features:
            out = self.layer_maker(i, out, 0)

        pre = out
        torch.cswappre(pre,0)
        out = pre.view(pre.size(0), -1)
        torch.cswappost(pre,out,0)
        out = self.layer_maker(self.classifier, out, 0)
        synchronize()
        print(time.time() - t1)
        return out


    def layer_maker(self, layer, input, type):
        if "Conv" in str(layer):
            type = 2 # 6 = ZFP, 3=ZVC, 2 = vDNN
        pre = input
        torch.cswappre(pre,type)
        out = layer(pre)
        torch.cswappost(pre,out,type)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                        #    nn.Tanh()
                        # nn.ELU()
                        # nn.PReLU()
                        #    nn.ReLU(inplace=True)
                        #    nn.Sigmoid()
                        nn.LeakyReLU(inplace=True)
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
