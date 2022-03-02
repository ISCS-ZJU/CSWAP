import torch
from torch.cuda import synchronize
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import tensorboard

import torchvision
import torchvision.transforms as transforms
import sys
import os
import time
# python3 cifar10.py 1 4 RandAug_record 300
# 1 means use randagugment, 4 means N, Epoch
from torch.utils.tensorboard import SummaryWriter
# from RandAugment import RandAugment
from models import *




tensorboard_name = "./ZFP_Schem_Eval/AlexNet_128"
tb_writer = SummaryWriter(tensorboard_name)
print("Record is saved in ",tensorboard_name)
train_tags = ['Train-Acc','Train-Loss']
test_tags = ['Test-Acc', 'Test-Loss']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# print("Epoch is ",str(sys.argv[4]))

# Data
print('==> Preparing data..')


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # # Can be saved
    transforms.RandomHorizontalFlip(),
    # Can be saved
    transforms.ToTensor(),
    # Can be saved
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # Can be saved
])

# if RandAugment_Bool:
#     print("RandAugment N value ",RandAugment_N)
#     transform_train.transforms.insert(0, RandAugment(RandAugment_N, 10))


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/data/DataSet', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='/data/DataSet', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = LeNet()
net = AlexNet()
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
# net = VGG('VGG11')
net = net.to(device)
print(net)
torch.cswapinit_()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# Training
def train(epoch):
    print('Epoch {}/{}'.format(epoch + 1, 120))
    print('-' * 10)
    start_time = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    ratio = 128
    # if ratio > 128:
    #     ratio = 128
    torch.cswapsetratio(ratio) # 1/4大小
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print('TrainLoss: %.3f | TrainAcc: %.3f%% ' % (train_loss/(batch_idx+1), 100.*correct/total))
        
    end_time = time.time()
    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time-start_time))
    for tag, value in zip(train_tags, [100.*correct/total, loss.item()]):
                tb_writer.add_scalars(tag, {'Train': value}, epoch)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        for tag, value in zip(test_tags, [100.*correct/total, loss.item()]):
                tb_writer.add_scalars(tag, {'Test': value}, epoch)
    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
        
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(net.state_dict(), './checkpoint/ckpt.pth')
    #     best_acc = acc

for epoch in range(start_epoch, start_epoch+100):
    # synchronize()
    # t1 = time.time()
    train(epoch)
    # synchronize()
    # print(time.time() - t1)
    test(epoch)

