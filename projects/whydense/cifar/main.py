'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
# from utils import progress_bar
from htmresearch.frameworks.pytorch.sparse_net import SparseNet
from htmresearch.frameworks.pytorch.image_transforms import RandomNoise


def getSparseNet():
    """This gets about 72% accuracy on the test set after 60-70 epochs"""
    net = SparseNet(
        inputSize=(3, 32, 32),
        outChannels=[30, 40],
        c_k=[400, 400],
        dropout=False,
        n=400,
        k=50,
        boostStrength=1.5,
        weightSparsity=0.3,
        boostStrengthFactor=0.85,
        kInferenceFactor=1.5,
        useBatchNorm=True,
        normalizeWeights=False
    )

    return net


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.95, type=float, help='learning rate gamma')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Use this for domino
if "DOMINO_WORKING_DIR" in os.environ:
    trainset = torchvision.datasets.CIFAR10(root='./projects/whydense/cifar/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./projects/whydense/cifar/data', train=False, download=True, transform=transform_test)

else:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# netInstance = VGG('VGG19')
# netInstance = ResNet18()
# netInstance = PreActResNet18()
# netInstance = GoogLeNet()
# netInstance = LeNet()
# netInstance = DenseNet121()
# netInstance = densenet_cifar()
netInstance = sparse_densenet_cifar()
# netInstance = ResNeXt29_2x64d()
# netInstance = MobileNet()
# netInstance = MobileNetV2()
# netInstance = DPN92()
# netInstance = ShuffleNetG2()
# netInstance = SENet18()
# netInstance = ShuffleNetV2(1)

# netInstance = getSparseNet()
netInstance = netInstance.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(netInstance)
    cudnn.benchmark = True
else:
    net = netInstance

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/ckptSmallDensenetSDR.t7',
    #                         map_location=device)
    # net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
    net = torch.load('./checkpoint/modelSmallDensenetSDR.pt', map_location=device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

# Training
def train(epoch):
    print('\nEpoch: %d, learningRate=%g' % (epoch, scheduler.get_lr()[0]))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
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

        if batch_idx % 100 == 0:
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    netInstance.postEpoch()

def test(epoch):
    # print('\nTesting')
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

            # if batch_idx % 50 == 0:
            #     print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss, 100.*correct/total, correct, total))
            #     # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     # % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print('Test Done. Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
          % (test_loss, 100. * correct / total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckptSparseDensenet_cifar.t7')
        best_acc = acc
        torch.save(net, './checkpoint/modelSparseDensenet_cifar.pt')


def testNoise(net, noiseLevel=0.3):

    transform_noise_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        RandomNoise(noiseLevel, whiteValue=0.5 + 2 * 0.20),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_noise_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    print("Running noise test with noise level:", noiseLevel)

    if isinstance(net, torch.nn.DataParallel) and device == "cpu":
        net = net.module

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

    print('Noise test Done. Noise: %g | Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
          % (noiseLevel, test_loss, 100. * correct / total, correct, total))


for epoch in range(start_epoch, start_epoch+30):
    scheduler.step()
    train(epoch)
    test(epoch)

print("Running noise tests with sparse net")
for noiseLevel in [0.0, 0.1, 0.2]:
    testNoise(net, noiseLevel)

