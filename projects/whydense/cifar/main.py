'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import numpy as np
import time
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
from htmresearch.frameworks.pytorch.image_transforms import RandomNoise


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.9, type=float, help='learning rate gamma')
parser.add_argument('--epochs', default=80, type=int, help='number of epochs to run')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--quick', '-q', action='store_true', help='one batch epochs, for debugging')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Model
print('==> Building model..')
# netInstance = VGG('VGG19')
# netInstance = ResNet18()
# netInstance = PreActResNet18()
# netInstance = GoogLeNet()
# netInstance = LeNet()
# netInstance = DenseNet121()
# netInstance = densenet_cifar(growth_rate=12)
netInstance = notso_densenet_cifar(growth_rate=12)
# netInstance = ResNeXt29_2x64d()
# netInstance = MobileNet()
# netInstance = MobileNetV2()
# netInstance = DPN92()
# netInstance = ShuffleNetG2()
# netInstance = SENet18()
# netInstance = ShuffleNetV2(1)

print(netInstance)

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

trainloader1 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
trainloader128 = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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
    # checkpoint = torch.load('./checkpoint/ckptSparseDensenet_cifar.t7',
    #                         map_location=device)
    # net.load_state_dict(checkpoint['net'])
    best_acc = 88.2
    start_epoch = 30
    net = torch.load('./checkpoint/modelSparseDensenet_cifar.pt', map_location=device)
    if isinstance(net, torch.nn.DataParallel) and device == "cpu":
        net = net.module



criterion = nn.CrossEntropyLoss()

def createOptimizer(net, lr, gamma):
    print("Resetting optimizer learning rate")
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    return optimizer, scheduler


# Training
def train(epoch, optimizer, scheduler):
    start_time = time.time()
    if epoch == 0:
        trainloader = trainloader1
    else:
        trainloader = trainloader128

    print('\n\nEpoch: %d, learningRate=%g' % (epoch, scheduler.get_lr()[0]))
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

        if args.quick and batch_idx % 1 == 0:
          break

    netInstance.postEpoch()
    print("Training time for epoch=", time.time() - start_time)

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

            if args.quick and batch_idx % 1 == 0:
              break


    print('Test Done. Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss, 100. * correct / total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..',acc,epoch)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckptSparseDensenet_cifar_s15_k50_gr18.t7')
        best_acc = acc
        torch.save(net, './checkpoint/modelSparseDensenet_cifar_s15_k50_gr18.pt')


def testNoise(net, noiseLevel=0.3):

    transform_noise_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        RandomNoise(noiseLevel, whiteValue=0.5 + 2 * 0.20,
                    blackValue=0.5 - 2*0.2),
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


optimizer, scheduler = createOptimizer(net, args.lr, args.gamma)
for epoch in range(start_epoch, start_epoch+args.epochs):

    start_time = time.time()

    scheduler.step()
    train(epoch, optimizer, scheduler)
    test(epoch)

    # Reset learning rate and run noise tests
    if scheduler.get_lr()[0] < 0.0000:
        print("Running noise tests at epoch", epoch)
        for noiseLevel in np.arange(0.0, 0.2, 0.025):
            testNoise(net, noiseLevel)
        print("-----\n\n")

        optimizer, scheduler = createOptimizer(net, args.lr / 10.0, args.gamma)

    print("Full epoch time=", time.time() - start_time)


print("Running final noise tests", epoch)
for noiseLevel in np.arange(0.0, 0.2, 0.025):
    testNoise(net, noiseLevel)
print("-----\n\n")
