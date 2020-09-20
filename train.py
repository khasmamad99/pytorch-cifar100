# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if args.dp:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
            print(
                f"Train Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"   
            )
            return epsilon, best_alpha

        # n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         # writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         # writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        # writer.add_scalar('Train/loss', loss.item(), n_iter)

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
        # writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    # writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    # writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-dp', default=False, action='store_true')
    parser.add_argument('-save_path', type=str, default='/content/drive/My Drive')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-sigma', type=float, default=0.001)
    parser.add_argument('-c', type=float, default=100.)
    args = parser.parse_args()

    net = get_network(args)

    if args.dp:
        net = convert_batchnorm_modules(net)

    net.to(device)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.2) #learning rate decay



    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.dp:
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.b,
            sample_size=len(cifar100_training_loader.dataset),
            alphas=[1 + x / 100.0 for x in range(1, 1000)] + list(range(12, 100)),
            noise_multiplier=args.sigma,
            max_grad_norm=args.c,
            clip_per_layer=False,
            enable_stat=False
        )
        privacy_engine.attach(optimizer)
        print("Attached privacy engine")

    checkpoint_path = os.path.join(args.save_path, args.net)

    # #use tensorboard
    # if not os.path.exists(settings.LOG_DIR):
    #     os.mkdir(settings.LOG_DIR)
    # writer = SummaryWriter(log_dir=os.path.join(
    #         settings.LOG_DIR, args.net, settings.TIME_NOW))
    # input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    # # writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    numpy_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}')

    best_acc = 0.0
    stats = []
    print(checkpoint_path.format(net=args.net, epoch=0, type='regular'))
    for epoch in range(1, args.epochs + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        stat = []
        if args.dp:
            epsilon, alpha = train(epoch)
            stat.append(epsilon)
            stat.append(alpha)
        else:
            train(epoch)
        acc = eval_training(epoch)
        stat.append(acc)
        stats.append(tuple(stat))

        if not args.dp:
            #start to save best performance model after learning rate decay to 0.01
            if epoch > 60 and best_acc < acc:
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                best_acc = acc
                continue

            if not epoch % settings.SAVE_EPOCH:
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
                print(checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

        else:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='dp'))
            np.save(numpy_path.format(net=args.net, epoch=epoch, type='dp'), stats)


    # writer.close()
