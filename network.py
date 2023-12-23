'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import random
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
from PIL import Image
from os import path

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


def network(test_image):
    # Models
    default_model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    customized_models_names = sorted(name for name in customized_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(customized_models.__dict__[name]))
    for name in customized_models.__dict__:
        if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
            models.__dict__[name] = customized_models.__dict__[name]

    # model_names = default_model_names + customized_models_names
    model_names = customized_models_names

    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Datasets
    parser.add_argument('-d', '--data', default=r'D:\学习资料\大学功课\软件课设\AttentionBasedNetwork\data\datasets', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Optimization options
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batchsize (default: 256)')
    parser.add_argument('--test-batch', default=10, type=int, metavar='N',
                        help='test batchsize (default: 200)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[31, 61],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default=r'D:\学习资料\大学功课\软件课设\AttentionBasedNetwork\checkpoint\imagenet_denoise\resnet152', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default=r'D:\学习资料\大学功课\软件课设\AttentionBasedNetwork\checkpoint\imagenet_denoise\resnet152\model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
    parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_false',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    #Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
    
    

    def main(test_image):
        global best_acc
        start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

        if not os.path.isdir(args.checkpoint):
            mkdir_p(args.checkpoint)

        # Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        # 创建一个转换函数来同样处理测试图像
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,  
        ])
        # 读取测试图像并转换称适合模型的格式
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_image = Image.fromarray(test_image)
        test_image = transform(test_image)

        # create model
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        elif args.arch.startswith('resnext'):
            model = models.__dict__[args.arch](
                        baseWidth=args.base_width,
                        cardinality=args.cardinality,
                    )
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

        #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #    model.features = torch.nn.DataParallel(model.features)
        #    model.cuda()
        # else:
        model = torch.nn.DataParallel(model).cuda()

        cudnn.benchmark = True
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Resume
        title = 'ImageNet-' + args.arch
        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            args.checkpoint = os.path.dirname(args.resume)
            checkpoint = torch.load(args.resume)
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


        if args.evaluate:
            print('\nEvaluation only')
            result_image = test(val_loader, model, criterion, start_epoch, use_cuda,test_image)
            return result_image

    def test(val_loader, model, criterion, epoch, use_cuda,test_image):
        global best_acc
        softmax = nn.Softmax(dim=1)
        data_time = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        bar = Bar('Processing', max=len(val_loader))
        count = 0
        info_count = 0
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # 将测试图像与添加到原本测试集输入中
            test_image = test_image.unsqueeze(0)
            inputs = torch.cat([inputs, test_image], dim=0)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            with torch.no_grad():
                _, outputs, attention = model(inputs)
                outputs = softmax(outputs)
                attention, fe, per = attention

            c_att = attention.data.cpu()
            c_att = c_att.numpy()
            d_inputs = inputs.data.cpu()
            d_inputs = d_inputs.numpy()

            in_b, in_c, in_y, in_x = inputs.shape
            for item_img, item_att in zip(d_inputs, c_att):

                v_img = ((item_img.transpose((1,2,0)) + 0.5 + [0.485, 0.456, 0.406]) * [0.229, 0.224, 0.225])* 256
                v_img = v_img[:, :, ::-1]
                resize_att = cv2.resize(item_att[0], (in_x, in_y))
                resize_att *= 255.

                cv2.imwrite('stock1.png', v_img)
                cv2.imwrite('stock2.png', resize_att)
                v_img = cv2.imread('stock1.png')
                vis_map = cv2.imread('stock2.png', 0)
                jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
                jet_map = cv2.add(v_img, jet_map)

            return jet_map

    result_image=main(test_image)
    return result_image
