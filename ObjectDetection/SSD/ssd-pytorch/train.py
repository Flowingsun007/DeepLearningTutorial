from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import time
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def weights_init(m):
    """conv2d层权重初始化"""
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()


def frozen_basenet(base_net, flag=True):
    if(flag):
        for param in base_net.parameters():
            param.requires_grad = False


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--frozen', default=True,
                    help='Frozen base net during training')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=6, type=int,
                    help='Number of workers used in loading data')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--logdir', default='data/logs',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

model_env = torch.device("cpu")  # 初始模型训练环境为cpu
if torch.cuda.is_available():
    if args.cuda:  # args.cuda = True则训练环境为gpu:0
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model_env = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def create_dataset():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    print('Loading the dataset...')
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    return cfg, data_loader


def build_network(cfg):
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    if args.cuda:       # 将模型放入指定的训练环境(cpu或gpu)
        ssd_net.to(model_env)

    # 从checkpoint开始训练
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
        base_net = ssd_net.vgg
        frozen_basenet(base_net, args.frozen)

    # 从头开始训练(迁移训练) >> 加载backbone vgg网络预训练权重
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        base_net = ssd_net.vgg
        base_net.load_state_dict(vgg_weights)
        frozen_basenet(base_net, args.frozen)
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    return ssd_net


def train():

    # 初始化data_loader、cfg
    cfg, data_loader = create_dataset()
    # build network
    ssd_net = build_network(cfg)
    # tf log
    writer = SummaryWriter(args.logdir)
    # 优化器
    optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # 损失函数
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    ssd_net.train()
    # loss counters
    loc_loss, conf_loss, step_index = 0, 0, 0

    batch_iterator = iter(data_loader)  # create batch dataset
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        # images, targets = next(batch_iterator)
        try:
            images, targets = next(batch_iterator)
        except StopIteration as e:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        # forward得到网络输出
        t0 = time.time()

        out = ssd_net(images)
        optimizer.zero_grad()   # 清空优化器之前累积的梯度
        loss_l, loss_c = criterion(out, targets)  # 将输出的out和target输入损失函数并前向传播，得到box回归和分类conf的loss
        loss = loss_l + loss_c
        loss.backward()         # 总损失应用backward反向传播
        optimizer.step()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        # print loss and write tf log
        t1 = time.time()
        lr = optimizer.param_groups[0]['lr']
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('lr: ' + str(lr) + ' || iter: ' + repr(iteration) +
                  ' || box_loss: %.4f || conf loss: %.4f || total loss: %.4f ||' % (loss_l, loss_c, loss.item()), end=' ')
            writer.add_scalar("lr", lr, global_step=iteration)
            writer.add_scalar("loss/box_loss", loss_l, global_step=iteration)
            writer.add_scalar("loss/conf_loss", loss_c, global_step=iteration)
            writer.add_scalar("loss/total_loss", loss, global_step=iteration)
            writer.flush()

        # 每迭1000轮多少轮存一次模型
        if iteration != 0 and iteration % 1000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_voc_iter' + repr(iteration) + '_loss' + loss.item() + '.pth')
    # 保存最终模型
    torch.save(ssd_net.state_dict(), args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
