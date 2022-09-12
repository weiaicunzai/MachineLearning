import argparse
from functools import partial
import os

import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import Caltech256
import torchvision.transforms as transforms
# torch.utils.data.random_split

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from moco import MoCo
from alexnet import alexnet



parser = argparse.ArgumentParser()
#parser.add_argument('data', metavar='DIR',
                    #help='path to dataset')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet50)')
#parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    #help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--local_rank', default=0, type=int, 
                    help='local gpu id')
#parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    #metavar='W', help='weight decay (default: 1e-4)',
                    #dest='weight_decay')
#parser.add_argument('-p', '--print-freq', default=10, type=int,
                    #metavar='N', help='print frequency (default: 10)')
#parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    #help='path to latest checkpoint (default: none)')
#parser.add_argument('--world-size', default=-1, type=int,
                    #help='number of nodes for distributed training')
#parser.add_argument('--rank', default=-1, type=int,
                    #help='node rank for distributed training')
#parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                    help='url used to set up distributed training')
#parser.add_argument('--dist-backend', default='nccl', type=str,
#                    help='distributed backend')
#parser.add_argument('--seed', default=None, type=int,
#                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
#parser.add_argument('--multiprocessing-distributed', action='store_true',
#                    help='Use multi-processing distributed training to launch '
#                         'N processes per node, which has N GPUs. This is the '
#                         'fastest way to use PyTorch for either single node or '
#                         'multi node data parallel training')

args = parser.parse_args()
# moco specific configs:
#parser.add_argument('--moco-dim', default=128, type=int,
#                    help='feature dimension (default: 128)')
#parser.add_argument('--moco-k', default=65536, type=int,
#                    help='queue size; number of negative keys (default: 65536)')
#parser.add_argument('--moco-m', default=0.999, type=float,
#                    help='moco momentum of updating key encoder (default: 0.999)')
#parser.add_argument('--moco-t', default=0.07, type=float,
#                    help='softmax temperature (default: 0.07)')


def train(args, net, train_loader):
    for epoch in range(args.epochs):
        # print(epoch)
        train_loader.sampler.set_epoch(epoch)
        loss_fn = nn.CrossEntropyLoss()
        for idx, ((img_q, img_k), label) in enumerate(train_loader):
            #print(idx)
            #print(img_q.shape)
        #    #print(idx, len(image), len(label))
            gpu_idx = torch.distributed.get_rank()
            #print(gpu_idx, 'dfdfdfdfdfdfdfdf', )
            #print(img_q.shape, gpu_idx, idx)
            logits = net(img_q.cuda(), img_k.cuda())
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = loss_fn(logits, labels)
            print(loss)
            loss.backward()


            #print(output.shape)

    # import sys; sys.exit()

# def init_distributed():

#     # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#     # dist_url = "env://" # default

#     # only works with torch.distributed.launch // torch.run
#     rank = int(os.environ["RANK"])
#     # print(rank, 'rank')
#     world_size = int(os.environ['WORLD_SIZE'])
#     local_rank = int(os.environ['LOCAL_RANK'])
#     # print(local_rank)

#     dist.init_process_group(
#             backend="nccl",
#             # init_method=dist_url,
#             world_size=world_size,
#             rank=rank)

#     # this will make all .cuda() calls work properly
#     torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    # setup_for_distributed(rank == 0)

def main():

    # print('here?')
    # init_distributed()
    # import sys; sys.exit()

    #world_size = 2
    #rank = 0
    rank = int(os.environ["RANK"])
    # print(rank, 'rank')
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend='nccl', 
        # init_method = "env://",
        world_size=world_size, 
        rank=rank)
    # print(dist.get_rank())
    # import sys; sys.exit()

    

    def to_rgb(img):
        return img.convert('RGB')

    train_transform = transforms.Compose([
                # transforms.ToPILImage(),
                to_rgb,
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            ])

    def two_crop(image, trans):
        img_q = trans(image)
        img_k = trans(image)
        return img_q, img_k


    #test_transform = transforms.Compose([
    #            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    #            transforms.RandomGrayscale(p=0.2),
    #            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    #            transforms.RandomHorizontalFlip(),
    #            transforms.ToTensor(),
    #            transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    #        ])



    # print(train_transform)
    dataset = Caltech256(
        'data', 
        transform=partial(two_crop, trans=train_transform),
        download=True,
    )

    #for i in dataset:
    #    print(i[0][0].shape, i[0][1].shape, i[1])

    # import sys; sys.exit()
        #print(len(i))
        #print(i[0].shape, i)
    # print(len(dataset))

    # train_num = int(len(dataset) * 0.8)
    #train_set, test_set = torch.utils.data.random_split(
    #    dataset, 
    #    [train_num, len(dataset) - train_num],
    #    generator=torch.Generator().manual_seed(42))

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
    # test_loader = torch.utils.data.DataLoader(test_set)
    sampler = DistributedSampler(dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2 ** 4, 
        pin_memory=True, 
        drop_last=True,
        num_workers=8, 
        sampler=sampler)

    torch.cuda.set_device(local_rank)


    #net = alexnet(128)
    net = MoCo(alexnet)
    # print(rank, 'ranking....')
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda()
    # net
    net = DDP(net, device_ids=[rank])
    # print(net)
    # print(local_rank, rank, 'dffffff', next(net.parameters()).device)
    # import sys; sys.exit()

    train(args, net, train_loader)


    #for epoch in range()

if __name__ == '__main__':
    # import torchvision.models as models
    # net = models.__dict__['resnet50']
    # print(net)
    main()