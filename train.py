import os
import sys
import shutil
from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import *
from utils import generate_dboxes, Encoder
from transform import SSDTransformer
from loss import Loss
from process import train, evaluate
from datasets import CocoDataset, KittiDataset

from configs.utils import parse_config

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = ArgumentParser(description="Implementation of SSD")
    parser.add_argument('--config', type=str, help='Model config')
    parser.add_argument('--resume', type=str, help='Checkpoint to resume')
    args = parser.parse_args()

    return args


def train_detector(cfg):
    num_gpus = torch.cuda.device_count()

    train_params = {"batch_size": cfg.BATCH_SIZE * num_gpus,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": cfg.NUM_WORKERS}

    test_params = {"batch_size": cfg.BATCH_SIZE * num_gpus,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": cfg.NUM_WORKERS}

    dboxes = generate_dboxes(model="ssd")
    model = SSDConcat(backbone=ResNetParallel(), num_classes=len(cfg.KITTI_CLASSES))

    if cfg.DATASET == 'KITTI':
        train_set = KittiDataset(cfg.ROOT, "train", SSDTransformer(dboxes, (300, 300), val=False))
        test_set = KittiDataset(cfg.ROOT, "val", SSDTransformer(dboxes, (300, 300), val=True))
    elif cfg.DATASET == 'COCO':
        train_set = CocoDataset(cfg.ROOT, 2017, "train", SSDTransformer(dboxes, (300, 300), val=False))
        test_set = CocoDataset(cfg.ROOT, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))

    train_loader = DataLoader(train_set, **train_params)
    test_loader = DataLoader(test_set, **test_params)

    encoder = Encoder(dboxes)

    cfg.lr = cfg.lr * num_gpus * (cfg.batch_size / 32)
    criterion = Loss(dboxes)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay,
                                nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=cfg.multistep, gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    if os.path.isdir(cfg.LOG_PATH):
        shutil.rmtree(cfg.LOG_PATH)
    os.makedirs(cfg.LOG_PATH)

    if not os.path.isdir(cfg.SAVE_FOLDER):
        os.makedirs(cfg.SAVE_FOLDER)

    writer = SummaryWriter(cfg.log_path)

    if args.resume.endswith('.pth') and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        first_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0

    for epoch in range(first_epoch, cfg.NUM_EPOCHS):
        train(model, train_loader, epoch, writer, criterion, optimizer, scheduler)
        if epoch % cfg.SAVE_INTERVAL == 0:
            print('Evaluating ...')
            evaluate(model, test_loader, epoch, writer, encoder, cfg.NMS_THRESHOLD, cfg.CLASSES)
            
            checkpoint = {"epoch": epoch,
                      "model_state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
            checkpoint_path = os.path.join(cfg.SAVE_FOLDER, cfg.NAME + f'_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'Saved checkpoint at {checkpoint_path}')


if __name__ == "__main__":
    args = get_args()
    cfg = parse_config(args.config)
    train_detector(cfg)
