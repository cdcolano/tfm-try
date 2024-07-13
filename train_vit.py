# Copyright (c) 2024 Ankan Bhunia
# This code is licensed under MIT license (see LICENSE file for details)

import os
import warnings

warnings.filterwarnings("ignore")

import time
import torch
import wandb
import sys
import copy
import math
import torch.distributed as dist
from torch import nn, optim
from tqdm import tqdm
import numpy as np
from data.dataset_e2e import get_dataloaders
import kornia
import torchvision
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, precision_score, recall_score, f1_score
from bounding_box import bounding_box as bb
from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.visualize import obtain_vis_maps
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
from transformers import ViTConfig, ViTForImageClassification

# os.environ["WANDB_API_KEY"] = "XXXX" ## enter your wandb token here.
os.environ["WANDB_MODE"] = "offline"

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://"  # default
    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_main_process():
    try:
        if dist.get_rank() == 0:
            return True
        else:
            return False
    except:
        return True

def build_vit_model(num_labels, resume_ckpt, device):
    config = ViTConfig(image_size=256, num_labels=num_labels, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = ViTForImageClassification(config)
    model = model.to(device)

    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model_state_dict"])
        if is_main_process():  
            print('model loaded successfully')

    return model

def train_vit(train_dataset, test_dataset, model, optimizer, lr_scheduler, device, wandb):
    model.train()
    for epoch in range(args.epochs):
        if is_main_process: 
            print(f'#Epoch - {epoch}')

        start_time = time.time()

        for batch in train_dataset:
            optimizer.zero_grad()
            inputs = batch['imgs'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        if is_main_process:
            print(f'Epoch Time {int(time.time() - start_time)} secs')

        lr_scheduler.step()

        if (epoch + 1) % args.save_checkpoints_every_epoch == 0 and is_main_process():
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                },
                os.path.join(args.ckpt_path, f"model_{str(epoch).zfill(6)}.pt")
            )

def main(args):
    if is_main_process(): 
        wandb.init(project="Looking3D", dir=f'./{args.exp_path}', name=args.exp_name, settings=wandb.Settings(code_dir="."))

    if args.distributed: 
        local_rank = int(os.environ['LOCAL_RANK'])

    num_mesh_images = [args.num_mesh_images, args.num_mesh_images]
    train_dataset, test_dataset = get_dataloaders(args, num_mesh_images=num_mesh_images)

    model = build_vit_model(num_labels=2, resume_ckpt=args.resume_ckpt, device=args.device)  # Adjust num_labels based on your classification task

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True
        )

    effective_lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    train_vit(train_dataset, test_dataset, model, optimizer, lr_scheduler, args.device, wandb)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='CMT-final')
    parser.add_argument('--data_path', type=str, default='/disk/scratch_ssd/s2514643/brokenchairs/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--pred_box', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lr_drop', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_wandb_logs_every_iters', type=int, default=100)
    parser.add_argument('--save_checkpoints_every_epoch', type=int, default=1)
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--num_mesh_images', type=int, default=5)
    parser.add_argument('--n_pnts', type=int, default=32)
    parser.add_argument('--no_contr_loss', action='store_true')

    args = parser.parse_args()
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    print('Experiment:', args.exp_name)

    if args.distributed:  
        init_distributed()

    args.exp_path = f'experiments/{args.exp_name}'
    args.ckpt_path = f'experiments/{args.exp_name}/checkpoints'

    if is_main_process():
        os.makedirs(args.ckpt_path, exist_ok=True)
        with open(f'experiments/{args.exp_name}/command', 'w') as f:
            f.write(" ".join(sys.argv[:]))

    main(args)
