from models.loca import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser
from utils.losses import ObjectNormalizedL2Loss

from time import perf_counter
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist

import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def train(args):
    
    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        # world_size = int(os.environ['WORLD_SIZE'])
        # rank = int(os.environ['RANK'])
        # gpu = int(os.environ['LOCAL_RANK'])
        world_size = 1
        rank = 0
        gpu = 0

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    # model = DistributedDataParallel(
    #     build_model(args).to(device),
    #     device_ids=[gpu],
    #     output_device=gpu
    # )
    model = build_model(args).to(device)

    backbone_params = dict()
    non_backbone_params = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
        else:
            non_backbone_params[n] = p
    #测试;查看参数更新
    # for name, param in model.named_parameters():
    #     print(name + ' ' + str(param.requires_grad)) 
    # t0 = torch.load('/root/loca/pre_models/rtdetr_r50vd_6x_coco_from_paddle.pth')
    pretrained_dict_feat = {k.split("backbone.")[1]: v for k, v in
        torch.load('/root/loca/pre_models/rtdetr_r50vd_6x_coco_from_paddle.pth')['ema'][
            'module'].items() if 'backbone' in k}
    model.backbone.load_state_dict(pretrained_dict_feat)    

    optimizer = torch.optim.AdamW(
        [
            {'params': non_backbone_params.values()},
            {'params': backbone_params.values(), 'lr': args.backbone_lr}
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.1)
    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best = checkpoint['best_val_ae']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best = 10000000000000

    criterion = ObjectNormalizedL2Loss()

    train = FSC147Dataset(
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects, #3
        tiling_p=args.tiling_p, # 0.5
        zero_shot=args.zero_shot
    )
    val = FSC147Dataset(
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p
    )
    train_loader = DataLoader(
        train,
        sampler=DistributedSampler(train),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val,
        sampler=DistributedSampler(val),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers
    )

    print(rank)
    for epoch in range(start_epoch + 1, args.epochs + 1): # 200
        if rank == 0:
            start = perf_counter()
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)
        aux_train_loss = torch.tensor(0.0).to(device)
        aux_val_loss = torch.tensor(0.0).to(device)
        train_ae = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)

        train_loader.sampler.set_epoch(epoch)
        model.train()
        for img, bboxes, density_map in train_loader:
            img = img.to(device)  #torch.Size([4, 3, 512, 512])
            bboxes = bboxes.to(device)  # [4,3,4]
            density_map = density_map.to(device)  #torch.Size([4, 1, 512, 512])

            optimizer.zero_grad()
            out, aux_out = model(img, bboxes) #[2, 1, 512, 512], 0: 1:

            # obtain the number of objects in batch
            with torch.no_grad():
                num_objects = density_map.sum() #tensor(100.0349, device='cuda:0')
                # dist.all_reduce_multigpu([num_objects])

            main_loss = criterion(out, density_map, num_objects)
            aux_loss = sum([
                args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
            ])
            loss = main_loss + aux_loss
            loss.backward()
            if args.max_grad_norm > 0: #0.1
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  #0.1
            optimizer.step()

            train_loss += main_loss * img.size(0)
            aux_train_loss += aux_loss * img.size(0)
            train_ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()

        model.eval()
        with torch.no_grad():
            for img, bboxes, density_map in val_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)
                out, aux_out = model(img, bboxes)
                with torch.no_grad():
                    num_objects = density_map.sum()
                    # dist.all_reduce_multigpu([num_objects])

                main_loss = criterion(out, density_map, num_objects)
                aux_loss = sum([
                    args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
                ])
                loss = main_loss + aux_loss

                val_loss += main_loss * img.size(0)
                aux_val_loss += aux_loss * img.size(0)
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()

        # dist.all_reduce_multigpu([train_loss])
        # dist.all_reduce_multigpu([val_loss])
        # dist.all_reduce_multigpu([aux_train_loss])
        # dist.all_reduce_multigpu([aux_val_loss])
        # dist.all_reduce_multigpu([train_ae])
        # dist.all_reduce_multigpu([val_ae])

        scheduler.step()

        if rank == 0:
            end = perf_counter()
            best_epoch = False
            if val_ae.item() / len(val) < best:
                best = val_ae.item() / len(val)
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_ae': val_ae.item() / len(val)
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.model_path, f'{args.model_name}.pt')
                )
                best_epoch = True

            print(
                f"Epoch: {epoch}",
                f"Train loss: {train_loss.item():.3f}",
                f"Aux train loss: {aux_train_loss.item():.3f}",
                f"Val loss: {val_loss.item():.3f}",
                f"Aux val loss: {aux_val_loss.item():.3f}",
                f"Train MAE: {train_ae.item() / len(train):.3f}",
                f"Val MAE: {val_ae.item() / len(val):.3f}",
                f"Epoch time: {end - start:.3f} seconds",
                'best' if best_epoch else ''
            )

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    args.swav_backbone = True
    args.pre_norm = True
    args.batch_size = 4
    args.image_size = 640
    #enc_layers: 
    args.num_enc_layers = 3 #待测试 【没有作用】。 1hyEnc_frozenBbone_bs4
    args.model_path = '/root/loca/output/640_size/PResNet640'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    train(args)
