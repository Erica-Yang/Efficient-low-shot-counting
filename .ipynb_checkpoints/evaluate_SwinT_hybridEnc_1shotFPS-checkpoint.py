from models.loca import build_model
from utils.data import FSC147Dataset, FSC147Dataset_Val
from utils.arg_parser import get_argparser

import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist


@torch.no_grad()
def evaluate(args):

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

    # state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))['model']
    state_dict = torch.load('/root/loca/output/SwinT_hybridEnc_1shot/locaRT_1_shot.pt')['model']
    # state_dict = {k[7:] if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # #计算量【一】；用这个
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: {:.2f}M".format(total_params / 1e6))
    print("Trainable parameters: {:.2f}M".format(trainable_params / 1e6))
    #===========================
    # # 计算量【二】; GFLOPs分析 (thop)
    from thop import profile
    # import torchsummary

    input1 = torch.randn(1, 3, 512,512).to(device)
    input2 = torch.randn(1, 1, 4).to(device)
    # input2 = torch.tensor([[[217.3154,  48.9602, 277.1949,  63.2716],
    #      [115.7221,  45.9473, 176.2744,  61.0119],
    #      [ 24.8937, 100.1801,  92.1740, 114.4915]]]).to(device)
    flops, params = profile(model, inputs=(input1,input2))

    print("参数量(M)：", params/1e6)
    print("FLOPS：", flops)
    print("GFLOPs(G)：", flops/1e9)

    for split in ['val', 'test']:
    # for split in ['val']:
        test = FSC147Dataset_Val(
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
        )
        test_loader = DataLoader(
            test,
            sampler=DistributedSampler(test),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()

        #计算FPS
        import time
        total_forward_time = 0.0  # 使用time来测试
        # 记录开始时间
        start_event = time.time()
        t_all = []
        for img, bboxes, density_map in test_loader:
            
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            
            start_forward_time = time.time()

            out, _ = model(img, bboxes)  #[bs,1,512,512]

            end_forward_time = time.time()
            forward_time = end_forward_time - start_forward_time
            # t_all.append(forward_time)
            total_forward_time += forward_time # 转换为毫秒

            # c_sum = out.sum()
            # ae += torch.abs(
            #     density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            # ).sum()
            # se += ((
            #     density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            # ) ** 2).sum()

        # dist.all_reduce_multigpu([ae])
        # dist.all_reduce_multigpu([se])
            
        # 记录结束时间
        end_event = time.time() 
        elapsed_time = (end_event - start_event)  # 转换为秒
        # num_imgs = len(test_loader)
        # num_iterations =num_imgs
        # fps = test_loader.sampler.num_samples / elapsed_time
        # elapsed_time_ms = elapsed_time / (1 * test_loader.sampler.num_samples)
        fps = test_loader.sampler.num_samples/total_forward_time
        avg_forward_time = total_forward_time*1000 / (1 * test_loader.sampler.num_samples)

        # print('average fps:',1 / np.mean(t_all))
        print(f"FPS: {fps}")
        print("elapsed_time_ms:", elapsed_time * 1000)
        print(f"Avg Forward Time per Image: {avg_forward_time} ms")


        # if rank == 0:
        #     print(
        #         f"{split.capitalize()} set",
        #         f"MAE: {ae.item() / len(test):.2f}",
        #         f"RMSE: {torch.sqrt(se / len(test)).item():.2f}",
        #     )

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    # 自己测试
    args.swav_backbone = True
    args.pre_norm = True
    args.num_objects = 1
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    evaluate(args)
