from models.loca import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser

import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import cv2
import numpy as np

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

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    # state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))['model']
    state_dict = torch.load('/root/loca/pretrained/loca_few_shot.pt')['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    for split in ['val', 'test']:
        test = FSC147Dataset(
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
        for img, bboxes, density_map in test_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)

            out, _ = model(img, bboxes)  #[bs,1,512,512]
            ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()
            se += ((
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ) ** 2).sum()
            #可视化
            cnt_pred = round(out.sum().item(), 1)
            #测试 pred dmap  '/root/loca/self_test/eval_zeroshot_results/npys/5744.npy'
            # dmap_path = os.path.join('/root/loca/self_test/eval_fewshot_results/npys',image_name+'.npy')
            # out = out.squeeze(0)
            # np.save(dmap_path,out.squeeze().cpu().numpy())
            
            output = out.permute(1, 2, 0)  # c x h x w -> h x w x c torch.Size([512, 512, 1])
            output = output.cpu().detach().numpy()
            output = cv2.resize(output, (512, 512))
            #sctivation_fn            
            output = 1 / (1 + np.exp(-output))
            #normalization:
            output = (output - output.min()) / (output.max() - output.min())
            #with_image:
            cnt_gt = int(density_map.sum().round()) #9
            # cnt_pred = round(output.sum().item(), 1) #0.0
            resname = "{}_gt{}_pred{}.png".format(image_name, cnt_gt, cnt_pred)
            filepath = os.path.join('/root/loca/self_test/eval_fewshot_results', split, resname) #'/root/SAFECount/experiments/FSC147/vis/6292_gt9_pred0.0.png'
            img_path = os.path.join('/root/loca/data/FSC147/images_384_VarV2', img_name[0])
            image = cv2.imread(img_path)
            image = cv2.resize(image,(512,512))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = apply_scoremap(image, output, alpha=0.5)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.putText(output, str(cnt_gt), (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(output, str(cnt_pred), (image.shape[1] - 100, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(filepath, output)

        dist.all_reduce_multigpu([ae])
        dist.all_reduce_multigpu([se])

        if rank == 0:
            print(
                f"{split.capitalize()} set",
                f"MAE: {ae.item() / len(test):.2f}",
                f"RMSE: {torch.sqrt(se / len(test)).item():.2f}",
            )

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    # 自己测试
    args.swav_backbone = True
    args.pre_norm = True
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    evaluate(args)
