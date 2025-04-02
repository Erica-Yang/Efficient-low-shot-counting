from .backbone import Backbone
from .transformer import TransformerEncoder
from .ope import OPEModule
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor

import torch
from torch import nn
from torch.nn import functional as F
#import PResNet; encoder
from .presnet import PResNet
from .hybrid_encoder import HybridEncoder


class LOCA(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_encoder_layers: int,
        num_ope_iterative_steps: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        backbone_name: str,
        swav_backbone: bool,
        train_backbone: bool,
        reduction: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
    ):

        super(LOCA, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers

        #RT-DETR的backbone
        self.backbone = PResNet(
            depth=50, variant='d', num_stages=4, 
            return_idx=[1, 2, 3],act='relu',freeze_at=4,
            freeze_norm=True, pretrained=True)
        self.backbone.num_channels=[512, 1024, 2048] #swinT/RT; loca:
        # self.backbone.num_channels=3584 #swinT/RT; loca:

        # self.input_proj = nn.Conv2d(
        #     self.backbone.num_channels, emb_dim, kernel_size=1
        # )

        self.feature_map_proj = nn.Conv2d((256 + 256 + 256), self.emb_dim, kernel_size=1)

        if num_encoder_layers > 0: #注意eval时eval_spatial_size
            self.encoder = HybridEncoder( #只使用了一层HybridEncoder
                in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], 
                hidden_dim=256, nhead=8,dim_feedforward = 1024,
                dropout=0.0, enc_act='gelu', use_encoder_idx=[2],
                num_encoder_layers=1, pe_temperature=10000,
                expansion=1.0, depth_mult=1.0,
                act='silu', eval_spatial_size=[640, 640]) #[512,512]/[640, 640]
            encoder2_num_encoder_layers = 1
            self.encoder2 = TransformerEncoder(
                encoder2_num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )

        self.ope = OPEModule(
            num_ope_iterative_steps, emb_dim, kernel_dim, num_objects, num_heads,
            reduction, layer_norm_eps, mlp_factor, norm_first, activation, norm, zero_shot
        )

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction)
            for _ in range(num_ope_iterative_steps - 1)
        ])

        self.pos_emb = PositionalEncodingsFixed(emb_dim)

    def forward(self, x, bboxes):
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects
        # backbone
        backbone_features = self.backbone(x) #[2, 512, 64, 64]/[2, 1024, 32, 32]/[2, 2048, 16, 16]
        # prepare the encoder input
        
        # push through the encoder
        if self.num_encoder_layers > 0:
            image_features = self.encoder(backbone_features) #[2, 256, 64, 64]/[2, 256, 32, 32]/[2, 256, 16, 16]
        else:
            image_features = backbone_features

        bs, c, h, w = image_features[0].size()
        image_features = torch.cat(
            [
                F.interpolate(
                    image_features[i],
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
                for i in range(len(image_features))
            ],
            dim=1,
        ) #[1, 1792, 100, 150])/640:[4, 768, 80, 80])

        # prepare OPE input
        f_e = self.feature_map_proj(image_features) #torch.Size([2, 256, 64, 64])
        f_e = self.encoder2(f_e)
        
        pos_emb = self.pos_emb(bs, h, w, f_e.device).flatten(2).permute(2, 0, 1) #[4,256,64,64]-->([4096, 4, 256])
        all_prototypes = self.ope(f_e, pos_emb, bboxes)

        outputs = list()
        for i in range(all_prototypes.size(0)):
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                bs, num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]  #torch.Size([1536, 1, 3, 3])

            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
                prototypes,
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)
            ).view(
                bs, num_objects, self.emb_dim, h, w
            ).max(dim=1)[0] #torch.Size([2, 256, 64, 64])

            # send through regression heads
            if i == all_prototypes.size(0) - 1:
                predicted_dmaps = self.regression_head(response_maps)
            else:
                predicted_dmaps = self.aux_heads[i](response_maps)
            outputs.append(predicted_dmaps) #[bs,1,512,512]

        return outputs[-1], outputs[:-1]


def build_model(args):

    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    return LOCA(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers, #hybrid_encoder layer
        num_ope_iterative_steps=args.num_ope_iterative_steps,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        swav_backbone=args.swav_backbone,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        dropout=args.dropout,
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first=args.pre_norm,
        activation=nn.GELU,
        norm=True,
    )