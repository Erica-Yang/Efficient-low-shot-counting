from .backbone import Backbone
from .transformer import Transformer
from .ope import OPEModule
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor

import torch
from torch import nn
from torch.nn import functional as F

#GDINO测试
from GroundingDINO.groundingdino.util.inference import load_model2


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

        self.emb_dim = emb_dim # 256
        self.num_objects = num_objects  #3
        self.reduction = reduction #8
        self.kernel_dim = kernel_dim #3
        self.image_size = image_size #512
        self.zero_shot = zero_shot #False
        self.num_heads = num_heads #8
        self.num_encoder_layers = num_encoder_layers #3

        # self.backbone = Backbone(
        #     backbone_name, pretrained=True, dilation=False, reduction=reduction,
        #     swav=swav_backbone, requires_grad=train_backbone
        # )
        CONFIG_PATH = "/root/loca/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"    #源码自带的配置文件
        CHECKPOINT_PATH = "/root/loca/GroundingDINO/groundingdino_swint_ogc.pth"   #下载的权重文件
        self.backbone = load_model2(CONFIG_PATH, CHECKPOINT_PATH).backbone[0]
        # self.backbone.num_channels = 1344
        self.backbone.num_channels = [192, 384, 768]
        
        # self.input_proj = nn.Conv2d(
        #     self.backbone.num_channels, emb_dim, kernel_size=1
        # )
        num_feature_levels = 3 #SwinT取3 level特征
        if num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.emb_dim, kernel_size=1),
                        nn.GroupNorm(32, self.emb_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, self.emb_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, self.emb_dim),
                    )
                )
                in_channels = self.emb_dim
            self.input_proj = nn.ModuleList(input_proj_list)

        self.feature_map_proj = nn.Conv2d((256 + 256 + 256), self.emb_dim, kernel_size=1)

        if num_encoder_layers > 0:
            # self.encoder = TransformerEncoder(
            #     num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
            #     mlp_factor, norm_first, activation, norm
            # )
            #测试deformable
           self.encoder = Transformer(
                d_model=256,
                nhead=8,
                num_encoder_layers=2,
                num_unicoder_layers=0,
                dim_feedforward=2048,
                dropout=0.0,
                activation="relu",
                normalize_before=False,
                query_dim=4,
                num_patterns=0,
                # for deformable encoder
                num_feature_levels=3,
                enc_n_points=4,
                # two stage
                two_stage_type="standard",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                use_transformer_ckpt=True
           )
            
        self.ope = OPEModule( #3, 256,       3;            3;           8;
            num_ope_iterative_steps, emb_dim, kernel_dim, num_objects, num_heads,
            reduction, layer_norm_eps, mlp_factor, norm_first, activation, norm, zero_shot
        )  # 8;         1e-05;          8;  

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction)
            for _ in range(num_ope_iterative_steps - 1)
        ])

        self.pos_emb = PositionalEncodingsFixed(emb_dim)

    def combine_features(self, features): ##[4, 256, 64,64],[4, 256, 32,32]，[4, 256, 16,16]
        # (bs, c, h, w) = ( #1,256,100,150
        #     features[0].decompose()[0].shape[-4],
        #     features[0].decompose()[0].shape[-3],
        #     features[0].decompose()[0].shape[-2],
        #     features[0].decompose()[0].shape[-1],
        # )

        bs, c, h, w = features[0].size()

        x = torch.cat(
            [
                F.interpolate(
                    features[i],
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
                for i in range(len(features))
            ],
            dim=1,
        ) #torch.Size([1, 1792, 100, 150])

        x = self.feature_map_proj(x)

        return x #torch.Size([bs, 256, 64, 64])

    def forward(self, x, bboxes):
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects
        # backbone
        backbone_features = self.backbone(x) #[4,3,512,512]--> //torch.Size([4, 3584, 64, 64]);torch.Size([4, 1344, 64, 64])
        # prepare the encoder input
        # src = self.input_proj(backbone_features) #torch.Size([4, 256, 64, 64])
        # bs, c, h, w = src.size() #[4, 256, 64, 64]
        # pos_emb = self.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1) #[4,256,64,64]-->([4096, 4, 256])
        # src = src.flatten(2).permute(2, 0, 1)

        #多level准备
        srcs = []
        masks = []
        poss = []
        for l in range(len(backbone_features)):
            src, mask = backbone_features[l].decompose() #[1, 256, 100, 150],[1,100,150]
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
            bs, c, h, w = src.size()
            poss.append(self.pos_emb(bs, h, w, src.device))

        # push through the encoder
        if self.num_encoder_layers > 0:
            image_features = self.encoder(srcs, masks, poss) #torch.Size([4096, 2, 256]) /#[4, 256, 64,64],[4, 256, 32,32]，[4, 256, 16,16]
        else:
            image_features = src

        #
        combined_features = self.combine_features(image_features) #原src [bs,256,64,64]
        # prepare OPE input
        # f_e = combined_features.permute(1, 2, 0).reshape(-1, self.emb_dim, h, w) #torch.Size([2, 256, 64, 64])
        f_e = combined_features #torch.Size([bs, 256, 64, 64])
        bs, _, h2, w2 = f_e.size() #[4, 256, 64, 64]
        pos_emb = self.pos_emb(bs, h2, w2, f_e.device).flatten(2).permute(2, 0, 1) #[4096，4，256]
        all_prototypes = self.ope(f_e, pos_emb, bboxes)  #torch.Size([3, 27, bs, 256])

        outputs = list()
        for i in range(all_prototypes.size(0)):  #3
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                bs, num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]  #torch.Size([1536, 1, 3, 3]) #//torch.Size([3072, 1, 3, 3])
            # [27,2,256]-->[2,27,256]-->[2,3,3,3,256]-->[bs,3,256,3,3]-->[1536, 3, 3]-->[1536, 1, 3, 3]
            # response_maps = F.conv2d(
            #     torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),  #[1, 1536, 64, 64]/ [1, 3072, 64, 64]
            #     prototypes,
            #     bias=None,
            #     padding=self.kernel_dim // 2,
            #     groups=prototypes.size(0)
            # ).view(
            #     bs, num_objects, self.emb_dim, h2, w2
            # ).max(dim=1)[0]  #torch.Size([2, 256, 64, 64])
            #用DAVE的softmax
            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),  #[1, 1536, 64, 64]/ [1, 3072, 64, 64]
                prototypes,
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)
            ).view(
                bs, num_objects, self.emb_dim, h2, w2
            ) #[bs,3,256,64,64]

            softmaxed_correlation_maps = response_maps.softmax(dim=1) #[bs, 3, 256, 64, 64]
            response_maps = torch.mul(softmaxed_correlation_maps, response_maps).sum(dim=1) #torch.Size([bs, 256, 64, 64])
            # send through regression heads
            if i == all_prototypes.size(0) - 1:  #2
                predicted_dmaps = self.regression_head(response_maps)
            else:
                predicted_dmaps = self.aux_heads[i](response_maps) #[4,1,512,512]
            outputs.append(predicted_dmaps)

        return outputs[-1], outputs[:-1]


def build_model(args):

    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    return LOCA(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers,
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
