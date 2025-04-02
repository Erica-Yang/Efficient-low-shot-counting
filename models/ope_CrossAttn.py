from .mlp import MLP
from .positional_encoding import PositionalEncodingsFixed

import torch
from torch import nn

from torchvision.ops import roi_align


class OPEModule(nn.Module):

    def __init__(
        self,
        num_iterative_steps: int,
        emb_dim: int,
        kernel_dim: int,
        num_objects: int,
        num_heads: int,
        reduction: int,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
    ):

        super(OPEModule, self).__init__()

        self.num_iterative_steps = num_iterative_steps
        self.zero_shot = zero_shot
        self.kernel_dim = kernel_dim
        self.num_objects = num_objects
        self.emb_dim = emb_dim
        self.reduction = reduction

        if num_iterative_steps > 0: #3
            self.iterative_adaptation = IterativeAdaptationModule(
                num_layers=num_iterative_steps, emb_dim=emb_dim, num_heads=num_heads,
                dropout=0, layer_norm_eps=layer_norm_eps,
                mlp_factor=mlp_factor, norm_first=norm_first,
                activation=activation, norm=norm,
                zero_shot=zero_shot
            )

        if not self.zero_shot:
            self.shape_or_objectness = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, self.kernel_dim**2 * emb_dim)
            )
        else:
            self.shape_or_objectness = nn.Parameter(
                torch.empty((self.num_objects, self.kernel_dim**2, emb_dim))
            )
            nn.init.normal_(self.shape_or_objectness)

        self.pos_emb = PositionalEncodingsFixed(emb_dim)

    def forward(self, f_e, pos_emb, bboxes):
        bs, _, h, w = f_e.size()  #torch.Size([2, 256, 64, 64])
        # extract the shape features or objectness
        if not self.zero_shot:
            box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device) #torch.Size([2, 3, 2])
            box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
            box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]  #torch.Size([2, 3, 2])
            shape_or_objectness = self.shape_or_objectness(box_hw).reshape(
                bs, -1, self.kernel_dim ** 2, self.emb_dim
            ).flatten(1, 2).transpose(0, 1)  #torch.Size([27, bs, 256])
        else:
            shape_or_objectness = self.shape_or_objectness.expand(
                bs, -1, -1, -1
            ).flatten(1, 2).transpose(0, 1) #[3,9,256]--> torch.Size([27, 2, 256])

        # if not zero shot add appearance
        if not self.zero_shot:
            # reshape bboxes into the format suitable for roi_align
            bboxes = torch.cat([  #[0,1(bs)]-->[0, 0, 0, 1, 1, 1]-->[6,1]; cat(;[bs*3,4])
                torch.arange(
                    bs, requires_grad=False
                ).to(bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1),
                bboxes.flatten(0, 1),
            ], dim=1)  #[2,3,4]-->[6,5]
            appearance = roi_align( #[6, 256, 3, 3]-->[2, 27, 256]-->([27, 2, 256])
                f_e,  #torch.Size([2, 256, 64, 64])
                boxes=bboxes, output_size=self.kernel_dim,  #[6,5]; 3
                spatial_scale=1.0 / self.reduction, aligned=True  #/8
            ).permute(0, 2, 3, 1).reshape(
                bs, self.num_objects * self.kernel_dim ** 2, -1
            ).transpose(0, 1)
        else:
            appearance = None

        query_pos_emb = self.pos_emb(  #[2, 256, 64, 64]-->[2, 256, 3, 3]-->([9, 2, 256])-->[27, 2, 256]
            bs, self.kernel_dim, self.kernel_dim, f_e.device
        ).flatten(2).permute(2, 0, 1).repeat(self.num_objects, 1, 1)

        if self.num_iterative_steps > 0:
            memory = f_e.flatten(2).permute(2, 0, 1) #torch.Size([4096, bs, 256])
            all_prototypes = self.iterative_adaptation(  #torch.Size([3, 27, 2, 256])
                shape_or_objectness, appearance, memory, pos_emb, query_pos_emb
            )
        else:
            if shape_or_objectness is not None and appearance is not None:
                all_prototypes = (shape_or_objectness + appearance).unsqueeze(0)
            else:
                all_prototypes = (
                    shape_or_objectness if shape_or_objectness is not None else appearance
                ).unsqueeze(0)

        return all_prototypes #torch.Size([3, 27, bs, 256])


class IterativeAdaptationModule(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool
    ):

        super(IterativeAdaptationModule, self).__init__()

        # self.layers = nn.ModuleList([
        #     IterativeAdaptationLayer(
        #         emb_dim, num_heads, dropout, layer_norm_eps,
        #         mlp_factor, norm_first, activation, zero_shot
        #     ) for i in range(num_layers)
        # ])
        self.layers = nn.ModuleList([
            IterativeAdaptationLayer(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, zero_shot
            ) for i in range(num_layers-1)
        ])
        self.layers.append(
            IterativeAdaptationLayer2(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, zero_shot
            )
        )

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):

        output = tgt  #torch.Size([27, 2, 256])
        outputs = list()
        for i, layer in enumerate(self.layers):
            output = layer(
                output, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask
            )
            outputs.append(self.norm(output))

        return torch.stack(outputs)  #torch.Size([3, 27, 2, 256])


class IterativeAdaptationLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        zero_shot: bool
    ):
        super(IterativeAdaptationLayer, self).__init__()

        self.norm_first = norm_first
        self.zero_shot = zero_shot

        if not self.zero_shot:
            self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm3 = nn.LayerNorm(emb_dim, layer_norm_eps)
        if not self.zero_shot:
            self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if not self.zero_shot:
            self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)

        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask
    ):
        if self.norm_first:
            if not self.zero_shot:
                tgt_norm = self.norm1(tgt)
                tgt = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt_norm, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])  #torch.Size([27, 2, 256])

            tgt_norm = self.norm2(tgt)
            tgt = tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt_norm, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0])
            tgt_norm = self.norm3(tgt)
            tgt = tgt + self.dropout3(self.mlp(tgt_norm))

        else:
            if not self.zero_shot:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key=self.with_emb(appearance),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

            tgt = self.norm2(tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0]))

            tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))

        return tgt


class IterativeAdaptationLayer2(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        zero_shot: bool
    ):
        super(IterativeAdaptationLayer2, self).__init__()

        self.norm_first = norm_first
        self.zero_shot = zero_shot

        if not self.zero_shot:
            self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm3 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm4 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm5 = nn.LayerNorm(emb_dim, layer_norm_eps)
        if not self.zero_shot:
            self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        if not self.zero_shot:
            self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.enc_dec_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout)
        self.enc_dec_attn2 = nn.MultiheadAttention(emb_dim, num_heads, dropout)

        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)
        self.mlp2 = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(
        self, tgt, appearance, memory, pos_emb, query_pos_emb, tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask
    ):
        if self.norm_first:
            if not self.zero_shot:
                tgt_norm = self.norm1(tgt)
                tgt_p = tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt_norm, query_pos_emb),
                    key=self.with_emb(appearance, query_pos_emb),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0])  #torch.Size([27, 2, 256])

            tgt_norm = self.norm2(tgt_p)
            tgt_ca, A_t = self.enc_dec_attn( ##[27, bs, 256]; torch.Size([4, 27, 4096])
                query=self.with_emb(tgt_norm, query_pos_emb), #[27, bs, 256]
                key=memory+pos_emb,  #[4096,bs,256]
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
            tgt = tgt + self.dropout2(tgt_ca)
            tgt_norm = self.norm3(tgt) #[27, bs, 256]
            tgt = tgt + self.dropout3(self.mlp(tgt_norm))

            #直接用attn
            memory_norm = self.norm4(memory)
            memory = memory + self.dropout4(self.enc_dec_attn2( ##[27, bs, 256]; torch.Size([4, 27, 4096])
                query=self.with_emb(memory_norm, pos_emb), #[4096,bs,256]
                key=tgt+query_pos_emb,  #[27, bs, 256]
                value=tgt,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0])
            memory_norm = self.norm5(memory) #[27, bs, 256]
            memory = memory + self.dropout5(self.mlp2(memory_norm))

            # 用attn转置
            # A_m = torch.transpose(A_t,1,2).contiguous() #torch.Size([4, 4096, 27])
            # A_m = A_m.softmax(dim=-1)
            # memory_ca = torch.matmul(A_m, tgt_p.permute(1,0,2)).permute(1,0,2) # *[27, 4, 256]-->[4,4096,256]
            # memory = memory + self.dropout4(memory_ca)
            # memory_norm = self.norm4(memory)
            # memory = memory +self.dropout5(self.mlp2(memory_norm))

            


        else:
            if not self.zero_shot:
                tgt = self.norm1(tgt + self.dropout1(self.self_attn(
                    query=self.with_emb(tgt, query_pos_emb),
                    key=self.with_emb(appearance),
                    value=appearance,
                    attn_mask=tgt_mask,
                    key_padding_mask=tgt_key_padding_mask
                )[0]))

            tgt = self.norm2(tgt + self.dropout2(self.enc_dec_attn(
                query=self.with_emb(tgt, query_pos_emb),
                key=memory+pos_emb,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0]))

            tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))

        return tgt, memory

