import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec, get_norm



class NeRFSemSegFPNHead(nn.Module):

    def __init__(self, args, feature_strides=[2,4,8,16], feature_channels=[128,128,128,128], num_classes = 20):
        super(NeRFSemSegFPNHead, self).__init__()

        self.n_p = args.n_p
        conv_dims = 128
        self.scale_heads = nn.ModuleList()
        self.common_stride = 2
        for stride, channels in zip(feature_strides, feature_channels):
            head_ops = []
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                norm_module = get_norm('GN', conv_dims)
                conv = Conv2d(
                    channels if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not 'GN',
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if stride != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
        
        self.predictor = Conv2d(conv_dims, num_classes + 1, kernel_size=1, stride=1, padding=0)

    def original_index_to_downsampled_index(self, original_index):
        original_width, downsample_factor = 320, 2
        original_row = (original_index-1) // original_width
        original_col = (original_index-1) % original_width
        
        downsampled_row = original_row // downsample_factor
        downsampled_col = original_col // downsample_factor
        
        downsampled_index = downsampled_row * (original_width // downsample_factor) + downsampled_col
        
        return downsampled_index
    
    def forward(self, deep_feats, agg_sem_feats, select_inds):

        #######   replace feature map           #######
        if select_inds is not None:
            agg_feats_3d = agg_sem_feats['feats_out_3d']
            agg_feats_2d = agg_sem_feats['feats_out']
            depth_weights = agg_sem_feats['weights']
            deep_feats = deep_feats.reshape(1, deep_feats.shape[1], -1).squeeze(0).permute(1,0)

            re_select_inds = []
            for select_ind in select_inds:
                re_select_inds.append(self.original_index_to_downsampled_index(select_ind))
            
            # distill loss
            device = deep_feats.device
            novel_feats = deep_feats[re_select_inds].detach()
            loss_distillation = F.cosine_embedding_loss(novel_feats, agg_feats_2d, torch.ones((len(re_select_inds))).to(device), reduction='mean')

            # max_weights_inds = depth_weights.argsort(dim=-1)[:,:5]
            max_weights_inds = depth_weights.argmax(dim=-1).unsqueeze(-1)
            for i in range(1, self.n_p + 1):
                max_weights_inds[max_weights_inds == 63] -= i         # 63 for sampled points in a ray (64 - 1)
                max_weights_inds[max_weights_inds == 63 - i] -= 1
                max_weights_inds[max_weights_inds == i] += 1
                max_weights_inds[max_weights_inds == 0] += i

            offsets = torch.arange(-self.n_p, self.n_p + 1)
            max_weights_inds = torch.cat([max_weights_inds + offset for offset in offsets], dim=1)
            similarity_targets = torch.zeros_like(depth_weights).to(device)
            similarity_targets[np.arange(512)[:, np.newaxis], max_weights_inds]=1

            loss_depth_guided_sem = F.cosine_embedding_loss(novel_feats.unsqueeze(1).repeat(1, 64, 1).reshape(-1, 512), agg_feats_3d.reshape(-1, 512), similarity_targets.reshape(-1), reduction='mean')
        else:
            deep_feats = deep_feats.reshape(1, deep_feats.shape[1], -1).squeeze(0).permute(1,0)

        ####### constrcut feature pyramids and Decoder  #######
        chunks = torch.chunk(deep_feats, 4, dim=1)
        for i, chunk in enumerate(chunks):
            chunk = chunk.reshape(120, 160, chunk.shape[-1]).permute(2,0,1).unsqueeze(0) # 1, 512, h, w
            if i == 0:
                x = self.scale_heads[i](chunk)
            else:
                chunk = F.interpolate(chunk, scale_factor = 1/(2**i), mode='bilinear', align_corners=True, recompute_scale_factor=True)
                x = x + self.scale_heads[i](chunk)

        out = self.predictor(x)
        out = F.interpolate(out, scale_factor = 2, mode='bilinear', align_corners=True)
        if select_inds is not None:
            return out, loss_distillation, loss_depth_guided_sem
        else:
            return out
