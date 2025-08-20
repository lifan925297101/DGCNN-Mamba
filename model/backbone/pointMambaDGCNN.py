import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math
from util.logger import *
from util.misc import fps  # Add import for FPS

from mamba_ssm.modules.mamba_simple import Mamba
from functools import partial
from knn_cuda import KNN
try:
    from models.block_scan import Block
except ImportError:
    from models.block import Block
from models.serialization import Point


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

# 从pointMamba.py导入必要的函数和类
def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    # ... (使用原始代码) ...
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
    except ImportError:
        norm_cls = partial(
            nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
        )
        
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block

class MixerModel(nn.Module):
    # ... (使用原始代码) ...
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out: int = 0.,
            drop_path=0.,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon, **factory_kwargs)
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids + pos

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, inference_params=inference_params
            )
            hidden_states = self.drop_out(hidden_states)

        hidden_states = self.norm_f(hidden_states)

        return hidden_states

# OrderScale相关函数
def init_OrderScale(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta

def apply_OrderScale(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

# 序列化函数
def serialization(pos, feat=None, x_res=None, order="hilbert", layers_outputs=[], grid_size=0.02):
    bs, n_p, _ = pos.size()
    if not isinstance(order, list):
        order = [order]

    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

    point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)

    order = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if feat is not None:
        feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

    for i in range(len(layers_outputs)):
        layers_outputs[i] = layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    return pos, order, inverse_order, feat, x_res


# 新的混合模型类
class PointMambaDGCNN(nn.Module):
    def __init__(self, k=20, drop_path=0.1, drop_out=0.2, rms_norm=False):
        super(PointMambaDGCNN, self).__init__()
        self.k = k
        
        # 严格按照point_base.py的设置构建DGCNN部分
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # 特征转换
        self.feat_transform = nn.Linear(1024, 256)
        
        # bin数量与point_base保持一致
        self.bin_num = [1, 2, 4, 8, 16, 32]
        
        # Mamba块的设置
        self.mamba_dims = [64, 64, 128, 256, 1024]  # 将第5层从512改为1024
        self.mamba_blocks = nn.ModuleList()
        
        # 位置编码
        self.pos_embeds = nn.ModuleList()
        for dim in self.mamba_dims:
            self.pos_embeds.append(nn.Sequential(
                nn.Linear(3, dim//2),
                nn.GELU(),
                nn.Linear(dim//2, dim)
            ))
        
        # OrderScale参数
        self.order_scales = nn.ModuleList()
        for dim in self.mamba_dims:
            gamma, beta = init_OrderScale(dim)
            self.order_scales.append(nn.ParameterList([gamma, beta]))
        
        # Mamba SSM配置
        ssm_cfg = {
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
        }
        
        # 计算drop_path的值
        dpr = [x.item() for x in torch.linspace(0, drop_path,len(self.mamba_dims))]
        
        
        # 为每一层特征创建对应的Mamba块
        for i, dim in enumerate(self.mamba_dims):
            self.mamba_blocks.append(MixerModel(
                d_model=dim,
                n_layer=1,  # 每个位置使用一个Mamba层
                ssm_cfg=ssm_cfg,
                rms_norm=rms_norm,
                residual_in_fp32=True,
                drop_out=drop_out,
                drop_path=dpr[i]
            ))
        
    def forward(self, x):  # (B, 3, 1024)
        batch_size = x.size(0)
        x0 = x
        x0_t = x0.transpose(2, 1).contiguous()
        
        # === 多层级DGCNN+Mamba融合 ===
        # DGCNN第1层
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        # Mamba增强第1层
        # x1_t = x1.transpose(2, 1)
        # pos1 = self.pos_embeds[0](x0_t)
        #_, _, _, x1_serial, pos1_serial = serialization(x0_t, x1_t, pos1, order="hilbert", layers_outputs=[])
        #x1_serial = apply_OrderScale(x1_serial, self.order_scales[0][0], self.order_scales[0][1])
        # x1 = self.mamba_blocks[0](x1_t, pos1).transpose(2, 1)
        
        
        # DGCNN第2层
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        # Mamba增强第2层
        # x2_t = x2.transpose(2, 1)
        # pos2 = self.pos_embeds[1](x0_t)
        # _, _, _, x2_serial, pos2_serial = serialization(x0_t, x2_t, pos2, order="hilbert", layers_outputs=[])
        # x2_serial = apply_OrderScale(x2_serial, self.order_scales[1][0], self.order_scales[1][1])
        # x2_mamba = self.mamba_blocks[1](x2_t, pos2).transpose(2, 1)
        
        
        # DGCNN第3层
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        # Mamba增强第3层
        # x3_t = x3.transpose(2, 1)
        # pos3 = self.pos_embeds[2](x0_t)
        # _, _, _, x3_serial, pos3_serial = serialization(x0_t, x3_t, pos3, order="hilbert", layers_outputs=[])
        # x3_serial = apply_OrderScale(x3_serial, self.order_scales[2][0], self.order_scales[2][1])
        # x3_mamba = self.mamba_blocks[2](x3_t, pos3).transpose(2, 1)

        
        # DGCNN第4层
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        
        # Mamba增强第4层
        # x4_t = x4.transpose(2, 1)
        # pos4 = self.pos_embeds[3](x0_t)
        # _, _, _, x4_serial, pos4_serial = serialization(x0_t, x4_t, pos4, order="hilbert", layers_outputs=[])
        # x4_serial = apply_OrderScale(x4_serial, self.order_scales[3][0], self.order_scales[3][1])
        # x4_mamba = self.mamba_blocks[3](x4_t, pos4).transpose(2, 1)

        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # x_mamba_l = torch.cat((x1_mamba, x2_mamba, x3_mamba, x4_mamba), dim=1)
        x_dgcnn = self.conv5(x)
        # x_mamba_l = self.conv5(x_mamba_l)
        
        # 最终Mamba处理
        x_t = x_dgcnn.transpose(2, 1)
        pos5 = self.pos_embeds[4](x0_t)
        _, _, _, x_serial_forward, pos5_forward = serialization(x0_t, x_t, pos5, order="hilbert", layers_outputs=[])
        _, _, _, x_serial_backward, pos5_backward = serialization(x0_t, x_t, pos5, order="hilbert-trans", layers_outputs=[])
        
        x_serial_forward = apply_OrderScale(x_serial_forward, self.order_scales[4][0], self.order_scales[4][1])
        x_serial_backward = apply_OrderScale(x_serial_backward, self.order_scales[4][0], self.order_scales[4][1])
        
        pos5 = torch.cat([pos5_forward, pos5_backward], dim=1)
        x_serial = torch.cat([x_serial_forward, x_serial_backward], dim=1)
        
        x_serial = self.mamba_blocks[4](x_serial, pos5)
        x_mamba = x_serial.transpose(2, 1)

        
        x = torch.cat([x_mamba,  x_dgcnn], dim=2)  # [B, 1024, 3N]

        
        
        # 特征分箱和转换
        bin_feat = []
        for bin in self.bin_num:
            z = x.view(batch_size, x.size(1), bin, -1)
            z_max, _ = z.max(3)
            z = z.mean(3) + z_max
            bin_feat.append(z)
        
        bin_feat = torch.cat(bin_feat, 2).permute(2, 0, 1).contiguous()
        bin_feat = self.feat_transform(bin_feat)
        
        return bin_feat 