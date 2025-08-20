# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from util.dist import cos_sim

class PCIA(nn.Module):
    """
    Point-cloud Correlation Interaction (PCIA) for Few-Shot 3D Point Cloud Classification
    实现论文"A Closer Look at Few-Shot 3D Point Cloud Classification"中的创新模块
    包含三个核心组件:
    1. Salient-Part Fusion (SPF) 显著部分融合模块
    2. Self-Channel Interaction Plus (SCI+) 自通道交互增强模块 
    3. Cross-Instance Fusion Plus (CIF+) 跨实例融合增强模块
    """
    def __init__(self, k_way, n_shot, query, feat_dim=256, use_spf=True, use_sci=True, use_cif=True):
        super().__init__()
        self.loss_fn = torch.nn.NLLLoss()
        
        self.k = k_way
        self.n = n_shot
        self.query = query
        self.feat_dim = feat_dim
        
        # 使用标志，控制是否使用各个模块
        self.use_spf = use_spf
        self.use_sci = use_sci
        self.use_cif = use_cif
        
        # SPF 模块参数
        self.SPFks = 8     # SPF中局部特征的数量
        self.SPFknn = 16     # SPF中中心点的数量
        self.Conv11 = nn.Sequential(nn.Conv2d(self.SPFks + 1, 1, kernel_size=1, bias=False))
        
        # SCIp 模块参数
        self.fuse1 = nn.Sequential(
            nn.Conv2d(self.k * 1, 1, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(0.3)  # 添加二维dropout - 对通道维度进行随机置零
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # 增强FC层后的dropout
        self.FC1 = nn.Linear(32, 32, bias=False)
        self.FC2 = nn.Linear(32, 32, bias=False)
        self.FC3 = nn.Linear(32, 32, bias=False)
        
        # 基本dropout增加到0.5 (从原来的0.4)
        self.dropout = nn.Dropout(0.5)
        
        # 添加额外的dropout层用于不同模块
        self.input_dropout = nn.Dropout(0.2)  # 输入特征的dropout
        self.attn_dropout = nn.Dropout(0.4)   # 注意力机制的dropout  
        self.feature_dropout = nn.Dropout(0.3)  # 特征融合后的dropout
        
        # CIFp 模块参数
        self.CIFk = 15      # CIF中相似实例的数量
        self.CIFh = 64      # CIF中隐藏层的维度
        
        self.sq_dims = self.CIFk + 1
        self.qs_dims = self.k + 1
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(self.sq_dims, self.CIFh, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.CIFh),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(0.25)  # 添加dropout在第一个卷积后
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(self.qs_dims, self.CIFh, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.CIFh),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(0.25)  # 添加dropout在第二个卷积后
        )
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(self.CIFh, self.sq_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.sq_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(self.CIFh, self.qs_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.qs_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.Conv5 = nn.Sequential(
            nn.Conv2d(16, self.CIFh, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.CIFh),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout2d(0.25)  # 添加dropout
        )
        self.Conv6 = nn.Sequential(
            nn.Conv2d(self.CIFh, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
        # 三元组损失的参数
        self.margin = 0.2  # 设置margin为0.2
    
    def forward(self, embedding, label):
        """
        处理多个bin特征的PCIA前向传播，保持[bins, batch_size, feat_dim]的结构
        
        Args:
            embedding: 输入的特征嵌入，形状为[bins, batch_size, feat_dim]
            label: 标签 [support_label, query_label]
            
        Returns:
            y_pred: 预测结果
            loss: 三元组损失值
        """
        if len(embedding.shape) == 3:
            # [bins, batch_size, feat_dim]
            bins, batch_size, feat_dim = embedding.shape
            
            
            
            # 分别处理每个bin
            bin_prototypes = []
            bin_queries = []
            
            for bin_idx in range(bins):
                bin_embed = embedding[bin_idx]  # [batch_size, feat_dim]
                support = bin_embed[:self.n*self.k]  # [n_shot*k_way, feat_dim]
                query = bin_embed[self.n*self.k:]    # [query_num, feat_dim]
                
                # 对每个bin应用PCIA处理
                s, q, _, _ = self._pcia_forward(support, query)
                bin_prototypes.append(s)  # [way, feat_dim]
                bin_queries.append(q)     # [query_num, feat_dim]
            
            # 将结果堆叠成[bins, way, feat_dim]和[bins, query_num, feat_dim]的形状
            s = torch.stack(bin_prototypes, dim=0)  # [bins, way, feat_dim]
            q = torch.stack(bin_queries, dim=0)     # [bins, query_num, feat_dim]
            
            # 使用3D特征计算距离和损失
            dist = self.get_dist(s, q)  # [way]
            
            
            y_pred = (-dist).softmax(1)  # softmax维度改为0
            
                
            # 计算三元组损失
            loss = self.get_loss(s, q, label[1].to(s.device))
        else:
            # 对于2D输入，直接处理
            support = embedding[:self.n*self.k]
            query = embedding[self.n*self.k:]
            
            # 在训练模式下添加输入特征的dropout
            if self.training:
                support = self.input_dropout(support)
                query = self.input_dropout(query)
            
            s, q, _, _ = self._pcia_forward(support, query)
            
            # 添加bin维度以使用相同的处理逻辑
            s = s.unsqueeze(0)  # [1, way, feat_dim]
            q = q.unsqueeze(0)  # [1, query_num, feat_dim]
            
            # 计算距离和损失
            dist = self.get_dist(s, q)
            
            
            y_pred = (-dist).softmax(0)
            
                
            # 计算三元组损失
            loss = self.get_loss(s, q, label[1].to(s.device))
        
        return y_pred, loss
    
    def _pcia_forward(self, support, query, p_fea=None, coords=None):
        """
        PCIA内部实现的前向传播
        
        Args:
            support: 支持集特征 [n_shot*k_way, feat_dim]
            query: 查询集特征 [query_num, feat_dim]
            p_fea: 点特征 (SPF使用)
            coords: 坐标信息 (SPF使用)
            
        Returns:
            s: 更新后的支持集原型特征 [way, feat_dim]
            q: 更新后的查询集特征 [way*query, feat_dim]
            center_idx: 中心点索引
            local_idx: 局部特征索引
        """
        way, shot = self.k, self.n
        query_size = query.size(0)
        dims = support.size(-1)
        b = way * (shot + query_size // way)
        
       
        
        center_idx, local_idx = None, None
        
        # 如果提供了点特征，应用SPF
        if p_fea is not None and coords is not None and self.use_spf:
            # SPF (Salient-Part Fusion)
            p_fea = p_fea.transpose(2, 1)                                              
            s_mean = support.view(way, shot, dims).mean(1, keepdim=True)
            s = s_mean.expand(way, shot, dims).reshape(way * shot, dims)
            g_fea = torch.cat((s, query), dim=0).view(b, 1, dims)                           

            # 1. 获取关键点索引
            center_idx = cos_similarity_batch(g_fea, p_fea, nk=self.SPFknn).squeeze()

            # 2. KNN查找
            c_fea = index_points(p_fea, center_idx.long())       
            local_idx = knn_point(self.SPFks, p_fea, c_fea)
            grouped_fea = index_points(p_fea, local_idx) 

            # 3. 融合局部特征
            grouped_fea = grouped_fea.view(-1, self.SPFks, dims).unsqueeze(2) 

            g_fea = g_fea.expand(b, self.SPFknn, dims)
            g_fea = g_fea.reshape(-1, 1, dims).unsqueeze(2) 
            g_l_feature = torch.cat((grouped_fea, g_fea), dim=1)
            mean_part_fea = self.Conv11(g_l_feature).view(b, self.SPFknn, dims) 
            
            # 在SPF中间结果应用dropout
            if self.training:
                mean_part_fea = self.dropout(mean_part_fea)
            
            spf_out_fea = torch.max(mean_part_fea, dim=1)[0]

            # 4. 更新支持集和查询集特征
            s = (spf_out_fea[:way * shot, :].view(way, shot, dims).mean(1) + s_mean.squeeze()) / 2.0
            q = (spf_out_fea[way * shot:, :] + query) / 2.0
            
            # 对融合后的特征应用dropout
            if self.training:
                s = self.feature_dropout(s)
                q = self.feature_dropout(q)
            
        else:
            # 如果没有点特征，直接计算原型
            # 由于输入格式可能有变化，使用安全的方式计算原型
            s_reshaped = support.reshape(way, shot, -1)
            s = s_reshaped.mean(1)  # [way, feat_dim]
            q = query
        
        if self.use_sci:
            # SCIp (Self-Channel Interaction Plus)
            s_meta = self.fuse1(s.view(way, 1, 1, dims).transpose(0, 1)).view(1, 1, dims)
            s_cat = torch.cat((s.view(way, 1, -1), s_meta.repeat(way, 1, 1)), dim=1)
            q_cat = torch.cat((q.view(query_size, 1, -1), s_meta.repeat(query_size, 1, 1)), dim=1)

            s_fuse = self.fuse2(s_cat.view(way, 2, 1, dims)).view(-1, 32, dims)
            q_fuse = self.fuse2(q_cat.view(query_size, 2, 1, dims)).view(-1, 32, dims)
            
           

            s_fuse_q = self.FC1(s_fuse.transpose(2, 1))
            s_fuse_k = self.FC2(s_fuse.transpose(2, 1))
            s_fuse_v = self.FC3(s_fuse.transpose(2, 1))

            q_fuse_q = self.FC1(q_fuse.transpose(2, 1))
            q_fuse_k = self.FC2(q_fuse.transpose(2, 1))
            q_fuse_v = self.FC3(q_fuse.transpose(2, 1))

            s_att_map = torch.bmm(s_fuse_q, s_fuse_k.transpose(2, 1)) / np.power(32, 0.5)
            q_att_map = torch.bmm(q_fuse_q, q_fuse_k.transpose(2, 1)) / np.power(32, 0.5)

            

            s_att_map = F.softmax(s_att_map, dim=2)
            q_att_map = F.softmax(q_att_map, dim=2)

            s_atten = torch.bmm(s_att_map, s_fuse_v)
            q_atten = torch.bmm(q_att_map, q_fuse_v)

            s_atten = self.fuse3(s_atten.view(-1, dims, 32, 1).transpose(2, 1)).squeeze()
            q_atten = self.fuse3(q_atten.view(-1, dims, 32, 1).transpose(2, 1)).squeeze()

            s = (s_atten + s) / 2.0
            q = (q_atten + q) / 2.0
            
            
        
        if self.use_cif:
            # CIFp (Cross-Instance Fusion Plus)
            s_init = s
            q_init = q

            sq_emd_att_map = torch.mm(s, q.transpose(0, 1)) / np.power(dims, 0.5)
            qs_emd_att_map = torch.mm(q, s.transpose(0, 1)) / np.power(dims, 0.5)
            
           

            sq_emd_att_map = F.softmax(sq_emd_att_map, dim=-1)
            qs_emd_att_map = F.softmax(qs_emd_att_map, dim=-1)

            sq_atten = torch.mm(sq_emd_att_map, q)
            qs_atten = torch.mm(qs_emd_att_map, s)

            idxsq = top_cos_similarity(s, q, self.CIFk)
            idxqs = top_cos_similarity(q, s, s.size(0))

            sq_a = torch.cat((s.view(way, 1, 1, dims), q[idxsq].view(way, self.CIFk, 1, dims)), dim=1)
            qs_a = torch.cat((s[idxqs].view(query_size, way, 1, dims), q.view(query_size, 1, 1, dims)), dim=1)

            sq_aa = F.softmax(self.Conv3(self.Conv1(sq_a)).squeeze(), dim=1)
            qs_aa = F.softmax(self.Conv4(self.Conv2(qs_a)).squeeze(), dim=1)
            
           

            s_att = torch.mul(sq_aa, sq_a.squeeze()).sum(dim=1)
            q_att = torch.mul(qs_aa, qs_a.squeeze()).sum(dim=1)

            s = (s_att + s_init + sq_atten) / 3.0
            q = (q_att + q_init + qs_atten) / 3.0
            
           

            idxsq = top_cos_similarity(s, q, 15)
            sq_a = torch.cat((s.view(way, 1, 1, dims), q[idxsq].view(way, 15, 1, dims)), dim=1)
            sq_aa = F.softmax(self.Conv6(self.Conv5(sq_a)).squeeze(), dim=1)
            
            
            
            s_att = torch.mul(sq_aa, sq_a.squeeze()).sum(dim=1)

            s = (s_att + s) / 2.0
            
           
        
        return s, q, center_idx, local_idx
    
    def get_loss(self, s, q, labels):
        """
        计算损失函数 (使用新的三元组损失方法)
        保持3D结构 [bins, batch_size, feat_dim]
        
        Args:
            s: 支持集原型特征 [bins, way, feat_dim]
            q: 查询集特征 [bins, query_num, feat_dim]
            labels: 查询集标签 [query_num]
            
        Returns:
            loss: 三元组损失值
        """
        bins, way, _ = s.shape
        _, query_num, _ = q.shape
        
        # 构建所有特征 [bins, way+query_num, feat_dim]
        feature = torch.cat([s, q], dim=1)
        
        # 构建标签 [way+query_num]
        support_labels = torch.arange(way).to(s.device)
        all_labels = torch.cat([support_labels, labels], dim=0)
        
        # 复制标签到每个bin [bins, way+query_num]
        all_labels = all_labels.unsqueeze(0).repeat(bins, 1)
        
        # 构建掩码
        hp_mask = (all_labels.unsqueeze(1) == all_labels.unsqueeze(2)).bool().view(-1)  # 相同类掩码
        hn_mask = (all_labels.unsqueeze(1) != all_labels.unsqueeze(2)).bool().view(-1)  # 不同类掩码
        
        # 计算距离矩阵
        dist = self.batch_dist(feature)  # [bins, way+query_num, way+query_num]
        dist = dist.view(-1)  # 展平为一维
        
        # 获取正样本和负样本的距离
        batch_size = way + query_num
        full_hp_dist = torch.masked_select(dist, hp_mask).reshape(bins, batch_size, -1, 1)  # 正样本距离
        full_hn_dist = torch.masked_select(dist, hn_mask).reshape(bins, batch_size, 1, -1)  # 负样本距离
        
        # 计算三元组损失
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(bins, -1)
        
        # 计算非零损失的均值
        full_loss_metric_sum = torch.sum(full_loss_metric, 1)
        full_loss_num = (full_loss_metric != 0).sum(1).float()
        full_loss_mean = full_loss_metric_sum / full_loss_num
        full_loss_mean[full_loss_num == 0] = 0
        
        return full_loss_mean.mean()

    def batch_dist(self, feat):
        """计算特征之间的距离矩阵，支持3D输入"""
        return torch.cdist(feat, feat)

    def get_dist(self, prototype, queries):
        """
        计算查询样本与原型之间的距离，处理3D输入，对查询维度取平均
        无循环实现，使用批量矩阵运算
        
        Args:
            prototype: 原型特征 [bins, way, feat_dim]
            queries: 查询特征 [bins, query_num, feat_dim]
            
        Returns:
            distance: 距离矩阵 [way]，对所有bins和所有查询样本取平均
        """
        # 直接使用批量距离计算，无需循环
        # 计算每个bin中queries和prototype之间的距离
        # [bins, query_num, way]
        dist_matrix = torch.cdist(queries, prototype)
        
        # 首先对查询样本维度取平均
        dist_per_bin = torch.mean(dist_matrix, dim=0)  # [bins, way]
        
        # 然后对所有bin取平均
        
        return dist_per_bin
        
    def adapt_to_batch(self, sample_inpt, label):
        """
        适应原始代码结构的接口函数，处理[点数量, batch_size, 点维度]格式的输入
        
        Args:
            sample_inpt: 输入样本 [点数量, batch_size, 点维度]
            label: 标签 [support_label, query_label]
            
        Returns:
            预测结果和损失
        """
        # 获取形状信息
        num_points, batch_size, point_dim = sample_inpt.shape
        
        # 将点云特征转换为实例特征 [点数量, batch_size, 点维度] -> [batch_size, 点维度]
        # 这里我们简单地对所有点求平均，获取全局特征
        sample_inpt = torch.mean(sample_inpt, dim=0)  # [batch_size, 点维度]
        
        # 分离支持集和查询集
        support = sample_inpt[:self.n*self.k]  # [n_shot*k_way, 点维度]
        query = sample_inpt[self.n*self.k:]    # [query_num, 点维度]
        
        # 应用PCIA处理
        s, q, _, _ = self._pcia_forward(support, query)
        
        # 添加bin维度以使用相同的处理逻辑
        s = s.unsqueeze(0)  # [1, way, feat_dim]
        q = q.unsqueeze(0)  # [1, query_num, feat_dim]
        
        # 计算距离和损失
        dist = self.get_dist(s, q)
        y_pred = (-dist).softmax(0)  # 对维度0应用softmax
        # 使用三元组损失
        loss = self.get_loss(s, q, label[1].to(s.device))
        
        return y_pred, loss


# 辅助函数
def euclidean_dist(x, y):
    """
    计算欧氏距离
    x: N x D
    y: M x D
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def cos_similarity_batch(x, y, nk, nomal=True):
    """
    计算批量余弦相似度
    x: B x N x D
    y: B x M x D
    """
    cos = torch.bmm(F.normalize(x, dim=-1),
                    F.normalize(y, dim=-1).transpose(2, 1))
    cos = 0.5 * cos + 0.5 if nomal else cos
    index = cos.topk(nk, dim=-1)[1]
    return index


def top_cos_similarity(x, y, k, nomal=True):
    """
    计算余弦相似度并返回前k个索引
    x: N x D
    y: M x D
    """
    cos = torch.mm(F.normalize(x, dim=-1),
                   F.normalize(y, dim=-1).transpose(1, 0))
    cos = 0.5 * cos + 0.5 if nomal else cos
    index = cos.topk(k, dim=-1)[1]
    return index


def knn_point(nsample, xyz, new_xyz):
    """
    查找K近邻点
    nsample: 局部区域最大采样点数
    xyz: 所有点, [B, N, C]
    new_xyz: 查询点, [B, S, C]
    Return:
        group_idx: 分组点索引, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    group_idx = topk(sqrdists, nsample, dim=-1, largest=False)
    return group_idx


def topk(inputs, k, dim=None, largest=True):
    """求取topk"""
    if dim is None:
        dim = -1
    if dim < 0:
        dim += inputs.ndim
    transpose_dims = [i for i in range(inputs.ndim)]
    transpose_dims[0] = dim
    transpose_dims[dim] = 0
    inputs = inputs.permute(2, 1, 0)
    index = torch.argsort(inputs, dim=0, descending=largest)
    indices = index[:k]
    indices = indices.permute(2, 1, 0)
    return indices


def square_distance(src, dst):
    """
    计算两组点之间的欧氏距离
    src: 源点, [B, N, C]
    dst: 目标点, [B, M, C]
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    根据索引提取点
    points: 输入点, [B, N, C]
    idx: 采样索引, [B, S]
    new_points: 索引点, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points 