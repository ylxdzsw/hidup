# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
ResNet implementation heavily inspired by the torchvision ResNet implementation
(needed for ResNeXt model implementation)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    """An implementation of a basic residual block
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BasicBlockVGG(nn.Module):
    expansion = 1
    """A BasicBlock without the residual connections
       Args:
           inplanes (int): input channels
           planes (int): output channels
           stride (int): filter stride (default is 1)
    """

    def __init__(self, in_planes, planes, stride=1, option='cifar10'):
        super(BasicBlockVGG, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, option='imagenet'):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckVGG(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, option='imagenet'):
        super(BottleneckVGG, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


# class ResNet(nn.Module):

#     def __init__(self, block, layers, num_classes= 1000):
#         super(ResNet, self).__init__()

#         self.in_planes = 64

#         ip = self.in_planes
#         self.relu = nn.ReLU(inplace=False)
#         self.conv1 = nn.Conv2d(3, ip, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_planes)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(block, ip, layers[0], stride=1)
#         self.layer2 = self._make_layer(block, ip * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, ip * 4, layers[2], stride=2)

#         self.layer4 = self._make_layer(block, ip * 8, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.linear = nn.Linear(ip * 8 * block.expansion, num_classes)

#         #Initialize the weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for i in range(len(strides)):
#             stride = strides[i]
#             layers.append(block(self.in_planes, planes, stride))
#             if i == 0: self.in_planes = planes * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x, get_features = False):
#         if get_features: features = OrderedDict()

#         out = self.relu(self.bn1(self.conv1(x)))
#         if self.layer4: out = self.maxpool(out)
#         if get_features:
#             features[0] = out.detach()

#         out = self.layer1(out)
#         if get_features:
#             features[1] = out.detach()

#         out = self.layer2(out)
#         if get_features:
#             features[2] = out.detach()

#         out = self.layer3(out)
#         if get_features:
#             features[3] = out.detach()

#         if self.layer4:
#             out = self.layer4(out)
#             if get_features:
#                 features[4] = out.detach()
#             out = self.avgpool(out)
#         else:
#             avgpool_module = nn.AvgPool2d(out.size()[3])
#             out = avgpool_module(out)
#         if get_features:
#             return features
#         out = out.view(out.size(0), -1)
#         # Fully connected layer to get to the class
#         out = self.linear(out)
#         return out



import math
import torch
import torch.nn.functional as F

class VMoE(torch.nn.Module):
    def __init__(self, nclasses, seqlen, emsize, nheads, nhid, dropout, n_expert, capacity, nlayers):
        super().__init__()
        self.emsize = emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')

        assert seqlen == 8 * 8

        self.patch_embed = PatchEmbed((32, 32), (4, 4), embed_dim=emsize)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emsize))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, seqlen + 1, emsize)) # seqlen patches + 1 cls token

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True)
            if i % 2 == 0 else
            Top2TransformerEncoderLayer(emsize, nheads, nhid, dropout, n_expert=n_expert, capacity=capacity)
            for i in range(nlayers)
        ])
        self.decoder = torch.nn.Linear(emsize, nclasses)

    def forward(self, x, get_features = False):
        # x: N, 3, 32, 32
        x = self.patch_embed(x)
        x = append_cls_token(x, self.cls_token)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = get_cls_token(x) # embedding of the class token
        x = self.decoder(x)
        return x


class Front(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = x[:, :2048]
        return x

class VMLP(torch.nn.Module):
    def __init__(self, nclasses, seqlen, emsize=2048, nheads=4, nhid=2048, dropout=0.2, nlayers=2):
        super().__init__()

        layers = [
            torch.nn.Linear(nhid, nhid)
            if i % 2 == 0 else
            torch.nn.Sigmoid()
            for i in range(nlayers)
        ]

        layers.insert(0, Front())

        self.layers = torch.nn.ModuleList(layers)

        self.decoder = torch.nn.Linear(emsize, nclasses)

    def forward(self, x, y = False):
        for layer in self.layers:
            x = layer(x)

        x = self.decoder(x)
        return x


def densenet121(pretrained=False, **kwargs):
    return VMLP(1000, 64, 2048, 4, 2048, 0.0, 8)

class Top2TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nheads, d_hidden=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, n_expert=4, capacity=None):
        super().__init__()

        self.self_attn = torch.nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

        self.gate_weight = torch.nn.Parameter(torch.empty((d_model, n_expert)))
        torch.nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

        self.w1 = torch.nn.Parameter(torch.empty((n_expert, d_model, d_hidden)))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

        self.dropout = torch.nn.Dropout(dropout)

        self.w2 = torch.nn.Parameter(torch.empty((n_expert, d_hidden, d_model)))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.n_expert = n_expert
        self.capacity = capacity
        self.activation = activation

    def forward(self, src, src_mask, src_key_padding_mask):
        """
        gate_input: (batch, seq_len, d_model)
        dispatch_tensor: (batch, seq_len, n_expert, capacity)
        expert_inputs: (batch, n_expert, capacity, d_model)
        expert_outputs: (batch, n_expert, capacity, d_model)
        combine_tensor: (batch, seq_len, n_expert, capacity)
        outputs: (batch, seq_len, d_model)
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._moe_block(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _moe_block(self, x):
        dispatch_tensor, combine_tensor = top_2_gating(x, self.n_expert, self.capacity, self.gate_weight, train=True) # (batch, seq_len, n_expert, capacity)

        expert_inputs = torch.einsum("bsd,bsec->becd", x, dispatch_tensor) # (batch, n_expert, capacity, d_model)

        h = torch.einsum("edh,becd->bech", self.w1, expert_inputs)

        h = self.activation(h)

        expert_outputs = torch.einsum("ehd,bech->becd", self.w2, h)

        output = torch.einsum("becd,bsec->bsd", expert_outputs, combine_tensor)

        return output

def top_2_gating(gate_input, n_expert, capacity, gate_weight, train = True):
    return _top_2_gating(gate_input, n_expert, capacity, gate_weight, train)

def append_cls_token(x, cls_token):
    return torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

def get_cls_token(x):
    return x[:, 0, :]

def _top_2_gating(gate_input, n_expert, capacity, gate_weight, train = True):
    gate_logits = torch.matmul(gate_input, gate_weight) # (batch, seq_len, n_expert)
    raw_gates = F.softmax(gate_logits, dim=2) # (batch, seq_len, n_expert)

    expert_gate, expert_index = torch.topk(raw_gates, k=2, dim=2, largest=True) # (batch, seq_len, 2)

    expert_mask = F.one_hot(expert_index, n_expert) # (batch, seq_len, 2, n_expert)

    position_in_expert = torch.cumsum(expert_mask, dim=1) * expert_mask # (batch, seqlen, 2, n_expert)
    expert_1_count = torch.sum(position_in_expert[:, :, 0, :], dim=1, keepdim=True) # (batch, 1, n_expert)
    position_in_expert[:, :, 1, :] += expert_1_count
    position_in_expert = position_in_expert * expert_mask

    expert_mask *= position_in_expert < capacity # (batch, seq_len, 2, n_expert)
    position_in_expert *= position_in_expert < capacity # (batch, seq_len, 2, n_expert)

    expert_mask_flat = torch.sum(expert_mask, dim=3, keepdim=False) # (batch, seq_len, 2)

    combine_tensor = torch.sum(( # (batch, seq_len, n_expert, capacity)
        torch.unsqueeze(torch.unsqueeze(expert_gate * expert_mask_flat, 3), 4) * # (batch, seq_len, 2, 1, 1)
        torch.unsqueeze(F.one_hot(expert_index, n_expert), 4) * # (batch, seq_len, 2, n_expert, 1) # TODO: why not use expert_mask?
        F.one_hot(position_in_expert, capacity)), dim=2, keepdim=False) # (batch, seq_len, 2, n_expert, capacity)

    dispatch_tensor = (combine_tensor > 0).to(torch.float32)

    return dispatch_tensor, combine_tensor

def positional_encoding(seqlen, emsize):
    import numpy as np
    p = np.array([[pos / np.power(10000, 2 * (j // 2) / emsize) for j in range(emsize)] for pos in range(seqlen)])
    p[:, 0::2] = np.sin(p[:, 0::2])
    p[:, 1::2] = np.cos(p[:, 1::2])
    return p


# https://github.com/rwightman/pytorch-image-models/blob/f670d98cb8ec70ed6e03b4be60a18faf4dc913b5/timm/models/layers/patch_embed.py#L15
class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x
