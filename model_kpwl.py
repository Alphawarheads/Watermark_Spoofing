import random
import sys
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq

class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        cp_path = '/public/home/qinxy/zhangzs/SLSforASVspoof-2021-DF-main/SLSforASVspoof-2021-DF-main/models/xlsr2_300m.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        self._frozen = False  # 冻结标记

    def set_freeze(self, freeze: bool):
        """供训练脚本调用：冻结/解冻前端并切换 train/eval。"""
        freeze = bool(freeze)
        if self._frozen == freeze:
            return
        self._frozen = freeze
        for p in self.model.parameters():
            p.requires_grad_(not freeze)
        if freeze:
            self.model.eval()
        else:
            self.model.train()

    def extract_feat(self, input_data: Tensor):
        # 确保设备/精度一致
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            # 保持与冻结状态一致
            self.model.eval() if self._frozen else self.model.train()

        # 输入整理
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # 冻结时不建图，解冻时保留梯度
        if self._frozen:
            with torch.no_grad():
                outs = self.model(input_tmp, mask=False, features_only=True)
        else:
            outs = self.model(input_tmp, mask=False, features_only=True)

        emb = outs['x']
        layerresult = outs['layer_results']
        return emb, layerresult

def getAttenF(layerResult):
    poollayerResult = []
    fullf = []
    for layer in layerResult:
        # (T,B,C)->(B,T,C)->(B,C,T)
        layery = layer[0].transpose(0, 1).transpose(1, 2)
        layery = F.adaptive_avg_pool1d(layery, 1)  # (B,C,1)
        layery = layery.transpose(1, 2)            # (B,1,C)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1)               # (B,T,C)
        x = x.view(x.size(0), -1, x.size(1), x.size(2))  # (B,L,T,C)
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)     # (B,L,1,C)
    fullfeature = torch.cat(fullf, dim=1)          # (B,L,T,C)
    return layery, fullfeature

class Model(nn.Module):
    """
    任务-only 版本，兼容训练脚本：
      - forward(x, grl_lambda=0.0)  # 忽略 grl_lambda
      - 返回 (task_logprob, domain_logit_placeholder, h)
        * task_logprob: (B,2) 的 LogSoftmax
        * domain_logit_placeholder: 空张量，占位
        * h: (B,1024) 共享表征（selu(fc1) 之后）
    """
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)  # 注意：与当前工程特征尺寸匹配
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def set_ssl_freeze(self, flag: bool):
        """供训练脚本调用，与原有接口一致。"""
        self.ssl_model.set_freeze(bool(flag))

    def forward(self, x: Tensor, grl_lambda: float = 0.0) -> Tuple[Tensor, Tensor, Tensor]:
        # 忽略 grl_lambda，仅作接口兼容
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1) if x.ndim == 3 else x)

        y0, fullfeature = getAttenF(layerResult)
        y0 = self.fc0(y0)               # (B,L,1,1)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)   # (B,T,C)
        fullfeature = fullfeature.unsqueeze(dim=1)# (B,1,T,C)

        z = self.first_bn(fullfeature)
        z = self.selu(z)
        z = F.max_pool2d(z, (3, 3))
        z = torch.flatten(z, 1)         # (B, 22847)

        h = self.fc1(z)                 # (B,1024)
        h = self.selu(h)                # 共享表征

        task_logit = self.fc3(h)        # (B,2)
        task_logprob = self.logsoftmax(self.selu(task_logit))  # 与原脚本/旧模型保持一致

        # 占位的域 logit（空张量），确保 “task_logprob, *_ = model(...)” 不报错
        domain_logit_placeholder = torch.empty(0, device=h.device)

        return task_logprob, domain_logit_placeholder, h
