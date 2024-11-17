import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..arch.resnet18_encoder import *
import timm

import clip
import torch.nn.init as init

from ..arch.vision_transformer import *



class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args

        if self.args.dataset in ['cifar100']:

            if self.args.arch == 'timm_vit_base_patch16_224':
                state_dict = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
                self.backbone = vit_base(img_size=[224], patch_size=16, num_classes=1000)
                self.backbone.load_state_dict(state_dict, strict=True) 
                self.num_features = 1000






        if self.args.dataset in ['mini_imagenet']:

            if self.args.arch == 'timm_vit_base_patch16_224':
                state_dict = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
                self.backbone = vit_base(img_size=[224], patch_size=16, num_classes=1000)
                self.backbone.load_state_dict(state_dict, strict=True)
                self.num_features = 1000


        if self.args.dataset == 'cub200':

            if self.args.arch == 'timm_vit_base_patch16_224':
                state_dict = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
                self.backbone = vit_base(img_size=[224], patch_size=16, num_classes=1000)
                self.backbone.load_state_dict(state_dict, strict=True) 
                self.num_features = 1000



        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.Tanh()
        self.ln = nn.LayerNorm(self.num_features)
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

    def forward_metric(self, x):
        x = self.encode(x)
        embed = x
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x
        elif 'dot' in self.mode:
            x = self.fc(x)
        return x, embed


    def encode(self, x):
        x = self.backbone(x)
        x = self.ln(x)
        x = self.activation(x)


        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input, embed = self.forward_metric(input)
            return input, embed
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

