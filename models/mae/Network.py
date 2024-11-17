import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


from ..arch.mae.mae import *


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            self.backbone = mae_vit_tiny_cifar(norm_pix_loss=True, concept=5)
            self.num_features = 384


        if self.args.dataset in ['mini_imagenet']:

            self.backbone = mae_vit_tiny_mini(norm_pix_loss=True)
            self.num_features = 384


        if self.args.dataset == 'cub200':
            self.backbone = mae_vit_tiny_cifar()  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790


        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

        # self.label = nn.Parameter(torch.randn(self.args.base_class, self.args.base_class).cuda())



        
    def criterion(self, embed):
        embed = embed.view(embed.size(0), -1, 276)
        norm = torch.norm(embed, dim=2)
        loss = (torch.sigmoid(norm) -norm)**2
        

        return loss.mean()
        
    def forward(self, input):
        
        loss, embed, _, _ = self.backbone(input)


        x = embed

        if 'cos' in self.mode:
            x = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(embed)

        
        return loss, embed, x
        
    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            _, data, _, _ = self.backbone(data)
        
        new_fc = self.update_fc_avg(data.detach(), label, class_list)


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
   
