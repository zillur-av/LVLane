import torch
from torch import nn
import torch.nn.functional as F
from lanedet.core.lane import Lane
import cv2
import numpy as np

from ..registry import HEADS, build_head

@HEADS.register_module
class LaneSeg(nn.Module):
    def __init__(self, decoder, exist=None, thr=0.6, 
            sample_y=None, cat_dim = None, cfg=None, in_channels=6, out_channels=6):
        super(LaneSeg, self).__init__()
        self.cfg = cfg
        self.thr = thr
        self.sample_y = sample_y
        self.cat_dim = cat_dim

        self.decoder = build_head(decoder, cfg)
        self.exist = build_head(exist, cfg) if exist else None 
        if self.cfg.classification:
            
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.conv1 = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3, 
                stride=1,
                padding=1,
                bias=False
            )
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU(inplace=True)

            self.category = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(353280, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, np.prod(self.cat_dim))
               ) 

    def get_lanes(self, output):
        segs = output['seg']
        segs = F.softmax(segs, dim=1)
        segs = segs.detach().cpu().numpy()
        #print(segs.shape)
        #print("------------")
        if 'exist' in output:
            exists = output['exist']
            exists = exists.detach().cpu().numpy()
            exists = exists > 0.5
        else:
            exists = [None for _ in segs]
        ret= {}

        lane_output = []
        lane_indexes = []
        for seg, exist in zip(segs, exists):
            #print(seg.shape)
            lanes, lane_indx = self.probmap2lane(seg, exist)
            lane_output.append(lanes)
            lane_indexes.append(lane_indx)
        ret.update({'lane_output': lane_output, 'lane indexes': lane_indexes})
        return ret

    def probmap2lane(self, probmaps, exists=None):
        lanes = []
        probmaps = probmaps[1:, ...]
        #print(probmaps.shape)
        if exists is None:
            exists = [True for _ in probmaps]
        
        lane_indx = []
        for i, (probmap, exist) in enumerate(zip(probmaps, exists)):
            if exist == 0:
                continue
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)
            cut_height = self.cfg.cut_height
            ori_h = self.cfg.ori_img_h - cut_height
            coord = []
            for y in self.sample_y:
                proj_y = round((y - cut_height) * self.cfg.img_height/ori_h)
                line = probmap[proj_y]
                if np.max(line) < self.thr:
                    continue
                value = np.argmax(line)
                x = value*self.cfg.ori_img_w/self.cfg.img_width#-1.
                if x > 0:
                    coord.append([x, y])
            if len(coord) < 5:
                continue

            coord = np.array(coord)
            coord = np.flip(coord, axis=0)
            coord[:, 0] /= self.cfg.ori_img_w
            coord[:, 1] /= self.cfg.ori_img_h
            lanes.append(Lane(coord))
            lane_indx.append(i)
    
        return lanes, lane_indx

    def loss(self, output, batch):
        weights = torch.ones(self.cfg.num_lanes)
        weights[0] = self.cfg.bg_weight
        weights = weights.cuda()
        criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label,
                                          weight=weights).cuda()
        criterion_exist = torch.nn.BCEWithLogitsLoss().cuda()
        loss = 0.
        loss_stats = {}
        seg_loss = criterion(F.log_softmax(
            output['seg'], dim=1), batch['mask'].long())
        loss += seg_loss
        loss_stats.update({'seg_loss': seg_loss})
        
        if self.cfg.classification:       
            loss_fn = torch.nn.CrossEntropyLoss()
            classification_output = output['category'].reshape(self.cfg.batch_size*(self.cfg.num_lanes - 1), self.cfg.num_classes)
            targets = batch['category'].reshape(self.cfg.batch_size*(self.cfg.num_lanes - 1))

            cat_loss = loss_fn(classification_output, targets)
            loss += cat_loss*0.7
            loss_stats.update({'cls_loss': cat_loss})

        if 'exist' in output:
            exist_loss = 0.1 * \
                criterion_exist(output['exist'], batch['lane_exist'].float())
            loss += exist_loss
            loss_stats.update({'exist_loss': exist_loss})

        ret = {'loss': loss, 'loss_stats': loss_stats}
        return ret


    def forward(self, x, **kwargs):
        output = {}
        x = x[-1]
        output.update(self.decoder(x))
        if self.exist:
            output.update(self.exist(x))
        
        if self.cfg.classification:
            x= output['seg'][:,1:, ...]
            print(x.shape)
            x = self.maxpool(x)
            print(x.shape)
            x = self.conv1(x)
            print(x.shape)
            x = self.bn1(x)
            x = self.relu(x).view(-1, 353280)
            print(x.shape)
            category = self.category(x).view(-1, *self.cat_dim)
            output.update({'category': category})

        return output 
