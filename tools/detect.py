import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.infer_process, cfg)
        self.net = build_net(self.cfg)
        #print(self.net)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)                 # add one dimension; make it (1,3,h,w) shape
        data.update({'img_path':img_path, 'ori_img':ori_img})  # add image path and original image in the dict
        return data

    def inference(self, data):
        with torch.no_grad():
            with autocast(enabled=self.cfg.autocast):
                data = self.net(data)
                lane_detection, lane_indx = self.net.module.get_lanes(data)
            if self.cfg.classification:
                lane_classes = self.get_lane_class(data, lane_indx)
                return lane_detection[0], lane_classes
        return lane_detection

    def get_lane_class(self, predictions, lane_indx):
        score = F.softmax(predictions['category'], dim=2)
        y_pred = score.argmax(dim=2).squeeze()
        return y_pred[lane_indx].detach().cpu().numpy()

    def show(self, data, lane_classes=None):
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        #print(lanes)
        imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file, lane_classes=lane_classes)

    def run(self, data):
        data = self.preprocess(data)
        lane_classes = None
        if self.cfg.classification:
            data['lanes'], lane_classes = self.inference(data)
        else:
            data['lanes'] = self.inference(data)
        if self.cfg.show or self.cfg.savedir:
            self.show(data, lane_classes)
        #return data

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def process(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir 
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        detect.run(p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', default=False, help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)
