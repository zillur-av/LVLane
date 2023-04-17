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
from lanedet.utils.visualization_vid import imshow_lanes
from lanedet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm
import time

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, ori_img):
        img_path = "None"
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path':img_path, 'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        img = imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)
        return img

    def run(self, data):
        t1 = time.time()
        data = self.preprocess(data)
        t2 = time.time()
        #print(t2-t1)
        data['lanes'] = self.inference(data)[0]
        img = self.show(data)
        return img

def process(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    vid = cv2.VideoCapture(args.img)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    result = cv2.VideoWriter('filename.avi', fourcc, 30.0, (1280,720))
    while(True):
        ret, frame = vid.read()

        img = detect.run(frame)
        cv2.imshow("view", img)
        result.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    result.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', default=True, help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)
