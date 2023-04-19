import os.path as osp
import numpy as np
import cv2
import os
import json
import torchvision
import inspect
from ..registry import PROCESS

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

def find_start_pos(row_sample,start_line):
    l,r = 0,len(row_sample)-1
    while True:
        mid = int((l+r)/2)
        if r - l == 1:
            return r
        if row_sample[mid] < start_line:    
            l = mid
        if row_sample[mid] > start_line:     
            r = mid
        if row_sample[mid] == start_line:
            return mid

def _grid_pts(pts, num_cols, w):
    # pts : numlane,n,2
    num_lane, n, n2 = pts.shape
    col_sample = np.linspace(0, w - 1, num_cols)

    assert n2 == 2
    to_pts = np.zeros((n, num_lane))
    tot_len = col_sample[1] - col_sample[0]
    
    for i in range(num_lane):
        pti = pts[i, :, 1]
        to_pts[:, i] = np.asarray(
            [int(pt // tot_len) if pt != -1 else num_cols for pt in pti])
    return to_pts.astype(int)

@PROCESS.register_module
class GenerateLaneCls(object):
    def __init__(self, row_anchor, num_cols, num_lanes, cfg):
        self.row_anchor = eval(row_anchor)
        self.num_cols = num_cols        #100
        self.num_lanes = num_lanes  #6

    def __call__(self, sample):
        label = sample['mask']    # seg_mask
        h, w = label.shape        # 720x1280
        if h != 288:
            scale_f = lambda x : int((x * 1.0/288) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))       # list [160, ..... 710]

        all_idx = np.zeros((self.num_lanes, len(sample_tmp),2))  # 6x56x2
        
        for i,r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]     # 1280 pixels in each row anchor: shape = 1280x1
                                                           # pixels are actually lane numbers like 1 to 6
            for lane_idx in range(1, self.num_lanes+1):    # 1 to 6
                pos = np.where(label_r == lane_idx)[0]     # x pixels of the lane location
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r        # if no lane, just put y values like 160, 170 
                    all_idx[lane_idx - 1, i, 1] = -1       # in x values, put -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos          # in x values, put mean of x pixels of the lane      

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i,:,1] == -1):            # if all x values are -1, ignore that lane
                continue

            valid = all_idx_cp[i,:,1] != -1
            valid_idx = all_idx_cp[i,valid,:]              # index of valid lanes (y,x)
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:      # if last y value is 710, ignore that too
                continue
            if len(valid_idx) < 6:
                continue

            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)     # create a line using upper half of the lane
            start_line = valid_idx_half[-1,0]
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])                          # get x values from the 1D poly using y values
            fitted = np.array([-1  if  x< 0 or x > w-1 else x for x in fitted])  # if x value is out of bound, make it -1

            assert np.all(all_idx_cp[i,pos:,1] == -1)                        
            all_idx_cp[i,pos:,1] = fitted                                        # make all x values after pos equal to fitted x values
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        sample['cls_label'] = _grid_pts(all_idx_cp, self.num_cols, w)            # return int 6x56x2 array as final class label

        return sample
