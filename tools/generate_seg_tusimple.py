import json
import numpy as np
import cv2
import os
import argparse

TRAIN_SET = ['label_data_0313.json', 'label_data_0601.json']
VAL_SET = ['label_data_0531.json']

# create mask labels and class labels

def gen_label_from_json(data_root, data_savedir, file_name):
    H, W = 720, 1280
    SEG_WIDTH = 30
    save_dir = data_savedir
    annot_file = file_name + '.json'
    class_file = file_name + '_classes.txt'

    os.makedirs(os.path.join(data_root, data_savedir, "list"), exist_ok=True)

    with open(os.path.join(data_root, data_savedir, 'list', annot_file), 'w') as outfile:
        json_path = os.path.join(data_root, annot_file)
        with open(json_path) as f, open(data_root + class_file) as cat_file:
            json_lines = f.readlines()
            class_lines = cat_file.readlines()
            line_index = 0
            while line_index < len(json_lines):
                line = json_lines[line_index]
                label = json.loads(line)
                class_line = class_lines[line_index]

                class_line = class_line.strip()
                class_list = class_line.split(' ')

                # ---------- clean and sort lanes -------------
                lanes = []
                _lanes = []
                slope = [] # identify 0th, 1st, 2nd, 3rd, 4th, 5th lane through slope
                for i in range(len(label['lanes'])):
                    l = [(x, y) for x, y in zip(label['lanes'][i], label['h_samples']) if x >= 0]
                    if (len(l)>1):
                        _lanes.append(l)
                        slope.append(np.arctan2(l[-1][1]-l[0][1], l[0][0]-l[-1][0]) / np.pi * 180)
                _lanes = [_lanes[i] for i in np.argsort(slope)]# arrange lanes based on slope
                data = [(slp, cls) for slp, cls in zip(slope, class_list)]
                data.sort(key = lambda x: x[0])                # arrange (slope, class_list) based on slope
                slope = [slope[i] for i in np.argsort(slope)]  # arrange slope low to high
                #print(data)
                #print(_lanes)
                
                idx = [None for i in range(6)]
                for i in range(len(slope)):
                    if slope[i] <= 90:
                        idx[2] = i
                        idx[1] = i-1 if i > 0 else None
                        idx[0] = i-2 if i > 1 else None
                    else:
                        idx[3] = i
                        idx[4] = i+1 if i+1 < len(slope) else None
                        idx[5] = i+2 if i+2 < len(slope) else None
                        break
                for i in range(6):
                    lanes.append([] if idx[i] is None else _lanes[idx[i]])  # keep max 3 on left and 3 on right

                # ---------------------------------------------
                data = [data[i] for i in idx if i is not None]
                
                img_path = label['raw_file']
                seg_img = np.zeros((H, W, 3))
                list_str = []  # str to be written to list.txt
                for i in range(len(lanes)):
                    coords = lanes[i]
                    if len(coords) < 4:
                        list_str.append(0)
                        continue
                    for j in range(len(coords)-1):
                        cv2.line(seg_img, coords[j], coords[j+1], (i+1, i+1, i+1), SEG_WIDTH//2)
                    list_str.append(1)   # from left 3 to right 3, put 1 if there is a lane
                
                seg_path = img_path.split("/")
                seg_path, img_name = os.path.join(data_root, data_savedir, seg_path[1]), seg_path[2]
                os.makedirs(seg_path, exist_ok=True)
                seg_path = os.path.join(seg_path, img_name[:-3]+"png")
                cv2.imwrite(seg_path, seg_img)

                cls = [c[1] for c in data]
                non_zero_ind = [i for i, e in enumerate(list_str) if e != 0]
                for i,val in enumerate(non_zero_ind):
                    list_str[val]= int(cls[i])
                
                line_index += 1
                label['categories']= list_str
                json_object = json.dumps(label)
                outfile.write(json_object)
                outfile.write('\n')

def generate_label(args):
    save_dir = os.path.join(args.root, args.savedir)
    os.makedirs(save_dir, exist_ok=True)

    print("generating masks...")
    gen_label_from_json(args.root, save_dir, file_name = 'LVLane_test_sunny')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the Tusimple dataset')
    parser.add_argument('--savedir', type=str, default='seg_label', help='The root of the Tusimple dataset')
    args = parser.parse_args()

    generate_label(args)
