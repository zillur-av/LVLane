import os.path as osp
import numpy as np
import cv2
import os
import json
import torch
import torchvision
from .base_dataset import BaseDataset
from lanedet.utils.tusimple_metric import LaneEval
from .registry import DATASETS
import logging
import random
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

SPLIT_FILES = {    

    'trainval': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'val': ['LVLane_test_sunny.json'],
    'test': ['LVLane_test_sunny.json']
}


@DATASETS.register_module
class TuSimple(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes, cfg)
        self.anno_files = SPLIT_FILES[split] 
        self.load_annotations()
        self.h_samples = list(range(160, 720, 10))

    def load_annotations(self):
        self.logger.info('Loading TuSimple annotations...')
        self.data_infos = []
        max_lanes = 0
        #df = {0:0, 1:1, 2:2, 3:3, 4:3, 5:4, 6:5, 7:6}       # for 6 class
        df = {0:0, 1:1, 2:1, 3:2, 4:2, 5:2, 6:1, 7:1}      # for 2 class
        #df = {0:0, 1:1, 2:1, 3:2, 4:2, 5:1}                # for caltech 2 class
        for anno_file in self.anno_files:
            anno_file = osp.join(self.data_root, anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                category = data['categories']
                category = list(map(df.get,category))

                mask_path = data['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append({
                    'img_path': osp.join(self.data_root, data['raw_file']),  #append all the samples in all the json files
                    'img_name': data['raw_file'],
                    'mask_path': osp.join(self.data_root, mask_path),
                    'lanes': lanes,
                    'categories':category
                })
        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

    def pred2lanes(self, pred):
        ys = np.array(self.h_samples) / self.cfg.ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.cfg.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate_detection(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir, 'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result, acc = LaneEval.bench_one_submit(pred_filename, self.cfg.test_json_file)
        self.logger.info(result)
        return acc

    # Calculate accuracy (a classification metric)
    def accuracy_fn(self, y_true, y_pred):
        """Calculates accuracy between truth labels and predictions.
        Args:
            y_true (torch.Tensor): Truth labels for predictions.
            y_pred (torch.Tensor): Predictions to be compared to predictions.
        Returns:
            [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
        """
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / torch.numel(y_pred))
        return acc

    def evaluate_classification(self, predictions, ground_truth):
        score = F.softmax(predictions, dim=1)
        y_pred = score.argmax(dim=1)
        return self.accuracy_fn(ground_truth, y_pred)

    def plot_confusion_matrix(self, y_true, y_pred):

        cf_matrix = confusion_matrix(y_true, y_pred)
        #class_names = ('background','solid-yellow', 'solid-white', 'dashed','botts\'-dots', 'double-solid-yellow','unknown')
        class_names = ('background', 'solid', 'dashed')
        # Create pandas dataframe
        dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
        total_number_of_instances = dataframe.sum(1)[1:].sum()
        
        
        #df = {0:0, 1:1, 2:1, 3:2, 4:2, 5:1, 6:1}
        #y_true = list(map(df.get,y_true))
        #y_pred = list(map(df.get,y_pred))
        #cf_matrix_2 = confusion_matrix(y_true, y_pred)
        #true_positives_2 = np.diag(cf_matrix_2)[1:].sum()
        #accuracy_2 = true_positives_2 / total_number_of_instances
        #print(f"Accuracy for 2 classes: {accuracy_2}")

        true_positives = np.diag(cf_matrix)[1:].sum()
        accuracy = true_positives / total_number_of_instances
        print(f"Accuracy for 2 classes: {accuracy}")

        # compute metrices from confusion matrix
        FP = cf_matrix.sum(axis=0) - np.diag(cf_matrix)  
        FN = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
        TP = np.diag(cf_matrix)
        TN = cf_matrix.sum() - (FP + FN + TP)

        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        # plot the confusion matrix
        plt.figure(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(dataframe, annot=True, cbar=None,cmap="YlGnBu",fmt="d")

        plt.title("Confusion Matrix"), plt.tight_layout()

        plt.ylabel("True Class"), 
        plt.xlabel("Predicted Class")
        plt.show()  
