import time
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import cv2

from lanedet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from lanedet.datasets import build_dataloader
from lanedet.utils.recorder import build_recorder
from lanedet.utils.net_utils import save_model, load_network
from mmcv.parallel import MMDataParallel 
import torch.nn.functional as F

class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        # self.net.to(torch.device('cuda'))
        # self.net = torch.nn.parallel.DataParallel(
        #         self.net, device_ids = range(self.cfg.gpus)).cuda()
        self.net = MMDataParallel(
                self.net, device_ids = range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.warmup_scheduler = None
        # TODO(zhengtu): remove this hard code
        if self.cfg.optimizer.type == 'SGD':
            self.warmup_scheduler = warmup.LinearWarmup(
                self.optimizer, warmup_period=5000)
        self.detection_metric = 0.
        self.classification_metric = 0.
        self.val_loader = None
        self.test_loader = None

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from,
                finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(k, torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch
    
    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            self.optimizer.zero_grad()
            
            output = self.net(data)
            loss = output['loss']          
            loss.backward()
            self.optimizer.step()
                
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            if self.warmup_scheduler:
                self.warmup_scheduler.dampen()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)

        self.recorder.logger.info('Start training...')
        for epoch in range(self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)
        self.net.eval()
        detection_predictions = []
        classification_acc = 0
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                detection_output = self.net.module.get_lanes(output)['lane_output']
                detection_predictions.extend(detection_output)
                if self.cfg.classification:
                    classification_acc += self.val_loader.dataset.evaluate_classification(output['category'].cuda(), data['category'].cuda())
 
            if self.cfg.view:
                self.val_loader.dataset.view(detection_output, data['meta'])

        detection_out = self.val_loader.dataset.evaluate_detection(detection_predictions, self.cfg.work_dir)
        detection_metric = detection_out
        if detection_metric > self.detection_metric:
            self.detection_metric = detection_metric

        if self.cfg.classification:
            classification_acc /= len(self.val_loader)
            self.recorder.logger.info("Detection: " +str(detection_out) + "  "+ "classification accuracy: " + str(classification_acc))      
            classification_metric = classification_acc
            if classification_metric > self.classification_metric:
                self.classification_metric = classification_metric
                self.save_ckpt(is_best=True)
            self.recorder.logger.info('Best detection metric: ' + str(self.detection_metric) + "  " + 'Best classification metric: ' + str(self.classification_metric))
        else:
            self.recorder.logger.info("Detection: " +str(detection_out))  
            self.recorder.logger.info('Best detection metric: ' + str(self.detection_metric))

    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)
        self.recorder.logger.info('Start testing...')
        classification_acc = 0
        y_true = []
        y_pred = []
        self.net.eval()
        detection_predictions = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i, data in enumerate(tqdm(self.test_loader, desc=f'test')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                detection_output = self.net.module.get_lanes(output)['lane_output']
                detection_predictions.extend(detection_output)

                if self.cfg.classification:
                    y_true.extend((data['category'].cpu().numpy()).flatten('C').tolist())
                    score = F.softmax(output['category'].cuda(), dim=1)
                    score = score.argmax(dim=1)
                    y_pred.extend((score.cpu().numpy()).flatten('C').tolist())

                    classification_acc += self.test_loader.dataset.evaluate_classification(output['category'].cuda(), data['category'].cuda())
        
        end.record()
        torch.cuda.synchronize()  
        print('execution time in milliseconds per image: {}'. format(start.elapsed_time(end)/2782))
        
        detection_out = self.test_loader.dataset.evaluate_detection(detection_predictions, self.cfg.work_dir)

        if self.cfg.classification:
            classification_acc /= len(self.test_loader)
            self.recorder.logger.info("Detection: " +str(detection_out) + "  "+ "classification accuracy: " + str(classification_acc))  
            self.test_loader.dataset.plot_confusion_matrix(y_true, y_pred)
        else:
            self.recorder.logger.info("Detection: " +str(detection_out))   


    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler,
                self.recorder, is_best)
