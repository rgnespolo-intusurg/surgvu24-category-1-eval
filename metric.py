from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from evalutils.stats import dice_from_confusion_matrix

import numpy as np

import sys, traceback

class ClassificationMetrics():

    def __init__(self, y_true, y_pred, class_names=None):
        # y_true – Target 
        # y_pred – Predicted 
        # class_names (List[str]) – Inclusive list of N labels to compute the confusion matrix for.
        self.y_true, self.y_pred = y_true, y_pred
        self.class_names = class_names

        # confusion matrix, default normalized
        self.cm = self.calculate_confusion_matrix(self.y_true, self.y_pred, 
                                                    normalize=True)

        # compute dice score
        self.dice_score = dice_from_confusion_matrix(self.cm)

        # compute accuacy 
        self.acc = accuracy_score(self.y_true, self.y_pred)

        # compute avgerage precision, recall, and f1 scores
        self.avg_precision =  precision_score(self.y_true, self.y_pred, average = 'macro')
        self.avg_recall = recall_score(self.y_true, self.y_pred, average = 'macro') 
        self.avg_f1 = f1_score(self.y_true, self.y_pred, average = 'macro') 
        
        # compute precision, recall, and f1 scores for each class. Return a list
        self.precision = precision_score(self.y_true, self.y_pred, average = None).tolist() 
        self.recall = recall_score(self.y_true, self.y_pred, average = None).tolist()
        self.f1 = f1_score(self.y_true, self.y_pred, average = None).tolist()

    def calculate_confusion_matrix(self, y_true, y_pred, normalize=True):
        cm = confusion_matrix(y_true, y_pred)
        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy
        if normalize:
            cm_unnorm = cm
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os,json
class DetectionMetrics():
    def __init__(self, anno_file, res_file, verbose=False):

        assert os.path.exists(anno_file)
        assert os.path.exists(res_file)
        
        self._verbose = verbose

        # init COCO ground truth api
        self.cocoGt=COCO(anno_file)
        print('coco ground truth initialized.')

        # init COCO detect api
        self.cocoDt=self.cocoGt.loadRes(res_file)
        print('coco detect initialized.')

        print('calculating coco metric...')
        self.coco_eval, self.coco_metric = self.get_coco_detection_metrics(self.cocoGt, self.cocoDt)
        # coco_eval.eval['precision'], coco_eval.eval['recall']


    def get_coco_detection_metrics(self, cocoGt, cocoDt, iouType='bbox'): 
        # initiate COCO evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,iouType)

        # reconfigure maximum detection per image
        cocoEval.params.maxDets = [5, 10, 100]

        # run evaluation 
        cocoEval.evaluate()
        cocoEval.accumulate()
        if self._verbose:
            cocoEval.summarize()
        else:
            with Suppressor():
                cocoEval.summarize()
            

        # acquire coco results 
        results =cocoEval.stats
        return cocoEval, results

    
    def intersection_over_union(self, gt_box, pred_box):
        inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
        inter_box_bottom_right = [min(gt_box[0]+gt_box[2], pred_box[0]+pred_box[2]), min(gt_box[1]+gt_box[3], pred_box[1]+pred_box[3])]

        inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
        inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

        intersection = inter_box_w * inter_box_h
        union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection
        
        iou = intersection / union

        return iou, intersection, union
    
    
class Suppressor(object):

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout

    def write(self, x): pass