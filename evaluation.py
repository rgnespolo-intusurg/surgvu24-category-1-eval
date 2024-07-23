from evalutils import DetectionEvaluation
from sklearn.metrics import accuracy_score
import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
import logging

from metric import DetectionMetrics
from gc_to_coco import GC_to_COCO

LOG = logging.getLogger(__name__)
logging.basicConfig(format='[%(filename)s:%(lineno)s - %(funcName)20s()] %(asctime)s %(message)s', level="INFO")


class Surgtoolloc(DetectionEvaluation):
    def __init__(self):
        self._gt_json_list= self.get_list_gt_jsons()
        self._pred_json_list, self._name_map = self.get_list_pred_jsons()
        logging.info("Found the following predictions jsons: " + str(self._pred_json_list))
        logging.info("Found the following name mapping: " + str(self._name_map))

    
    
    
    def evaluate_one_video(self, gt_coco_file, pred_gc_file, name_map=None):
        logging.info("Evaluating pair of gt and pred files:")
        logging.info("    gt file: " + gt_coco_file)
        logging.info("    gc pred file: " + pred_gc_file)
        if name_map:
            tru_fname = name_map[pred_gc_file]
            logging.info("   gc mapped test set file: " + tru_fname)
        logging.info("   converting gc json to coco format...")
        pred_gc_json = self.adjust_pred_gc_json(gt_coco_file, pred_gc_file,
                                                name_map=name_map)
        pred_coco_file = self.resave_gc_json_as_coco(gc_json_file=pred_gc_file,
                                                     gc_json=pred_gc_json, name_map=name_map)
        logging.info("   evaluating ground truth coco and converted gc coco format jsons...")
        my_det_met = DetectionMetrics(gt_coco_file, pred_coco_file)
        logging.info("   completed evaluation. Removing converted coco json from filesystem...")
        os.remove(pred_coco_file)
        return my_det_met
    
    
    
    
    
    def evaluate_all(self, gt_json_list, pred_json_list, name_map=None):
        all_dets = []
        for gt_file in gt_json_list:
            gt_name = gt_file.split('/')[-1].split('_coco.json')[0]
            found_match = False
            for pred_file in pred_json_list:
                if name_map:
                    true_pred_name = name_map[pred_file].split('.')[0]
                else:
                    true_pred_name = pred_file.split('/')[-1].split('dummy_')[-1].split('_gc.json')[0]
                if gt_name == true_pred_name:
                    found_match = True
                    break
            if found_match:
                logging.info("Matched " + gt_name + " with " + true_pred_name + "...")
                one_det = self.evaluate_one_video(gt_file, pred_file, name_map=name_map)
                all_dets.append(one_det)
            else:
                logging.info("Could not match " + gt_name + " with any prediction files...")
                logging.info("Not evaluating on this file...")
        self._all_dets = all_dets
                    
    
    
    def evaluate(self):
        self.evaluate_all(self._gt_json_list, self._pred_json_list, name_map=self._name_map)
        
        avg_prec = []
        for det in self._all_dets:
            logging.info("Computing average precision from each video...")
            avg_prec.append(np.mean(det.coco_eval.stats[0]))
            
        tot_avg_prec = np.mean(avg_prec)
        logging.info("   Found average precision across all videos: " + str(tot_avg_prec))
        
        out_dict = {
            'mean_mAP': tot_avg_prec
        }
        
        if os.path.isdir('/output/'):
            logging.info("Saving away output...")
            filename = '/output/metrics.json'
            logging.info("  saving output metrics file: " + filename)
            with open(filename, 'w') as handle:
                json.dump(out_dict, handle)
            handle.close()
        else:
            logging.info("Created the following cumulative result...")
            logging.info(str(out_dict))
            logging.info("Not saving away output. Did not find /output/ directory...")
        
    
    
    
    
    def resave_gc_json_as_coco(self, gc_json_file=None, name_map=None, set_index_0=False,
                                     gc_json=None):
        logging.info("Saving one gc_json prediction file as coco json file...")
        logging.info("    gc pred json file name: " + gc_json_file)
        if 'inference_output' in gc_json_file:
            logging.info(" offsetting image number to 0 instead of 1...")
            set_index_0 = True
        my_to_coco = GC_to_COCO(gc_json_fname=gc_json_file, width=640, height=512, 
                               set_index_0 = set_index_0, gc_json=gc_json)
        pred_coco_json = my_to_coco.convert()
        if name_map:
            logging.info("      found a name mapping to map pred file name...")
            fname = name_map[gc_json_file].split('.')[0] + '_coco.json'
        else:
            fname = gc_json_file.split('/')[-1].split('.')[0] + '_coco.json'
        if os.path.isdir('/output/'):
            fname_path = '/output/' + fname
        else:
            logging.info("Did not find /output/ file directory...")
            fname_path = "inference_output/" + fname
        logging.info("    saving to path: " + fname_path)
        with open(fname_path, 'w') as handle:
            json.dump(pred_coco_json, handle)
        handle.close()
        return fname_path
    
    
    
    def get_list_gt_jsons(self):
        gt_jsons = glob.glob('ground-truth/*_coco.json')
        logging.info(" Found the following ground truth jsons:")
        logging.info("    " + str(gt_jsons))
        return gt_jsons
    
    
    def get_list_pred_jsons(self, pred_json_loc="/input/predictions.json"):
        if os.path.isfile(pred_json_loc):
            logging.info("    found the prediction json location: " + pred_json_loc)
            cases = self.load_predictions_json(fname=pred_json_loc)
            return list(cases.keys()), cases
        else:
            logging.info("     did not find prediciton json location...")
            logging.info("     loading examples predictions from local repo...")
            pred_jsons = glob.glob('inference_output/*.json')
            return pred_jsons, None
        
    
    def load_predictions_json(self, fname="/input/predictions.json"):

        cases = {}

        with open(fname, "r") as f:
            entries = json.load(f)
        f.close()

        if isinstance(entries, float):
            raise TypeError(f"entries of type float for file: {fname}")

        logging.info("  Here's the structure of the predictions.json file generated by GC...")
        logging.info(str(entries))
        
        for e in entries:
            # Find case name through input file name
            inputs = e["inputs"]
            name = None
            for input in inputs:
                if input["interface"]["slug"] == "endoscopic-robotic-surgery-video":
                    name = str(input["file"]).split('/')[-1]
                    logging.info("   found input file: " + name)
                    break  # expecting only a single input
            if name is None:
                raise ValueError(f"No filename found for entry: {e}")

            entry = {"name": name}

            # Find output value for this case
            outputs = e["outputs"]

            for output in outputs:
                pk = e["pk"]
                relative_path = output["interface"]["relative_path"]
                full_path_output_json = "/input/" + pk + "/output/" + relative_path
                cases[full_path_output_json] = name

        return cases
    
    
    
    def adjust_pred_gc_json(self, gt_coco_file, pred_gc_file, name_map=None):
        logging.info("Adjusting prediction gc json to have same frames as ground truth gc json:")
        logging.info("    gt file: " + gt_coco_file)
        logging.info("    gc pred file: " + pred_gc_file)
        if name_map:
            tru_fname = name_map[pred_gc_file]
            logging.info("   gc mapped test set pred file: " + tru_fname)
        gt_gc_file = gt_coco_file.split('_coco')[0] + '_gc.json'
        logging.info("    gt gc file: " + gt_gc_file)
        # open gt_gc_file
        with open(gt_gc_file, 'r') as handle:
            gt_gc_json = json.load(handle)
        handle.close()
        gt_boxes = gt_gc_json['boxes']
        gt_frame_numbers = []
        for box in gt_boxes:
            frame_number = int(box['name'].split('slice_nr_')[-1].split('_')[0]) - 1
            gt_frame_numbers.append(frame_number)
        logging.info('   found number of frames in ground truth labels: ' + str(len(gt_frame_numbers)))
        self._gt_frame_numbers = set(gt_frame_numbers)
        # open pred gc file 
        with open(pred_gc_file, 'r') as handle:
            pred_gc_json = json.load(handle)
        handle.close()
        new_boxes = []
        pred_boxes = pred_gc_json['boxes']
        logging.info('   forcing prediction gc json to have the same frames...')
        pred_frame_numbers = []
        pred_frame_numbers_adj = []
        for box in pred_boxes:
            frame_number = int(box['name'].split('slice_nr_')[-1].split('_')[0])
            pred_frame_numbers.append(frame_number)
            if frame_number in gt_frame_numbers:
                pred_frame_numbers_adj.append(frame_number)
                new_boxes.append(box)
        self._pred_frame_numbers = set(pred_frame_numbers)
        self._pred_frame_numbers_adj = set(pred_frame_numbers_adj)
        pred_gc_json['boxes'] = new_boxes
        return pred_gc_json

if __name__ == "__main__":
    Surgtoolloc().evaluate()
