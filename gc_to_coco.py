import numpy as np
import pandas as pd
import json 
import os
import glob
from datetime import datetime
import logging

class GC_to_COCO:
    
    def __init__(self, gc_json_fname=None, width=640, height=512, set_index_0=False, gc_json=None):
        self._gc_fname = gc_json_fname
        self._width = width
        self._height = height
        self._set_index_0 = set_index_0
        
        if gc_json is not None:
            logging.info("  GC_to_COCO object found gc json (in addition to filename). Will convert gc_json and not open the file")
            self._gc_json = gc_json
        else:
            with open(self._gc_fname, 'r') as handle:
                self._gc_json = json.load(handle)
            handle.close()
        
        
        
        
    def name_to_tool(self, slice_name):
        tool_name = ' '.join(slice_name.split('_')[3:])
        return tool_name
    
    
    def map_tool_to_id(self):
        tools = ['grasping retractor', 'cadiere forceps', 
                 'bipolar forceps', 'force bipolar', 'clip applier', 
                 'stapler', 'permanent cautery hook spatula', 
                 'monopolar curved scissor', 'vessel sealer', 
                 'tip up fenestrated grasper', 'bipolar dissector', 
                 'needle driver', 'prograsp forceps',
                 'suction irrigator']
        tool_to_id = {}
        id_to_tool = {}
        for i,tool in enumerate(tools):
            tool_to_id[tool] = i
            id_to_tool[i] = tool
        return tool_to_id, id_to_tool
    
    
    def create_categories_json(self,):
        self._tool_to_id, self._id_to_tool = self.map_tool_to_id()
        categories = []
        for tool in self._tool_to_id:
            this_dict = {
                "id": self._tool_to_id[tool],
                "name": tool
            }
            categories.append(this_dict)
        self._categories = categories
        
        
    def one_frame_to_coco(self, boxes_json_elem, for_gt=False):
        this_name = boxes_json_elem["name"]
        corners = boxes_json_elem["corners"]
        # for images json
        image_id = int(this_name.split('slice_nr_')[-1].split('_')[0])
        if self._set_index_0:
            image_id -= 1
        filename = str(image_id).zfill(10) + '.jpg'
        date_captured = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # for annotations json
        category_id = self._tool_to_id[self.name_to_tool(this_name)]
        x1 = corners[0][0]
        y1 = corners[0][1]
        w = corners[1][0] - x1
        h = corners[2][1] - y1
        area = w*h
        bbox = [x1, y1, w, h]
        # create json from one frame
        images_json = {
            "id": image_id,
            "width": self._width,
            "height": self._height,
            "file_name": filename,
            "date_captured": date_captured
        }
        if "probability" in boxes_json_elem.keys():
            score = boxes_json_elem["probability"]
        else:
            score = 0.5
        anno_json = {
            "category_id": category_id,
            "image_id": image_id,
            "bbox": bbox,
            "score": score,
            "segmentation": []
        }
        if for_gt:
            anno_json = {
                "category_id": category_id,
                "image_id": image_id,
                "bbox": bbox,
                "area": area
            }
        return images_json, anno_json
    
    
    def gc_to_coco_boxes(self, for_gt=False):
        all_image_ids = []
        images = []
        annotations = []
        for i,one_json in enumerate(self._gc_json["boxes"]):
            image_json, anno_json = self.one_frame_to_coco(one_json, 
                                                           for_gt=for_gt)
            anno_json["id"] = i
            if for_gt:
                anno_json["iscrowd"] = 0
            if image_json["id"] not in all_image_ids:
                images.append(image_json)
                all_image_ids.append(image_json["id"])
            annotations.append(anno_json)
        self._images = images
        self._annotations = annotations
        
        
    def convert(self, for_gt=False):
        self.create_categories_json()
        self.gc_to_coco_boxes(for_gt=for_gt)
        return self._annotations
            