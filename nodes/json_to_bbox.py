# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm JSON to Bounding Box Node

JSON转边界框节点，支持将JSON格式的目标检测结果转换为边界框
并可在图像上绘制边界框

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import parse_json, draw_bbox, qwen3bbox

class json_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple","Qwen3-VL", "Qwen2.5-VL"], {"default": "Qwen3-VL"}),
                "label": ("STRING", {"default": "", "tooltip": "Filter by label"}),
            },
            "optional": {"image": ("IMAGE",)},
        }
    
    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, json, mode, label, image=None):
        mode = mode[0]
        label = label[0]
        flat_images_list = []
        original_structure = []
        
        if image is not None:
            for img_batch in image:
                if img_batch.ndim == 3:
                    flat_images_list.append(img_batch.unsqueeze(0))
                    original_structure.append(1)
                else:
                    count = img_batch.shape[0]
                    original_structure.append(count)
                    for n in range(count):
                        flat_images_list.append(img_batch[n:n+1])
        
        total_images = len(flat_images_list)
        output_bboxes = []
        processed_flat_results = []
        
        for i, j in enumerate(json):
            bboxes = parse_json(j)
            if label != "":
                bboxes = [item for item in bboxes if item.get("label") == label or item.get("text_content") == label]
            
            if total_images > 0:
                curr_idx = i if i < total_images else (total_images - 1)
                curr_img = flat_images_list[curr_idx]
                try:
                    res_img = draw_bbox(curr_img[0], bboxes, mode)
                    processed_flat_results.append(res_img)
                except Exception as e:
                    print(f"【提示】绘制Bounding Box失败：{e}")
                    processed_flat_results.append(curr_img)
            
            bbox = qwen3bbox(flat_images_list[curr_idx][0], bboxes) if mode in ["Qwen3-VL", "Qwen2.5-VL"] else [tuple(item["bbox_2d"]) for item in bboxes]
            output_bboxes.append(bbox)
        
        restructured_images_list = []
        cursor = 0
        for count in original_structure:
            chunk = processed_flat_results[cursor : cursor + count]
            if chunk:
                restructured_images_list.append(torch.cat(chunk, dim=0))
            cursor += count
        
        return (output_bboxes, restructured_images_list)

NODE_CLASS_MAPPINGS = {
    "json_to_bbox": json_to_bbox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "json_to_bbox": "JSON to Bounding Box"
}
