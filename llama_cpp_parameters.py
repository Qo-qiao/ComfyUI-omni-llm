# -*- coding: utf-8 -*-
"""
Llama-cpp Parameters Node
"""
from .common import HARDWARE_INFO

class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1, "tooltip": "Maximum number of generated tokens, affects output text length"}),
                "top_k": ("INT", {"default": 20 if HARDWARE_INFO["is_low_perf"] else 30, "min": 0, "max": 1000, "step": 1, "tooltip": "Number of sampling candidates, smaller values are more focused"}),
                "top_p": ("FLOAT", {"default": 0.85 if HARDWARE_INFO["is_low_perf"] else 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling threshold, controls diversity"}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum sampling probability, avoids completely ignoring low-probability words"}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Typical sampling threshold, controls generation typicality"}),
                "temperature": ("FLOAT", {"default": 0.6 if HARDWARE_INFO["is_low_perf"] else 0.8, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Generation temperature, higher values are more random"}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Repeat penalty, avoids repetitive content"}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Frequency penalty, reduces high-frequency words"}),
                "presence_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Presence penalty, encourages new content"}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1, "tooltip": "Mirostat sampling mode: 0=off, 1=basic, 2=version 2"}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mirostat learning rate"}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Mirostat target perplexity"}),
                "state_uid": ("INT", {"default": -1, "min": -1, "max": 999999, "step": 1, "tooltip": "Conversation state ID, -1=use node unique ID"}),
            }
        }
    
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, **kwargs):
        return (kwargs,)
