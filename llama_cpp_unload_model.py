# -*- coding: utf-8 -*-
"""
Llama-cpp Unload Model Node
"""
from .common import any_type, LLAMA_CPP_STORAGE

class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,)}}
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any):
        LLAMA_CPP_STORAGE.clean()
        return (any,)
