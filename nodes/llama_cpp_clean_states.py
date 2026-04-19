# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Clean States Node

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import any_type, LLAMA_CPP_STORAGE
from engine import clear_all_caches

class llama_cpp_clean_states:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "state_uid": ("INT", {"default": -1, "min": -1, "max": 999999, "step": 1}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def process(self, any, state_uid):
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        clear_all_caches()
        return (any,)
