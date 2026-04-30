# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Clean States Node

状态清理节点，用于清理模型缓存和状态，释放内存资源

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import LLAMA_CPP_STORAGE, clear_all_caches

class llama_cpp_clean_states:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",),
                "state_uid": ("INT", {"default": -1, "min": -1, "max": 999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def process(self, any, state_uid):
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        clear_all_caches()
        return (any,)
