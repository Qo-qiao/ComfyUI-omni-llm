# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Parameters Node

参数配置节点，用于设置LLM模型的生成参数，如温度、采样策略等

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import HARDWARE_INFO

class omni_llm_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 1, "tooltip": "最大生成token数，影响输出文本长度。API推理模型（如deepseek-flash）的思考token也计入此配额，生成长篇内容（完整模板/分镜/歌词）时建议设为4096-8192以避免思考占满配额导致正文为空；留默认1024时后端也会在正文为空时自动翻倍扩容重试"}),
                "top_k": ("INT", {"default": 20 if HARDWARE_INFO["is_low_perf"] else 30, "min": 0, "max": 1000, "step": 1, "tooltip": "采样候选数，值越小生成越集中，top_k"}),
                "top_p": ("FLOAT", {"default": 0.85 if HARDWARE_INFO["is_low_perf"] else 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "核心采样阈值，控制生成多样性，top_p"}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "最小采样概率，避免完全忽略低概率词汇"}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "典型采样阈值，控制生成的典型性"}),
                "temperature": ("FLOAT", {"default": 0.6 if HARDWARE_INFO["is_low_perf"] else 0.8, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "生成温度，值越高越随机"}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "重复惩罚，避免生成重复内容"}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "频率惩罚，减少高频词汇出现"}),
                "presence_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "存在惩罚，鼓励生成新内容"}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1, "tooltip": "Mirostat采样模式：0=关闭，1=基础版，2=版本2"}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mirostat学习率，影响生成多样性"}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Mirostat目标困惑度，影响生成多样性"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "tooltip": "随机种子：≥0=固定种子用于复现结果，连接随机种子节点可实现每次随机"}),
                "state_uid": ("INT", {"default": -1, "min": -1, "max": 999999, "step": 1, "tooltip": "对话状态ID，-1=使用节点唯一ID"}),
            },
            "optional": {
                "reasoning_budget": ("INT", {"default": -1, "min": -1, "max": 4096, "step": 1, "tooltip": "推理预算：-1=无限制，0=关闭思考模式，N=限制N个思考token"}),
            }
        }
    
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "omni-llm"
    
    def process(self, **kwargs):
        return (kwargs,)

NODE_CLASS_MAPPINGS = {
    "omni_llm_parameters": omni_llm_parameters
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "omni_llm_parameters": "Omni LLM Model Parameters"
}