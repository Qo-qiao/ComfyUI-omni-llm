# -*- coding: utf-8 -*-
"""
TTS时间对齐节点

使用强制对齐推理节点输出的时间戳数据，实现按文本时间轴生成配音，支持精确时间对齐和自然流对齐两种模式。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import sys
import torch
import numpy as np
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class tts_align:
    """TTS对齐节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_model": ("TTSMODEL", {"tooltip": "TTS模型"}),
                "timestamps": ("JSON", {"default": "", "tooltip": "时间戳数据\n• 从强制对齐推理节点获取\n• 或导入JSON格式文件"}),
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "要合成的文本内容"}),
                "sample_rate": ("INT", {"default": 24000, "min": 8000, "max": 48000, "step": 100, "tooltip": "音频采样率（Hz）"}),
            },
            "optional": {
                "align_mode": (["Exact Timing", "Natural Flow"], {"default": "Natural Flow", "tooltip": "选择对齐模式\n• 精确时间对齐：严格按照时间戳生成音频\n• 自然流对齐：保持自然语音节奏"}),
                "silence_padding": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "段落后静音填充（秒）"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "JSON")
    RETURN_NAMES = ("aligned_audio", "alignment_info")
    FUNCTION = "align_tts"
    CATEGORY = "llama-cpp-vlm"
    
    def align_tts(self, tts_model, timestamps, text, sample_rate=24000, 
                  align_mode="Natural Flow", silence_padding=0.1):
        """
        使用时间戳进行TTS配音对齐
        
        Args:
            tts_model: TTS模型
            timestamps: JSON格式的时间戳数据
            text: 要合成的文本
            sample_rate: 音频采样率
            speed: TTS速度
            align_mode: 对齐模式
            silence_padding: 段落后静音填充
            
        Returns:
            aligned_audio: 对齐后的音频
            alignment_info: 对齐信息
        """
        try:
            # 检查TTS模型是否为None
            if tts_model is None:
                print("【TTS对齐】TTS模型未加载")
                # 返回空音频对象
                empty_waveform = torch.zeros(0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                empty_audio = {"waveform": empty_waveform, "sample_rate": sample_rate}
                return (empty_audio, json.dumps({"error": "TTS模型未加载"}))
            
            # 解析时间戳数据
            if isinstance(timestamps, str):
                # 检查是否是文件路径
                if timestamps and timestamps.lower().endswith('.json') and os.path.isfile(timestamps):
                    # 从文件读取JSON数据
                    try:
                        with open(timestamps, 'r', encoding='utf-8') as f:
                            timestamps_data = json.load(f)
                        print(f"【TTS对齐】从文件读取时间戳数据: {timestamps}")
                    except Exception as e:
                        print(f"【TTS对齐】读取文件失败: {str(e)}")
                        # 返回空音频对象而不是None
                        empty_waveform = torch.zeros(0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        empty_audio = {"waveform": empty_waveform, "sample_rate": sample_rate}
                        return (empty_audio, json.dumps({"error": f"读取文件失败: {str(e)}"}))
                else:
                    # 直接解析JSON字符串
                    try:
                        timestamps_data = json.loads(timestamps)
                    except Exception as e:
                        print(f"【TTS对齐】解析JSON字符串失败: {str(e)}")
                        # 返回空音频对象而不是None
                        empty_waveform = torch.zeros(0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        empty_audio = {"waveform": empty_waveform, "sample_rate": sample_rate}
                        return (empty_audio, json.dumps({"error": f"解析JSON字符串失败: {str(e)}"}))
            else:
                timestamps_data = timestamps
            
            # 检查状态
            if timestamps_data.get("status") != "success":
                error_msg = timestamps_data.get("error_msg", "时间戳数据无效")
                print(f"【TTS对齐】{error_msg}")
                # 返回空音频对象而不是None
                empty_waveform = torch.zeros(0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                empty_audio = {"waveform": empty_waveform, "sample_rate": sample_rate}
                return (empty_audio, json.dumps({"error": error_msg}))
            
            # 获取段落数据
            segments = timestamps_data.get("segments", [])
            
            if not segments:
                print("【TTS对齐】没有可用的段落数据")
                # 返回空音频对象而不是None
                empty_waveform = torch.zeros(0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                empty_audio = {"waveform": empty_waveform, "sample_rate": sample_rate}
                return (empty_audio, json.dumps({"segments": 0}))
            
            # 按时间顺序排序
            segments.sort(key=lambda x: x.get("start", 0))
            
            audio_segments = []
            alignment_details = []
            
            # 处理每个段落
            for i, segment in enumerate(segments):
                segment_text = segment.get("text", "").strip()
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                target_duration = end_time - start_time
                
                if not segment_text:
                    continue
                
                try:
                    # 使用TTS模型合成该段落的音频
                    segment_audio = tts_model.synthesize(
                        text=segment_text
                    )
                    
                    if segment_audio and "waveform" in segment_audio:
                        waveform = segment_audio["waveform"]
                        
                        # 转换为torch张量（如果是numpy数组）
                        if isinstance(waveform, np.ndarray):
                            waveform = torch.from_numpy(waveform)
                        
                        # 确保波形是单通道
                        if waveform.dim() > 1:
                            waveform = waveform.squeeze(0)
                        
                        # 获取合成音频的实际时长
                        actual_duration = len(waveform) / sample_rate
                        
                        # 根据对齐模式处理
                        if align_mode == "Exact Timing":
                            # 精确时间对齐
                            target_samples = int(target_duration * sample_rate)
                            
                            if len(waveform) > target_samples:
                                # 截断过长的音频
                                waveform = waveform[:target_samples]
                            elif len(waveform) < target_samples:
                                # 填充静音
                                padding_samples = target_samples - len(waveform)
                                padding = torch.zeros(padding_samples, dtype=waveform.dtype)
                                waveform = torch.cat([waveform, padding])
                        
                        audio_segments.append(waveform)
                        
                        alignment_details.append({
                            "segment_index": i,
                            "text": segment_text,
                            "target_start": start_time,
                            "target_end": end_time,
                            "target_duration": target_duration,
                            "actual_duration": actual_duration,
                            "status": "success"
                        })
                    else:
                        alignment_details.append({
                            "segment_index": i,
                            "text": segment_text,
                            "status": "failed",
                            "error": "TTS合成失败"
                        })
                    
                except Exception as e:
                    print(f"【TTS对齐】段落 {i} 合成失败: {str(e)}")
                    alignment_details.append({
                        "segment_index": i,
                        "text": segment_text,
                        "status": "failed",
                        "error": str(e)
                    })
            
            # 添加段落后的静音填充
            if audio_segments and silence_padding > 0:
                padding_samples = int(silence_padding * sample_rate)
                padding = torch.zeros(padding_samples, dtype=torch.float32)
                
                # 在每个段落后添加静音
                padded_segments = []
                for i, segment in enumerate(audio_segments):
                    padded_segments.append(segment)
                    if i < len(audio_segments) - 1:
                        padded_segments.append(padding)
                
                audio_segments = padded_segments
            
            # 合并所有音频段
            if audio_segments:
                aligned_waveform = torch.cat(audio_segments)
            else:
                # 如果没有成功的音频段，创建空音频
                aligned_waveform = torch.zeros(0, dtype=torch.float32)
            
            # 构建ComfyUI标准的音频格式
            # 需要添加batch维度 [batch, channels, samples]
            if aligned_waveform.dim() == 1:
                aligned_waveform = aligned_waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            elif aligned_waveform.dim() == 2:
                aligned_waveform = aligned_waveform.unsqueeze(0)  # [1, channels, samples]
            
            aligned_audio = {
                "waveform": aligned_waveform,
                "sample_rate": sample_rate
            }
            
            # 构建对齐信息
            alignment_info = {
                "total_segments": len(segments),
                "successful_segments": len([d for d in alignment_details if d.get("status") == "success"]),
                "total_duration": aligned_waveform.shape[-1] / sample_rate,
                "segments": alignment_details
            }
            
            print(f"【TTS对齐】完成，成功合成 {alignment_info['successful_segments']}/{alignment_info['total_segments']} 个段落")
            
            return (aligned_audio, json.dumps(alignment_info))
            
        except Exception as e:
            print(f"【TTS对齐错误】{str(e)}")
            import traceback
            traceback.print_exc()
            # 返回空音频对象而不是None
            empty_waveform = torch.zeros(0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            empty_audio = {"waveform": empty_waveform, "sample_rate": sample_rate}
            return (empty_audio, json.dumps({"error": str(e)}))


# 节点映射
NODE_CLASS_MAPPINGS = {
    "tts_align": tts_align,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "tts_align": "TTS Align",
}
