# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Forced Aligner Inference Node

强制对齐推理节点，支持将音频和文本输入ForcedAligner
输出细粒度时间戳（音素/单词级）

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import sys
import torch
import numpy as np
import uuid
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class forced_aligner_inference:
    """强制对齐推理节点"""
    
    
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aligner_model": ("ALIGNERMODEL", {"tooltip": "加载的强制对齐模型"}),
                "audio": ("AUDIO", {"tooltip": "输入音频数据"}),
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "要对齐的文本内容"}),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 16000, "min": 8000, "max": 48000, "step": 100, "tooltip": "音频采样率（Hz）"}),
                "output_format": (["Text", "SRT", "VTT", "JSON"], {"default": "Text", "tooltip": "输出格式：Text=对齐文本，SRT=SRT字幕，VTT=VTT字幕，JSON=JSON数据"}),
                "language": (["zh", "en"], {"default": "zh", "tooltip": "文本语言：zh=中文，en=英文"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "JSON", "STRING")
    RETURN_NAMES = ("aligned_text", "timestamps", "subtitle_text")
    FUNCTION = "run_forced_aligner"
    CATEGORY = "llama-cpp-vlm"
    
    def run_forced_aligner(self, aligner_model, audio, text, sample_rate=16000, output_format="Text", language="zh"):
        """
        执行强制对齐推理
        
        Args:
            aligner_model: 加载的强制对齐模型
            audio: 输入音频数据（torch张量）
            text: 要对齐的文本
            sample_rate: 音频采样率
            output_format: 输出格式（Text/SRT/VTT）
            
        Returns:
            aligned_text: 对齐后的文本
            timestamps: 时间戳数据（JSON格式）
            subtitle_text: 字幕格式文本（SRT或VTT）
        """
        if aligner_model is None:
            error_msg = "未加载强制对齐模型"
            print(f"【强制对齐推理】{error_msg}")
            task_id = str(uuid.uuid4())
            error_output = {
                "task_id": task_id,
                "audio_info": {
                    "audio_duration": 0,
                    "audio_sample_rate": sample_rate
                },
                "segments": [],
                "status": "failed",
                "error_msg": error_msg
            }
            return ("", json.dumps(error_output, ensure_ascii=False), "")
        
        if not text:
            error_msg = "文本内容为空"
            print(f"【强制对齐推理】{error_msg}")
            task_id = str(uuid.uuid4())
            error_output = {
                "task_id": task_id,
                "audio_info": {
                    "audio_duration": 0,
                    "audio_sample_rate": sample_rate
                },
                "segments": [],
                "status": "failed",
                "error_msg": error_msg
            }
            return ("", json.dumps(error_output, ensure_ascii=False), "")
        
        if audio is None:
            error_msg = "音频数据为空"
            print(f"【强制对齐推理】{error_msg}")
            task_id = str(uuid.uuid4())
            error_output = {
                "task_id": task_id,
                "audio_info": {
                    "audio_duration": 0,
                    "audio_sample_rate": sample_rate
                },
                "segments": [],
                "status": "failed",
                "error_msg": error_msg
            }
            return ("", json.dumps(error_output, ensure_ascii=False), "")
        
        try:
            print(f"【强制对齐推理】开始处理，文本长度: {len(text)}")
            
            # 使用统一的音频数据提取方法
            waveform, actual_sample_rate = self._extract_audio_data(audio)
            if waveform is None:
                error_msg = "音频数据提取失败"
                print(f"【强制对齐推理】{error_msg}")
                task_id = str(uuid.uuid4())
                error_output = {
                    "task_id": task_id,
                    "audio_info": {
                        "audio_duration": 0,
                        "audio_sample_rate": sample_rate
                    },
                    "segments": [],
                    "status": "failed",
                    "error_msg": error_msg
                }
                return (text, json.dumps(error_output, ensure_ascii=False), "")
            
            # 使用提取到的采样率
            sample_rate = actual_sample_rate
            
            # 执行强制对齐
            result = aligner_model.align(waveform, text, language=language)
            
            print(f"【强制对齐推理】对齐结果类型: {type(result)}")
            print(f"【强制对齐推理】对齐结果内容: {result}")
            
            if result is None:
                error_msg = "强制对齐失败"
                print(f"【强制对齐推理】{error_msg}")
                task_id = str(uuid.uuid4())
                error_output = {
                    "task_id": task_id,
                    "audio_info": {
                        "audio_duration": 0,
                        "audio_sample_rate": sample_rate
                    },
                    "segments": [],
                    "status": "failed",
                    "error_msg": error_msg
                }
                return (text, json.dumps(error_output, ensure_ascii=False), "")
            
            # 生成任务ID
            task_id = str(uuid.uuid4())
            
            # 计算音频时长（秒）
            audio_duration = len(waveform) / sample_rate
            
            # 处理对齐结果
            aligned_text = text
            segments_data = []
            
            # 提取时间戳信息
            if hasattr(result, '__dict__'):
                # 如果是对象类型
                if hasattr(result, 'segments'):
                    for segment in result.segments:
                        segment_data = {
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text
                        }
                        if hasattr(segment, 'words'):
                            segment_data["words"] = []
                            for word in segment.words:
                                word_data = {
                                    "word": word.word,
                                    "start": word.start,
                                    "end": word.end
                                }
                                if hasattr(word, 'phonemes'):
                                    word_data["phonemes"] = []
                                    for phoneme in word.phonemes:
                                        phoneme_data = {
                                            "phoneme": phoneme.phoneme,
                                            "start": phoneme.start,
                                            "end": phoneme.end
                                        }
                                        word_data["phonemes"].append(phoneme_data)
                                segment_data["words"].append(word_data)
                        segments_data.append(segment_data)
                # 尝试另一种对象结构
                elif hasattr(result, 'word_segments'):
                    for segment in result.word_segments:
                        segment_data = {
                            "start": segment.start_time,
                            "end": segment.end_time,
                            "text": segment.word
                        }
                        segments_data.append(segment_data)
            elif isinstance(result, dict):
                # 如果是字典类型
                segments_data = result.get('segments', [])
                if not segments_data:
                    segments_data = result.get('word_segments', [])
                    if not segments_data:
                        # 尝试直接提取时间戳信息
                        if 'start' in result and 'end' in result and 'text' in result:
                            segment_data = {
                                "start": result['start'],
                                "end": result['end'],
                                "text": result['text']
                            }
                            segments_data.append(segment_data)
            
            # 如果仍然没有提取到segments，尝试创建一个包含原始文本的段落
            if not segments_data and text:
                segment_data = {
                    "start": 0,
                    "end": audio_duration,
                    "text": text
                }
                segments_data.append(segment_data)
            
            print(f"【强制对齐推理】提取到 {len(segments_data)} 个段落")
            print(f"【强制对齐推理】segments_data内容: {segments_data}")
            
            # 组装标准化输出
            timestamps = {
                "task_id": task_id,
                "audio_info": {
                    "audio_duration": round(audio_duration, 3),
                    "audio_sample_rate": sample_rate
                },
                "segments": segments_data,
                "status": "success",
                "error_msg": ""
            }
            
            # 生成字幕文本
            subtitle_text = ""
            print(f"【强制对齐推理】输出格式: {output_format}")
            if output_format == "SRT":
                subtitle_text = self._generate_srt(segments_data)
                print(f"【强制对齐推理】生成的SRT字幕: {subtitle_text}")
                if not subtitle_text.strip():
                    subtitle_text = "1\n00:00:00,000 --> 00:00:00,000\n未检测到对齐段落\n\n"
            elif output_format == "VTT":
                subtitle_text = self._generate_vtt(segments_data)
                print(f"【强制对齐推理】生成的VTT字幕: {subtitle_text}")
                if not subtitle_text.strip() or subtitle_text == "WEBVTT\n\n":
                    subtitle_text = "WEBVTT\n\n00:00:00.000 --> 00:00:00.000\n未检测到对齐段落\n\n"
            elif output_format == "JSON":
                subtitle_text = json.dumps(timestamps, ensure_ascii=False, indent=2)
                print(f"【强制对齐推理】生成的JSON字幕: {subtitle_text[:100]}...")  # 只显示前100个字符
            else:  # Text格式
                # 生成包含时间戳的文本
                if segments_data:
                    for segment in segments_data:
                        start_time = segment.get("start", 0)
                        end_time = segment.get("end", 0)
                        text = segment.get("text", "")
                        subtitle_text += f"[{start_time:.3f} - {end_time:.3f}] {text}\n"
                    print(f"【强制对齐推理】生成的Text字幕: {subtitle_text}")
                else:
                    subtitle_text = "未检测到对齐段落"
                    print(f"【强制对齐推理】未检测到对齐段落")
            
            # 确保aligned_text也包含有用信息
            if aligned_text == text and segments_data:
                aligned_text_with_timestamps = ""
                for segment in segments_data:
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    seg_text = segment.get("text", "")
                    aligned_text_with_timestamps += f"[{start_time:.3f} - {end_time:.3f}] {seg_text}\n"
                aligned_text = aligned_text_with_timestamps.strip() or text
            
            print(f"【强制对齐推理】完成，生成 {len(segments_data)} 个段落")
            
            return (aligned_text, json.dumps(timestamps, ensure_ascii=False), subtitle_text)
            
        except Exception as e:
            error_msg = str(e)
            print(f"【强制对齐推理错误】执行失败: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # 生成错误情况下的标准化输出
            task_id = str(uuid.uuid4())
            error_output = {
                "task_id": task_id,
                "audio_info": {
                    "audio_duration": 0,
                    "audio_sample_rate": sample_rate
                },
                "segments": [],
                "status": "failed",
                "error_msg": error_msg
            }
            
            return (text, json.dumps(error_output, ensure_ascii=False), "")


    def _generate_srt(self, segments):
        """生成SRT格式字幕"""
        srt_content = ""
        
        for i, segment in enumerate(segments, 1):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "")
            
            if text.strip():
                # 转换时间格式（秒 -> SRT时间码）
                start_time_str = self._seconds_to_srt_time(start_time)
                end_time_str = self._seconds_to_srt_time(end_time)
                
                # 添加到SRT内容
                srt_content += f"{i}\n"
                srt_content += f"{start_time_str} --> {end_time_str}\n"
                srt_content += f"{text}\n\n"
        
        return srt_content
    
    def _generate_vtt(self, segments):
        """生成VTT格式字幕"""
        vtt_content = "WEBVTT\n\n"
        
        for segment in segments:
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "")
            
            if text.strip():
                # 转换时间格式（秒 -> VTT时间码）
                start_time_str = self._seconds_to_vtt_time(start_time)
                end_time_str = self._seconds_to_vtt_time(end_time)
                
                # 添加到VTT内容
                vtt_content += f"{start_time_str} --> {end_time_str}\n"
                vtt_content += f"{text}\n\n"
        
        return vtt_content
    
    def _seconds_to_srt_time(self, seconds):
        """将秒转换为SRT时间格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _seconds_to_vtt_time(self, seconds):
        """将秒转换为VTT时间格式 (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def _extract_audio_data(self, audio_input):
        """提取音频数据和采样率
        
        Args:
            audio_input: 音频输入，支持以下格式：
                - dict: {"waveform": tensor, "sample_rate": int}
                - torch.Tensor: 音频波形张量
                - numpy.ndarray: 音频波形数组
                - str: 音频文件路径
        
        Returns:
            tuple: (waveform_numpy_array, sample_rate)
        """
        try:
            waveform = None
            sample_rate = 16000  # 默认16kHz
            
            if isinstance(audio_input, dict):
                waveform = audio_input.get("waveform")
                if waveform is None:
                    waveform = audio_input.get("samples")
                sample_rate = audio_input.get("sample_rate", 16000)
            elif isinstance(audio_input, torch.Tensor):
                waveform = audio_input
            elif isinstance(audio_input, np.ndarray):
                waveform = audio_input
            elif isinstance(audio_input, str) and os.path.exists(audio_input):
                # 支持从文件加载音频
                try:
                    import soundfile as sf
                    waveform, sample_rate = sf.read(audio_input)
                except ImportError:
                    from scipy.io import wavfile
                    sample_rate, waveform = wavfile.read(audio_input)
            else:
                print(f"【强制对齐推理错误】不支持的音频输入类型: {type(audio_input)}")
                return None, 16000
            
            # 转换为numpy数组
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            
            # 处理维度
            if waveform.ndim > 1:
                if waveform.ndim == 3:  # [batch, channels, samples]
                    waveform = waveform.squeeze(0)
                if waveform.ndim == 2:  # [channels, samples]
                    waveform = waveform.mean(axis=0)  # 转为单声道
            
            # 确保是float32
            if waveform.dtype != np.float32:
                if waveform.dtype == np.int16:
                    waveform = waveform.astype(np.float32) / 32768.0
                else:
                    waveform = waveform.astype(np.float32)
            
            return waveform, sample_rate
            
        except Exception as e:
            print(f"【强制对齐推理错误】音频数据提取失败: {str(e)}")
            return None, 16000


# 节点映射
NODE_CLASS_MAPPINGS = {
    "forced_aligner_inference": forced_aligner_inference,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "forced_aligner_inference": "强制对齐推理",
}