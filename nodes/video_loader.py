# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Video Loader Node

视频加载节点，用于加载视频文件并输出图像序列、视频和音频

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import av
import torch
import numpy as np
import folder_paths
from typing import Tuple, Optional


class VideoLoader:
    """
    视频加载节点
    - 无输入端口
    - 输出端口：图像序列、视频、音频
    - 支持视频文件选择和上传
    - 支持自定义帧率、分辨率、帧数限制等参数
    """
    
    @classmethod
    def INPUT_TYPES(s):
        # 获取输入目录中的视频文件
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            # 过滤视频文件
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.3gp')
            files = [f for f in files if f.lower().endswith(video_extensions)]
        
        return {
            "required": {
                "video": (sorted(files) if files else ["None"], {"video_upload": True}),
            },
            "optional": {
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 120, "step": 1, "tooltip": "强制帧率，0=使用视频原始帧率"}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64, "tooltip": "自定义宽度，0=保持原始尺寸"}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64, "tooltip": "自定义高度，0=保持原始尺寸"}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "tooltip": "帧数读取上限，0=无限制"}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "tooltip": "跳过前X帧"}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "间隔，每N帧取1帧"}),
                "format": (["Mochi", "LTX-2", "WAN2.2", "Original"], {"default": "Mochi", "tooltip": "输出格式，影响颜色通道顺序"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "AUDIO")
    RETURN_NAMES = ("images", "video", "audio")
    FUNCTION = "load_video"
    CATEGORY = "llama-cpp-vlm"
    
    def load_video(self, video, force_rate=0, custom_width=0, custom_height=0, 
                   frame_load_cap=0, skip_first_frames=0, select_every_nth=1, format="Mochi"):
        """
        加载视频并返回图像序列、视频对象和音频
        """
        if video == "None" or not video:
            # 返回空数据
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_audio = {"sample_rate": 44100, "waveform": torch.zeros((1, 1), dtype=torch.float32)}
            return (empty_image, None, empty_audio)
        
        # 获取视频完整路径
        video_path = folder_paths.get_annotated_filepath(video)
        
        if not os.path.exists(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")
        
        # 使用 PyAV 读取视频
        container = av.open(video_path)
        
        # 获取视频流
        video_stream = None
        audio_stream = None
        
        for stream in container.streams:
            if isinstance(stream, av.video.stream.VideoStream):
                video_stream = stream
            elif isinstance(stream, av.audio.stream.AudioStream):
                audio_stream = stream
        
        if video_stream is None:
            raise ValueError("未找到视频流")
        
        # 获取视频信息
        original_fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        target_fps = force_rate if force_rate > 0 else original_fps
        
        # 计算目标尺寸
        target_width = custom_width if custom_width > 0 else video_stream.width
        target_height = custom_height if custom_height > 0 else video_stream.height
        
        # 读取视频帧
        frames = []
        loaded_frames = 0
        frame_count = 0
        
        for frame in container.decode(video_stream):
            frame_count += 1
            
            # 跳过前N帧
            if frame_count <= skip_first_frames:
                continue
            
            # 间隔采样
            if (frame_count - skip_first_frames - 1) % select_every_nth != 0:
                continue
            
            # 帧数限制
            if frame_load_cap > 0 and loaded_frames >= frame_load_cap:
                break
            
            # 转换帧为 numpy 数组
            img = frame.to_ndarray(format='rgb24')
            
            # 调整尺寸
            if img.shape[1] != target_width or img.shape[0] != target_height:
                import cv2
                img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            # 归一化到 0-1 范围
            img = img.astype(np.float32) / 255.0
            
            # 根据格式调整通道顺序
            if format == "Mochi":
                # RGB 顺序，保持原样
                pass
            elif format == "LTX-2":
                # RGB 顺序，保持原样
                pass
            elif format == "WAN2.2":
                # RGB 顺序，保持原样
                pass
            else:
                # Original，保持原样
                pass
            
            frames.append(img)
            loaded_frames += 1
            frame_count += 1
        
        # 关闭视频容器
        container.close()
        
        if not frames:
            raise ValueError("未能从视频中读取任何帧")
        
        # 转换为 torch tensor
        images = torch.from_numpy(np.stack(frames, axis=0))
        
        # 重新打开容器读取音频
        audio_data = self._extract_audio(video_path)
        
        # 创建视频对象（用于 VIDEO 输出）
        # 这里我们返回图像序列作为视频表示
        video_output = {
            "frames": images,
            "fps": target_fps,
            "width": target_width,
            "height": target_height,
            "duration": len(frames) / target_fps if target_fps > 0 else 0
        }
        
        return (images, video_output, audio_data)

    def _extract_audio(self, video_path: str) -> dict:
        """
        从视频中提取音频
        """
        try:
            container = av.open(video_path)
            
            # 查找音频流
            audio_stream = None
            for stream in container.streams:
                if isinstance(stream, av.audio.stream.AudioStream):
                    audio_stream = stream
                    break
            
            if audio_stream is None:
                # 没有音频流，返回空音频
                return {
                    "sample_rate": 44100,
                    "waveform": torch.zeros((1, 1), dtype=torch.float32)
                }
            
            # 获取音频信息
            sample_rate = audio_stream.sample_rate
            
            # 读取音频帧
            audio_frames = []
            for frame in container.decode(audio_stream):
                # 转换为 numpy 数组
                audio_data = frame.to_ndarray()
                audio_frames.append(audio_data)
            
            container.close()
            
            if not audio_frames:
                return {
                    "sample_rate": sample_rate,
                    "waveform": torch.zeros((1, 1), dtype=torch.float32)
                }
            
            # 合并所有音频帧
            audio_array = np.concatenate(audio_frames, axis=0)
            
            # 如果是多声道，转换为单声道
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1, keepdims=True)
            
            # 转换为 torch tensor，添加 batch 维度
            waveform = torch.from_numpy(audio_array).float()
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
            elif len(waveform.shape) == 2:
                if waveform.shape[0] > waveform.shape[1]:
                    # 转置为 (channels, samples)
                    waveform = waveform.T
                waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
            
            return {
                "sample_rate": sample_rate,
                "waveform": waveform
            }
            
        except Exception as e:
            print(f"音频提取失败: {e}")
            return {
                "sample_rate": 44100,
                "waveform": torch.zeros((1, 1), dtype=torch.float32)
            }

NODE_CLASS_MAPPINGS = {
    "VideoLoader": VideoLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLoader": "Video Loader"
}