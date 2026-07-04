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
import json
import torch
import numpy as np
import folder_paths
from fractions import Fraction
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
            }
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "AUDIO")
    RETURN_NAMES = ("images", "video", "audio")
    FUNCTION = "load_video"
    CATEGORY = "omni-llm"
    
    def load_video(self, video, force_rate=0, custom_width=0, custom_height=0, 
                   frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
        """
        加载视频并返回图像序列、视频对象和音频
        """
        if video == "None" or not video:
            # 返回空数据
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_audio = {"sample_rate": 44100, "waveform": torch.zeros((1, 2, 100), dtype=torch.float32)}
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
        # 实现 ComfyUI 标准视频对象接口
        class VideoOutput:
            def __init__(self, frames, fps, width, height, duration, audio=None):
                self.frames = frames  # shape: [frames, height, width, channels]
                self.fps = fps
                self.width = width
                self.height = height
                self.duration = duration
                self.audio = audio  # {"waveform": tensor [B, C, T], "sample_rate": int}
            
            def get_dimensions(self):
                return (self.width, self.height)
            
            def get_frames(self):
                return self.frames
            
            def get_fps(self):
                return self.fps
            
            def save_to(self, path, **kwargs):
                """保存视频到文件，使用 PyAV 库（与官方实现一致）"""
                import math
                
                format_val = kwargs.get('format')
                codec = kwargs.get('codec')
                
                # 只支持 MP4 格式和 H264 编码器
                if hasattr(format_val, 'value'):
                    format_str = format_val.value
                else:
                    format_str = str(format_val) if format_val else "auto"
                
                if hasattr(codec, 'value'):
                    codec_str = codec.value
                else:
                    codec_str = str(codec) if codec else "auto"
                
                if format_str != "auto" and format_str != "mp4":
                    raise ValueError("Only MP4 format is supported for now")
                if codec_str != "auto" and codec_str != "h264":
                    raise ValueError("Only H264 codec is supported for now")
                
                # 设置输出格式
                extra_kwargs = {}
                if isinstance(format_val, object) and hasattr(format_val, 'value'):
                    if format_val.value != "auto":
                        extra_kwargs["format"] = format_val.value
                elif format_str != "auto":
                    extra_kwargs["format"] = format_str
                
                with av.open(path, mode='w', options={'movflags': 'use_metadata_tags'}, **extra_kwargs) as output:
                    # 添加元数据
                    metadata = kwargs.get('metadata')
                    if metadata is not None:
                        for key, value in metadata.items():
                            output.metadata[key] = json.dumps(value)
                    
                    # 获取帧速率
                    frame_rate = Fraction(round(self.fps * 1000), 1000)
                    
                    # 创建视频流
                    video_stream = output.add_stream('h264', rate=frame_rate)
                    video_stream.width = self.width
                    video_stream.height = self.height
                    video_stream.pix_fmt = 'yuv420p'
                    
                    # 创建音频流
                    audio_sample_rate = 1
                    audio_stream = None
                    waveform = None
                    layout = 'stereo'
                    
                    if self.audio is not None and 'waveform' in self.audio:
                        audio_sample_rate = int(self.audio.get('sample_rate', 44100))
                        waveform = self.audio['waveform']
                        # 裁剪音频到视频长度
                        waveform = waveform[0, :, :math.ceil((audio_sample_rate / frame_rate) * self.frames.shape[0])]
                        layout = {1: 'mono', 2: 'stereo', 6: '5.1'}.get(waveform.shape[0], 'stereo')
                        audio_stream = output.add_stream('aac', rate=audio_sample_rate, layout=layout)
                    
                    # 编码视频帧
                    for i in range(self.frames.shape[0]):
                        frame_tensor = self.frames[i]
                        img = (frame_tensor * 255).clamp(0, 255).byte().cpu().numpy()
                        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                        frame = frame.reformat(format='yuv420p')
                        packet = video_stream.encode(frame)
                        output.mux(packet)
                    
                    # 刷新视频编码器
                    packet = video_stream.encode(None)
                    output.mux(packet)
                    
                    # 编码音频
                    if audio_stream and waveform is not None:
                        frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format='fltp', layout=layout)
                        frame.sample_rate = audio_sample_rate
                        frame.pts = 0
                        output.mux(audio_stream.encode(frame))
                        
                        # 刷新音频编码器
                        output.mux(audio_stream.encode(None))
        
        video_output = VideoOutput(images, target_fps, target_width, target_height, 
                                  len(frames) / target_fps if target_fps > 0 else 0, 
                                  audio_data)
        
        return (images, video_output, audio_data)

    def _extract_audio(self, video_path: str) -> dict:
        """
        从视频中提取音频
        返回格式: {"waveform": torch.Tensor [B, C, T], "sample_rate": int}
        B: batch size, C: channels, T: time/samples
        与 VideoHelperSuite 保持一致
        """
        import subprocess
        import re
        
        try:
            # 使用 ffmpeg 提取音频，与 VHS 保持一致
            args = ["ffmpeg", "-i", video_path, "-f", "f32le", "-"]
            res = subprocess.run(args, capture_output=True, check=True)
            
            # 从原始字节数据创建 torch tensor
            audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
            
            # 解析 ffmpeg 输出获取采样率和通道数
            match = re.search(r', (\d+) Hz, (\w+),', res.stderr.decode('utf-8', 'backslashreplace'))
            
            if match:
                sample_rate = int(match.group(1))
                channels = {"mono": 1, "stereo": 2}.get(match.group(2), 2)
            else:
                sample_rate = 44100
                channels = 2
            
            # 重组音频数据: [samples] -> [samples, channels] -> [channels, samples] -> [1, channels, samples]
            audio = audio.reshape((-1, channels)).transpose(0, 1).unsqueeze(0)
            
            return {
                "sample_rate": sample_rate,
                "waveform": audio
            }
            
        except subprocess.CalledProcessError as e:
            print(f"音频提取失败: {e.stderr.decode('utf-8', 'backslashreplace')}")
            return {
                "sample_rate": 44100,
                "waveform": torch.zeros((1, 2, 100), dtype=torch.float32)
            }
        except Exception as e:
            print(f"音频提取失败: {e}")
            return {
                "sample_rate": 44100,
                "waveform": torch.zeros((1, 2, 100), dtype=torch.float32)
            }

NODE_CLASS_MAPPINGS = {
    "VideoLoader": VideoLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLoader": "Video Loader"
}
