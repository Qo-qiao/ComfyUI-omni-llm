# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm 多模型音频合成节点

多模型音频合成节点，支持多种TTS模型的音频生成
可根据角色配置和对话内容生成情感丰富的语音

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import torch
import numpy as np
import sys
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import folder_paths
from common import DialogueParser


class multi_model_tts:
    """
    多模型音频合成节点
    支持加载多个TTS模型，为不同角色分配不同模型，合成多角色对话音频
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dialogue_text": ("STRING", {"multiline": True, "default": "角色1: 你好，今天天气怎么样？\n角色2: 今天天气很好，阳光明媚。", "tooltip": "多人对话文本（使用格式：角色名: 文本 或 角色名(情感): 文本）"}),
            },
            "optional": {
                # 支持动态添加模型（最多4个）
                "tts_model_1": ("TTSMODEL", {"tooltip": "TTS模型1"}),
                "tts_model_2": ("TTSMODEL", {"tooltip": "TTS模型2"}),
                "tts_model_3": ("TTSMODEL", {"tooltip": "TTS模型3"}),
                "tts_model_4": ("TTSMODEL", {"tooltip": "TTS模型4"}),

                "pause_duration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "说话人之间的停顿时间（秒）"}),
                "add_silence": ("BOOLEAN", {"default": True, "tooltip": "是否在说话人之间添加静音"}),
                "force_offload": ("BOOLEAN", {"default": False, "tooltip": "强制卸载模型释放显存"}),
                "output_format": (["wav", "flac", "mp3", "ogg"], {"default": "wav", "tooltip": "输出音频格式"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "AUDIO", "DICT")
    RETURN_NAMES = ("text", "audio", "dialogue_info")
    FUNCTION = "generate_multi_model_audio"
    CATEGORY = "llama-cpp-vlm"
    
    def generate_multi_model_audio(self, dialogue_text,
                                  tts_model_1=None, tts_model_2=None, tts_model_3=None, tts_model_4=None,
                                  pause_duration=0.5, add_silence=True, force_offload=False, output_format="wav"):
        """
        生成多模型多角色对话音频
        根据连接的模型和角色配置数量动态处理
        """
        # VOICE_MAP: 与 llama_cpp_tts_loader 保持一致
        VOICE_MAP = {
            "Vivian": 0,
            "Serena": 1,
            "Uncle_Fu": 2,
            "Dylan": 3,
            "Eric": 4,
            "Ryan": 5,
            "Aiden": 6,
            "Ono_Anna": 7,
            "Sohee": 8,
        }

        # 预定义的默认音色列表（轮流分配）
        default_voices = ["Vivian", "Uncle_Fu", "Serena", "Dylan"]

        # 构建角色配置
        role_configs = {}

        # 收集所有模型
        models = [tts_model_1, tts_model_2, tts_model_3, tts_model_4]

        # 从TTS模型的config中读取角色配置
        for i, tts_model in enumerate(models):
            if tts_model:
                role_name = f"角色{i+1}"

                # 从TTS模型配置中获取音色信息
                model_config = getattr(tts_model, 'config', {})
                speaker_id = model_config.get('speaker_id', 0)

                # 将speaker_id映射回音色名称
                voice_id_to_name = {v: k for k, v in VOICE_MAP.items()}
                voice_name = voice_id_to_name.get(speaker_id, default_voices[i % len(default_voices)])

                # 读取其他配置
                emotion = model_config.get('emotion', 'default')
                speed = model_config.get('speed', 1.0)
                pitch = model_config.get('pitch', 0.0)
                volume = model_config.get('volume', 1.0)

                role_configs[role_name] = {
                    "model": tts_model,
                    "voice": voice_name,
                    "speaker_id": speaker_id,
                    "emotion": emotion,
                    "speed": speed,
                    "pitch": pitch,
                    "volume": volume
                }
                print(f"【多模型TTS】从模型读取配置: {role_name} -> {voice_name}(id={speaker_id}), 情感={emotion}, 速度={speed}")
        
        # 过滤掉没有模型的角色
        valid_roles = {name: config for name, config in role_configs.items() if config["model"] is not None}
        
        if not valid_roles:
            print("【多模型TTS】未提供有效的TTS模型")
            empty_audio = {
                "waveform": torch.tensor([], dtype=torch.float32),
                "sample_rate": 24000
            }
            return ("", empty_audio, {"error": "未提供有效的TTS模型"})
        
        if not dialogue_text or not dialogue_text.strip():
            print("【多模型TTS】对话文本为空")
            empty_audio = {
                "waveform": torch.tensor([], dtype=torch.float32),
                "sample_rate": 24000
            }
            return ("", empty_audio, {"error": "对话文本为空"})
        
        try:
            print(f"【多模型TTS】正在解析对话...")
            
            # 解析对话
            dialogue = DialogueParser.parse_dialogue(
                dialogue_text,
                default_speaker=list(valid_roles.keys())[0] if valid_roles else "角色1",
                default_emotion="默认"
            )
            
            if not dialogue:
                print("【多模型TTS】未能解析对话，使用默认说话人")
                if not valid_roles:
                    empty_audio = {
                        "waveform": torch.tensor([], dtype=torch.float32),
                        "sample_rate": 24000
                    }
                    return ("", empty_audio, {"error": "未提供有效的TTS模型"})
                default_role = list(valid_roles.keys())[0]
                dialogue = [{
                    "speaker": default_role,
                    "emotion": "默认",
                    "text": dialogue_text.strip()
                }]
            
            # 验证对话
            is_valid, errors = DialogueParser.validate_dialogue(dialogue)
            if not is_valid:
                print(f"【多模型TTS】对话验证失败：")
                for error in errors:
                    print(f"  - {error}")
                # 返回空的音频对象，避免下游节点报错
                empty_audio = {
                    "waveform": torch.tensor([], dtype=torch.float32),
                    "sample_rate": 24000
                }
                return ("", empty_audio, {"error": "对话验证失败", "errors": errors})
            
            # 检查对话中的角色是否有对应的模型
            for segment in dialogue:
                speaker = segment["speaker"]
                if speaker not in valid_roles:
                    print(f"【多模型TTS】角色 {speaker} 没有对应的TTS模型")
                    empty_audio = {
                        "waveform": torch.tensor([], dtype=torch.float32),
                        "sample_rate": 24000
                    }
                    return ("", empty_audio, {"error": f"角色 {speaker} 没有对应的TTS模型"})
            
            # 打印对话信息
            dialogue_info = DialogueParser.format_dialogue_info(dialogue)
            print(f"【多模型TTS】解析结果：\n{dialogue_info}")
            
            # 为每个说话人生成音频（顺序处理，TTS模型不支持并发）
            print(f"【多模型TTS】顺序生成音频（共{len(dialogue)}段）...")

            audio_segments = []
            total_duration = 0.0

            for i, segment in enumerate(dialogue):
                speaker = segment["speaker"]
                segment_emotion = segment["emotion"]
                text = segment["text"]

                # 获取角色配置
                role_config = valid_roles[speaker]
                tts_model = role_config["model"]

                # 从TTS模型配置中获取参数
                voice_name = role_config["voice"]
                speaker_id = role_config["speaker_id"]

                # 合并配置
                emotion = segment_emotion if segment_emotion != "默认" else role_config["emotion"]
                speed = role_config["speed"]
                pitch = role_config["pitch"]
                volume = role_config["volume"]

                print(f"【多模型TTS】开始生成第{i+1}段音频: [{speaker}]({emotion}) 音色={voice_name}(id={speaker_id})")

                # 调用TTS模型生成音频
                try:
                    audio_result = tts_model.synthesize(
                        text=text,
                        speaker_id=speaker_id,
                        speed=speed,
                        emotion=emotion,
                        pitch=pitch,
                        volume=volume
                    )
                except Exception as e:
                    print(f"【多模型TTS】第{i+1}段音频生成失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

                if audio_result is None:
                    print(f"【多模型TTS】第{i+1}段音频生成失败（返回None）")
                    continue

                waveform = audio_result.get("waveform")
                sample_rate = audio_result.get("sample_rate", 24000)

                if waveform is None:
                    print(f"【多模型TTS】第{i+1}段音频波形为空")
                    continue

                # 处理waveform格式
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.cpu().numpy() if waveform.is_cuda else waveform.numpy()

                # 确保是一维数组
                if waveform.ndim > 1:
                    waveform = waveform.flatten()
                    if waveform.ndim > 1:
                        waveform = waveform.mean(axis=0)

                # 计算音频时长
                segment_duration = len(waveform) / sample_rate

                audio_segments.append({
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                    "duration": segment_duration,
                    "speaker": speaker,
                    "emotion": emotion,
                    "text": text,
                    "voice": voice_name,
                    "speaker_id": speaker_id,
                    "model_type": tts_model.model_type
                })
                total_duration += segment_duration

                print(f"【多模型TTS】第{i+1}段完成: 时长={segment_duration:.2f}秒")

            if not audio_segments:
                print("【多模型TTS】没有生成任何音频片段")
                empty_audio = {
                    "waveform": torch.tensor([], dtype=torch.float32),
                    "sample_rate": 24000
                }
                return ("", empty_audio, {"error": "没有生成任何音频片段"})
            
            # 拼接音频
            print(f"【多模型TTS】正在拼接音频...")
            final_audio = self._concatenate_audio_segments(
                audio_segments,
                pause_duration=pause_duration,
                add_silence=add_silence
            )
            
            if final_audio is None or len(final_audio) == 0:
                print("【多模型TTS】拼接后的音频为空")
                empty_audio = {
                    "waveform": torch.tensor([], dtype=torch.float32),
                    "sample_rate": 24000
                }
                return ("", empty_audio, {"error": "拼接后的音频为空"})
            
            # 保存音频文件
            print(f"【多模型TTS】正在保存音频...")
            audio_path = self._save_audio(
                final_audio,
                audio_segments[0]["sample_rate"],
                output_format=output_format
            )
            
            # 创建AUDIO格式输出
            waveform = torch.from_numpy(final_audio)
            # 确保waveform是三维张量 [batch, channels, samples]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
            elif waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
            audio_output = {
                "waveform": waveform,
                "sample_rate": audio_segments[0]["sample_rate"]
            }
            
            # 构建对话信息
            dialogue_info_dict = {
                "total_segments": len(dialogue),
                "total_duration": total_duration,
                "sample_rate": audio_segments[0]["sample_rate"],
                "segments": [
                    {
                        "speaker": seg["speaker"],
                        "emotion": seg["emotion"],
                        "text": seg["text"],
                        "duration": seg["duration"],
                        "voice": seg.get("voice", "Unknown"),
                        "speaker_id": seg.get("speaker_id", 0),
                        "model_type": seg["model_type"]
                    }
                    for seg in audio_segments
                ],
                "pause_duration": pause_duration,
                "add_silence": add_silence,
                "role_configs": {
                    role: {
                        "voice": config["voice"],
                        "emotion": config["emotion"],
                        "speed": config["speed"],
                        "pitch": config["pitch"],
                        "volume": config["volume"],
                        "model_type": config["model"].model_type if config["model"] else None
                    }
                    for role, config in role_configs.items()
                }
            }
            
            print(f"【多模型TTS】音频生成成功: {audio_path}")
            print(f"【多模型TTS】总时长: {total_duration:.2f}秒")
            
            # 强制卸载模型
            if force_offload:
                print("【多模型TTS】强制卸载模型释放显存...")
                # 卸载所有TTS模型
                for model in models:
                    if model and hasattr(model, 'unload'):
                        try:
                            model.unload()
                            print(f"【多模型TTS】模型已卸载")
                        except Exception as e:
                            print(f"【多模型TTS】卸载模型时出错: {str(e)}")
            
            return (dialogue_text, audio_output, dialogue_info_dict)
            
        except Exception as e:
            print(f"【多模型TTS错误】生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            empty_audio = {
                "waveform": torch.tensor([], dtype=torch.float32),
                "sample_rate": 24000
            }
            return ("", empty_audio, {"error": f"生成失败: {str(e)}"})
    
    def _concatenate_audio_segments(self, audio_segments, pause_duration=0.5, 
                                   add_silence=True):
        """
        拼接音频片段
        
        Args:
            audio_segments: 音频片段列表
            pause_duration: 停顿时长（秒）
            add_silence: 是否添加静音
        
        Returns:
            拼接后的音频波形
        """
        if not audio_segments:
            return None
        
        sample_rate = audio_segments[0]["sample_rate"]
        
        # 计算静音样本数
        silence_samples = int(pause_duration * sample_rate) if add_silence else 0
        
        # 创建静音片段
        silence = np.zeros(silence_samples, dtype=np.float32)
        
        # 拼接所有音频片段
        concatenated_audio = []
        
        for i, segment in enumerate(audio_segments):
            waveform = segment["waveform"]
            
            # 确保波形是一维数组
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0)
            
            # 添加音频片段
            concatenated_audio.append(waveform)
            
            # 如果不是最后一个片段，添加静音
            if i < len(audio_segments) - 1 and add_silence:
                concatenated_audio.append(silence)
        
        # 拼接所有片段
        final_audio = np.concatenate(concatenated_audio)
        
        return final_audio
    
    def _save_audio(self, waveform, sample_rate, output_format="wav"):
        """
        保存音频文件
        
        Args:
            waveform: 音频波形
            sample_rate: 采样率
            output_format: 输出音频格式 (wav, flac, mp3, ogg)
        
        Returns:
            音频文件路径
        """
        try:
            import soundfile as sf
            
            # 确定输出目录
            try:
                output_dir = folder_paths.get_output_directory()
                audio_output_dir = os.path.join(output_dir, "audio")
                os.makedirs(audio_output_dir, exist_ok=True)
            except:
                audio_output_dir = "."
                os.makedirs(audio_output_dir, exist_ok=True)
            
            # 生成输出文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"multi_model_dialogue_{timestamp}.{output_format}"
            audio_path = os.path.join(audio_output_dir, output_filename)
            
            # 保存音频文件
            sf.write(
                audio_path,
                waveform,
                sample_rate,
                format=output_format.upper(),
                subtype='PCM_16' if output_format == 'wav' else None
            )
            
            return audio_path
            
        except ImportError:
            print("【多模型TTS错误】未安装soundfile库，请运行: pip install soundfile")
            return ""
        except Exception as e:
            print(f"【多模型TTS错误】保存音频失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""

NODE_CLASS_MAPPINGS = {
    "multi_model_tts": multi_model_tts
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "multi_model_tts": "Multi-Model TTS"
}
