# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm 对话解析器

解析多人对话文本，识别说话人和情绪

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import List, Dict, Tuple, Optional

class DialogueParser:
    """
    对话解析器
    支持多种对话格式
    """
    
    # 说话人映射
    SPEAKER_MAP = {
        "女声1": 0,
        "女声2": 1,
        "男声1": 2,
        "男声2": 3,
    }
    
    # 情绪映射
    EMOTION_MAP = {
        "默认": "default",
        "开心": "happy",
        "悲伤": "sad",
        "愤怒": "angry",
        "惊讶": "surprised",
        "平静": "calm",
        "兴奋": "excited",
        "温柔": "gentle",
    }
    
    @classmethod
    def parse_dialogue(cls, text: str, default_speaker: str = "女声1", 
                    default_emotion: str = "默认") -> List[Dict]:
        """
        解析对话文本
        
        Args:
            text: 对话文本
            default_speaker: 默认说话人
            default_emotion: 默认情绪
        
        Returns:
            对话片段列表，每个片段包含说话人、情绪、文本
        """
        if not text or not text.strip():
            return []
        
        # 尝试不同的解析模式
        dialogue = cls._parse_bracket_format(text, default_speaker, default_emotion)
        if not dialogue:
            dialogue = cls._parse_colon_format(text, default_speaker, default_emotion)
        if not dialogue:
            dialogue = cls._parse_dash_format(text, default_speaker, default_emotion)
        if not dialogue:
            # 默认格式：整个文本作为一个对话
            dialogue = [{
                "speaker": default_speaker,
                "emotion": default_emotion,
                "text": text.strip()
            }]
        
        return dialogue
    
    @classmethod
    def _parse_bracket_format(cls, text: str, default_speaker: str, 
                             default_emotion: str) -> List[Dict]:
        """
        解析括号格式：[说话人] 文本
        """
        pattern = r'\[([^\]]+)\]\s*([^\[]+)'
        matches = re.findall(pattern, text)
        
        if not matches:
            return []
        
        dialogue = []
        for speaker, content in matches:
            # 检查是否包含情绪标记
            speaker, emotion = cls._parse_speaker_emotion(speaker, default_emotion)
            
            # 清理文本
            content = content.strip()
            
            dialogue.append({
                "speaker": speaker,
                "emotion": emotion,
                "text": content
            })
        
        return dialogue
    
    @classmethod
    def _parse_colon_format(cls, text: str, default_speaker: str, 
                           default_emotion: str) -> List[Dict]:
        """
        解析冒号格式：说话人: 文本
        """
        lines = text.split('\n')
        dialogue = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尝试匹配冒号格式
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    content = parts[1].strip()
                    
                    # 检查是否包含情绪标记
                    speaker, emotion = cls._parse_speaker_emotion(speaker, default_emotion)
                    
                    dialogue.append({
                        "speaker": speaker,
                        "emotion": emotion,
                        "text": content
                    })
                    continue
            
            # 如果不是冒号格式，使用默认说话人
            dialogue.append({
                "speaker": default_speaker,
                "emotion": default_emotion,
                "text": line
            })
        
        return dialogue
    
    @classmethod
    def _parse_dash_format(cls, text: str, default_speaker: str, 
                         default_emotion: str) -> List[Dict]:
        """
        解析破折号格式：说话人 - 文本
        """
        lines = text.split('\n')
        dialogue = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尝试匹配破折号格式
            if ' - ' in line:
                parts = line.split(' - ', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    content = parts[1].strip()
                    
                    # 检查是否包含情绪标记
                    speaker, emotion = cls._parse_speaker_emotion(speaker, default_emotion)
                    
                    dialogue.append({
                        "speaker": speaker,
                        "emotion": emotion,
                        "text": content
                    })
                    continue
            
            # 如果不是破折号格式，使用默认说话人
            dialogue.append({
                "speaker": default_speaker,
                "emotion": default_emotion,
                "text": line
            })
        
        return dialogue
    
    @classmethod
    def _parse_speaker_emotion(cls, speaker_text: str, 
                              default_emotion: str) -> Tuple[str, str]:
        """
        解析说话人和情绪
        
        格式：说话人(情绪) 或 说话人
        """
        # 检查括号中的情绪
        emotion_pattern = r'\(([^\)]+)\)$'
        emotion_match = re.search(emotion_pattern, speaker_text)
        
        if emotion_match:
            emotion = emotion_match.group(1)
            speaker = re.sub(r'\([^\)]+\)$', '', speaker_text).strip()
        else:
            speaker = speaker_text
            emotion = default_emotion
        
        # 验证情绪是否有效
        if emotion not in cls.EMOTION_MAP:
            emotion = default_emotion
        
        return (speaker, emotion)
    
    @classmethod
    def get_speaker_id(cls, speaker_name: str) -> int:
        """
        获取说话人ID
        """
        return cls.SPEAKER_MAP.get(speaker_name, 0)
    
    @classmethod
    def get_emotion_code(cls, emotion_name: str) -> str:
        """
        获取情绪代码
        """
        return cls.EMOTION_MAP.get(emotion_name, "default")
    
    @classmethod
    def validate_dialogue(cls, dialogue: List[Dict]) -> Tuple[bool, List[str]]:
        """
        验证对话格式
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        if not dialogue:
            errors.append("对话内容为空")
            return (False, errors)
        
        for i, segment in enumerate(dialogue):
            # 检查必需字段
            if "text" not in segment or not segment["text"].strip():
                errors.append(f"第{i+1}段对话：文本内容为空")
            
            # 检查说话人（不再验证是否在SPEAKER_MAP中，因为角色名称是动态的）
            if "speaker" not in segment or not segment["speaker"].strip():
                errors.append(f"第{i+1}段对话：缺少说话人")
            
            # 检查情绪
            if "emotion" not in segment:
                errors.append(f"第{i+1}段对话：缺少情绪")
            elif segment["emotion"] not in cls.EMOTION_MAP:
                errors.append(f"第{i+1}段对话：未知情绪 '{segment['emotion']}'")
        
        return (len(errors) == 0, errors)
    
    @classmethod
    def format_dialogue_info(cls, dialogue: List[Dict]) -> str:
        """
        格式化对话信息（用于调试）
        """
        info = []
        for i, segment in enumerate(dialogue, 1):
            speaker = segment.get("speaker", "未知")
            emotion = segment.get("emotion", "默认")
            text = segment.get("text", "")
            text_preview = text[:50] + "..." if len(text) > 50 else text
            
            info.append(f"{i}. [{speaker}]({emotion}): {text_preview}")
        
        return "\n".join(info)
