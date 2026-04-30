# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Multi-Image Input Node

多图输入节点 - 用于分析多张图像并创作故事内容

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    HARDWARE_INFO, image2base64, scale_image
)

class MultiImageInput:
    """
    多图输入节点
    - 支持输入多张图像进行分析
    - 支持无图像模式，通过选项设置生成提示词
    - 自动预处理和编码
    - 输出可直接连接到llama_cpp_instruct_adv节点的custom_prompt
    - 专为wan2.2视频生成优化
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "mode": (["图像模式", "文本模式"], {"default": "图像模式", "tooltip": "选择工作模式：图像模式分析图像内容，文本模式根据选项生成提示词"}),
                "story_type": ([
                    "连贯故事",
                    "分镜描述",
                    "场景分析",
                    "角色发展",
                    "情感递进",
                    "创意写作",
                    "脚本创作",
                    "广告文案",
                    "产品介绍",
                    "教育内容"
                ], {"default": "连贯故事", "tooltip": "选择内容创作类型"}),
                "story_length": ([
                    "简短（200字以内）",
                    "中等（400字以内）",
                    "详细（600字以内）",
                    "完整（1000字以内）"
                ], {"default": "中等（400字以内）", "tooltip": "选择内容长度"}),
                "language": (["中文", "English"], {"default": "中文", "tooltip": "选择输出语言"}),
                "max_size": ("INT", {"default": 256, "min": 128, "max": 512, "step": 32, "tooltip": "图像最大尺寸（像素）"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "自定义提示词，用于指导内容创作"}),
                "include_image_descriptions": ("BOOLEAN", {"default": True, "tooltip": "是否在故事前包含每张图像的描述"}),
                "story_theme": ([
                    "无特定主题",
                    "冒险故事",
                    "浪漫故事",
                    "悬疑故事",
                    "科幻故事",
                    "奇幻故事",
                    "日常生活",
                    "历史故事",
                    "未来科技",
                    "商业营销",
                    "科普教育",
                    "娱乐喜剧"
                ], {"default": "无特定主题", "tooltip": "选择内容主题"}),
                "narrative_style": ([
                    "第一人称",
                    "第三人称",
                    "全知视角",
                    "多视角切换"
                ], {"default": "第三人称", "tooltip": "选择叙事风格"}),
                "content_focus": ([
                    "平衡发展",
                    "强调情节",
                    "强调角色",
                    "强调情感",
                    "强调视觉",
                    "强调对话"
                ], {"default": "平衡发展", "tooltip": "选择内容重点"}),
                "target_audience": ([
                    "普通大众",
                    "青少年",
                    "儿童",
                    "专业人士",
                    "特定群体"
                ], {"default": "普通大众", "tooltip": "选择目标受众"}),
                "video_model": ([
                    "WAN2.2",
                    "LTX2",
                    "通用视频",
                    "自定义"
                ], {"default": "WAN2.2", "tooltip": "选择视频生成模型类型，不同模型需要不同的提示词格式"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "prompt")
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "process_multi_images"
    CATEGORY = "llama-cpp-vlm"
    
    def process_multi_images(self, image1=None, image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None, image9=None, image10=None, image11=None, image12=None, mode="Image Mode", story_type="Coherent Story", story_length="Medium (400 words or less)", language="中文", max_size=256, 
                          custom_prompt="", include_image_descriptions=True, 
                          story_theme="No Specific Theme", narrative_style="Third Person",
                          content_focus="Balanced Development", target_audience="General Public", video_model="WAN2.2"):
        """
        处理多张图像或文本模式，生成适合内容创作的提示词
        
        参数说明：
        - image1: 第一张输入图像
        - image2: 第二张输入图像
        - image3: 第三张输入图像
        - image4: 第四张输入图像
        - image5: 第五张输入图像
        - image6: 第六张输入图像
        - image7: 第七张输入图像
        - image8: 第八张输入图像
        - image9: 第九张输入图像
        - image10: 第十张输入图像
        - image11: 第十一张输入图像
        - image12: 第十二张输入图像
        - mode: 工作模式（图像模式/文本模式）
        - story_type: 内容创作类型
        - story_length: 内容长度
        - language: 输出语言
        - max_size: 图像最大尺寸
        - custom_prompt: 自定义提示词
        - include_image_descriptions: 是否包含图像描述
        - story_theme: 内容主题
        - narrative_style: 叙事风格
        - content_focus: 内容重点
        - target_audience: 目标受众
        - video_model: 视频生成模型类型（WAN2.2/LTX2/General Video/Custom）
        """
        
        if mode == "Image Mode":
            return self._process_image_mode(
                image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12, story_type, story_length, language, max_size,
                custom_prompt, include_image_descriptions, story_theme, narrative_style, video_model
            )
        else:
            return self._process_text_mode(
                story_type, story_length, language, custom_prompt,
                story_theme, narrative_style, content_focus, target_audience, video_model
            )
    
    def _process_image_mode(self, image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12, story_type, story_length, language, max_size,
                          custom_prompt, include_image_descriptions, story_theme, narrative_style, video_model):
        """
        图像模式：分析图像内容进行故事创作
        """
        
        # 收集所有图像
        all_images = []
        if image1 is not None:
            all_images.append(image1)
        if image2 is not None:
            all_images.append(image2)
        if image3 is not None:
            all_images.append(image3)
        if image4 is not None:
            all_images.append(image4)
        if image5 is not None:
            all_images.append(image5)
        if image6 is not None:
            all_images.append(image6)
        if image7 is not None:
            all_images.append(image7)
        if image8 is not None:
            all_images.append(image8)
        if image9 is not None:
            all_images.append(image9)
        if image10 is not None:
            all_images.append(image10)
        if image11 is not None:
            all_images.append(image11)
        if image12 is not None:
            all_images.append(image12)
        
        # 检查是否有图像输入
        if len(all_images) == 0:
            raise ValueError("【Error】Image Mode requires at least one image input")
        
        print(f"【Image Mode】Starting to process {len(all_images)} images")
        
        # 预处理图像
        preprocessed_images = []
        image_descriptions = []
        
        for i, image in enumerate(all_images):
            try:
                # 缩放图像
                img_np = scale_image(image, max_size)
                preprocessed_images.append(img_np)
                
                # 生成图像描述占位符
                image_descriptions.append(f"【图像 {i+1}】")
                
            except Exception as e:
                print(f"【警告】图像 {i+1} 预处理失败：{e}")
                continue
        
        # 构建多图分析提示词
        multi_image_prompt = self._build_multi_image_prompt(
            len(preprocessed_images),
            story_type,
            story_length,
            language,
            custom_prompt,
            include_image_descriptions,
            story_theme,
            narrative_style,
            video_model,
            content_focus,
            target_audience
        )
        
        print(f"【Image Mode】Completed, generated prompt length: {len(multi_image_prompt)} characters")
        
        # 将图像列表合并为单个tensor（如果有多张图像）
        if len(preprocessed_images) == 1:
            combined_images = preprocessed_images[0]
        else:
            # 将多张图像堆叠在一起
            combined_images = np.concatenate(preprocessed_images, axis=0)
        
        return (combined_images, multi_image_prompt)
    
    def _process_text_mode(self, story_type, story_length, language, custom_prompt,
                         story_theme, narrative_style, content_focus, target_audience, video_model):
        """
        文本模式：通过选项设置生成提示词
        """
        
        print(f"【Text Mode】Starting to generate prompt for {video_model}")
        
        # 构建文本模式提示词
        text_prompt = self._build_text_prompt(
            story_type,
            story_length,
            language,
            custom_prompt,
            story_theme,
            narrative_style,
            content_focus,
            target_audience,
            video_model
        )
        
        print(f"【Text Mode】Completed, generated prompt length: {len(text_prompt)} characters")
        
        # 文本模式不返回图像
        return (None, text_prompt)
    
    def _build_text_prompt(self, story_type, story_length, language, custom_prompt,
                         story_theme, narrative_style, content_focus, target_audience, video_model):
        """
        构建文本模式提示词
        """
        
        if language == "中文":
            return self._build_chinese_text_prompt(
                story_type, story_length, custom_prompt,
                story_theme, narrative_style, content_focus, target_audience, video_model
            )
        else:
            return self._build_english_text_prompt(
                story_type, story_length, custom_prompt,
                story_theme, narrative_style, content_focus, target_audience, video_model
            )
    
    def _build_chinese_text_prompt(self, story_type, story_length, custom_prompt,
                                  story_theme, narrative_style, content_focus, target_audience, video_model):
        """
        构建中文文本模式提示词
        """
        
        # 视频模型特定的提示词格式
        model_specific_instruction = self._get_video_model_instruction(video_model, language="中文")
        
        # 故事长度映射
        length_map = {
            "Short (200 words or less)": "200字以内",
            "Medium (400 words or less)": "400字以内",
            "Detailed (600 words or less)": "600字以内",
            "Complete (1000 words or less)": "1000字以内"
        }
        target_length = length_map.get(story_length, "400字以内")
        
        type_map = {
            "Coherent Story": "创作一个连贯完整的故事",
            "Storyboard Description": "按照时间顺序描述每个场景",
            "Scene Analysis": "深入分析每个场景的细节",
            "Character Development": "描述角色的成长和变化",
            "Emotional Progression": "展现情感的变化和发展",
            "Creative Writing": "进行创意写作，展现独特的想法和创意",
            "Script Creation": "创作一个完整的剧本，包含对话和场景描述",
            "Advertising Copy": "创作吸引人的广告文案",
            "Product Introduction": "撰写详细的产品介绍",
            "Educational Content": "创作教育性内容，易于理解和学习"
        }
        type_instruction = type_map.get(story_type, "创作一个连贯完整的故事")
        
        theme_map = {
            "No Specific Theme": "",
            "Adventure Story": "冒险主题，充满挑战和探索",
            "Romance Story": "浪漫主题，展现情感和关系",
            "Mystery Story": "悬疑主题，设置悬念和谜题",
            "Sci-Fi Story": "科幻主题，展现未来科技",
            "Fantasy Story": "奇幻主题，包含魔法和超自然元素",
            "Daily Life": "日常生活主题，展现平凡中的美好",
            "Historical Story": "历史主题，展现过去的文化和时代",
            "Future Technology": "未来科技主题，展现先进科技和人类发展",
            "Business Marketing": "商业营销主题，突出产品或服务优势",
            "Educational Popularization": "教育科普主题，传播知识和信息",
            "Entertainment Comedy": "娱乐搞笑主题，带来欢乐和轻松"
        }
        theme_instruction = theme_map.get(story_theme, "")
        
        style_map = {
            "First Person": "使用第一人称视角（我）",
            "Third Person": "使用第三人称视角（他/她）",
            "Omniscient Perspective": "使用全知视角，了解所有角色的想法",
            "Multi-Perspective Switching": "在不同角色之间切换视角"
        }
        style_instruction = style_map.get(narrative_style, "使用第三人称视角（他/她）")
        
        focus_map = {
            "Balanced Development": "平衡发展情节、角色、情感等各个方面",
            "Emphasize Plot": "重点突出情节的发展和转折",
            "Emphasize Characters": "重点刻画角色的性格和成长",
            "Emphasize Emotions": "重点展现情感的变化和表达",
            "Emphasize Visuals": "重点描述视觉元素和场景细节",
            "Emphasize Dialogue": "重点通过对话推动故事发展"
        }
        focus_instruction = focus_map.get(content_focus, "平衡发展情节、角色、情感等各个方面")
        
        audience_map = {
            "General Public": "适合普通大众阅读和理解",
            "Teenagers": "适合青少年阅读，内容积极向上",
            "Children": "适合儿童阅读，语言简单易懂",
            "Professionals": "适合专业人士阅读，内容专业深入",
            "Specific Groups": "适合特定群体阅读，针对性强"
        }
        audience_instruction = audience_map.get(target_audience, "适合普通大众阅读和理解")
        
        # 构建提示词
        prompt_parts = []
        
        # 添加主要任务
        prompt_parts.append(f"你是一位专业的内容创作专家，擅长根据要求创作高质量的内容。")
        prompt_parts.append(f"")
        prompt_parts.append(f"【任务要求】")
        prompt_parts.append(f"1. {type_instruction}。")
        prompt_parts.append(f"2. 内容长度控制在 {target_length}，确保内容充实但不冗长。")
        prompt_parts.append(f"3. {style_instruction}，保持叙事风格一致。")
        prompt_parts.append(f"4. {focus_instruction}。")
        prompt_parts.append(f"5. {audience_instruction}。")
        
        # 添加主题（如果有）
        if theme_instruction:
            prompt_parts.append(f"6. 内容采用{theme_instruction}的风格。")
        
        # 添加自定义提示词（如果有）
        if custom_prompt and custom_prompt.strip():
            prompt_parts.append(f"")
            prompt_parts.append(f"【自定义要求】")
            prompt_parts.append(f"{custom_prompt}")
        
        # 添加输出格式
        prompt_parts.append(f"")
        prompt_parts.append(f"【输出格式】")
        prompt_parts.append(f"1. 直接输出JSON格式内容，不添加任何额外说明或标记。")
        prompt_parts.append(f"2. JSON必须包含以下字段：")
        prompt_parts.append(f"   - task_type: 内容创作类型")
        prompt_parts.append(f"   - theme: 内容主题")
        prompt_parts.append(f"   - narrative_style: 叙事风格")
        prompt_parts.append(f"   - content_focus: 内容重点")
        prompt_parts.append(f"   - target_audience: 目标受众")
        prompt_parts.append(f"   - story_length: 故事长度")
        prompt_parts.append(f"   - video_model: 视频模型类型")
        prompt_parts.append(f"   - content: 创作的具体内容")
        prompt_parts.append(f"   - image_descriptions: 图像描述列表（图像模式时）")
        prompt_parts.append(f"3. 确保JSON格式正确，可被json.loads()解析。")
        prompt_parts.append(f"4. {model_specific_instruction}")
        
        return "\n".join(prompt_parts)
    
    def _build_english_text_prompt(self, story_type, story_length, custom_prompt,
                                  story_theme, narrative_style, content_focus, target_audience, video_model):
        """
        构建英文文本模式提示词
        """
        
        # 视频模型特定的提示词格式
        model_specific_instruction = self._get_video_model_instruction(video_model, language="English")
        # Story length mapping
        length_map = {
            "Short (200 words or less)": "200 words or less",
            "Medium (400 words or less)": "400 words or less",
            "Detailed (600 words or less)": "600 words or less",
            "Complete (1000 words or less)": "1000 words or less"
        }
        target_length = length_map.get(story_length, "400 words or less")
        
        type_map = {
            "Coherent Story": "Create a coherent and complete story",
            "Storyboard Description": "Describe each scene in chronological order",
            "Scene Analysis": "Analyze details of each scene in depth",
            "Character Development": "Describe the growth and changes of characters",
            "Emotional Progression": "Show the progression and development of emotions",
            "Creative Writing": "Engage in creative writing, showcasing unique ideas and creativity",
            "Script Creation": "Create a complete script with dialogue and scene descriptions",
            "Advertising Copy": "Create engaging advertising copy",
            "Product Introduction": "Write detailed product introductions",
            "Educational Content": "Create educational content that is easy to understand and learn"
        }
        type_instruction = type_map.get(story_type, "Create a coherent and complete story")
        
        theme_map = {
            "No Specific Theme": "",
            "Adventure Story": "Adventure theme, full of challenges and exploration",
            "Romance Story": "Romance theme, showing emotions and relationships",
            "Mystery Story": "Mystery theme, setting suspense and puzzles",
            "Sci-Fi Story": "Sci-fi theme, showing future technology",
            "Fantasy Story": "Fantasy theme, including magic and supernatural elements",
            "Daily Life": "Daily life theme, showing beauty in the ordinary",
            "Historical Story": "Historical theme, showing past culture and era",
            "Future Technology": "Future technology theme, showing advanced tech and human development",
            "Business Marketing": "Business marketing theme, highlighting product or service advantages",
            "Educational Popularization": "Educational popularization theme, spreading knowledge and information",
            "Entertainment Comedy": "Entertainment and comedy theme, bringing joy and relaxation"
        }
        theme_instruction = theme_map.get(story_theme, "")
        
        style_map = {
            "First Person": "Use first-person perspective (I)",
            "Third Person": "Use third-person perspective (he/she)",
            "Omniscient Perspective": "Use omniscient perspective, knowing all characters' thoughts",
            "Multi-Perspective Switching": "Switch perspectives between different characters"
        }
        style_instruction = style_map.get(narrative_style, "Use third-person perspective (he/she)")
        
        focus_map = {
            "Balanced Development": "Balance development of plot, characters, emotions, and other aspects",
            "Emphasize Plot": "Focus on highlighting plot development and twists",
            "Emphasize Characters": "Focus on characterizing personalities and growth",
            "Emphasize Emotions": "Focus on showcasing emotional changes and expressions",
            "Emphasize Visuals": "Focus on describing visual elements and scene details",
            "Emphasize Dialogue": "Focus on advancing story through dialogue"
        }
        focus_instruction = focus_map.get(content_focus, "Balance development of plot, characters, emotions, and other aspects")
        
        audience_map = {
            "General Public": "Suitable for general public to read and understand",
            "Teenagers": "Suitable for teenagers to read, content is positive and uplifting",
            "Children": "Suitable for children to read, language is simple and easy to understand",
            "Professionals": "Suitable for professionals to read, content is professional and in-depth",
            "Specific Groups": "Suitable for specific groups to read, highly targeted"
        }
        audience_instruction = audience_map.get(target_audience, "Suitable for general public to read and understand")
        
        # Build prompt
        prompt_parts = []
        
        # Add main task
        prompt_parts.append(f"You are a professional content creation expert, skilled at creating high-quality content based on requirements.")
        prompt_parts.append(f"")
        prompt_parts.append(f"【Task Requirements】")
        prompt_parts.append(f"1. {type_instruction}.")
        prompt_parts.append(f"2. Keep the content length within {target_length}, ensuring the content is substantial but not redundant.")
        prompt_parts.append(f"3. {style_instruction}, maintaining consistent narrative style.")
        prompt_parts.append(f"4. {focus_instruction}.")
        prompt_parts.append(f"5. {audience_instruction}.")
        
        # Add theme (if any)
        if theme_instruction:
            prompt_parts.append(f"6. The content should adopt a {theme_instruction} style.")
        
        # Add custom prompt (if any)
        if custom_prompt and custom_prompt.strip():
            prompt_parts.append(f"")
            prompt_parts.append(f"【Custom Requirements】")
            prompt_parts.append(f"{custom_prompt}")
        
        # Add output format
        prompt_parts.append(f"")
        prompt_parts.append(f"【Output Format】")
        prompt_parts.append(f"1. Output JSON format content directly, without adding any additional explanations or markers.")
        prompt_parts.append(f"2. JSON must contain the following fields:")
        prompt_parts.append(f"   - task_type: Content creation type")
        prompt_parts.append(f"   - theme: Content theme")
        prompt_parts.append(f"   - narrative_style: Narrative style")
        prompt_parts.append(f"   - content_focus: Content focus")
        prompt_parts.append(f"   - target_audience: Target audience")
        prompt_parts.append(f"   - story_length: Story length")
        prompt_parts.append(f"   - video_model: Video model type")
        prompt_parts.append(f"   - content: The specific content created")
        prompt_parts.append(f"   - image_descriptions: List of image descriptions (for image mode)")
        prompt_parts.append(f"3. Ensure the JSON format is correct and can be parsed by json.loads().")
        prompt_parts.append(f"4. {model_specific_instruction}")
        
        return "\n".join(prompt_parts)
    
    def _build_multi_image_prompt(self, image_count, story_type, story_length,
                               language, custom_prompt, include_image_descriptions,
                               story_theme, narrative_style, video_model,
                               content_focus, target_audience):
        """
        构建多图分析提示词
        """

        # 语言设置
        if language == "中文":
            return self._build_chinese_prompt(
                image_count, story_type, story_length, custom_prompt,
                include_image_descriptions, story_theme, narrative_style, video_model,
                content_focus, target_audience
            )
        else:
            return self._build_english_prompt(
                image_count, story_type, story_length, custom_prompt,
                include_image_descriptions, story_theme, narrative_style, video_model,
                content_focus, target_audience
            )
    
    def _build_chinese_prompt(self, image_count, story_type, story_length,
                            custom_prompt, include_image_descriptions,
                            story_theme, narrative_style, video_model,
                            content_focus, target_audience):
        """
        构建中文提示词
        """

        # 视频模型特定的提示词格式
        model_specific_instruction = self._get_video_model_instruction(video_model, language="中文")

        # 故事长度映射
        length_map = {
            "Short (200 words or less)": "200字以内",
            "Medium (400 words or less)": "400字以内",
            "Detailed (600 words or less)": "600字以内",
            "Complete (1000 words or less)": "1000字以内"
        }
        target_length = length_map.get(story_length, "400字以内")

        type_map = {
            "Coherent Story": "创作一个连贯完整的故事",
            "Storyboard Description": "按照时间顺序描述每个场景",
            "Scene Analysis": "深入分析每个场景的细节",
            "Character Development": "描述角色的成长和变化",
            "Emotional Progression": "展现情感的变化和发展",
            "Creative Writing": "进行创意写作，展现独特的想法和创意",
            "Script Creation": "创作一个完整的剧本，包含对话和场景描述",
            "Advertising Copy": "创作吸引人的广告文案",
            "Product Introduction": "撰写详细的产品介绍",
            "Educational Content": "创作教育性内容，易于理解和学习"
        }
        type_instruction = type_map.get(story_type, "创作一个连贯完整的故事")

        theme_map = {
            "No Specific Theme": "",
            "Adventure Story": "冒险主题，充满挑战和探索",
            "Romance Story": "浪漫主题，展现情感和关系",
            "Mystery Story": "悬疑主题，设置悬念和谜题",
            "Sci-Fi Story": "科幻主题，展现未来科技",
            "Fantasy Story": "奇幻主题，包含魔法和超自然元素",
            "Daily Life": "日常生活主题，展现平凡中的美好",
            "Historical Story": "历史主题，展现过去的文化和时代",
            "Future Technology": "未来科技主题，展现先进科技和人类发展",
            "Business Marketing": "商业营销主题，突出产品或服务优势",
            "Educational Popularization": "教育科普主题，传播知识和信息",
            "Entertainment Comedy": "娱乐搞笑主题，带来欢乐和轻松"
        }
        theme_instruction = theme_map.get(story_theme, "")

        style_map = {
            "First Person": "使用第一人称视角（我）",
            "Third Person": "使用第三人称视角（他/她）",
            "Omniscient Perspective": "使用全知视角，了解所有角色的想法",
            "Multi-Perspective Switching": "在不同角色之间切换视角"
        }
        style_instruction = style_map.get(narrative_style, "使用第三人称视角（他/她）")

        focus_map = {
            "Balanced Development": "平衡发展情节、角色、情感等各个方面",
            "Emphasize Plot": "重点突出情节的发展和转折",
            "Emphasize Characters": "重点刻画角色的性格和成长",
            "Emphasize Emotions": "重点展现情感的变化和表达",
            "Emphasize Visuals": "重点描述视觉元素和场景细节",
            "Emphasize Dialogue": "重点通过对话推动故事发展"
        }
        focus_instruction = focus_map.get(content_focus, "平衡发展情节、角色、情感等各个方面")

        audience_map = {
            "General Public": "适合普通大众阅读和理解",
            "Teenagers": "适合青少年阅读，内容积极向上",
            "Children": "适合儿童阅读，语言简单易懂",
            "Professionals": "适合专业人士阅读，内容专业深入",
            "Specific Groups": "适合特定群体阅读，针对性强"
        }
        audience_instruction = audience_map.get(target_audience, "适合普通大众阅读和理解")

        # 构建提示词
        prompt_parts = []

        # 添加主要任务
        prompt_parts.append(f"你是一位专业的视频故事创作专家，擅长根据多张图片创作连贯的故事内容。")
        prompt_parts.append(f"")
        prompt_parts.append(f"【任务要求】")
        prompt_parts.append(f"1. 仔细分析提供的 {image_count} 张图像，理解每张图像的内容、场景、角色和情感。")
        prompt_parts.append(f"2. {type_instruction}，确保故事逻辑连贯、情节发展自然。")

        # 添加长度要求
        prompt_parts.append(f"3. 故事长度控制在 {target_length}，确保内容充实但不冗长。")

        # 添加叙事风格
        prompt_parts.append(f"4. {style_instruction}，保持叙事风格一致。")

        # 添加内容重点
        prompt_parts.append(f"5. {focus_instruction}。")

        # 添加目标受众
        prompt_parts.append(f"6. {audience_instruction}。")

        # 添加主题（如果有）
        if theme_instruction:
            prompt_parts.append(f"7. 故事采用{theme_instruction}的风格。")
        
        # 添加图片描述要求
        if include_image_descriptions:
            prompt_parts.append(f"")
            prompt_parts.append(f"【图片分析要求】")
            prompt_parts.append(f"对于每张图片，请提供：")
            prompt_parts.append(f"- 场景描述：详细说明环境、光线、色彩等视觉元素。")
            prompt_parts.append(f"- 主体识别：识别图片中的主要角色或物体。")
            prompt_parts.append(f"- 动作状态：描述角色的动作、表情和姿态。")
            prompt_parts.append(f"- 情感基调：分析图片传达的情感和氛围。")
        
        # 添加自定义提示词（如果有）
        if custom_prompt and custom_prompt.strip():
            prompt_parts.append(f"")
            prompt_parts.append(f"【自定义要求】")
            prompt_parts.append(f"{custom_prompt}")
        
        # 添加输出格式
        prompt_parts.append(f"")
        prompt_parts.append(f"【输出格式】")
        prompt_parts.append(f"1. 直接输出JSON格式内容，不添加任何额外说明或标记。")
        prompt_parts.append(f"2. JSON必须包含以下字段：")
        prompt_parts.append(f"   - task_type: 内容创作类型")
        prompt_parts.append(f"   - theme: 内容主题")
        prompt_parts.append(f"   - narrative_style: 叙事风格")
        prompt_parts.append(f"   - content_focus: 内容重点")
        prompt_parts.append(f"   - target_audience: 目标受众")
        prompt_parts.append(f"   - story_length: 故事长度")
        prompt_parts.append(f"   - video_model: 视频模型类型")
        prompt_parts.append(f"   - content: 创作的具体内容")
        prompt_parts.append(f"   - image_descriptions: 图像描述列表，每项包含scene_description、subject_identification、action_state、emotional_tone")
        prompt_parts.append(f"3. 确保JSON格式正确，可被json.loads()解析。")
        prompt_parts.append(f"4. {model_specific_instruction}")

        return "\n".join(prompt_parts)

    def _build_english_prompt(self, image_count, story_type, story_length,
                            custom_prompt, include_image_descriptions,
                            story_theme, narrative_style, video_model,
                            content_focus, target_audience):
        """
        构建英文提示词
        """

        # 视频模型特定的提示词格式
        model_specific_instruction = self._get_video_model_instruction(video_model, language="English")
        # Story length mapping
        length_map = {
            "Short (200 words or less)": "200 words or less",
            "Medium (400 words or less)": "400 words or less",
            "Detailed (600 words or less)": "600 words or less",
            "Complete (1000 words or less)": "1000 words or less"
        }
        target_length = length_map.get(story_length, "400 words or less")

        type_map = {
            "Coherent Story": "Create a coherent and complete story",
            "Storyboard Description": "Describe each scene in chronological order",
            "Scene Analysis": "Analyze the details of each scene in depth",
            "Character Development": "Describe the growth and changes of characters",
            "Emotional Progression": "Show the progression and development of emotions",
            "Creative Writing": "Engage in creative writing, showcasing unique ideas and creativity",
            "Script Creation": "Create a complete script with dialogue and scene descriptions",
            "Advertising Copy": "Create engaging advertising copy",
            "Product Introduction": "Write detailed product introductions",
            "Educational Content": "Create educational content that is easy to understand and learn"
        }
        type_instruction = type_map.get(story_type, "Create a coherent and complete story")

        theme_map = {
            "No Specific Theme": "",
            "Adventure Story": "Adventure theme, full of challenges and exploration",
            "Romance Story": "Romance theme, showing emotions and relationships",
            "Mystery Story": "Mystery theme, setting suspense and puzzles",
            "Sci-Fi Story": "Sci-fi theme, showing future technology",
            "Fantasy Story": "Fantasy theme, including magic and supernatural elements",
            "Daily Life": "Daily life theme, showing beauty in the ordinary",
            "Historical Story": "Historical theme, showing past culture and era",
            "Future Technology": "Future technology theme, showing advanced tech and human development",
            "Business Marketing": "Business marketing theme, highlighting product or service advantages",
            "Educational Popularization": "Educational popularization theme, spreading knowledge and information",
            "Entertainment Comedy": "Entertainment and comedy theme, bringing joy and relaxation"
        }
        theme_instruction = theme_map.get(story_theme, "")

        style_map = {
            "First Person": "Use first-person perspective (I)",
            "Third Person": "Use third-person perspective (he/she)",
            "Omniscient Perspective": "Use omniscient perspective, knowing all characters' thoughts",
            "Multi-Perspective Switching": "Switch perspectives between different characters"
        }
        style_instruction = style_map.get(narrative_style, "Use third-person perspective (he/she)")

        focus_map = {
            "Balanced Development": "Balance development of plot, characters, emotions, and other aspects",
            "Emphasize Plot": "Focus on highlighting plot development and twists",
            "Emphasize Characters": "Focus on characterizing personalities and growth",
            "Emphasize Emotions": "Focus on showcasing emotional changes and expressions",
            "Emphasize Visuals": "Focus on describing visual elements and scene details",
            "Emphasize Dialogue": "Focus on advancing story through dialogue"
        }
        focus_instruction = focus_map.get(content_focus, "Balance development of plot, characters, emotions, and other aspects")

        audience_map = {
            "General Public": "Suitable for general public to read and understand",
            "Teenagers": "Suitable for teenagers to read, content is positive and uplifting",
            "Children": "Suitable for children to read, language is simple and easy to understand",
            "Professionals": "Suitable for professionals to read, content is professional and in-depth",
            "Specific Groups": "Suitable for specific groups to read, highly targeted"
        }
        audience_instruction = audience_map.get(target_audience, "Suitable for general public to read and understand")

        # Build prompt
        prompt_parts = []

        # Add main task
        prompt_parts.append(f"You are a professional video story creation expert, skilled at creating coherent story content based on multiple images.")
        prompt_parts.append(f"")
        prompt_parts.append(f"【Task Requirements】")
        prompt_parts.append(f"1. Carefully analyze the provided {image_count} images, understanding the content, scenes, characters, and emotions in each image.")
        prompt_parts.append(f"2. {type_instruction}, ensuring the story is logically coherent and the plot develops naturally.")

        # Add length requirement
        prompt_parts.append(f"3. Keep the story length within {target_length}, ensuring the content is substantial but not redundant.")

        # Add narrative style
        prompt_parts.append(f"4. {style_instruction}, maintaining consistent narrative style.")

        # Add content focus
        prompt_parts.append(f"5. {focus_instruction}.")

        # Add target audience
        prompt_parts.append(f"6. {audience_instruction}.")

        # Add theme (if any)
        if theme_instruction:
            prompt_parts.append(f"7. The story should adopt a {theme_instruction} style.")
        
        # Add image description requirements
        if include_image_descriptions:
            prompt_parts.append(f"")
            prompt_parts.append(f"【Image Analysis Requirements】")
            prompt_parts.append(f"For each image, please provide:")
            prompt_parts.append(f"- Scene description: Detailed description of environment, lighting, colors, and other visual elements.")
            prompt_parts.append(f"- Subject identification: Identify the main characters or objects in the image.")
            prompt_parts.append(f"- Action state: Describe the character's actions, expressions, and poses.")
            prompt_parts.append(f"- Emotional tone: Analyze the emotions and atmosphere conveyed by the image.")
        
        # Add custom prompt (if any)
        if custom_prompt and custom_prompt.strip():
            prompt_parts.append(f"")
            prompt_parts.append(f"【Custom Requirements】")
            prompt_parts.append(f"{custom_prompt}")
        
        # Add output format
        prompt_parts.append(f"")
        prompt_parts.append(f"【Output Format】")
        prompt_parts.append(f"1. Output JSON format content directly, without adding any additional explanations or markers.")
        prompt_parts.append(f"2. JSON must contain the following fields:")
        prompt_parts.append(f"   - task_type: Content creation type")
        prompt_parts.append(f"   - theme: Content theme")
        prompt_parts.append(f"   - narrative_style: Narrative style")
        prompt_parts.append(f"   - content_focus: Content focus")
        prompt_parts.append(f"   - target_audience: Target audience")
        prompt_parts.append(f"   - story_length: Story length")
        prompt_parts.append(f"   - video_model: Video model type")
        prompt_parts.append(f"   - content: The specific content created")
        prompt_parts.append(f"   - image_descriptions: List of image descriptions, each containing scene_description, subject_identification, action_state, emotional_tone")
        prompt_parts.append(f"3. Ensure the JSON format is correct and can be parsed by json.loads().")
        prompt_parts.append(f"4. {model_specific_instruction}")
        
        return "\n".join(prompt_parts)
    
    def _get_video_model_instruction(self, video_model, language="中文"):
        """
        获取视频模型特定的提示词指令
        
        参数说明：
        - video_model: 视频生成模型类型（WAN2.2/LTX2/General Video/Custom）
        - language: 语言（中文/English）
        
        返回：
        - 模型特定的提示词指令字符串
        """
        
        if language == "中文":
            return self._get_chinese_video_model_instruction(video_model)
        else:
            return self._get_english_video_model_instruction(video_model)
    
    def _get_chinese_video_model_instruction(self, video_model):
        """
        获取中文视频模型特定的提示词指令
        """
        
        instructions = {
            "WAN2.2": "确保故事适合WAN2.2视频生成模型，强调场景描述和视觉元素，使用简洁有力的语言，避免过于复杂的句式。",
            "LTX2": "确保故事适合LTX2视频生成模型，注重细节描述和情感表达，使用流畅自然的语言，突出关键场景转换。",
            "General Video": "确保故事适合通用视频生成模型，平衡场景描述和叙事流畅性，使用清晰易懂的语言。",
            "Custom": "根据自定义视频生成模型的要求调整提示词格式和内容风格。"
        }
        
        return instructions.get(video_model, instructions["WAN2.2"])
    
    def _get_english_video_model_instruction(self, video_model):
        """
        获取英文视频模型特定的提示词指令
        """
        
        instructions = {
            "WAN2.2": "Ensure the story is suitable for the WAN2.2 video generation model. Emphasize scene descriptions and visual elements. Use concise and powerful language, avoiding overly complex sentence structures.",
            "LTX2": "Ensure the story is suitable for the LTX2 video generation model. Focus on detailed descriptions and emotional expression. Use fluent and natural language, highlighting key scene transitions.",
            "General Video": "Ensure the story is suitable for general video generation models. Balance scene descriptions and narrative flow. Use clear and easy-to-understand language.",
            "Custom": "Adjust the prompt format and content style according to the requirements of the custom video generation model."
        }
        
        return instructions.get(video_model, instructions["WAN2.2"])
