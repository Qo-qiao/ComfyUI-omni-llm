# AI绘图模型提示词格式规范

本文档详细说明各主流AI绘图模型的提示词格式、语法特点和最佳实践。

---

## 目录

1. [SD1.5](#sd15)
2. [SDXL](#sdxl)
3. [Anima](#anima)
4. [Flux1](#flux1)
5. [Krea2](#krea2)
6. [ZImage](#zimage)
7. [Qwen-Image-2512](#qwen-image-2512)
8. [Mage-Flow](#mage-flow)
9. [HunyuanImage_2.1](#hunyuanimage_21)
10. [HiDream-O1-Image](#hidream-o1-image)
11. [ERNIE-Image](#ernie-image)
12. [Boogu-Image](#boogu-image)
13. [LongCat-Image](#longcat-image)
14. [Flux2-Klein](#flux2-klein)
15. [GLM-Image](#glm-image)
16. [GPT-Image2](#gpt-image2)
17. [Nanobanana](#nanobanana)
18. [Pony Diffusion](#pony-diffusion)
19. [Midjourney](#midjourney)
20. [通用提示词结构](#通用提示词结构)

---

## SD1.5

### 基本格式

```
正向提示词: <主体描述>, <风格描述>, <质量词>, <技术参数>
反向提示词: <排除内容>, <负面质量词>
```

### 语法特点

- **权重语法**: `(word:1.2)` 增强权重, `(word:0.8)` 降低权重
- **嵌套权重**: `((word))` 或 `{{{word}}}` 多层增强
- **交替词**: `[word1|word2]` 在步数间交替
- **组合词**: `[word1::0.3]` 前30%步数使用, `[word1:word2:0.5]` 50%步数切换

### 推荐参数

- 分辨率: 512x512, 768x512, 512x768
- 需要较多质量修饰词
- 对负面提示词敏感
- 推荐步数: 20-30

### 示例

```
正向: (masterpiece:1.2), (best quality:1.2), a beautiful woman with long silver hair, intricate fantasy armor, glowing blue eyes, standing in enchanted forest, magical particles floating, cinematic lighting, 8k, highly detailed, sharp focus

反向: (worst quality:1.4), (low quality:1.4), (normal quality:1.4), ugly, deformed, noisy, blurry, low contrast, bad anatomy, extra limbs, poorly drawn face, mutation, watermark, text
```

---

## SDXL

### 基本格式

```
正向提示词: <主体描述>, <风格描述>, <质量词>, <技术参数>
反向提示词: <排除内容>, <负面质量词>
```

### 语法特点

- 与SD1.5相同的权重语法
- 更好的自然语言理解
- 原生支持更长的提示词
- refiner可增强细节

### 推荐参数

- 分辨率: 1024x1024, 896x1152, 1152x896
- 更自然的描述效果
- 推荐步数: 25-40

### 示例

```
正向: masterpiece, best quality, ultra highres, 1girl, solo, white dress, standing in flower field, sunlight, soft lighting, intricate details, cinematic composition

反向: (worst quality:1.4), (low quality:1.4), bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry
```

---

## Anima

### 基本格式

```
正向提示词: <主体描述>, <风格描述>, <质量词>
反向提示词: <排除内容>, <负面质量词>
```

### 语法特点

- 支持自然语言描述
- 动漫风格优化
- 支持角色一致性
- 色彩表现力强

### 最佳实践

- 使用动漫相关描述词
- 明确指定角色特征
- 描述表情和姿态
- 适合二次元创作

### 示例

```
正向: anime girl with pink hair, large expressive eyes, school uniform, cherry blossom background, soft pastel colors, detailed illustration, studio ghibli style, masterpiece

反向: worst quality, low quality, bad anatomy, bad hands, blurry, watermark, text, ugly, deformed
```

---

## Flux1

### 基本格式

```
<自然语言描述>, <风格关键词>
```

### 语法特点

- 支持自然语言描述
- 对提示词理解能力强
- 原生高分辨率输出
- 较少需要质量修饰词
- 支持简单权重语法

### 最佳实践

- 详细描述主体和场景
- 明确指定艺术风格
- 描述光照和氛围
- 可适当添加质量词增强效果

### 示例

```
A professional portrait photograph of a young woman with auburn hair, wearing an elegant black dress, standing in a modern art gallery. Soft diffused lighting from large windows, shallow depth of field, shot on medium format camera, editorial style, highly detailed, photorealistic
```

---

## Krea2

### 基本格式

```
<自然语言描述>
```

### 语法特点

- 实时生成预览
- 自然语言理解
- 风格混合能力
- 支持图像增强

### 最佳实践

- 使用描述性语言
- 描述具体的风格和氛围
- 利用实时预览调整
- 适合快速迭代创作

### 示例

```
A dreamy ethereal portrait of a woman with flowing silver hair, surrounded by floating luminescent particles, soft blue and purple color palette, cinematic lighting, fantasy art style, highly detailed
```

---

## ZImage

### 基本格式

```
<主体描述>, <风格描述>
```

### 语法特点

- 支持多风格融合
- 细节保留能力强
- 色彩还原准确
- 支持高分辨率输出

### 最佳实践

- 明确描述主体特征
- 指定艺术风格
- 描述光影效果
- 适合高质量图像生成

### 示例

```
Majestic dragon soaring through storm clouds, scales shimmering with iridescent colors, lightning illuminating the scene, epic fantasy art, dramatic composition, highly detailed, 8k resolution
```

---

## Qwen-Image-2512

### 基本格式

```
<自然语言描述>
```

### 语法特点

- 多模态理解能力强
- 支持中英文描述
- 细节丰富
- 风格多样性

### 最佳实践

- 使用详细的文字描述
- 描述场景、物体、光照
- 可指定艺术风格
- 适合复杂场景生成

### 示例

```
中国古代山水画风格，云雾缭绕的山峰，瀑布从悬崖倾泻而下，山间有古典亭台楼阁，墨色浓淡相宜，留白意境深远，水墨画质感，高清细节
```

---

## Mage-Flow

### 基本格式

```
<主体描述>, <风格描述>, <氛围>
```

### 语法特点

- 流式生成
- 实时预览
- 风格可调
- 支持图像编辑

### 最佳实践

- 描述主体和背景
- 明确风格和氛围
- 利用流式预览调整
- 适合快速原型设计

### 示例

```
Cyberpunk cityscape at night, neon lights reflecting on wet streets, flying vehicles in the sky, holographic advertisements, rain-soaked atmosphere, blade runner aesthetic, cinematic composition
```

---

## HunyuanImage_2.1

### 基本格式

```
<自然语言描述>
```

### 语法特点

- 中文理解优化
- 国风元素支持
- 细节丰富
- 高分辨率输出

### 最佳实践

- 使用中文或英文描述
- 描述具体的场景和元素
- 指定艺术风格
- 适合国风和东方美学创作

### 示例

```
仙侠风格，白衣剑仙立于云端，长发飘逸，手持仙剑，身后是壮丽的仙山琼阁，云海翻腾，霞光万道，仙气飘飘，高清细节，国风美学
```

---

## HiDream-O1-Image

### 基本格式

```
<主体描述>, <风格描述>, <技术参数>
```

### 语法特点

- 高质量图像生成
- 细节保留
- 色彩准确
- 支持多种风格

### 最佳实践

- 详细描述主体特征
- 指定风格和技术参数
- 描述光照和构图
- 适合高质量创作

### 示例

```
Ethereal forest scene with bioluminescent mushrooms, fireflies dancing in the air, crystal clear stream reflecting moonlight, magical atmosphere, fantasy illustration style, highly detailed, 8k
```

---

## ERNIE-Image

### 基本格式

```
<自然语言描述>
```

### 语法特点

- 中文理解优化
- 知识增强
- 细节丰富
- 风格多样

### 最佳实践

- 使用详细的文字描述
- 描述场景和氛围
- 可指定艺术风格
- 适合中文场景生成

### 示例

```
未来科技城市夜景，摩天大楼林立，霓虹灯闪烁，飞行汽车穿梭其间，全息广告牌悬浮空中，雨后的街道反射着五彩灯光，赛博朋克风格，电影级画面
```

---

## Boogu-Image

### 基本格式

```
<主体描述>, <风格描述>
```

### 语法特点

- 风格化生成
- 色彩表现力强
- 支持多种艺术风格
- 细节丰富

### 最佳实践

- 描述主体和风格
- 明确色彩和氛围
- 适合艺术创作
- 支持概念设计

### 示例

```
Steampunk mechanical city, brass gears and copper pipes, Victorian architecture with steam engines, foggy atmosphere, warm golden lighting, detailed mechanical elements, retro-futuristic style
```

---

## LongCat-Image

### 基本格式

```
<主体描述>, <风格描述>, <质量词>
```

### 语法特点

- 高分辨率输出
- 细节丰富
- 色彩准确
- 支持长文本描述

### 最佳实践

- 详细描述场景
- 指定风格和质量
- 描述光照和构图
- 适合高质量图像

### 示例

```
Panoramic landscape of floating islands in the sky, waterfalls cascading into clouds, ancient temples on each island, magical aurora in the background, fantasy art style, epic scale, highly detailed, 8k
```

---

## Flux2-Klein

### 埂本格式

```
<自然语言描述>, <风格关键词>
```

### 语法特点

- 增强的自然语言理解
- 更好的细节保留
- 高分辨率输出
- 支持复杂场景

### 最佳实践

- 使用详细的自然语言描述
- 描述具体的场景和元素
- 指定艺术风格
- 适合高质量创作

### 示例

```
A majestic phoenix rising from ashes, flames transitioning from deep red to brilliant gold, feathers detailed with intricate patterns, dramatic lighting from below, epic fantasy art, cinematic composition, highly detailed
```

---

## GLM-Image

### 基本格式

```
<自然语言描述>
```

### 语法特点

- 中文理解优化
- 知识增强
- 风格多样
- 细节丰富

### 最佳实践

- 使用详细的文字描述
- 描述场景和氛围
- 可指定艺术风格
- 适合中文场景生成

### 示例

```
水墨风格的江南水乡，小桥流水，白墙黛瓦，柳树依依，渔船点缀河面，远处山峦起伏，烟雨朦胧，中国传统绘画风格，意境悠远
```

---

## GPT-Image2

### 基本格式

```
<完整的自然语言描述>
```

### 语法特点

- **纯自然语言**: 无需特殊语法
- **对话式交互**: 可通过对话迭代修改
- **内置世界知识**: 理解文化、历史、科学等概念
- **多模态理解**: 可参考上传的图片进行生成
- **高保真度**: 对复杂场景和细节理解能力强

### 最佳实践

- 使用详细的自然语言描述
- 描述具体的场景、物体、光照、氛围
- 可以指定艺术风格和参考作品
- 适合商业应用和概念设计

### 示例

```
A cinematic wide shot of a futuristic city at night, with towering glass skyscrapers reflecting neon lights in pink and blue. Flying vehicles leave light trails across the sky, while pedestrians with holographic umbrellas walk on elevated walkways. Light rain creates a misty atmosphere, and holographic advertisements float between buildings.
```

---

## Nanobanana

### 基本格式

```
<主体描述>, <风格描述>
```

### 语法特点

- 轻量化生成
- 快速迭代
- 风格多样
- 支持图像编辑

### 最佳实践

- 使用简洁的描述
- 快速迭代调整
- 适合快速原型
- 支持风格探索

### 示例

```
Minimalist logo design, clean geometric shapes, modern aesthetic, gradient colors, professional branding, vector style, clean lines
```

---

## Pony Diffusion

### 基本格式

```
正向提示词: <tag格式描述>, <风格描述>, <质量词>
反向提示词: <排除内容>, <负面质量词>
```

### 语法特点

- **Tag格式**: 使用逗号分隔的标签式描述
- **触发词**: 每个模型有特定的触发词（trigger words）
- **评分系统**: 支持`score_9, score_8_up, score_7_up`等质量评分
- **nsfw分级**: 支持不同安全级别的内容生成
- **动漫风格优化**: 专为动漫/二次元风格优化

### 常见触发词

| 触发词 | 说明 |
|--------|------|
| `score_9, score_8_up, score_7_up` | 质量评分，分数越高质量越好 |
| `source_anime` | 动漫风格标记 |
| `source_pony` | Pony原生风格 |
| `score_6, score_5, score_4` | 低质量评分（用于反向提示） |

### 示例

```
正向: score_9, score_8_up, score_7_up, source_anime, 1girl, long silver hair, blue eyes, flowing dress, standing in flower field, soft lighting, detailed, beautiful

反向: score_6, score_5, score_4, worst quality, low quality, bad anatomy, bad hands, blurry, watermark
```

---

## Midjourney

### 基本格式

```
<主体描述>, <风格描述>, <参数>
```

### 参数语法

| 参数 | 说明 | 示例 |
|------|------|------|
| `--ar` | 宽高比 | `--ar 16:9` |
| `--s` | 风格化程度 (0-1000) | `--s 250` |
| `--c` | 混乱度 (0-100) | `--c 50` |
| `--q` | 质量 (.25/.5/1) | `--q 2` |
| `--v` | 版本号 | `--v 6` |
| `--niji` | 动漫风格 | `--niji 6` |
| `--no` | 负面提示 | `--no text, watermark` |
| `--style` | 风格变体 | `--style raw` |
| `--iw` | 图片权重 (0-2) | `--iw 1.5` |
| `--seed` | 种子值 | `--seed 12345` |
| `--tile` | 无缝纹理 | `--tile` |
| `--video` | 生成视频 | `--video` |

### 语法特点

- 使用逗号分隔描述词
- 支持URL作为参考图
- `::` 语法设置词权重: `sunset::2, ocean::1`
- `--no` 作为负面提示

### 示例

```
a majestic dragon soaring through clouds at sunset, scales shimmering with golden light, epic fantasy art, dramatic lighting, highly detailed, cinematic composition --ar 16:9 --v 6 --s 250 --no text, watermark
```

---

## 通用提示词结构

### 标准结构模板

```
[主体] + [动作/姿态] + [服装/外观] + [环境/背景] + [光照] + [风格] + [质量词]
```

### 描述顺序建议

1. **主体** (Subject): 人物、物体、场景焦点
2. **特征** (Features): 外观、服装、表情
3. **动作** (Action): 姿态、运动状态
4. **环境** (Environment): 背景、场景元素
5. **氛围** (Atmosphere): 光照、天气、情绪
6. **风格** (Style): 艺术风格、媒介
7. **技术** (Technical): 质量、分辨率、视角

### 权重分配原则

| 模型 | 核心主体 | 风格词 | 质量词 | 背景 |
|------|----------|--------|--------|------|
| SD1.5 | 1.3-1.5 | 1.1-1.2 | 1.1-1.3 | 0.8-1.0 |
| SDXL | 1.3-1.5 | 1.1-1.2 | 1.1-1.3 | 0.8-1.0 |
| Midjourney | ::3-5 | ::1-2 | 自然融入 | ::0.5-1 |
| Flux1 | 详细描述 | 明确指定 | 可选 | 详细描述 |
| Flux2-Klein | 详细描述 | 明确指定 | 可选 | 详细描述 |
| Pony Diffusion | 标签式 | 标签式 | 评分标签 | 标签式 |
| GPT-Image2 | 详细描述 | 详细描述 | 可选 | 详细描述 |
| 其他模型 | 详细描述 | 详细描述 | 可选 | 详细描述 |

### 模型选择建议

| 需求 | 推荐模型 |
|------|----------|
| 写实人像 | SDXL, Flux1, Flux2-Klein, GPT-Image2 |
| 动漫插画 | Anima, Pony Diffusion, Midjourney(--niji) |
| 风景摄影 | Midjourney, SDXL, Flux1, Flux2-Klein |
| 概念艺术 | Midjourney, Flux1, Flux2-Klein, GLM-Image |
| 商业产品 | GPT-Image2, Flux1, Krea2 |
| 中文场景 | Qwen-Image-2512, HunyuanImage_2.1, ERNIE-Image, GLM-Image |
| 国风创作 | HunyuanImage_2.1, GLM-Image, Qwen-Image-2512 |
| 快速原型 | Krea2, Nanobanana, Mage-Flow |
| 精细控制 | SD1.5, SDXL |
| 动漫角色 | Anima, Pony Diffusion |
| 蒸汽朋克 | Boogu-Image |
| 仙侠风格 | HunyuanImage_2.1 |
