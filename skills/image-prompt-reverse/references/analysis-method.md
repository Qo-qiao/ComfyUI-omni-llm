# 图片分析方法论

本文档提供系统化的图片分析方法，用于准确提取图片中的视觉元素并转换为提示词。

---

## 目录

1. [分析框架](#分析框架)
2. [主体识别](#主体识别)
3. [风格判断](#风格判断)
4. [技术参数推断](#技术参数推断)
5. [描述词选择原则](#描述词选择原则)

---

## 分析框架

### 五层分析法

按以下顺序分析图片，从整体到细节：

```
第一层: 整体印象 (风格、氛围、情绪)
第二层: 主体内容 (人物、物体、场景焦点)
第三层: 环境背景 (场景、地点、背景元素)
第四层: 技术细节 (光照、视角、构图)
第五层: 质量特征 (分辨率、细节程度、瑕疵)
```

### 分析检查清单

分析图片时，依次检查以下元素：

#### 必检项目
- [ ] 主体是什么？(人物/物体/场景)
- [ ] 主体数量？(单人/多人/无人物)
- [ ] 整体风格？(写实/动漫/艺术风格)
- [ ] 光照条件？(自然光/人造光/特殊光效)
- [ ] 视角构图？(特写/全景/仰视/俯视)

#### 人物图片额外检查
- [ ] 性别、年龄、种族特征
- [ ] 发型、发色、眼睛颜色
- [ ] 面部表情、情绪
- [ ] 服装款式、颜色、材质
- [ ] 姿态、动作、手势
- [ ] 身体朝向

#### 场景图片额外检查
- [ ] 地点类型 (室内/室外/幻想场景)
- [ ] 时间 (白天/夜晚/黄昏/黎明)
- [ ] 天气条件
- [ ] 主要物体和装饰
- [ ] 空间深度

---

## 主体识别

### 人物特征描述模板

```
[数量] + [性别] + [年龄] + [种族/民族] + [发型发色] + [眼睛颜色] + [面部特征] + [表情情绪] + [服装] + [姿态动作] + [身体朝向]
```

#### 发型描述词

| 长度 | 形状 | 质感 |
|------|------|------|
| bald, short, medium, long, very long | straight, wavy, curly, coily, braided, ponytail, twin tails, bun | silky, frizzy, thick, thin, voluminous |

| 发色 | |
|------|------|
| black, brown, blonde, red, auburn, white, silver, gray, pink, blue, purple, green, rainbow, ombre, highlighted |

#### 眼睛描述词

| 形状 | 颜色 | 特殊效果 |
|------|------|------|
| round, almond, hooded, upturned, downturned, wide, narrow | brown, black, blue, green, hazel, gray, amber, violet, red, heterochromia | glowing, sparkling, detailed iris, beautiful eyes |

#### 表情描述词

| 情绪类别 | 关键词 |
|----------|--------|
| 正面 | smiling, happy, joyful, cheerful, excited, confident, peaceful, serene |
| 负面 | sad, crying, angry, fearful, anxious, depressed, melancholic |
| 中性 | neutral, serious, thoughtful, contemplative, calm |
| 特殊 | seductive, mysterious, playful, mischievous, surprised, shocked |

#### 姿态描述词

| 类别 | 关键词 |
|------|--------|
| 站姿 | standing, leaning, arms crossed, hands on hips, hand on face |
| 坐姿 | sitting, cross-legged, kneeling, on chair, on ground |
| 躺姿 | lying down, reclining, on back, on side, on stomach |
| 动态 | walking, running, jumping, dancing, fighting, reaching |

### 物体特征描述模板

```
[物体名称] + [数量] + [大小] + [形状] + [颜色] + [材质] + [状态] + [位置关系]
```

### 场景特征描述模板

```
[场景类型] + [地点] + [时间] + [天气] + [主要元素] + [氛围]
```

---

## 风格判断

### 风格识别指南

#### 写实 vs 风格化判断

| 特征 | 写实风格 | 风格化 |
|------|----------|--------|
| 线条 | 无明显线条，边缘自然过渡 | 明显线条，边缘清晰 |
| 色彩 | 自然色彩，真实光影 | 夸张色彩，非真实光影 |
| 比例 | 符合真实人体比例 | 夸张或简化比例 |
| 细节 | 高度细节，真实纹理 | 简化或风格化细节 |

### 主流风格识别特征

| 风格 | 识别特征 |
|------|----------|
| 照片级写实 | 真实光影、皮肤纹理、自然色彩、景深效果 |
| 动漫/日式 | 大眼睛、简化面部、鲜明轮廓、鲜艳色彩 |
| 欧美卡通 | 夸张比例、粗线条、明快色彩 |
| 油画风格 | 笔触可见、厚重质感、古典色彩 |
| 水彩风格 | 淡雅色彩、水痕效果、柔和边缘 |
| 3D渲染 | 光滑表面、精确光影、材质质感 |
| 像素艺术 | 方块像素、有限色彩、复古感 |
| 概念艺术 | 氛围感强、快速笔触、设计感 |

### 艺术风格参考

识别到特定风格时，可引用以下艺术家或作品：

| 风格 | 参考关键词 |
|------|------------|
| 印象派 | monet, renoir, impressionist |
| 超现实 | dali, magritte, surrealism |
| 浮世绘 | hokusai, utamaro, ukiyo-e |
| 吉卜力 | miyazaki, studio ghibli |
| 赛博朋克 | blade runner, ghost in the shell |
| 蒸汽朋克 | victorian sci-fi, brass and gears |

---

## 技术参数推断

### 光照分析

| 光照类型 | 视觉特征 |
|----------|----------|
| 自然日光 | 柔和阴影，色温偏暖(日出/日落)或中性(正午) |
| 工作室光 | 均匀照明，可控阴影，专业质感 |
| 戏剧光 | 强对比，深阴影，高光突出 |
| 背光/逆光 | 轮廓光，主体暗，边缘发光 |
| 体积光 | 光束可见，空气中有颗粒感 |
| 霓虹光 | 彩色光源，赛博朋克感，反射光 |

### 视角分析

| 视角 | 特征 |
|------|------|
| 平视 | 自然视角，与人眼同高 |
| 仰视 | 主体显得高大，天空背景多 |
| 俯视 | 主体显得渺小，地面可见多 |
| 鸟瞰 | 从高处向下看，全景感 |
| 虫视 | 从低处向上看，夸张透视 |

### 构图分析

| 构图类型 | 特征 |
|----------|------|
| 中心构图 | 主体在画面中央 |
| 三分法 | 主体在三分线交点 |
| 对称构图 | 左右或上下对称 |
| 引导线 | 线条引导视线到主体 |
| 框架构图 | 主体被框架包围 |

### 景深分析

| 景深类型 | 特征 |
|----------|------|
| 浅景深 | 背景模糊，主体清晰，人像常用 |
| 深景深 | 前后都清晰，风景常用 |
| 移焦 | 部分清晰部分模糊 |

---

## 描述词选择原则

### 精准原则

1. **具体优于抽象**: "long flowing silver hair" > "nice hair"
2. **可量化优于模糊**: "35mm lens" > "normal lens"
3. **专业术语增强准确性**: "chiaroscuro lighting" > "dramatic lighting"

### 优先级排序

```
高优先级: 主体、风格、核心特征
中优先级: 环境、光照、构图
低优先级: 细节修饰、质量词
```

### 避免的描述方式

- ❌ 模糊描述: "beautiful", "nice", "good"
- ❌ 矛盾描述: "bright darkness", "colorful black and white"
- ❌ 过度描述: 堆砌过多形容词导致混乱
- ❌ 主观描述: "I think", "maybe", "probably"

### 推荐的描述方式

- ✅ 客观描述: 描述可见特征而非主观感受
- ✅ 结构化: 按主体→环境→风格→技术顺序
- ✅ 精确词汇: 使用专业术语和具体描述
- ✅ 适度修饰: 关键特征加强，次要特征简化

### 提示词长度建议

| 模型 | 推荐长度 |
|------|----------|
| Stable Diffusion | 50-150 tokens |
| SDXL | 30-100 tokens |
| Midjourney | 30-80 words |
| DALL-E 3 | 100-400 words (自然语言) |
| Flux | 50-150 words |
