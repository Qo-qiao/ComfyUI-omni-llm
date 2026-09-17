# 明星/名人识别指南

本文档提供系统化的明星识别方法，帮助准确识别图片中的明星/名人并生成精准的描述提示词。

---

## 目录

1. [识别原则](#识别原则)
2. [明星识别流程](#明星识别流程)
3. [面部特征匹配方法](#面部特征匹配方法)
4. [明星描述框架](#明星描述框架)
5. [明星提示词模板](#明星提示词模板)
6. [注意事项](#注意事项)

---

## 识别原则

### 核心原则

1. **面部特征匹配** - 通过五官比例、面部轮廓、标志性特征判断
2. **多角度验证** - 结合面部、体型、发型、常见造型综合判断
3. **不确定性标注** - 无法确定时详细描述特征而非猜测
4. **尊重隐私** - 客观描述可见特征，避免主观评价

### 识别置信度

| 置信度 | 判断标准 | 输出方式 |
|--------|----------|----------|
| 高 | 面部特征高度吻合，可确认身份 | 直接说出明星姓名 |
| 中 | 部分特征匹配，有一定相似度 | "疑似为XX" |
| 低 | 仅有模糊相似 | 不猜测，详细描述面部特征 |

---

## 明星识别流程

```
Step 1: 面部特征提取
  ├─ 面部轮廓（脸型、下颌线、颧骨）
  ├─ 眼睛（形状、大小、间距、眼角方向）
  ├─ 鼻子（形状、大小、鼻翼宽度）
  ├─ 嘴巴（唇形、大小、微笑线）
  ├─ 眉毛（形状、粗细、间距）
  └─ 标志性特征（痣、疤痕、纹身、独特特征）

Step 2: 整体特征提取
  ├─ 发型发色
  ├─ 体型比例
  ├─ 服装风格
  └─ 气质/风格

Step 3: 身份匹配
  ├─ 高置信度 → 输出明星名
  ├─ 中置信度 → 输出疑似明星+特征描述
  └─ 低置信度 → 详细描述所有可见特征

Step 4: 生成描述
  └─ 按明星描述框架组织输出
```

---

## 面部特征匹配方法

### 高辨识度面部特征

以下特征最具辨识度，优先匹配：

| 特征类型 | 说明 | 示例 |
|----------|------|------|
| 面部轮廓 | 脸型、下颌线形状 | 方脸、瓜子脸、圆脸、心形脸 |
| 眼睛特征 | 眼形、大小、间距、特殊瞳色 | 丹凤眼、圆眼、深陷眼窝、异色瞳 |
| 鼻子特征 | 鼻梁高度、鼻翼宽度、鼻尖形状 | 高鼻梁、鹰钩鼻、宽鼻翼 |
| 眉毛特征 | 眉形、粗细、间距 | 剑眉、弯眉、连心眉、粗眉 |
| 标志性特征 | 独一无二的识别特征 | 面部痣、伤疤、纹身、胎记 |

### 中辨识度特征

| 特征类型 | 说明 |
|----------|------|
| 发型 | 标志性发型（但明星经常变换） |
| 体型 | 特殊体型（异常高/矮/壮/瘦） |
| 皮肤 | 肤色、肤质特征 |
| 表情习惯 | 标志性表情或微笑方式 |

### 低辨识度特征

| 特征类型 | 说明 |
|----------|------|
| 服装 | 明星经常变换服装 |
| 妆容 | 不同场合妆容差异大 |
| 发色 | 染发频繁，不稳定 |

---

## 明星描述框架

### 识别成功时

```
明星姓名
├─ 基本信息: 性别、年龄外观、国籍/地区
├─ 面部特征: 脸型、五官特征、标志性面部特征
├─ 发型发色: 当前发型和发色
├─ 体型特征: 身高体型印象
├─ 服装造型: 当前穿着
├─ 气质风格: 整体气质和风格印象
└─ 图片风格: 写真/剧照/街拍/活动照
```

### 识别失败时

```
未知人物
├─ 基本信息: 性别、年龄外观、种族特征
├─ 面部特征: 脸型、五官详细描述
├─ 标志性特征: 最独特的1-2个面部特征
├─ 发型发色: 详细描述
├─ 体型特征: 身高体型
├─ 服装造型: 详细描述
├─ 气质风格: 整体气质印象
└─ 相似参考: "面部特征近似于XX风格"（可选）
```

---

## 明星提示词模板

### 已识别明星模板

**Stable Diffusion:**
```
(明星姓名:1.3), [面部特征描述], [发型发色], [服装描述], [表情姿态], [光照], [背景], [摄影风格], [质量词]

示例:
(Scarlett Johansson:1.3), (heart-shaped face:1.1), (full lips:1.1), (green eyes:1.1), (wavy blonde hair:1.1), wearing elegant red gown, confident expression, soft studio lighting, neutral background, professional portrait photography, 8k, photorealistic, highly detailed
```

**Midjourney:**
```
明星姓名, [面部特征], [发型发色], [服装], [表情], [摄影风格], [参数]

示例:
Scarlett Johansson, heart-shaped face, full lips, green eyes, wavy blonde hair, elegant red gown, confident expression, soft studio lighting, professional portrait photography --ar 2:3 --v 6 --s 250
```

**DALL-E 3:**
```
A professional portrait photograph of Scarlett Johansson. She has a heart-shaped face with full lips and green eyes. Her wavy blonde hair falls naturally around her shoulders. She is wearing an elegant red gown and has a confident expression. The lighting is soft studio lighting against a neutral background.
```

**Flux:**
```
Professional portrait photograph, Scarlett Johansson, heart-shaped face, full lips, green eyes, wavy blonde hair, elegant red gown, confident expression, soft studio lighting, neutral background, photorealistic, 8k, highly detailed
```

### 未识别人物模板

```
[性别] [年龄] [种族] person with [详细面部特征], [发型发色], wearing [服装], [表情], [摄影风格], [质量词]

示例:
Young East Asian woman with oval face, almond-shaped eyes, small nose, straight black shoulder-length hair, wearing white blouse, gentle smile, soft natural lighting, professional portrait photography, 8k, photorealistic
```

### 明星风格模仿模板

当用户想要生成类似某明星风格但非该明星本人的图片时：

```
[性别] person resembling [明星姓名] in [面部特征], [发型发色], [服装], [摄影风格]

示例:
Young woman resembling Scarlett Johansson in facial features, heart-shaped face, full lips, wavy auburn hair, wearing casual denim jacket, outdoor portrait photography, golden hour lighting
```

---

## 注意事项

### 面部变化因素

明星的面部会因以下因素变化，识别时需考虑：

| 因素 | 影响 | 处理方式 |
|------|------|----------|
| 年龄 | 面部随年龄变化 | 描述当前年龄外观 |
| 妆容 | 改变五官视觉效果 | 透过妆容描述底层面部结构 |
| 体重变化 | 面部胖瘦变化 | 描述当前面部状态 |
| 修图/滤镜 | 改变面部特征 | 描述图片中可见的特征 |
| 角色扮演 | 剧中造型可能差异大 | 描述当前造型，注明角色 |

### 特殊场景处理

**剧照/角色造型**:
- 描述当前角色造型
- 标注作品名和角色名
- 同时描述明星本人特征

**AI生成明星图片**:
- 识别是否为AI生成（面部不自然、手指异常等）
- 描述AI生成的风格特征
- 说明"AI生成图片"

**多人合照**:
- 分别识别每个人物
- 描述人物之间的关系和位置
- 按重要性排序描述

**侧脸/背影**:
- 描述可见特征
- 标注"侧面/背面视角"
- 不确定时不猜测身份

### 伦理注意事项

1. **客观描述** - 仅描述可见的物理特征，不做主观评价
2. **避免不当内容** - 不生成或描述不当内容
3. **尊重肖像权** - 提示词中标注明星姓名时，明确为"肖像参考"
4. **区分真人vs角色** - 明确区分明星本人和其饰演的角色
