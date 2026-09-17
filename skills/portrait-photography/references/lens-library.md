# 镜头语言库

焦段决定虚化、压缩与透视，景别决定叙事量。本指南帮助在提示词中精确描述镜头语言。

---

## 目录

1. [焦段与光圈参考](#焦段与光圈参考)
2. [景别分类](#景别分类)
3. [质感关键词](#质感关键词)
4. [描述规则](#描述规则)

---

## 焦段与光圈参考

| 焦段 | 光圈 | 画面语言 | 适用场景 | 提示词描述方式 |
|------|------|----------|----------|----------------|
| 24mm | f/2.8 | 广角环境人像，透视夸张 | 环境叙事大片 | wide angle environmental portrait |
| 35mm | f/1.4 | 环境人像，交代场景 | 街拍纪实 | 35mm street photography, environmental |
| 50mm | f/1.2 | 接近人眼视角 | 日常自然 | natural perspective, everyday |
| 85mm | f/1.4-1.8 | 人像皇：透视自然、奶油焦外 | 面部与半身主力 | 85mm portrait lens, shallow depth of field |
| 135mm | f/1.8 | 极致压缩、空气切割感 | 极简背景特写 | telephoto compression, isolated subject |
| 70-200mm | f/2.8 | 远距离"偷感"抓拍 | 街头/旅行 | telephoto candid, distant perspective |
| 手机主摄 26mm | f/1.8 | 日常直出、轻微畸变 | 抓拍生活流 | shot on iPhone, wide angle phone camera |

---

## 景别分类

| 景别 | 英文 | 画面范围 | 叙事量 | 适用场景 |
|------|------|----------|--------|----------|
| 特写 | close-up | 面部/局部 | 表情细节 | 情绪表达、美妆 |
| 半身 | half body / bust | 胸部以上 | 上半身姿态 | 人像主力 |
| 全身 | full body | 完整身体 | 姿态+服装 | 时尚、姿态展示 |
| 环境人像 | environmental portrait | 人物+环境 | 场景叙事 | 故事感、氛围 |

### 景别与焦段搭配建议

| 景别 | 推荐焦段 | 效果 |
|------|----------|------|
| 特写 | 85mm, 135mm | 压缩+虚化 |
| 半身 | 50mm, 85mm | 自然透视 |
| 全身 | 35mm, 50mm | 完整展示 |
| 环境人像 | 24mm, 35mm | 场景交代 |

---

## 质感关键词

### 焦外效果

- 浅景深焦点落在眼睛 → sharp focus on the eyes, shallow depth of field
- 奶油般焦外 → creamy bokeh
- 背景圆形光斑 → circular bokeh highlights, light orbs in background
- 长焦压缩感 → telephoto compression
- 前景虚化作框架 → foreground bokeh framing

### 光学特征

- 轻微暗角 → slight vignette, natural lens vignette
- 边缘色散 → chromatic aberration, color fringing
- 镜头光晕 → lens flare, natural light leak

---

## 描述规则

### 核心原则

**焦段光圈只决定描述方式，不罗列参数**

✅ 正确：`85mm 人像镜头，浅景深焦点落在眼睛`
❌ 错误：`f/1.4, ISO 200, 85mm`

### 描述模板

```
[焦段] [镜头类型]，[景深效果]，[焦点位置]
```

示例：
- `85mm portrait lens, shallow depth of field, sharp focus on the eyes`
- `35mm wide angle, environmental portrait, slight distortion`
- `telephoto lens, compressed background, isolated subject`
