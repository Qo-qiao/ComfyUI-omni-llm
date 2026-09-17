# 质量提示词词库

本文档包含经过验证的质量增强词和反向提示词，用于提升AI绘图输出的整体质量。

---

## 目录

1. [正向质量词](#正向质量词)
2. [反向提示词](#反向提示词)
3. [风格修饰词](#风格修饰词)
4. [光照与氛围词](#光照与氛围词)
5. [视角与构图词](#视角与构图词)

---

## 正向质量词

### 通用质量词

```
masterpiece, best quality, high quality, ultra highres, 8k, 4k, highly detailed, sharp focus, professional, award winning
```

### 细节增强词

```
intricate details, fine details, detailed texture, hyperdetailed, meticulous, precise lines, crisp details, high resolution
```

### 真实感词

```
photorealistic, hyperrealistic, ultra realistic, realistic, lifelike, true to life, photorealism, raw photo
```

### 艺术质量词

```
trending on artstation, deviantart, concept art, digital art, professional illustration, official art, fanart
```

### 按风格分类

#### 写实/摄影风格
```
raw photo, dslr, high quality, film grain, f/1.8, shallow depth of field, professional photography, canon, nikon, medium format, 85mm lens, studio lighting
```

#### 动漫/插画风格
```
anime style, manga style, detailed anime, cel shading, vibrant colors, clean lines, high quality anime art, studio quality
```

#### 数字艺术风格
```
digital painting, concept art, matte painting, digital art, artstation trending, cgsociety, intricate digital artwork
```

#### 传统艺术风格
```
oil painting, watercolor, acrylic, traditional art, canvas texture, brush strokes, fine art, museum quality
```

---

## 反向提示词

### 通用负面词 (推荐始终包含)

```
low quality, worst quality, bad quality, normal quality, lowres, blurry, jpeg artifacts
```

### 人物相关负面词

```
bad anatomy, bad proportions, deformed, disfigured, mutation, mutated, extra limbs, missing limbs, floating limbs, disconnected limbs, malformed hands, poorly drawn hands, bad hands, missing fingers, extra fingers, fused fingers, too many fingers, malformed face, poorly drawn face, bad face, double head, multiple heads, cloned face, long neck, bad body, bad posture
```

### 画面质量负面词

```
grainy, pixelated, compressed, overexposed, underexposed, bad lighting, harsh lighting, washed out, oversaturated, dull colors, noise, artifacts, watermark, signature, text, logo, username, error, glitch, distorted, warped
```

### 风格相关负面词

```
amateur, amateurish, beginner, low effort, messy, sketchy, rough, unfinished, incomplete, draft, doodle, scribble
```

### SD专用负面词 (Embedding)

```
ng_deepnegative_v1_75t, badhandv4, EasyNegative, verybadimagenegative_v1.3
```

### 完整反向提示词模板

#### 通用模板
```
(worst quality:1.4), (low quality:1.4), (normal quality:1.4), lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry
```

#### 人物专用模板
```
(worst quality:1.4), (low quality:1.4), bad anatomy, bad proportions, deformed, disfigured, mutation, extra limbs, missing limbs, floating limbs, poorly drawn hands, bad hands, missing fingers, extra fingers, fused fingers, malformed face, bad face, long neck, bad body
```

#### 风景专用模板
```
(worst quality:1.4), (low quality:1.4), blurry, grainy, pixelated, lowres, jpeg artifacts, watermark, text, signature, oversaturated, underexposed, overexposed
```

---

## 风格修饰词

### 艺术风格

| 风格 | 关键词 |
|------|--------|
| 印象派 | impressionist, monet style, soft brushstrokes, light and color |
| 超现实 | surrealist, dali style, dreamlike, impossible geometry |
| 赛博朋克 | cyberpunk, neon lights, futuristic, dystopian, high tech |
| 蒸汽朋克 | steampunk, victorian, brass, gears, mechanical |
| 极简主义 | minimalist, simple, clean, negative space, minimal |
| 巴洛克 | baroque, ornate, dramatic, rich details, gold |
| 浮世绘 | ukiyo-e, japanese woodblock, flat colors, bold lines |
| 波普艺术 | pop art, warhol style, bright colors, bold outlines |
| 水彩 | watercolor, wet on wet, soft edges, flowing colors |
| 油画 | oil painting, thick brushstrokes, rich textures, classical |

### 数字风格

| 风格 | 关键词 |
|------|--------|
| 3D渲染 | 3d render, octane render, unreal engine, cinema 4d, blender |
| 像素艺术 | pixel art, 8-bit, retro game style, limited palette |
| 矢量艺术 | vector art, flat design, clean lines, scalable |
| 概念艺术 | concept art, digital painting, matte painting, environment design |
| 动漫 | anime, manga, japanese animation style, cel shaded |
| 漫画 | comic book style, marvel style, dc style, inked |
| 吉卜力 | studio ghibli style, miyazaki, whimsical, hand-drawn |

---

## 光照与氛围词

### 光照类型

| 类型 | 关键词 |
|------|--------|
| 自然光 | natural lighting, sunlight, daylight, golden hour, blue hour |
| 戏剧光 | dramatic lighting, chiaroscuro, high contrast, rim lighting |
| 柔光 | soft lighting, diffused light, gentle light, ambient occlusion |
| 硬光 | hard lighting, harsh light, strong shadows |
| 背光 | backlighting, silhouette, contre-jour, rim light |
| 体积光 | volumetric lighting, god rays, light shafts, atmospheric |
| 霓虹 | neon lighting, neon glow, cyberpunk lights, rgb |
| 工作室 | studio lighting, three-point lighting, key light, fill light |

### 氛围词

| 氛围 | 关键词 |
|------|--------|
| 梦幻 | dreamy, ethereal, fantasy, magical, surreal |
| 神秘 | mysterious, enigmatic, dark, shadowy, ominous |
| 温馨 | cozy, warm, inviting, comfortable, homey |
| 史诗 | epic, grand, majestic, heroic, monumental |
| 忧郁 | melancholic, sad, somber, moody, emotional |
| 宁静 | serene, peaceful, calm, tranquil, quiet |
| 活力 | vibrant, energetic, dynamic, lively, colorful |

---

## 视角与构图词

### 相机角度

```
front view, side view, back view, three-quarter view, bird's eye view, worm's eye view, low angle, high angle, dutch angle, overhead view
```

### 镜头类型

```
close-up, extreme close-up, medium shot, full body shot, wide shot, establishing shot, portrait, headshot, bust shot, cowboy shot
```

### 焦距效果

```
wide angle, fisheye, telephoto, macro, 35mm, 50mm, 85mm, 135mm, 200mm
```

### 景深效果

```
shallow depth of field, deep focus, bokeh, background blur, foreground blur, focus on subject
```

### 构图法则

```
rule of thirds, golden ratio, centered composition, symmetrical composition, leading lines, framing, negative space, dynamic composition
```
