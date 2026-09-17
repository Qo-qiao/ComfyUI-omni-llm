# 提示词生成策略与模板

本文档提供系统化的提示词生成策略和各类场景的专用模板，帮助生成更精准、可复现的AI绘图提示词。

---

## 目录

1. [提示词生成策略](#提示词生成策略)
2. [人物肖像模板](#人物肖像模板)
3. [动漫角色模板](#动漫角色模板)
4. [风景摄影模板](#风景摄影模板)
5. [城市建筑模板](#城市建筑模板)
6. [静物产品模板](#静物产品模板)
7. [幻想科幻模板](#幻想科幻模板)
8. [海报设计模板](#海报设计模板)
9. [产品商业摄影模板](#产品商业摄影模板)
10. [食物摄影模板](#食物摄影模板)
11. [时尚摄影模板](#时尚摄影模板)
12. [UI/UX设计模板](#uiux设计模板)
13. [传统绘画模板](#传统绘画模板)
14. [3D渲染/像素艺术模板](#3d渲染像素艺术模板)
15. [多模型适配策略](#多模型适配策略)

---

## 提示词生成策略

### 核心原则

**1. 结构化组织**
```
[主体] → [特征] → [环境] → [光照] → [风格] → [技术] → [质量]
```

**2. 重要性排序**
- 核心主体（最高权重）
- 关键特征（高权重）
- 环境背景（中权重）
- 风格修饰（中权重）
- 技术参数（低权重）
- 质量词（默认权重）

**3. 精准描述原则**
- 使用具体可量化的描述
- 避免主观形容词
- 使用专业术语
- 保持描述一致性

### 提示词优化流程

```
Step 1: 提取核心元素
  └─ 确定主体、风格、关键特征

Step 2: 构建基础描述
  └─ 按结构组织基础提示词

Step 3: 添加细节修饰
  └─ 补充特征细节、环境描述

Step 4: 优化权重分配
  └─ 为核心元素添加权重

Step 5: 添加质量词
  └─ 补充正向质量词和反向提示词

Step 6: 模型适配调整
  └─ 根据目标模型调整格式
```

### 权重分配策略

| 元素类型 | SD权重 | MJ权重 | 说明 |
|----------|--------|--------|------|
| 核心主体 | 1.3-1.5 | ::3-5 | 最重要的识别特征 |
| 关键特征 | 1.2-1.3 | ::2-3 | 重要的外貌/特征描述 |
| 风格词 | 1.1-1.2 | ::1-2 | 艺术风格修饰 |
| 环境词 | 0.9-1.1 | ::0.5-1 | 背景环境描述 |
| 质量词 | 1.1-1.3 | 自然融入 | 质量增强词 |

---

## 人物肖像模板

### 基础模板结构

```
[镜头] portrait of a [年龄] [性别] with [面部特征], [发型], wearing [服装], [表情], [姿态], [光照], [背景], [摄影风格], [质量词]
```

### 各模型格式

**Stable Diffusion:**
```
(85mm portrait:1.3) of a (young woman:1.2) with (symmetrical facial features:1.2), (long flowing auburn hair:1.1), wearing (elegant black dress:1.1), (gentle smile:1.1), (three-quarter view:1.0), (soft natural lighting from window:1.2), (blurred bokeh background:0.9), (professional photography:1.1), 8k, highly detailed, sharp focus, masterpiece

Negative: (worst quality:1.4), (low quality:1.4), (normal quality:1.4), bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry
```

**Midjourney:**
```
85mm portrait of a young woman with symmetrical facial features, long flowing auburn hair, wearing elegant black dress, gentle smile, soft natural lighting from window, blurred bokeh background, professional photography, 8k, highly detailed --ar 2:3 --v 6 --s 250
```

**DALL-E 3:**
```
A professional 85mm portrait photograph of a young woman with symmetrical facial features and long flowing auburn hair. She is wearing an elegant black dress and has a gentle smile. The lighting is soft and natural, coming from a window to the side. The background is beautifully blurred with creamy bokeh. The image is shot in professional photography style with high detail and sharp focus.
```

**Flux:**
```
Professional portrait photography, 85mm lens, young woman with symmetrical features and flowing auburn hair, elegant black dress, gentle smile, soft window light, creamy bokeh background, photorealistic, 8k, highly detailed
```

### 人像细分模板

**时尚人像:**
```
[时尚风格] fashion portrait of a [模特描述], wearing [设计师服装], [夸张造型], [艺术光影], [时尚摄影风格], [杂志质感]

示例:
High fashion editorial portrait of a young Asian model, wearing avant-garde designer gown with dramatic silhouette, bold makeup with graphic eyeliner, dramatic side lighting creating strong shadows, Vogue magazine style, high-end fashion photography
```

**商业人像:**
```
Professional headshot of a [职业] [性别], wearing [商务着装], [自信表情], [专业布光], [中性背景], [企业形象风格]

示例:
Professional corporate headshot of a middle-aged businessman, wearing navy blue suit with white shirt and red tie, confident friendly smile, professional three-point lighting, neutral gray background, corporate photography style
```

---

## 动漫角色模板

### 基础模板结构

```
[风格] anime [角色类型] with [发型], [眼睛], [面部特征], wearing [服装], [表情], [姿势], [背景], [艺术风格], [质量词]
```

### 各模型格式

**Stable Diffusion:**
```
(Studio Ghibli style:1.3) anime girl with (long pink twin tails:1.2), (large expressive green eyes:1.2), (wearing school uniform with red ribbon:1.1), (cheerful expression:1.1), (standing pose:1.0), (cherry blossom background:1.0), soft pastel colors, detailed anime illustration, masterpiece, best quality

Negative: (worst quality:1.4), (low quality:1.4), bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, bad proportions, deformed
```

**Midjourney:**
```
Studio Ghibli style anime girl character, long pink twin tails, large expressive green eyes, wearing Japanese school uniform with red ribbon, cheerful expression, standing pose, cherry blossom petals falling in background, soft pastel colors, detailed anime illustration --ar 2:3 --niji 6 --style cute
```

**DALL-E 3:**
```
A Studio Ghibli style anime illustration of a young girl character with long pink hair styled in twin tails and large expressive green eyes. She is wearing a Japanese school uniform with a red ribbon tie. She has a cheerful expression and is standing in a relaxed pose. Cherry blossom petals are falling around her in the background. The art style features soft pastel colors and the characteristic warm, whimsical aesthetic of Studio Ghibli animation.
```

**NijiJourney:**
```
Beautiful anime girl, long pink twin tails, large green eyes, school uniform with red ribbon, cheerful smile, cherry blossom background, soft lighting, pastel colors, detailed illustration --niji 6 --ar 2:3 --style cute
```

### 动漫风格细分

**赛博朋克风格:**
```
Cyberpunk anime character with [科技特征], [霓虹光照], [未来城市背景], [科幻元素]

示例:
Cyberpunk anime girl with cybernetic eye implant and glowing blue circuit patterns on skin, short silver hair with neon blue highlights, wearing futuristic tactical outfit, serious expression, standing on rainy neon-lit street with holographic advertisements, blade runner aesthetic, neon color palette
```

**奇幻风格:**
```
Fantasy anime [角色职业] with [魔法元素], [奇幻服装], [神秘背景], [魔法效果]

示例:
Fantasy anime mage girl with long white hair and glowing purple eyes, wearing elaborate magical robes with star patterns, holding staff with glowing crystal, floating magical runes around her, mystical forest background with ethereal lighting, detailed fantasy illustration
```

---

## 风景摄影模板

### 基础模板结构

```
[时间] [季节] landscape of [地点], [天气], [前景], [中景], [背景], [光照], [色彩], [摄影风格], [质量词]
```

### 各模型格式

**Stable Diffusion:**
```
(golden hour:1.3) (autumn landscape:1.2) of (mountain lake:1.2), (clear sky with few clouds:1.0), (colorful fallen leaves in foreground:1.1), (crystal clear lake reflection:1.2), (distant snow-capped mountains:1.1), (warm side lighting:1.2), (rich orange and gold tones:1.1), panoramic view, landscape photography, 8k, photorealistic, highly detailed

Negative: (worst quality:1.4), (low quality:1.4), blurry, grainy, oversaturated, underexposed, overexposed, watermark, text
```

**Midjourney:**
```
Golden hour autumn landscape of mountain lake, clear sky with scattered clouds, colorful fallen maple leaves in foreground, crystal clear lake with perfect reflection, distant snow-capped mountains, warm golden side lighting, rich orange and gold color palette, panoramic landscape photography, 8k, photorealistic --ar 16:9 --v 6 --s 250
```

**DALL-E 3:**
```
A breathtaking golden hour autumn landscape photograph of a serene mountain lake. The foreground is covered with colorful fallen maple leaves in vibrant oranges, reds, and golds. The crystal-clear lake creates a perfect mirror reflection of the scene. In the background, majestic snow-capped mountains rise against a clear sky with scattered clouds. The warm golden side lighting creates a magical atmosphere with rich orange and gold tones throughout the scene. Shot as a panoramic landscape in photorealistic style with exceptional detail.
```

**Flux:**
```
Golden hour landscape photography, autumn mountain lake scene, colorful fallen leaves foreground, crystal clear reflective water, distant snow-capped peaks, warm golden lighting, rich autumn colors, panoramic composition, photorealistic, 8k, highly detailed
```

### 风景细分模板

**海景:**
```
[时间] seascape with [海洋状态], [天空状况], [海岸特征], [光照], [氛围]

示例:
Dramatic sunset seascape with crashing waves on rocky coastline, stormy sky with dramatic clouds breaking to reveal golden sunlight, sea spray and mist, long exposure silky water effect, powerful and moody atmosphere, professional landscape photography
```

**森林:**
```
[季节] forest scene with [树木类型], [光照条件], [地面植被], [氛围]

示例:
Misty morning forest scene with towering redwood trees, sunbeams filtering through dense canopy creating god rays, lush ferns covering forest floor, ethereal and peaceful atmosphere, soft diffused lighting, nature photography
```

---

## 城市建筑模板

### 基础模板结构

```
[时间] cityscape of [城市类型], [建筑特征], [光源], [天气/氛围], [视角], [技术], [风格], [质量词]
```

### 各模型格式

**Stable Diffusion:**
```
(night cityscape:1.3) of (cyberpunk metropolis:1.2), (towering skyscrapers with glass facades:1.2), (neon signs and street lights:1.1), (reflecting on wet pavement:1.1), (light rain creating atmospheric haze:1.0), (street level perspective:1.1), (long exposure light trails:1.0), blade runner aesthetic, cinematic composition, 8k, highly detailed

Negative: (worst quality:1.4), (low quality:1.4), blurry, grainy, oversaturated, underexposed, overexposed, watermark, text, bad architecture
```

**Midjourney:**
```
Night cityscape of cyberpunk metropolis, towering futuristic skyscrapers with illuminated glass facades, vibrant neon signs in multiple colors, wet streets reflecting city lights, light rain creating atmospheric haze and light blooms, street level perspective, long exposure light trails from passing vehicles, blade runner aesthetic, cinematic composition, dramatic lighting --ar 16:9 --v 6 --s 250
```

**DALL-E 3:**
```
A dramatic night cityscape of a futuristic cyberpunk metropolis. Towering skyscrapers with illuminated glass facades dominate the skyline, covered in vibrant neon signs displaying advertisements in multiple colors. The wet streets below reflect the kaleidoscope of city lights. Light rain creates an atmospheric haze with beautiful light blooms. Shot from street level perspective with long exposure capturing light trails from passing vehicles. The aesthetic is reminiscent of Blade Runner with cinematic composition and dramatic lighting.
```

### 建筑细分模板

**古典建筑:**
```
[角度] view of [建筑风格] [建筑类型], [建筑细节], [光照], [环境], [摄影风格]

示例:
Low angle view of Gothic cathedral architecture, intricate stone carvings and pointed arches, flying buttresses, stained glass windows, dramatic sunset lighting creating warm golden glow on stone facade, surrounded by historic European cityscape, architectural photography
```

**现代建筑:**
```
[视角] of modern [建筑类型], [材料特征], [设计特点], [环境], [摄影风格]

示例:
Dramatic upward perspective of modern glass skyscraper, reflective glass facade mirroring blue sky and clouds, sleek geometric design with clean lines, surrounded by contemporary urban plaza, minimalist architectural photography
```

---

## 静物产品模板

### 基础模板结构

```
[布光类型] product photography of [产品], [材质特征], [颜色], [背景], [构图], [摄影风格], [质量词]
```

### 各模型格式

**Stable Diffusion:**
```
(professional studio lighting:1.2) product photography of (vintage leather handbag:1.3), (rich brown textured leather:1.2), (brass hardware details:1.1), (neutral gray background:1.0), (three-quarter angle:1.1), (soft shadows:1.0), commercial photography style, 8k, highly detailed, sharp focus

Negative: (worst quality:1.4), (low quality:1.4), blurry, grainy, bad lighting, harsh shadows, overexposed, underexposed, watermark, text
```

**Midjourney:**
```
Professional studio product photography of vintage leather handbag, rich textured brown leather with natural grain, brass hardware and buckle details, neutral gray seamless background, three-quarter angle view, soft studio lighting with gentle shadows, commercial photography style, 8k, highly detailed --ar 1:1 --v 6
```

**DALL-E 3:**
```
A professional studio product photograph of a vintage leather handbag. The bag is crafted from rich textured brown leather with visible natural grain patterns. Brass hardware and buckle details add elegant accents. Shot against a neutral gray seamless background from a three-quarter angle. Soft studio lighting creates gentle shadows that emphasize the leather texture and bag structure. Commercial photography style with exceptional detail and sharp focus.
```

### 静物细分模板

**食物摄影:**
```
[风格] food photography of [菜品], [摆盘], [光线], [背景], [氛围]

示例:
Rustic food photography of gourmet burger with melted cheese and fresh vegetables, artisanal wooden board presentation, natural window light from side, dark textured background, steam rising from hot food, appetizing and delicious appearance, professional food photography
```

**珠宝摄影:**
```
[布光] jewelry photography of [珠宝类型], [材质], [设计], [背景], [效果]

示例:
Dramatic jewelry photography of diamond necklace, sparkling brilliant-cut diamonds in platinum setting, black reflective background creating mirror effect, focused lighting creating spectacular fire and brilliance, luxury jewelry advertisement style
```

---

## 幻想科幻模板

### 科幻场景模板

**赛博朋克:**
```
Cyberpunk [场景类型] with [科技元素], [光照], [氛围], [细节]

SD:
(cyberpunk street scene:1.3) with (flying vehicles:1.2), (holographic advertisements:1.1), (neon lights in pink and blue:1.2), (rainy night:1.1), (steam rising from vents:1.0), (distant futuristic skyline:1.1), blade runner aesthetic, cinematic lighting, 8k, highly detailed

MJ:
Cyberpunk street scene with flying vehicles and hovering drones, massive holographic advertisements covering buildings, vibrant neon lights in pink and cyan, rainy night with wet streets reflecting lights, steam rising from street vents, distant futuristic skyline with towering skyscrapers, blade runner aesthetic, cinematic lighting --ar 16:9 --v 6
```

**太空科幻:**
```
[类型] space scene with [天体], [飞船/空间站], [光照], [氛围]

SD:
(epic space scene:1.3) with (distant galaxy and nebula:1.2), (massive space station:1.2), (Earth visible in background:1.1), (dramatic rim lighting:1.2), (stars and cosmic dust:1.0), sci-fi concept art style, cinematic composition, 8k, highly detailed

MJ:
Epic space scene with colorful distant galaxy and purple nebula, massive ring-shaped space station in foreground, Earth visible in background with atmosphere glow, dramatic rim lighting from distant star, countless stars and cosmic dust, sci-fi concept art style, cinematic composition --ar 21:9 --v 6
```

### 奇幻场景模板

**西方奇幻:**
```
Fantasy [场景] with [魔法元素], [生物], [环境], [氛围]

SD:
(epic fantasy landscape:1.3) with (floating islands:1.2), (ancient magical castle:1.2), (dragons flying in sky:1.1), (waterfalls flowing upward:1.1), (magical aurora in sky:1.0), (lush impossible vegetation:1.0), fantasy concept art, dramatic lighting, 8k, masterpiece

MJ:
Epic fantasy landscape with floating islands connected by rope bridges, ancient magical castle perched on largest island, dragons flying between islands, waterfalls flowing upward defying gravity, magical aurora lighting the sky, lush impossible vegetation with giant flowers, fantasy concept art style --ar 16:9 --v 6
```

---

## 海报设计模板

### 电影海报模板

```
[风格] movie poster for [电影名称/类型], [主视觉元素], [标题排版], [色调], [氛围], [设计风格]

SD:
(cinematic movie poster:1.3) for [sci-fi thriller], (dramatic central character:1.2), (bold title typography:1.1), (dark moody color palette:1.1), (atmospheric lighting:1.2), movie poster design, professional layout, 8k, highly detailed

MJ:
Cinematic movie poster for sci-fi thriller, dramatic central character with intense expression, bold title typography at bottom, dark moody color palette with blue and orange accents, atmospheric volumetric lighting, movie poster design, professional layout --ar 2:3 --v 6

DALL-E 3:
A professional movie poster design for a sci-fi thriller film. The central character stands dramatically with an intense expression, bathed in atmospheric volumetric lighting. The color palette features dark moody tones with blue and orange accents. Bold title typography is placed at the bottom of the poster. The overall design follows professional movie poster layout conventions.
```

### 活动海报模板

```
[风格] event poster for [活动类型], [主要视觉], [文字信息], [色彩], [设计风格]

SD:
(graphic event poster:1.3) for [music festival], (vibrant abstract visuals:1.2), (bold event title and details:1.1), (neon color scheme:1.2), (modern graphic design:1.1), professional poster design, 8k

MJ:
Graphic event poster for music festival, vibrant abstract visuals with geometric shapes, bold event title and date details, neon color scheme with pink and cyan, modern minimalist graphic design --ar 2:3 --v 6
```

---

## 产品商业摄影模板

### 基础模板结构

```
[布光类型] product photography of [产品], [材质特征], [颜色], [背景], [构图], [摄影风格], [质量词]

SD:
(professional studio lighting:1.2) product photography of [luxury watch], (rose gold case and leather strap:1.2), (dramatic reflections:1.1), (dark gradient background:1.0), (three-quarter angle:1.1), commercial photography, 8k, highly detailed

MJ:
Professional studio product photography of luxury watch, rose gold case with black leather strap, dramatic reflections on polished surface, dark gradient background, three-quarter angle view, commercial photography style --ar 1:1 --v 6

DALL-E 3:
Professional studio product photograph of a luxury watch. The watch features a rose gold case and black leather strap with dramatic reflections on its polished surface. Shot against a dark gradient background from a three-quarter angle. Commercial photography style with exceptional detail and sharp focus.
```

### 电子产品模板

```
[布光] product shot of [电子产品], [材质], [设计特点], [背景], [风格]

SD:
(clean product shot:1.3) of [smartphone], (matte black finish:1.2), (sleek modern design:1.1), (minimalist white background:1.0), (soft studio lighting:1.1), tech product photography, 8k

MJ:
Clean product shot of modern smartphone, matte black finish with subtle texture, sleek minimalist design, pure white background with soft shadows, soft diffused studio lighting, tech product photography --ar 1:1 --v 6
```

---

## 食物摄影模板

### 基础模板结构

```
[风格] food photography of [食物], [摆盘], [光线], [背景], [氛围]

SD:
(rustic food photography:1.3) of [gourmet burger], (artisanal wooden board:1.2), (natural window light from side:1.1), (dark textured background:1.0), (steam rising from hot food:1.1), appetizing, professional food photography, 8k

MJ:
Rustic food photography of gourmet burger with melted cheese and fresh vegetables, artisanal wooden board presentation, natural window light from side, dark textured background, steam rising, appetizing and delicious appearance --ar 4:3 --v 6

DALL-E 3:
Rustic food photography of a gourmet burger with melted cheese and fresh vegetables. The burger is presented on an artisanal wooden board. Natural window light comes from the side, creating soft shadows. The background is dark and textured. Steam rises from the hot food, creating an appetizing and delicious appearance.
```

### 甜点模板

```
[风格] dessert photography of [甜点], [装饰], [光线], [背景]

SD:
(elegant dessert photography:1.3) of [chocolate cake], (gold leaf decoration:1.2), (soft diffused lighting:1.1), (marble surface:1.0), (crumbs and chocolate sauce:1.1), professional food photography, 8k

MJ:
Elegant dessert photography of chocolate layer cake with gold leaf decoration, soft diffused lighting, marble surface with scattered crumbs and chocolate sauce, professional food photography, appetizing --ar 1:1 --v 6
```

---

## 时尚摄影模板

### 基础模板结构

```
[风格] fashion photography of [模特描述], wearing [服装], [姿态], [布光], [背景], [摄影风格]

SD:
(high fashion editorial:1.3) of [young model], wearing [designer outfit], (confident pose:1.1), (dramatic studio lighting:1.2), (minimalist background:1.0), (Vogue magazine style:1.1), fashion photography, 8k

MJ:
High fashion editorial photography of young model wearing designer outfit, confident striking pose, dramatic studio lighting with strong shadows, minimalist gray background, Vogue magazine style, professional fashion photography --ar 2:3 --v 6

DALL-E 3:
High fashion editorial photography of a young model wearing a designer outfit. The model strikes a confident and striking pose. Dramatic studio lighting creates strong shadows and highlights. The background is minimalist gray. The overall style is reminiscent of Vogue magazine fashion photography.
```

### 珠宝摄影模板

```
[布光] jewelry photography of [珠宝类型], [材质], [设计], [背景], [效果]

SD:
(dramatic jewelry photography:1.3) of [diamond necklace], (sparkling brilliant-cut diamonds:1.2), (platinum setting:1.1), (black reflective background:1.0), (spectacular fire and brilliance:1.2), luxury jewelry advertisement, 8k

MJ:
Dramatic jewelry photography of diamond necklace, sparkling brilliant-cut diamonds in platinum setting, black reflective background creating mirror effect, focused lighting creating spectacular fire and brilliance, luxury jewelry advertisement style --ar 1:1 --v 6
```

---

## UI/UX设计模板

### APP界面模板

```
[风格] mobile app UI design for [应用类型], [界面元素], [色彩], [风格]

SD:
(clean mobile app UI:1.3) for [fitness tracker], (modern card-based layout:1.2), (vibrant gradient color scheme:1.1), (minimalist design:1.0), (flat design elements:1.1), UI/UX design, 8k

MJ:
Clean mobile app UI design for fitness tracker, modern card-based layout with health data visualization, vibrant gradient color scheme in blue and green, minimalist design with flat design elements --ar 9:16 --v 6

DALL-E 3:
Clean mobile app UI design for a fitness tracker application. The interface features a modern card-based layout with health data visualization including steps, heart rate, and calories burned. The color scheme uses vibrant gradients in blue and green. The design follows minimalist principles with flat design elements.
```

### 网页设计模板

```
[风格] web design for [网站类型], [布局], [色彩], [元素]

SD:
(modern web design:1.3) for [e-commerce], (hero section with product showcase:1.2), (clean white and accent color scheme:1.1), (responsive grid layout:1.0), (professional UI elements:1.1), web design, 8k

MJ:
Modern web design for e-commerce website, hero section with product showcase and call-to-action, clean white and accent color scheme, responsive grid layout with product cards, professional UI elements and navigation --ar 16:9 --v 6
```

---

## 传统绘画模板

### 水墨/国画模板

```
[风格] chinese ink wash painting of [主题], [笔触], [意境], [构图]

SD:
(traditional chinese ink wash:1.3) of [mountain landscape], (fluid brushstrokes:1.2), (misty atmospheric perspective:1.1), (generous white space:1.0), (monochrome with subtle color:1.1), sumi-e style, 8k

MJ:
Traditional chinese ink wash painting of mountain landscape, fluid brushstrokes creating misty peaks, atmospheric perspective with layered mountains, generous white space, monochrome with subtle warm accents, sumi-e style --ar 3:4 --v 6

DALL-E 3:
Traditional Chinese ink wash painting of a mountain landscape. Fluid brushstrokes create misty peaks that fade into the distance. The composition uses atmospheric perspective with layered mountains receding into white space. The painting is primarily monochrome with subtle warm accents, following the traditional sumi-e style.
```

### 油画模板

```
[风格] oil painting of [主题], [笔触], [色彩], [氛围]

SD:
(classical oil painting:1.3) of [portrait], (visible brushstrokes:1.2), (rich color palette:1.1), (chiaroscuro lighting:1.2), (canvas texture:1.0), oil painting, museum quality, 8k

MJ:
Classical oil painting of portrait subject, visible expressive brushstrokes, rich warm color palette, dramatic chiaroscuro lighting, canvas texture visible, museum quality oil painting --ar 3:4 --v 6
```

---

## 3D渲染/像素艺术模板

### 3D产品渲染模板

```
[引擎] 3D render of [产品], [材质], [光照], [背景], [风格]

SD:
(octane render:1.3) of [sneakers], (premium leather and mesh:1.2), (studio lighting with rim light:1.1), (gradient background:1.0), (photorealistic materials:1.2), 3D product visualization, 8k

MJ:
Octane render of premium sneakers, leather and mesh materials with realistic textures, studio lighting with dramatic rim light, clean gradient background, photorealistic 3D product visualization --ar 1:1 --v 6

DALL-E 3:
A photorealistic 3D render of premium sneakers. The shoes feature a combination of leather and mesh materials with highly realistic textures. Studio lighting with dramatic rim light creates depth and dimension. The background is a clean gradient, typical of professional 3D product visualization.
```

### 像素艺术模板

```
[风格] pixel art of [主题], [色彩], [细节], [氛围]

SD:
(retro pixel art:1.3) of [character], (limited color palette:1.2), (16-bit style:1.1), (clean pixel edges:1.0), (nostalgic game aesthetic:1.1), pixel art, 8k

MJ:
Retro pixel art of character, limited color palette with vibrant accents, 16-bit style reminiscent of classic games, clean pixel edges, nostalgic game aesthetic --ar 1:1 --v 6

DALL-E 3:
Retro pixel art illustration of a character. The artwork uses a limited color palette with vibrant accents, styled in 16-bit aesthetic reminiscent of classic video games. Clean pixel edges define the character against a simple background, creating a nostalgic game aesthetic.
```

---

## 多模型适配策略

### 模型特性对比

| 特性 | SD1.5 | SDXL | Flux1 | Flux2-Klein | Midjourney | Pony Diffusion | GPT-Image2 |
|------|-------|------|-------|-------------|------------|----------------|------------|
| 语法 | 权重语法 | 权重语法 | 自然+风格 | 自然+风格 | 自然+参数 | Tag格式 | 纯自然语言 |
| 长度 | 50-150 tokens | 50-150 tokens | 50-150 words | 50-150 words | 30-80 words | 标签列表 | 100-400 words |
| 质量词 | 必需 | 必需 | 可选 | 可选 | 可选 | 评分标签 | 内置 |
| 反向提示 | 必需 | 必需 | 可选 | 可选 | --no参数 | 必需 | 不支持 |
| 风格控制 | 权重调节 | 权重调节 | 风格词 | 风格词 | --s参数 | 标签控制 | 描述控制 |

### 自动适配规则

**从SD格式转换:**

1. **转Midjourney:**
   - 移除权重语法
   - 保留核心描述词
   - 添加--ar、--v、--s参数
   - 反向提示转为--no

2. **转DALL-E 3:**
   - 展开为完整句子
   - 添加连接词和修饰语
   - 保持自然语言流畅性
   - 详细描述视觉效果

3. **转Flux:**
   - 简化权重语法
   - 保留关键风格词
   - 自然语言为主
   - 适当添加质量词

4. **转NijiJourney:**
   - 添加--niji参数
   - 选择合适--style
   - 优化动漫相关描述
   - 简化复杂修饰

### 提示词优化检查清单

生成提示词后检查：

- [ ] 核心主体描述是否清晰具体
- [ ] 关键特征是否包含足够细节
- [ ] 权重分配是否合理（核心>次要）
- [ ] 质量词是否适合目标模型
- [ ] 反向提示词是否针对性强
- [ ] 整体长度是否符合模型建议
- [ ] 描述是否存在矛盾
- [ ] 是否避免了主观形容词
