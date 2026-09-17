# 场景识别专项指南

本文档提供针对不同类型图片的专项识别方法，帮助更精准地分析和描述各类场景。

---

## 目录

1. [人物肖像场景](#人物肖像场景)
2. [动漫/二次元场景](#动漫二次元场景)
3. [风景/自然场景](#风景自然场景)
4. [城市/建筑场景](#城市建筑场景)
5. [静物/产品场景](#静物产品场景)
6. [幻想/科幻场景](#幻想科幻场景)
7. [海报/平面设计场景](#海报平面设计场景)
8. [产品/商业摄影场景](#产品商业摄影场景)
9. [食物摄影场景](#食物摄影场景)
10. [时尚摄影场景](#时尚摄影场景)
11. [UI/UX设计场景](#uiux设计场景)
12. [传统绘画场景](#传统绘画场景)
13. [3D渲染/像素艺术场景](#3d渲染像素艺术场景)
14. [场景混合识别](#场景混合识别)

---

## 人物肖像场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 构图 | 人物占据画面主要部分，背景虚化或简单 |
| 焦距 | 中长焦（85mm-135mm等效），浅景深 |
| 光照 | 柔和的人像光，眼神光明显 |
| 焦点 | 眼睛或面部清晰，背景模糊 |

### 分析重点

**面部细节**:
- 五官比例和对称性
- 皮肤质感（光滑/有纹理/雀斑等）
- 眼睛细节（虹膜颜色、眼神光、睫毛）
- 嘴唇形状和颜色
- 面部轮廓（圆脸/方脸/瓜子脸等）

**发型分析**:
- 长度、层次、造型
- 发色（自然色/染色/渐变）
- 发质（直发/卷发/蓬松/服帖）
- 发型风格（时尚/复古/随意）

**表情情绪**:
- 基础情绪（喜/怒/哀/乐/平静）
- 微表情（嘴角上扬角度、眼角皱纹）
- 眼神情绪（温柔/锐利/迷离）
- 整体气质（优雅/活泼/冷峻）

**服装造型**:
- 服装风格（正式/休闲/时尚/复古）
- 颜色搭配
- 材质质感
- 配饰细节

### 人像类型细分

| 类型 | 特征 | 提示词重点 |
|------|------|------------|
| 证件照/头像 | 正面、均匀光照、中性背景 | 清晰面部特征、对称 |
| 时尚人像 | 夸张造型、艺术光影、强烈风格 | 时尚感、艺术感 |
| 生活照 | 自然表情、环境光、日常服装 | 自然、真实、生活化 |
| 艺术人像 | 创意构图、特殊光影、情绪表达 | 艺术性、情绪、氛围 |
| 商业人像 | 专业布光、精致妆容、商务着装 | 专业、精致、商务 |

---

## 动漫/二次元场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 线条 | 清晰轮廓线，线条粗细变化 |
| 眼睛 | 大眼睛，高光明显，简化细节 |
| 面部 | 简化面部结构，小鼻子小嘴 |
| 色彩 | 鲜艳或柔和，色块分明 |
| 比例 | 头身比（2-8头身不等） |

### 风格细分

**日式动漫风格**:
- 大眼睛，复杂发型
- 表情夸张
- 赛璐璐上色或厚涂
- 参考：新海诚、宫崎骏、京都动画

**美式卡通风格**:
- 夸张比例，粗线条
- 强烈色彩对比
- 几何化造型
- 参考：迪士尼、皮克斯、漫威漫画

**韩式插画风格**:
- 精致五官，柔和光影
- 唯美色彩
- 细腻质感
- 参考：韩式游戏原画、Webtoon

**像素艺术风格**:
- 方块像素
- 有限色彩
- 复古感
- 参考：8-bit/16-bit游戏

### 动漫人物分析要点

**特别注意性别识别**:
- 动漫人物面部特征高度中性化
- **必须依赖服装判断**：裙子/西装/领带/蝴蝶结等
- **发型是重要线索**：极短vs极长
- **配饰辅助**：发饰、领结、首饰等

**动漫特有元素**:
- 眼睛高光形状和数量
- 头发的高光和阴影表现
- 表情符号化（汗滴、青筋、脸红等）
- 背景风格（简化/详细/特效）

---

## 风景/自然场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 主体 | 自然景观为主，人物占比小或无 |
| 景深 | 通常深景深，前后景都清晰 |
| 光线 | 自然光为主，时间感明显 |
| 色彩 | 自然色调，季节特征 |

### 风景类型细分

**山水风景**:
- 山脉形态（险峻/平缓/层叠）
- 水体特征（湖泊/河流/瀑布/海洋）
- 植被类型（森林/草原/荒漠）
- 天气状况（晴朗/云雾/雨雪）

**季节特征**:

| 季节 | 视觉特征 | 关键词 |
|------|----------|--------|
| 春季 | 新绿、花朵、柔和光线 | spring, cherry blossoms, fresh green |
| 夏季 | 深绿、强烈阳光、蓝天 | summer, lush green, vibrant blue sky |
| 秋季 | 红叶、金黄、温暖色调 | autumn, fall colors, golden leaves |
| 冬季 | 雪景、枯枝、冷色调 | winter, snow, bare trees, cool tones |

**时间特征**:

| 时段 | 光线特征 | 色彩倾向 |
|------|----------|----------|
| 黎明 | 柔和蓝光，天空渐变 | 蓝紫色调 |
| 日出 | 暖色光线，长阴影 | 橙红金色调 |
| 上午 | 明亮白光，清晰阴影 | 中性色调 |
| 正午 | 强烈顶光，短阴影 | 高对比 |
| 下午 | 温暖侧光，较长阴影 | 暖色调 |
| 日落 | 金色光线，长阴影 | 橙红紫色调 |
| 黄昏 | 柔和光线，天空彩霞 | 粉紫色调 |
| 夜晚 | 人工光源，星空 | 深蓝黑色调 |

### 风景分析要点

**前景元素**:
- 引导视线的元素
- 框架式构图的元素
- 增加层次感的元素

**中景主体**:
- 视觉焦点
- 主要景物
- 兴趣中心

**背景层次**:
- 远山/天际线
- 天空/云层
- 空气透视效果

---

## 城市/建筑场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 主体 | 建筑物、城市景观、人造结构 |
| 线条 | 几何线条，透视明显 |
| 光线 | 人工光源与自然光混合 |
| 元素 | 街道、车辆、人群、标识 |

### 建筑类型细分

**现代建筑**:
- 玻璃幕墙，钢结构
- 简洁线条，几何造型
- 关键词：modern architecture, glass facade, steel structure

**古典建筑**:
- 柱式、拱门、装饰细节
- 石材、砖墙
- 关键词：classical architecture, columns, ornate details

**传统建筑** (按地域):
- 中式：飞檐、斗拱、红墙
- 日式：木质结构、和式屋顶
- 欧式：尖顶、钟楼、石墙
- 中东：圆顶、拱门、几何图案

**城市街景**:
- 街道透视
- 行人车辆
- 商铺招牌
- 城市氛围

### 城市夜景特殊分析

**光源分析**:
- 路灯（色温、分布）
- 建筑灯光（内透、轮廓灯）
- 霓虹招牌（颜色、风格）
- 车灯轨迹（长曝光效果）

**氛围营造**:
- 繁华都市 vs 寂静街道
- 赛博朋克风格（霓虹、雨夜）
- 温馨夜景（暖光、小店）

---

## 静物/产品场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 主体 | 单一或组合物体 |
| 背景 | 简洁，突出主体 |
| 布光 | 专业布光，质感突出 |
| 构图 | 精心安排，平衡感 |

### 静物类型

**食物摄影**:
- 摆盘艺术
- 食材新鲜度
- 色彩搭配
- 环境氛围（餐厅/家庭/户外）

**产品摄影**:
- 材质质感（金属/玻璃/布料/木材）
- 反光控制
- 品牌呈现
- 使用场景暗示

**艺术静物**:
- 花卉（种类、状态、花瓶）
- 水果（新鲜度、色彩、摆放）
- 器物（古董、工艺品、日常用品）
- 绘画质感（油画、素描）

### 质感描述词库

| 材质 | 描述词 |
|------|--------|
| 金属 | metallic, shiny, reflective, brushed, polished |
| 玻璃 | transparent, translucent, glossy, crystal clear |
| 木材 | wooden, grain texture, rustic, polished wood |
| 布料 | fabric texture, soft, woven, silky, linen |
| 陶瓷 | ceramic, glazed, porcelain, matte finish |
| 皮革 | leather, textured, grainy, smooth leather |

---

## 幻想/科幻场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 元素 | 超现实、未来感、魔法元素 |
| 风格 | 概念艺术、CG渲染 |
| 氛围 | 神秘、宏大、异世界感 |
| 技术 | 复杂光影、特效元素 |

### 科幻场景细分

**赛博朋克**:
- 霓虹灯光，雨夜城市
- 高科技低生活
- 义体、机械元素
- 关键词：cyberpunk, neon lights, futuristic city, rain

**太空科幻**:
- 宇宙、星球、飞船
- 科技感建筑
- 宇航员、外星生物
- 关键词：space, sci-fi, futuristic, spacecraft

**后启示录**:
- 废墟、荒废
- 自然 reclaim
- 幸存者元素
- 关键词：post-apocalyptic, ruins, abandoned, overgrown

### 奇幻场景细分

**西方奇幻**:
- 中世纪城堡、龙、魔法
- 精灵、矮人、兽人
- 史诗感
- 关键词：fantasy, medieval, dragon, magic, epic

**东方奇幻**:
- 仙侠、武侠元素
- 东方建筑、服饰
- 水墨风格
- 关键词：wuxia, xianxia, oriental fantasy, ink wash

**暗黑奇幻**:
- 哥特风格
- 黑暗氛围
- 神秘生物
- 关键词：dark fantasy, gothic, mysterious, ominous

---

## 海报/平面设计场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 构图 | 中心对称或黄金分割，强调视觉冲击 |
| 元素 | 文字+图像组合，图形化设计 |
| 色彩 | 对比强烈，色彩统一和谐 |
| 排版 | 文字层次分明，字体搭配协调 |

### 设计类型细分

**电影海报**:
- 戏剧性构图，人物/场景为主视觉
- 标题文字突出，排版讲究
- 色调统一，氛围感强
- 关键词：movie poster, cinematic, dramatic, title typography

**活动海报**:
- 信息层次清晰（时间、地点、活动名称）
- 图形元素与文字平衡
- 色彩鲜明，吸引眼球
- 关键词：event poster, graphic design, bold typography

**促销海报**:
- 商品/优惠信息突出
- 色彩鲜艳，对比强烈
- 标题醒目，CTA明确
- 关键词：sale poster, promotional, vibrant colors

**电影/游戏宣传海报**:
- 史诗感构图
- 人物/场景融合
- 光影戏剧性
- 关键词：movie poster, game poster, epic composition

### 海报分析要点

**文字分析**:
- 字体风格（衬线/无衬线/手写）
- 文字层级（标题/副标题/正文）
- 文字与图像的关系

**构图分析**:
- 视觉引导路径
- 焦点位置
- 留白运用

---

## 产品/商业摄影场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 主体 | 产品为主体，清晰展示 |
| 背景 | 简洁干净，突出产品 |
| 布光 | 专业布光，质感突出 |
| 构图 | 精心安排，平衡感 |

### 产品类型细分

**电子产品**:
- 金属质感，反光控制
- 科技感，未来感
- 关键词：electronics, tech product, sleek design

**化妆品/护肤品**:
- 玻璃/塑料质感
- 柔和光线，高级感
- 关键词：cosmetics, skincare, luxury beauty

**家居用品**:
- 生活场景融入
- 温馨氛围
- 关键词：home goods, lifestyle, cozy

**奢侈品**:
- 高级质感，精致细节
- 暗调/高级灰
- 关键词：luxury, premium, exquisite

### 质感描述词库

| 材质 | 描述词 |
|------|--------|
| 金属 | metallic, shiny, reflective, brushed, polished |
| 玻璃 | transparent, translucent, glossy, crystal clear |
| 木材 | wooden, grain texture, rustic, polished wood |
| 布料 | fabric texture, soft, woven, silky, linen |
| 陶瓷 | ceramic, glazed, porcelain, matte finish |
| 皮革 | leather, textured, grainy, smooth leather |

---

## 食物摄影场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 主体 | 食物为主体，摆盘讲究 |
| 色彩 | 色彩鲜艳，令人食欲 |
| 光线 | 柔和光线，质感突出 |
| 氛围 | 温馨，生活感 |

### 食物类型细分

**正餐/主菜**:
- 摆盘艺术，色彩搭配
- 食材新鲜度
- 热气/蒸汽效果
- 关键词：gourmet, plated food, restaurant quality

**甜点/烘焙**:
- 精致造型
- 奶油/巧克力质感
- 装饰细节
- 关键词：dessert, pastry, baking, sweet

**饮品**:
- 杯具质感
- 液体透明度/色彩
- 冰块/气泡效果
- 关键词：beverage, cocktail, coffee, refreshing

**街头小吃**:
- 烟火气
- 手持/即食感
- 本地特色
- 关键词：street food, local cuisine, casual

### 食物摄影分析要点

- 摆盘方式和构图
- 食材色彩搭配
- 质感表现（酥脆/柔软/多汁）
- 环境氛围（餐厅/户外/家庭）

---

## 时尚摄影场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 主体 | 服装/珠宝/配饰为核心 |
| 模特 | 专业姿态，表情到位 |
| 布光 | 专业棚拍或自然光 |
| 风格 | 杂志感，高级感 |

### 时尚类型细分

**高级时装**:
- 设计感强，造型夸张
- 艺术化构图
- 关键词：high fashion, couture, editorial

**街拍时尚**:
- 自然姿态
- 城市背景
- 关键词：street style, urban fashion, casual chic

**珠宝/腕表**:
- 精致细节
- 闪耀效果
- 关键词：jewelry, watch, luxury accessories

**美妆**:
- 面部特写
- 妆容细节
- 关键词：beauty, makeup, cosmetics closeup

### 时尚摄影分析要点

- 服装设计和材质
- 模特姿态和表情
- 妆容和造型
- 整体风格定位

---

## UI/UX设计场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 布局 | 网格系统，对齐规范 |
| 元素 | 按钮、卡片、图标、文字 |
| 色彩 | 品牌色，色彩系统 |
| 交互 | 可点击元素，导航 |

### 设计类型细分

**APP界面**:
- 移动端布局
- 触控友好
- 关键词：mobile app, UI design, app interface

**网页设计**:
- 响应式布局
- 桌面端优化
- 关键词：web design, landing page, dashboard

**仪表盘**:
- 数据可视化
- 图表组件
- 关键词：dashboard, data visualization, analytics

**电商界面**:
- 商品展示
- 购物车元素
- 关键词：e-commerce, product page, shopping

### UI设计分析要点

- 布局结构和网格
- 色彩系统和品牌一致性
- 字体层级和可读性
- 组件设计和交互模式

---

## 传统绘画场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 笔触 | 可见笔触，绘画质感 |
| 色彩 | 手绘色彩，非数码感 |
| 材质 | 画布/纸张纹理 |
| 技法 | 传统绘画技法 |

### 绘画类型细分

**水墨/国画**:
- 水墨晕染
- 留白意境
- 写意风格
- 关键词：ink wash, chinese painting, sumi-e, brush painting

**油画**:
- 厚重质感
- 笔触明显
- 色彩丰富
- 关键词：oil painting, thick brushstrokes, canvas texture

**水彩**:
- 透明感
- 水痕效果
- 色彩流动
- 关键词：watercolor, transparent, wet on wet

**素描/速写**:
- 线条为主
- 铅笔/炭笔质感
- 明暗关系
- 关键词：sketch, pencil drawing, charcoal, line art

**版画**:
- 线条清晰
- 色块分明
- 复古感
- 关键词：printmaking, woodcut, linocut, engraving

### 传统绘画分析要点

- 绘画媒介（纸/画布/绢）
- 笔触特征（粗细/干湿/浓淡）
- 色彩运用（调色/晕染/叠加）
- 构图留白

---

## 3D渲染/像素艺术场景

### 识别特征

| 特征类型 | 关键指标 |
|----------|----------|
| 质感 | 光滑表面，精确光影 |
| 细节 | 高精度建模 |
| 渲染 | CG渲染痕迹 |
| 风格 | 数字化，精确 |

### 3D渲染类型细分

**产品渲染**:
- 材质精确
- 光影真实
- 关键词：3D render, product visualization, octane render

**角色渲染**:
- 写实或风格化角色
- 毛发/皮肤细节
- 关键词：3D character, digital human, character render

**场景渲染**:
- 建筑/环境可视化
- 光影氛围
- 关键词：architectural visualization, environment render, interior render

**动画风格3D**:
- 皮克斯/迪士尼风格
- 卡通渲染
- 关键词：3D animation, pixar style, toon shading

### 像素艺术类型

**复古游戏风格**:
- 8-bit/16-bit像素
- 有限色彩
- 关键词：pixel art, retro game, 8-bit, 16-bit

**现代像素艺术**:
- 高分辨率像素
- 丰富色彩
- 关键词：hd pixel art, modern pixel, detailed pixel

### 3D/像素分析要点

- 渲染引擎特征（Octane/Blender/C4D）
- 材质和光影精度
- 像素大小和色彩数量
- 风格定位（写实/卡通/复古）

---

## 场景混合识别

### 混合场景处理

当图片包含多种场景元素时：

**主次判断**:
1. 确定主要场景类型（占画面比例最大的）
2. 识别次要元素（占比小但重要的）
3. 描述两者关系和互动

**常见混合场景**:

| 混合类型 | 分析重点 | 示例 |
|----------|----------|------|
| 人物+风景 | 人物与环境的比例关系 | 旅行照、环境人像 |
| 建筑+自然 | 人造与自然的融合 | 园林、山间寺庙 |
| 静物+场景 | 物体与环境的叙事 | 咖啡馆静物、书桌场景 |
| 动漫+现实 | 二次元与三次元的结合 | 手办摄影、cosplay |

### 场景识别决策树

```
开始
  │
  ├─ 有人物？
  │    ├─ 人物占比 > 50%？ → 人物肖像
  │    ├─ 人物占比 20-50%？ → 环境人像/时尚摄影
  │    └─ 人物占比 < 20%？ → 风景/场景带人物
  │
  ├─ 有建筑？
  │    ├─ 建筑为主？ → 建筑摄影
  │    └─ 建筑为背景？ → 城市街景/风景
  │
  ├─ 有动漫特征？
  │    ├─ 明显的线条和简化特征？ → 动漫/二次元
  │    └─ 写实但有动漫元素？ → 3D渲染/CG
  │
  ├─ 有超现实元素？
  │    ├─ 科幻特征？ → 科幻场景
  │    └─ 奇幻特征？ → 奇幻场景
  │
  ├─ 有文字+图像？
  │    ├─ 电影/活动宣传？ → 海报设计
  │    ├─ 书籍相关内容？ → 书籍封面
  │    └─ 品牌/促销信息？ → 平面设计
  │
  ├─ 有产品？
  │    ├─ 电子产品/化妆品？ → 产品摄影
  │    ├─ 美食/饮品？ → 食物摄影
  │    └─ 服装/珠宝？ → 时尚摄影
  │
  ├─ 有界面元素？
  │    ├─ APP/网页界面？ → UI/UX设计
  │    └─ 数据图表？ → 仪表盘设计
  │
  ├─ 有绘画特征？
  │    ├─ 水墨/国画？ → 传统绘画
  │    ├─ 油画/水彩？ → 传统绘画
  │    └─ 素描/速写？ → 传统绘画
  │
  ├─ 有3D/像素特征？
  │    ├─ 光滑CG渲染？ → 3D渲染
  │    └─ 方块像素？ → 像素艺术
  │
  └─ 以上都不是？
       ├─ 单一物体？ → 静物/产品
       └─ 自然景观？ → 风景/自然
```

---

## 场景专用提示词模板

### 人像摄影模板

```
[镜头类型] portrait of a [年龄] [性别] with [面部特征], [发型描述], wearing [服装], [表情情绪], [光照条件], [背景描述], [摄影风格], [质量词]

示例:
85mm portrait of a young woman with symmetrical features, long flowing auburn hair, wearing elegant black dress, gentle smile, soft natural lighting from window, blurred bokeh background, professional photography, 8k, highly detailed
```

### 动漫角色模板

```
[风格] anime [性别] character with [发型], [眼睛特征], wearing [服装], [表情], [姿势], [背景], [艺术风格], [质量词]

示例:
Studio Ghibli style anime girl character with long pink twin tails, large expressive green eyes, wearing school uniform with red ribbon, cheerful expression, standing pose, cherry blossom background, soft pastel colors, detailed illustration, masterpiece
```

### 风景摄影模板

```
[时间] [季节] landscape of [地点], [天气], [前景元素], [中景主体], [背景层次], [光照], [色彩氛围], [摄影风格], [质量词]

示例:
Golden hour autumn landscape of mountain lake, clear sky with few clouds, colorful fallen leaves in foreground, crystal clear lake reflection, distant snow-capped mountains, warm side lighting, rich orange and gold tones, panoramic view, 8k, photorealistic
```

### 城市夜景模板

```
[时间] cityscape of [城市类型], [建筑特征], [光源描述], [天气/氛围], [视角], [摄影技术], [风格], [质量词]

示例:
Night cityscape of cyberpunk metropolis, towering skyscrapers with glass facades, neon signs and street lights reflecting on wet pavement, light rain creating atmospheric haze, street level perspective, long exposure light trails, blade runner aesthetic, cinematic composition, 8k
```
