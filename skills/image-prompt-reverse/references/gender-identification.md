# 性别识别精准指南

本文档提供系统化的性别识别方法，帮助准确判断图片中人物的性别特征，避免误认。

---

## 目录

1. [性别识别原则](#性别识别原则)
2. [男性特征识别](#男性特征识别)
3. [女性特征识别](#女性特征识别)
4. [模糊/中性特征处理](#模糊中性特征处理)
5. [常见误认场景](#常见误认场景)
6. [性别描述词库](#性别描述词库)

---

## 性别识别原则

### 核心原则

1. **多重验证原则** - 不要依赖单一特征，需要至少2-3个特征交叉验证
2. **服装优先原则** - 服装款式通常是最明显的性别指示器
3. **发型辅助原则** - 发型是重要但非决定性特征
4. **面部特征谨慎原则** - 动漫/插画风格中面部特征可能中性化，需谨慎判断
5. **不确定性标注** - 当无法确定时，使用中性描述或明确标注不确定性

### 识别优先级

```
高优先级 (决定性特征):
- 服装款式 (裙子/西装等)
- 身体轮廓 (明显性别特征)
- 配饰 (领带/项链等)

中优先级 (辅助特征):
- 发型长度和款式
- 面部轮廓
- 姿态和肢体语言

低优先级 (参考特征):
- 色彩偏好
- 装饰元素
- 背景暗示
```

---

## 男性特征识别

### 服装特征

| 类别 | 关键词 | 确定性 |
|------|--------|--------|
| 上装 | suit, tuxedo, dress shirt, tie, bow tie, vest, blazer | 高 |
| 下装 | trousers, slacks, suit pants, shorts (男款) | 高 |
| 外套 | overcoat, trench coat, leather jacket (男款) | 中 |
| 鞋履 | dress shoes, oxfords, loafers, boots (男款) | 中 |
| 配饰 | necktie, bow tie, cufflinks, wristwatch, belt | 高 |

### 发型特征

| 类型 | 描述 | 确定性 |
|------|------|--------|
| 短发 | very short hair, buzz cut, crew cut | 高 |
| 中短发 | short hair, neat haircut | 中 |
| 造型 | slicked back, side part, undercut | 中 |
| 面部毛发 | beard, mustache, stubble, goatee | 极高 |

### 身体轮廓

- 宽肩窄臀的轮廓
- 明显的肌肉线条 (若可见)
- 较直的腰线

### 面部特征 (写实风格)

- 较方的下颌线
- 较粗的眉骨
- 较明显的喉结 (若可见)
- 较短的睫毛

---

## 女性特征识别

### 服装特征

| 类别 | 关键词 | 确定性 |
|------|--------|--------|
| 裙装 | dress, skirt, gown, evening dress | 极高 |
| 上装 | blouse, camisole, crop top, off-shoulder | 高 |
| 内衣 | bra, lingerie, corset, bodysuit | 极高 |
| 外套 | cardigan, shawl, bolero, feminine coat | 中 |
| 鞋履 | high heels, pumps, stilettos, ballet flats | 高 |
| 配饰 | necklace, earrings, bracelet, hair accessories | 中 |

### 发型特征

| 类型 | 描述 | 确定性 |
|------|------|--------|
| 长发 | long hair, very long hair, flowing hair | 高 |
| 造型 | ponytail, braids, updo, bun, pigtails | 高 |
| 装饰 | hair ribbon, hair clip, hairband, flower in hair | 高 |

### 身体轮廓

- 窄肩宽臀的轮廓 (若可见)
- 明显的胸部曲线 (若可见)
- 较明显的腰线

### 面部特征 (写实风格)

- 较柔和的下颌线
- 较长的睫毛
- 较饱满的嘴唇
- 较细的眉毛

---

## 模糊/中性特征处理

### 中性服装

以下服装**不能**作为性别判断依据：

- T-shirt, hoodie, sweater (基础款)
- jeans, pants (基础款)
- jacket, coat (无明显性别特征)
- sneakers, boots (基础款)

### 中性发型

以下发型**不能**单独作为性别判断依据：

- medium length hair
- straight hair (无明显造型)
- simple ponytail (无装饰)
- bangs/fringe (单独存在时)

### 动漫/插画风格的特殊性

在动漫风格中，以下特征可能高度中性化：

- 大眼睛 (男女通用)
- 光滑皮肤 (男女通用)
- 纤细身材 (男女通用)
- 柔和面部轮廓 (男女通用)

**处理策略**:
1. 优先依赖服装判断
2. 结合发型长度和款式
3. 查看是否有明确的性别化配饰
4. 若仍不确定，使用中性描述

### 不确定性处理

当无法确定性别时，使用以下策略：

1. **使用中性描述词**:
   - "a person" 代替 "a man/woman"
   - "androgynous appearance"
   - "gender-neutral style"

2. **描述可见特征而非推断性别**:
   - 描述服装而非说"穿着女装"
   - 描述发型而非说"女性发型"

3. **明确标注不确定性**:
   - "性别特征不明显，可能为..."
   - "中性风格，难以确定性别"

---

## 常见误认场景

### 场景1: 动漫少年/少女

**问题**: 动漫风格中少年和少女特征相似

**解决方案**:
- 查看服装是否有明显性别特征
- 注意发型长度 (极短=男性概率高，极长=女性概率高)
- 查看是否有面部毛发
- 查看配饰 (领带vs蝴蝶结等)

### 场景2: 中性风格角色

**问题**: 刻意的中性化设计

**解决方案**:
- 优先描述可见特征
- 避免强行指定性别
- 使用"androgynous"等中性描述词

### 场景3: 背影/侧影

**问题**: 无法看到面部和正面服装

**解决方案**:
- 依赖发型轮廓
- 依赖身体轮廓
- 依赖可见的服装部分
- 不确定时标注"从背影看..."

### 场景4: 非二元性别/跨性别表达

**问题**: 性别表达与生理性别可能不一致

**解决方案**:
- 描述可见的性别表达特征
- 避免假设生理性别
- 尊重图片呈现的性别表达

---

## 性别描述词库

### 男性描述词

```
man, male, boy, guy, gentleman, young man, middle-aged man, elderly man
```

### 女性描述词

```
woman, female, girl, lady, young woman, middle-aged woman, elderly woman
```

### 中性/通用描述词

```
person, individual, figure, character, youth, teen, child
```

### 年龄+性别组合

| 年龄段 | 男性 | 女性 | 中性 |
|--------|------|------|------|
| 儿童 | young boy | young girl | child, kid |
| 青少年 | teenage boy | teenage girl | teenager, youth |
| 青年 | young man | young woman | young person |
| 中年 | middle-aged man | middle-aged woman | middle-aged person |
| 老年 | elderly man, old man | elderly woman, old woman | elderly person |

### 性别表达描述词

```
masculine, feminine, androgynous, gender-neutral, tomboy, effeminate
```

---

## 快速判断流程

```
Step 1: 检查服装
  ├─ 有明显性别特征服装 → 按服装判断
  └─ 中性服装 → 进入Step 2

Step 2: 检查发型
  ├─ 极短/有明显男性造型 → 男性概率高
  ├─ 极长/有明显女性造型 → 女性概率高
  └─ 中长发/中性发型 → 进入Step 3

Step 3: 检查面部特征 (写实风格)
  ├─ 有胡须/明显男性特征 → 男性
  ├─ 有明显女性特征 → 女性
  └─ 中性/动漫风格 → 进入Step 4

Step 4: 检查配饰和姿态
  ├─ 有明显性别化配饰 → 按配饰判断
  └─ 仍不确定 → 使用中性描述
```
