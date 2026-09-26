# Qwen-Image-2.1 Vocabulary & Parameter Cheat Sheet

Quick lookups for composing precise prompts matching Qwen-Image-2.1's semantic latent space and native 2K pipeline.

## 1. Medium & Style Vocabulary

Used primarily in the opening sentence (t2i):

| Category | Recommended Terms | Notes |
|---|---|---|
| **Mediums** | photograph, poster, illustration, scene, portrait, infographic, close-up, graphic, page, card, sheet, logo, editorial | Medium is NEVER omitted. |
| **Photographic Styles** | photorealistic, editorial fashion, cinematic, documentary, macro close-up, tilt-shift, vintage film (Kodak Portra, Fujifilm) | Describe optical traits instead of empty boosters. |
| **Illustration & Art** | flat-vector, watercolour, hand-drawn sketch, ink wash, woodcut, ukiyo-e, gouache, retro comic, storybook | Pair with paper/stroke texture. |
| **Digital & 3D** | isometric 3D render, claymation, octane render, low-poly, cyberpunk neon, stylized character model | Specify shader/material qualities. |

---

## 2. Materials & Physical Textures

Always give the material/texture, not just the bare noun:

- **Metals**: brushed stainless steel, anodized aluminum, tarnished brass, polished copper, cast iron, matte chrome.
- **Glass & Liquids**: frosted glass, fluted glass, leaded crystal, condensation droplets, murky puddle, glossy glaze.
- **Fabrics & Wearables**: coarse linen, ribbed cotton knit, distressed denim, supple lambskin leather, sheer silk chiffon, houndstooth wool.
- **Surfaces & Architecture**: weathered teak wood, exposed aggregate concrete, polished travertine marble, cracked asphalt, terracotta tiles.
- **Paper & Graphics**: matte recycled paper fibre, high-gloss coated cardstock, embossed vellum, aged parchment with deckled edges.

---

## 3. Spatial & Positional Phrases

Used in the spatial inventory and frame walk (t2i long form). Aim for 8–14 positional anchors covering corners, edges, and centre:

```
[upper-left corner]      [across the top band]       [upper-right corner]
[along the left edge]    [in the absolute centre]    [along the right edge]
[lower-left corner]      [across the lower third]    [lower-right corner]
```

- **Relative positioning**: `directly in front of`, `tucked behind`, `nestled between`, `receding into the background`, `angled slightly toward the camera`, `jutting outward from the bottom edge`.
- **Compositional distribution**: anchor objects in the four quadrants and borders — don't cluster solely in the centre.

---

## 4. Lighting Descriptors

- **Natural Light**: soft diffused daylight filtering through an overcast sky, harsh midday summer sunlight casting sharp short shadows, warm golden-hour glow coming low from the left, cool blue twilight ambient light.
- **Interior & Studio**: three-point studio lighting with a large softbox, directional rim lighting accentuating edges, warm overhead incandescent pendant lamp, single dramatic spotlight cutting through darkness.
- **Atmospheric**: volumetric light rays (god rays) streaming through misty air, neon ambient spill from adjacent street signage, flickering firelight illuminating one side of the subject.

---

## 5. Aspect Ratio (`wh_ratio`) Quick Matrix — Native 2K

| Ratio | Native pixels | | Ratio | Native pixels |
|---|---|---|---|---|
| 1:1 | 2048×2048 | | 3:2 | 2528×1696 |
| 4:3 | 2400×1792 | | 2:3 | 1696×2528 |
| 16:9 | 2752×1536 | | 9:16 | 1536×2752 |

- The official rewriter may also emit 2:1, 21:9, 9:21, 4:5, 3:1, 5:4, 1:3, 18:39, 9:20, 7:3, 9:5, 5:7 — map to the nearest supported resolution in your pipeline.
- vLLM-style APIs accept `"size": "1024x1024"` strings directly.
- **"2K / 4K / 8K" are quality descriptors, not ratio hints** — never infer a ratio from them.

---

## 6. Transparent (RGBA) Template

Official template — use verbatim, substituting only the description:

```text
This is an RGBA image with transparency. <your description>.
The image has alpha channel and the background is transparent.
```

Also powers subject cutout and transparent-layer editing.
