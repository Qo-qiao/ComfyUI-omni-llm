# Working example prompts

Curated from official Qwen sources and high-signal community posts. Each is labeled with its origin. Adapt freely; keep the quoting and preserve-clause patterns intact.

## Generation — official Qwen-Image-2.1 (2026-09-20)

```text
A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement
A capybara reading a book by candlelight
A ceramic teapot on a wooden table
Clean flat vector infographic titled "FROM CHERRY TO CUP" showing five numbered steps left to right
```

## Text-heavy generation — official Qwen-Image blog (2025-08-04)

Bookstore window with four readable text layers:

```text
Bookstore window display. A sign displays "New Arrivals This Week". Below, a shelf
tag with the text "Best-Selling Novels Here". To the side, a colorful poster
advertises "Author Meet And Greet on Saturday" with a central portrait of the
author. There are four books on the bookshelf, namely "The light between worlds"
"When stars are scattered" "The slient patient" "The night circus"
```

Movie poster with row-by-row text:

```text
A movie poster. The first row is the movie title, which reads "Imagination
Unleashed". The second row is the movie subtitle, which reads "Enter a world
beyond your imagination". The third row reads "Cast: Qwen-Image". The fourth row
reads "Director: The Collective Imagination of Humanity". … At the bottom edge,
the text "Launching in the Cloud, August 2025" appears in bold, modern
sans-serif font …
```

Mixed media on a storefront (chalk + neon + poster + digits):

```text
A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per
cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster
showing a beautiful Chinese woman, and beneath the poster is written
"π≈3.1415926-53589793-23846264-33832795-02384197".
```

## Editing — official 2.1

```text
Change the background to a sunset beach
Let this mascot dance under the moon
```

## Multi-reference — official 2.1

```text
These three characters are sitting around a campfire in a forest
```

With tag references (PE-I2I convention, N≥2):

```text
Place <image1>'s character in the forest camp of <image2>. Keep hairstyle,
clothing, and facial features identical.
```

## Transparent — official 2.1

```text
This is an RGBA image with transparency. A cute cartoon dragon sticker.
The image has alpha channel and the background is transparent.
```

## Community editing patterns

```text
Replace the sign text with 'GRAND OPENING'. Keep original font, size, color, and
perspective. Do not alter background or signboard.

Re-render this scene in a Studio Ghibli art style. Preserve character identity,
clothing, and layout.

Within the red box, replace the lower component of the character '稽' with '旨'.
Match stroke thickness and calligraphy style. Leave everything else unchanged.

Relight the scene with a warm key light from the right and cool rim light from
the back. Keep pose and background unchanged.

Render with a 35 mm lens, shallow depth of field, focus on subject's face.
Preserve environment blur.

Place the same character in a desert environment. Keep hairstyle, clothing, and
facial features identical.
```

## Camera / quality phrasing (apiyi, fal.ai — Qwen-Image-2512 era)

```text
shot on Canon EOS R5, 85mm f/1.4 lens, professional photography, RAW format
, Ultra HD, 4K, cinematic composition.        ← first-generation official suffix;
                                                omit for 2.1 (quality boosters discouraged)
```

Negative prompt that shipped with an official 2512 portrait example:

```text
低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，
画面具有AI感。构图混乱。文字模糊，扭曲。
```

## Japanese-text workaround (Zenn, Edit-2509, 2025-10-01)

Convert rendered text to calligraphy, tracing strokes exactly (instruction itself in Japanese is fine for the edit path):

```text
画像に書かれたテキストを習字風のフォントに変換してください。1画ごとの配置を忠実に
なぞり、抜け漏れがないようにしてください。左下に、赤い四角形の「通义千问」という
印をつけてください
```

Composite a text image into a room picture using numbered references:

```text
1枚目の部屋に飾られている絵について、額縁は残して、その内部を2枚目の画像で表す文字に
置き換えてください。フォントは入力されたゴシック体ではなく、習字のような行書体に変更
してください。最後に、赤い四角のハンコを絵の左下端に加えてください
```

## Source index

| Source | URL |
|---|---|
| Qwen-Image-2.1 GitHub | https://github.com/QwenLM/Qwen-Image-2.1 |
| Qwen-Image-2.1 HF model card | https://huggingface.co/Qwen/Qwen-Image-2.1 |
| PE-T2I / PE-I2I model cards | https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I ・ …-PE-I2I |
| Official blog (zh) | https://qwen.ai/blog?id=qwen-image-2.1 |
| ComfyUI official post | https://blog.comfy.org/p/qwen-image-21-in-comfyui-open-weight |
| Qwen-Image blog (text rendering) | https://qwenlm.github.io/blog/qwen-image/ |
| Reddit edit playbook | https://www.reddit.com/r/StableDiffusion/comments/1n1n81o/ |
| Zenn: Japanese text via Edit | https://zenn.dev/kota_iizuka/articles/33219ebb8aff99 |
| Zenn: prompt guide (ja) | https://zenn.dev/rick_lyric/articles/ffd10bbb59e8b6 |
