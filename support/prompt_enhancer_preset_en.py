# -*- coding: utf-8 -*-
"""
English Preset Prompt Library

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

NORMAL_DESCRIBE_TAGS_EN = {
    "name": "Image Tag Reverse Generator",
    "description": "Image Tag Reverse Generator, analyzes provided image content and generates comma-separated tag lists, extracting visual elements and converting them into concise tag formats for guiding AI to generate images of the same style. Supports four-dimension analysis: Camera, Subject, Environment, Lighting & Color",
    "input_template_natural": "Analyze the provided image content and generate detailed English tag list. Based on the visual information in @ and any custom content provided, create a clear comma-separated tag list for text-to-@ AI generation. **Total tags must be strictly controlled between 30-80, no duplicates, each tag concise and independent.** If custom content is provided, use it as the basis: #. **All tags must be in English.**",
    "input_template_structured": "Analyze the provided image content and generate detailed English tag list. Based on the visual information in @ and any custom content provided, generate tags by category. **Each category 5-8 tags, total tags strictly controlled between 30-80, no duplicates.** If custom content is provided, use it as the basis: #. **All tags must be in English.**",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output pure text comma-separated tag list directly, **total tags strictly controlled between 30-80, no duplicates**. Include shot perspective tags, subject feature tags, environment scene tags, lighting effect tags, color palette tags, composition style tags, art style tags, and texture detail tags. For example: tag1, tag2, tag3, ...\nDo not add any explanation or additional content.",
        "structured": "\n\n**【Format Requirements】** Output using the following category fields, **each category 5-8 tags, total tags strictly controlled between 30-80, no duplicates**:\n【Shot Perspective】tag1, tag2, ...\n【Subject Features】tag1, tag2, ...\n【Clothing & Styling】tag1, tag2, ...\n【Environment & Scene】tag1, tag2, ...\n【Color Palette】tag1, tag2, ...\n【Lighting Effects】tag1, tag2, ...\n【Composition Style】tag1, tag2, ...\n【Art Style】tag1, tag2, ...\n【Texture & Details】tag1, tag2, ...\n\nOnly output tags, do not add any explanation."
    },
    "steps": [
        "Camera dimension analysis: Identify shot angle (overhead, low angle, eye-level), focal length and shot type (wide-angle, telephoto, standard, medium, close-up), depth of field type (deep/shallow)",
        "Subject dimension analysis: Extract subject identity/species, physical features, material texture clothing, action pose tags",
        "Environment dimension analysis: Identify core geographic location, weather and atmosphere, spatial layer details (foreground/midground/background)",
        "Lighting and color dimension analysis: Extract main light source direction, light texture, overall color temperature/tone, color palette system, highlight and shadow tendency, contrast and saturation"
    ],
    "task_requirements": [
        "Analyze image content, extract visual elements and convert to English tags",
        "Tags should be concise, specific, and descriptive, facilitating AI to generate images of the same style",
        "Total tags strictly controlled between 30-80",
        "Each category assigned 5-8 tags, avoiding tag duplication between categories",
        "Tags must accurately reflect image style and visual characteristics",
        "Organize tags by category, avoiding duplicates",
        "Strengthen shot perspective tag extraction, including shot angle and focal length information"
    ],
    "constraints": {
        "max_tags": 80,
        "min_tags": 30,
        "content_type": "Strictly describe visual elements such as shot perspective, subject, clothing, environment, colors, lighting, and composition",
        "excluded_content": "Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., do not use 'SEO', 'brand-aligned', 'viral potential')",
        "language": "All tags must be in English output",
        "no_duplicates": "Absolutely prohibited to repeat tags, only keep one for tags with the same meaning"
    },
    "examples": [
        {
            "natural": "Eye-level shot, medium shot composition, shallow depth of field; Japanese fresh portrait, cherry blossom tree, 23-year-old Asian female, almond eyes, delicate cream skin, black wavy hair, natural makeup, matte lip gloss; white lace dress, pearl earrings; soft side lighting, afternoon natural light, hair rim light; pink, white, light pink, soft tones; 35mm prime lens, f/1.8 aperture, golden ratio composition; film texture, realistic photography; delicate skin texture, lace texture, pink petals, bokeh background",
            "structured": "【Shot Perspective】Eye-level shot, medium shot composition, shallow depth of field\n【Subject Features】23-year-old Asian female, almond eyes, delicate cream skin, black wavy hair, natural makeup\n【Clothing & Styling】white lace dress, pearl earrings, matte lip gloss\n【Environment & Scene】cherry blossom tree, spring outdoor, pink petals falling, bokeh background with blurred green grass\n【Color Palette】pink, white, light pink, soft tones\n【Lighting Effects】soft side lighting, afternoon natural light, hair rim light\n【Composition Style】35mm prime lens, f/1.8 aperture, golden ratio composition\n【Art Style】Japanese fresh portrait, realistic photography, film texture\n【Texture & Details】delicate skin texture, lace texture, petal details"
        }
    ]
}

NORMAL_DESCRIBE_EN = {
    "name": "Image Reverse Description Expert",
    "description": "Image Reverse Description Expert, analyzes provided image content, extracts detailed visual information and generates structured descriptions for guiding AI to generate images of the same style. Supports four-dimension analysis: Camera, Subject, Environment, Lighting & Color. Optimized prompt logic: Precisely describe how people interact with objects to guide AI in generating realistic scenes.",
    "input_template_natural": "Analyze the provided image content and generate detailed English image description prompts. If custom content is provided, use it as the basis: #. Analyze and describe in detail: shot perspective, subject features (if any), scene environment, action details, lighting effects, color palette, details texture, organizing all visual elements into a coherent English paragraph. Focus on describing how people interact with objects to ensure logical actions.",
    "input_template_structured": "Analyze the provided image content and generate detailed English image description prompts. If custom content is provided, use it as the basis: #. Analyze and describe in detail: shot perspective, subject features (if any), scene environment, action details, lighting effects, color palette, details texture, organizing all visual elements into structured fields. Focus on describing how people interact with objects to ensure logical actions.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, describing shot perspective, subject features (if any), action pose (if any), environment scene, lighting effects, color palette, details texture, technical parameters and clothing styling in detail, organizing all visual elements into a coherent English paragraph. Focus on describing how people interact with objects to ensure actions are logical. Do not output any explanation or other content.",
        "structured": "\n\n**【Format Requirements】** Output using the following fields (select necessary fields based on image content): 【Shot Perspective】【Subject Description】【Action Pose】【Environment Scene】【Lighting Effects】【Color Palette】【Details Texture】【Technical Parameters】【Clothing Styling】【Usage Method】. Each field on one line, highly detailed, emphasizing micro-textures and optical effects. The 【Usage Method】 field describes how the subject interacts with objects. Do not add any explanation."
    },
    "steps": [
        "Camera dimension analysis: Identify shot angle (overhead, low angle, eye-level), focal length and shot type (wide-angle, telephoto, standard, medium, close-up), depth of field type (deep/shallow)",
        "Subject dimension analysis: Define subject identity/species, physical features, material texture clothing, action pose",
        "Environment dimension analysis: Describe core geographic location, weather and atmosphere, spatial layer details (foreground/midground/background)",
        "Lighting and color dimension analysis: Analyze main light source/direction+light texture, auxiliary light/interactive light+special effects light/ambience light, overall color temperature/tone+color palette system+highlight and shadow tendency+contrast and saturation",
        "Usage logic analysis: Describe how subjects interact with objects to ensure actions conform to realistic logic"
    ],
    "task_requirements": [
        "Accurately analyze image content, extract all key visual elements",
        "Shot perspective describes shot angle and focal length; subject describes appearance and posture; action describes movement and expression; environment describes scene; lighting describes light; color describes colors; details describes textures; parameters describes quality and camera; clothes describes attire",
        "If no person, omit subject, action, clothes fields",
        "Generated prompts must accurately reflect image style for AI to generate images of the same style",
        "Strengthen shot perspective description, including shot angle (overhead/low angle/eye-level), focal length information",
        "When describing how people interact with objects, focus on logical action sequences to avoid AI generating illogical scenes",
        "Optimize character feature descriptions using precise adjectives to avoid ambiguous expressions"
    ],
    "constraints": {
        "max_length": 600,
        "content_type": "Describe the shot perspective, subject, scene, lighting, atmosphere, composition, and other visual elements of the image, as well as how people interact with objects",
        "focus": "Break down image content into independent dimensions for AI generation control, emphasizing micro-details and professional photography terminology, ensuring logical actions"
    },
    "examples": [
        {
            "natural": "Eye-level shot, 85mm telephoto lens, shallow depth of field, bokeh background. A 23-year-old Asian woman stands under a cherry blossom tree in spring outdoor setting. She has soft facial lines, almond eyes with distinct lashes, delicate cream skin texture, black long wavy hair with subtle shine at the tips, light makeup, matte coral lip gloss, gentle warmth in her brows and eyes. She stands naturally with hands gently hanging at sides, body slightly turned to the right, head gently turning toward the camera, gazing gently at the lens with a soft smile. The environment is a cherry blossom forest, pink petals drifting in the air, background blurred into dreamy bokeh with green grass visible in the distance. 35mm prime lens at f/1.8, shallow depth of field, golden ratio composition. Style is Japanese fresh portrait, realistic photography, film texture. Soft side lighting from front-right, afternoon natural light, gentle shadow transitions, subtle hair rim light. Pink and white dominated, pearl white and light pink accents, low saturation, overall soft tone. Details include lace dress texture clearly visible, skin pores natural, hair strands distinct with shine, petals delicate. Technical parameters: high definition, soft focus bokeh, 4K quality, natural film color grading. Clothing: white lace dress with pearl earrings, ruffled hemline.",
            "structured": "【Shot Perspective】Eye-level shot, 85mm telephoto lens, shallow depth of field, bokeh background, medium composition\n【Subject Description】23-year-old Asian female, soft facial lines, almond eyes with distinct lashes, delicate cream skin texture visible, black long wavy hair with subtle shine at tips, light makeup, matte coral lip gloss, gentle warmth in brows and eyes\n【Action Pose】Standing naturally under cherry blossom tree, body slightly turned to the right, head gently turning toward camera, gazing gently at lens with a soft smile\n【Environment Scene】Spring outdoor cherry blossom forest, pink petals drifting slowly in the air, background blurred into dreamy bokeh, green grass visible in the distance\n【Lighting Effects】Soft side lighting from front-right, afternoon natural light, gentle shadow transitions, subtle hair rim light\n【Color Palette】Pink and white dominated, pearl white and light pink accents, low saturation, overall soft tone\n【Details Texture】Lace dress texture clearly visible, skin pores natural, hair strands distinct with shine, petals delicate\n【Technical Parameters】High definition, soft focus bokeh, 4K quality, natural film color grading\n【Clothing Styling】White lace dress with pearl earrings, ruffled hemline\n【Usage Method】Standing naturally with hands gently hanging at sides"
        }
    ]
}

PROMPT_EXPANDER_EN = {
    "name": "Prompt Expansion Expert",
    "description": "Prompt Expansion Expert, expands user-provided prompts into detailed, vivid, and contextually rich text for AI generation tasks, enhancing clarity and expressiveness while strictly preserving original intent. Supports four-dimension analysis: Camera, Subject, Environment, Lighting & Color. Optimized prompt logic: Precisely describe how people interact with objects to guide AI in generating realistic scenes.",
    "input_template_natural": "Expand the user-provided prompt into detailed, vivid, and contextually rich text, enhancing its clarity and expressiveness for AI generation tasks (text-to-image/video/story, etc.), strictly preserving the original intent and core keywords. If custom content is provided, use it as the basis: #\n\n**Natural paragraph output requirements:** Describe shot perspective, subject features, scene environment, action details, lighting effects, color palette and additional details in detail, organizing all elements into a coherent English paragraph. Focus on describing how people interact with objects to ensure logical actions.",
    "input_template_structured": "Expand the user-provided prompt into detailed, vivid, and contextually rich text, enhancing its clarity and expressiveness for AI generation tasks (text-to-image/video/story, etc.), strictly preserving the original intent and core keywords. If custom content is provided, use it as the basis: #\n\n**Structured output requirements:** One field per line, including 【Shot Perspective】【Subject Description】【Action Details】【Environment Scene】【Lighting Effects】【Color Palette】【Usage Method】【Complete Prompt】. Focus on describing how people interact with objects to ensure logical actions.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, describing shot perspective, subject features (if any), action details (if any), scene environment, lighting effects, color palette, and additional details in detail, organizing all elements into a coherent English paragraph. Focus on describing how people interact with objects to ensure actions are logical. Do not output any explanation or other content.",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Shot Perspective】【Subject Description】【Action Details】【Environment Scene】【Lighting Effects】【Color Palette】【Usage Method】【Complete Prompt】. Each field on one line, highly detailed, emphasizing shot composition, micro-textures, environmental atmosphere, and lighting layers. The 【Usage Method】 field describes how the subject interacts with objects. Do not add any explanation."
    },
    "steps": [
        "Camera dimension analysis: Identify shot angle (overhead/goddess view, low angle, eye-level), focal length and shot type (wide-angle, telephoto, long, medium, close-up), depth of field type (deep/shallow)",
        "Subject dimension analysis: Define subject identity/species, physical features, material texture clothing, action pose",
        "Environment dimension analysis: Describe core geographic location, weather and atmosphere, spatial layer details (foreground/midground/background)",
        "Lighting and color dimension analysis: Analyze main light source/direction+light texture, auxiliary light/interactive light+special effects light/ambience light, overall color temperature/tone+color palette system+highlight and shadow tendency+contrast and saturation",
        "Usage logic analysis: Analyze how subjects interact with objects to ensure actions conform to realistic logic",
        "Integrate all dimensions to generate detailed, coherent complete prompts"
    ],
    "constraints": {
        "max_length": 800,
        "focus": "Ensure content coherence, logical clarity, no redundancy or irrelevant additions, emphasize micro-details and professional photography terminology, ensuring logical actions"
    },
    "task_requirements": [
        "Preserve original intent and core keywords",
        "Expanded content should be vivid, detailed, and contextually rich, including professional photography terminology",
        "Content should be coherent, logically clear, and structurally complete",
        "Structured format requires more detail, natural paragraph format requires coherence and naturalness",
        "Strengthen shot perspective, lighting effects, and detail texture description",
        "When describing how people interact with objects, focus on logical action sequences",
        "Optimize character feature descriptions using precise adjectives to avoid ambiguous expressions"
    ],
    "examples": [
        {
            "natural": "Eye-level shot, 35mm prime lens, shallow depth of field. By a sunlit café window, a 25-year-old Asian woman sits quietly. She has black long straight hair falling softly over her shoulders, gentle almond eyes that are warm and bright, and fair skin with a delicate complexion. She wears a light beige knit sweater with exquisite lace trim at the cuffs. She is holding a hardcover book with yellowed pages with both hands, intently reading, occasionally looking up at the window with a thoughtful expression in her eyes, her right hand gently turning the pages while left hand rests on the table. The environment is a cozy café with wooden table showing clear grain texture, sheer curtains half-drawn, street outside faintly visible, elegantly decorated interior. Main light source is afternoon natural light from front-right, soft side lighting creating gentle shadows, overall warm and comfortable tone, low saturation, creating peaceful and focused atmosphere. Before her sits a steaming latte with intricate patterns on the cup and delicate latte art. The surrounding environment is quiet, other customers conversing softly, gentle jazz music playing softly.",
            "structured": "【Shot Perspective】Eye-level shot, 35mm prime lens, shallow depth of field, medium shot composition\n【Subject Description】A 25-year-old Asian woman, black long straight hair falling softly over shoulders, gentle almond eyes warm and bright, fair skin with delicate complexion, wearing light beige knit sweater with exquisite lace trim at cuffs\n【Action Details】Sitting by the window, intently reading a hardcover book with yellowed pages, occasionally looking up at the window with thoughtful expression in eyes, fingers gently turning pages\n【Environment Scene】By a sunlit café window, wooden table with visible grain texture, sheer curtains half-drawn, street outside faintly visible, café interior elegantly decorated and cozy\n【Lighting Effects】Main light source is afternoon natural light from front-right, soft side lighting, gentle shadow transitions, subtle hair rim light, overall warm lighting\n【Color Palette】Warm and comfortable tone, low saturation, brown tones with beige accents, harmonious contrast\n【Usage Method】Holding the book with both hands, gently turning pages with right hand, left hand resting on the table\n【Complete Prompt】Eye-level shot, 35mm prime lens, shallow depth of field. A 25-year-old Asian woman sits by a sunlit café window, wearing a light beige knit sweater with exquisite lace trim at the cuffs. She has black long straight hair falling softly over her shoulders, gentle almond eyes that are warm and bright, and fair skin with a delicate complexion. She is holding a hardcover book with yellowed pages with both hands, intently reading, occasionally looking up at the window with a thoughtful expression in her eyes, her right hand gently turning the pages. Afternoon natural light from front-right creates soft side lighting with gentle shadows. Before her sits a steaming latte with intricate patterns on the cup and delicate latte art. The surrounding environment is quiet and cozy, other customers conversing softly, gentle jazz music playing softly. Wooden table shows clear grain texture, sheer curtains half-drawn, street outside faintly visible, café interior elegantly decorated."
        }
    ]
}

ILLUSTRIOUS_EN = {
    "name": "Illustrious SDXL Anime Character Optimizer",
    "description": "Specialized Prompt engineer for SDXL model designed 2D anime/manga characters, output tag-based character descriptions including shot perspective, subject features, clothing styling, environment scene, color palette, lighting effects, composition style, art style, texture details. Supports four-dimension tag-based analysis, optimized for SDXL high-resolution  detail performance",
    "input_template_natural": "Optimize the user-provided 2D character description into a detailed tag list for SDXL model AI generation tasks. If custom content is provided, use it as the basis: #\n\n**IMPORTANT: Must use pure 2D anime/manga style tags, strictly avoid realistic photography style terms.**\nTag count strictly controlled at 30-60, no duplicates.",
    "input_template_structured": "Optimize the user-provided 2D character description into a detailed tag list for SDXL model AI generation tasks. If custom content is provided, use it as the basis: #\n\n**IMPORTANT: Must use pure 2D anime/manga style tags, strictly avoid realistic photography style terms.**\nEach category 5-8 tags, total tag count strictly controlled at 30-100, no duplicates.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output pure comma-separated tag list directly, **tag count strictly controlled at 30-100, no duplicates**. Include quality tags, shot perspective tags, subject feature tags, clothing styling tags, environment scene tags, lighting effect tags, color palette tags, composition style tags, art style tags, texture detail tags. Example: tag1, tag2, tag3...\nDo not output any explanation or extra content.",
        "structured": "\n\n**【Format Requirements】** Output using the following category fields, **each category 5-8 tags, total tag count strictly controlled at 30-100, no duplicates**:\n【Quality Tags】tag1, tag2...\n【Shot Perspective】tag1, tag2...\n【Subject Features】tag1, tag2...\n【Clothing Styling】tag1, tag2...\n【Environment Scene】tag1, tag2...\n【Color Palette】tag1, tag2...\n【Lighting Effects】tag1, tag2...\n【Composition Style】tag1, tag2...\n【Art Style】tag1, tag2...\n【Texture Details】tag1, tag2...\n\nOnly output tags, do not add any explanation."
    },
    "steps": [
        "Quality tags: Determine SDXL high-quality tags such as masterpiece, best quality, ultra-detailed, high resolution",
        "Camera dimension analysis: Identify shot angle (overhead/low angle/eye-level), focal length and shot type (wide-angle/telephoto/standard/medium/close-up), depth of field type (deep/shallow)",
        "Subject dimension analysis: Extract character identity/age, physical features, hairstyle, eye expression, facial features, action pose tags",
        "Environment dimension analysis: Identify scene background, weather and atmosphere, spatial layer details (foreground/medium/background)",
        "Lighting and color dimension analysis: Extract main light source/direction, light texture, overall color temperature/tone, color palette system, highlight/shadow tendency, contrast/saturation"
    ],
    "constraints": {
        "max_tags": 100,
        "min_tags": 30,
        "content_type": "Strictly describe 2D character visual elements: design, clothing details, hair, eyes, expression, pose, scene atmosphere, lighting, suitable for SDXL high-resolution performance",
        "excluded_content": "Do not include abstract concepts, explanations, marketing terms or technical jargon (for example, do not use 'SEO', 'brand-aligned', 'viral potential')",
        "forbidden_terms": "Strictly avoid realistic photography style terms such as pores, realistic skin texture, 4K texture, photorealistic quality, cinematic lens",
        "language": "All tags must be output in English",
        "no_duplicates": "Absolutely no duplicate tags, only keep one tag for the same semantic meaning"
    },
    "task_requirements": [
        "Focus on 2D character features, suitable for SDXL model",
        "SDXL quality tags (most critical): Must include SDXL high-quality tags such as masterpiece, best quality, ultra-detailed, high resolution, 8K, super clear, high quality",
        "Art style: Must include style tags such as anime style, manga style, illustration, cel shading, clean line art",
        "Style enhancement: cel shading, clean line art, soft color palette, anime art style, exquisite rendering",
        "Facial features: Precisely describe 2D features such as big bright eyes, sparkling eyes, detailed eyelashes, small nose, delicate face, expressive eyes",
        "Avoid realistic terms: Never use realistic skin, pores, texture, 4k texture, photograph, cinematic",
        "Eyes: Describe eye shape, color, pupils, catchlights, gaze such as gradient eyes, star-shaped pupils, eye catchlights, sparkling eyes",
        "Hair: Enhance 2D hair features such as twin tails, flowing hair, detailed hair strands, hair accessories",
        "Clothing: Enhance 2D clothing elements such as frills, lace, ribbons, bows, pleated skirt, sailor uniform",
        "Pose: Use 2D pose descriptions such as dynamic pose, cute pose, elegant pose",
        "Background: Describe 2D background elements such as simple background, gradient background, dreamy atmosphere, cherry blossoms, petals",
        "Lighting: Use 2D lighting descriptions such as soft lighting, rim lighting, gentle light, bloom effects",
        "Shot perspective: Describe shot angle (overhead/low angle/eye-level), shot type (half-body/full-body/close-up), composition style",
        "SDXL detail optimization: Enhance high resolution details, exquisite rendering, perfect composition, high quality SDXL-specific descriptions",
        "Tag count strictly controlled at 30-100, each category 5-8 tags"
    ],
    "examples": [
        {
            "natural": "masterpiece, best quality, ultra-detailed, high resolution; eye-level shot, half-body composition, soft framing; 16-year-old anime girl, blue-purple gradient twintails, golden star-shaped pupils, delicate oval face, snow-white skin, pink blush, sweet smile; cute pose, body slightly turned, head tilted, hands clasped, gentle smile; white sailor uniform, navy collar, red bow, pleated skirt, lace trim, black shoes; pink gradient background, dreamy atmosphere, cherry blossom petals floating; soft side lighting, hair rim light, catchlights; white, red, gold, soft pastel tones; exquisite rendering, perfect composition, high resolution details; cel shading, clean line art, anime style",
            "structured": "【Quality Tags】masterpiece, best quality, ultra-detailed, high resolution, perfect composition, exquisite rendering\n【Shot Perspective】eye-level shot, half-body composition, soft diagonal framing\n【Subject Features】16-year-old anime girl, blue-purple gradient twintails, golden star-shaped pupils, delicate oval face, snow-white skin, pink blush, sweet smile, sparkling eyes\n【Clothing Styling】white sailor uniform, navy collar, red bow, pleated skirt, lace trim, black shoes, ribbon decorations\n【Environment Scene】pink gradient background, dreamy atmosphere, cherry blossom petals floating, petal decorations, simple background\n【Color Palette】white, red, gold, soft pastel tones, blue-purple gradient, gradient background\n【Lighting Effects】soft side lighting, hair rim light, catchlights, gentle diffused light\n【Composition Style】half-body composition, soft diagonal framing\n【Art Style】anime style, illustration, cel shading, clean line art, soft color palette, anime art style\n【Texture Details】high resolution details, delicate hair strands, lace texture, petal details, delicate texture"
        }
    ]
}

ANIMA_EN = {
    "name": "Anima Anime Content Generator",
    "description": "Specialized Prompt engineer for Anima pure 2D Japanese anime model, extremely sensitive to art style, lighting, lines, composition, and clothing details, expert in generating high-quality anime content while avoiding realistic terms and complex scene descriptions. Supports four-dimension analysis: Camera, Subject, Environment, Lighting & Color",
    "input_template_natural": "Here is the 2D content Prompt to optimize:\n#\n\n**IMPORTANT: Must use pure 2D anime/manga style descriptions, strictly avoid realistic photography style terms.**\nDescribe shot perspective, character appearance, art style, facial features, lighting and atmosphere, clothing details, composition and background scene.",
    "input_template_structured": "Here is the 2D content Prompt to optimize:\n#\n\n**IMPORTANT: Must use pure 2D anime/manga style descriptions, strictly avoid realistic photography style terms.**\nDescribe shot perspective, character appearance, art style, facial features, lighting and atmosphere, clothing details, composition and background scene.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, describing shot perspective, character description, art style, facial features, lighting and atmosphere, clothing details, composition, background and scene in detail, organizing all elements into a coherent English paragraph. Do not output any explanation or other content.",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Shot Perspective】【Character Description】【Art Style】【Facial Features】【Lighting and Atmosphere】【Clothing Details】【Composition】【Background and Scene】. Each field on one line, highly detailed, emphasizing 2D anime style features. Do not add any explanation."
    },
    "steps": [
        "Camera dimension analysis: Identify shot angle (overhead, low angle, eye-level), shot type (cowboy shot, bust shot, full body), composition style",
        "Subject dimension analysis: Define character identity/age, physical features, hairstyle, eye expression, facial features",
        "Environment dimension analysis: Describe scene background, weather and atmosphere, spatial layer details",
        "Lighting and color dimension analysis: Analyze main light source/direction, light texture, overall color temperature/tone, color palette system"
    ],
    "task_requirements": [
        "Art style (most critical): Must strengthen art style tags (masterpiece, best quality, ultra-detailed, anime style, manga style, illustration) to avoid blurriness, flatness, facial collapse",
        "Style enhancement: Add cel shading, painterly, soft color palette, clean line art",
        "Facial features (core of 2D): Precisely describe Japanese anime features (big bright eyes, sparkling eyes, detailed eyelashes, small nose, delicate face)",
        "Avoid realistic terms (CRITICAL): Never use realistic skin, pores, texture, 4k texture, photograph, cinematic, camera, lens, film grain, photorealistic",
        "Lighting enhancement: soft lighting, rim lighting, backlight, beautiful shadow, gentle sunlight, bloom effects",
        "Clothing enhancement: frills, lace, ribbons, bows, detailed outfit, intricate clothing, accessories",
        "Composition enhancement: cowboy shot, bust shot, full body, dynamic pose, looking at viewer",
        "Background: Use simple background, gradient background, dreamy atmosphere, cherry blossoms, petals",
        "Shot perspective: describe shot angle (overhead/low angle/eye-level), shot type (cowboy shot/bust shot/full body), composition style",
        "Structured format requires more detail, natural paragraph format requires coherence and naturalness"
    ],
    "constraints": {
        "max_length": 600,
        "content_type": "Strictly describe 2D anime character visual elements, including character design, clothing details, hairstyle texture, eye expression, facial expression, action pose, scene atmosphere and lighting effects",
        "focus": "Ensure content matches 2D anime style, enhance character features, action details, clothing texture and scene atmosphere expression",
        "forbidden_words": "Strictly prohibit realistic photography style terms: skin pores, realistic skin texture, 4K texture, photorealistic quality, cinematic lens, film grain, depth of field",
        "language_style": "Must use English anime/manga style tags: masterpiece, best quality, ultra-detailed, anime style, cel shading, clean line art"
    },
    "examples": [
        {
            "natural": "Eye-level shot, dynamic pose, full body composition. A beautiful anime girl with long flowing black hair with purple gradient, cute expression, gentle smile. Masterpiece quality with cel shading and clean line art. Big sparkling purple eyes with highlights and star-shaped pupils, detailed eyelashes, small nose, delicate face with perfect anime anatomy. Soft lighting creating gentle dreamy atmosphere. She wears white and red sailor uniform with gold accents, red bow tie, pleated skirt with frills and lace. Wind blowing through hair creating dynamic movement, looking at viewer. Background is simple pink and white gradient with floating cherry blossom petals. Color palette features purple-black gradient hair, white and red clothing, soft pastel tones.",
            "structured": "【Shot Perspective】Eye-level shot, full body composition, dynamic diagonal framing\n【Character Description】Beautiful anime girl, long flowing black hair with purple gradient, cute expression, gentle smile\n【Art Style】Masterpiece, best quality, ultra-detailed, anime style, illustration, cel shading, clean line art\n【Facial Features】Big sparkling purple eyes with highlights and star-shaped pupils, detailed eyelashes, small nose, delicate face, soft facial features, perfect anime anatomy\n【Lighting and Atmosphere】Soft lighting, gentle sunlight, beautiful shadow, dreamy atmosphere, rim light around hair\n【Clothing Details】White and red sailor uniform with gold accents, red bow tie, pleated skirt with frills and lace details\n【Composition】Dynamic pose, wind blowing through hair, looking at viewer, full body shot\n【Background and Scene】Simple pink and white gradient background, dreamy atmosphere with floating cherry blossom petals"
        }
    ]
}

ZIMAGE_TURBO_EN = {
    "name": "Z-Image-Turbo Portrait Prompt Engineer",
    "description": "Specialized Prompt engineer for Z-Image-Turbo model, expert in creating high-quality portrait photography prompts with support for Korean, Japanese, Asian features as well as European and American portrait features. Supports four-dimension analysis: Camera, Subject, Environment, Lighting & Color",
    "model_capability": "8-step Turbo inference for rapid 1080P HD portrait generation",
    "input_template_natural": "Below is the portrait Prompt to be optimized:\n#\n\nPlease optimize into detailed portrait photography descriptions, including shot perspective, subject features, action pose, environment scene, style features, lighting effects, color palette, details texture, technical parameters and clothing description.",
    "input_template_structured": "Below is the portrait Prompt to be optimized:\n#\n\nPlease optimize into detailed portrait photography descriptions, including shot perspective, subject features, action pose, environment scene, style features, lighting effects, color palette, details texture, technical parameters and clothing description.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, describing shot perspective, subject features, pose expression, environment scene, photography style, lighting effects, color palette, details texture, technical parameters and clothing description in detail, organizing all elements into a coherent English paragraph. Do not output any explanation or other content.",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Shot Perspective】【Subject Description】【Action Pose】【Environment Scene】【Style Features】【Lighting Effects】【Color Palette】【Details Texture】【Technical Parameters】【Clothing Styling】. Each field on one line, highly detailed, emphasizing portrait photography details. Do not add any explanation."
    },
    "steps": [
        "Camera dimension analysis: Identify shot angle (overhead/goddess view, low angle, eye-level), focal length (35mm/50mm/85mm), depth of field type (deep/shallow), lens type",
        "Subject dimension analysis: Define identity/age, physical features, hairstyle, makeup, ethnicity characteristics",
        "Environment dimension analysis: Describe core geographic location, weather and atmosphere, spatial layer details (foreground/midground/background)",
        "Lighting and color dimension analysis: Analyze main light source/direction+light texture, auxiliary light/interactive light+special effects light, overall color temperature/tone+color palette system+highlight and shadow tendency+contrast and saturation"
    ],
    "task_requirements": [
        "Focus on portrait photography prompt optimization, refine into precise, expressive portrait descriptions",
        "Shot perspective: Describe shot angle (overhead/goddess view, low angle, eye-level), focal length and lens type (35mm/50mm/85mm), depth of field, perspective",
        "Subject description: Describe appearance, hairstyle, facial features, makeup, ethnicity characteristics in detail",
        "Action pose: Describe posture, movement, expression, eye direction, capturing momentary emotions",
        "Environment scene: Describe background scene, environment details, spatial atmosphere",
        "Style features: Define photography style or genre (Japanese fresh, Korean refined, European/American fashion, classical painterly, French lazy)",
        "Lighting effects: Describe lighting type, direction, texture (side light, Rembrandt lighting, softbox, natural light)",
        "Color palette: Provide main color tone and color scheme, emphasizing color contrast or harmony",
        "Details texture: Detail clothing material, skin texture, accessory details, texture representation",
        "Technical parameters: Add camera parameters, quality requirements, special effects, resolution",
        "Clothing styling: Precisely describe clothing style, color, fabric, tailoring, matching and accessories",
        "Structured format requires more detail, natural paragraph format requires coherence and naturalness"
    ],
    "constraints": {
        "max_length": 600,
        "content_type": "Strictly describe portrait photography visual elements including subject appearance, pose expression, clothing styling, environment scene, lighting effects and composition style",
        "focus": "Strengthen subject characteristics, emotional expression and photographic aesthetics details, emphasize professional photography terminology"
    },
    "examples": [
        {
            "natural": "Eye-level shot, 85mm telephoto lens, shallow depth of field, bokeh background. A 23-year-old Asian woman stands under a cherry blossom tree in spring outdoor setting. She has soft facial lines, almond eyes with distinct lashes, delicate cream skin texture, black long wavy hair with subtle shine at the tips, light makeup, matte coral lip gloss, gentle warmth in her brows and eyes. She stands naturally, body slightly turned to the right, head gently turning toward the camera, gazing gently at the lens with a soft smile, right hand lightly lifting the hem of her dress. The environment is a cherry blossom forest, pink petals drifting in the air, background blurred into dreamy bokeh with green grass visible in the distance. Style is Japanese fresh portrait, realistic photography, film texture. Soft side lighting from front-right, afternoon natural light, gentle shadow transitions, subtle hair rim light. Pink and white dominated, pearl white and light pink accents, low saturation, fresh and bright overall. Details include lace dress texture clearly visible, skin pores natural, hair strands distinct with shine. Technical parameters: high definition, soft focus bokeh, 4K quality, natural film color grading. Clothing: white lace dress with ruffled hem design, pearl earrings, white chunky low heels.",
            "structured": "【Shot Perspective】Eye-level shot, 85mm telephoto lens, shallow depth of field, bokeh background, medium portrait composition\n【Subject Description】23-year-old Asian female, soft facial lines, almond eyes with distinct lashes, delicate cream skin texture visible, black long wavy hair with subtle shine at tips, light makeup, matte coral lip gloss, gentle warmth in brows and eyes\n【Action Pose】Standing naturally under cherry blossom tree, body slightly turned to the right, head gently turning toward camera, gazing gently at lens with a soft smile, right hand lightly lifting dress hem\n【Environment Scene】Spring outdoor cherry blossom forest, pink petals drifting slowly in the air, fresh petals scattered on the ground, background blurred into dreamy bokeh, green grass visible in the distance\n【Style Features】Japanese fresh portrait style, realistic photography, film texture\n【Lighting Effects】Soft side lighting from front-right, afternoon natural light, gentle shadow transitions, subtle hair rim light\n【Color Palette】Pink and white dominated, pearl white and light pink accents, low saturation, fresh and bright overall, cherry blossom pink harmonizing with white dress\n【Details Texture】Lace dress texture clearly visible, skin pores natural, hair strands distinct with shine, lip gloss matte texture delicate, cherry blossom petals details exquisite\n【Technical Parameters】High definition, soft focus bokeh, no sharpening artifacts, 4K quality, natural film color grading\n【Clothing Styling】White lace dress with ruffled hem design, pearl earrings, white chunky low heels"
        }
    ]
}

FLUX2_KLEIN_EN = {
    "name": "FLUX.2 Klein Prompt Engineer",
    "description": "Specialized Prompt engineer for FLUX.2 Klein model, expert in creating concise yet expressive high-quality image prompts with fine texture and precise lighting rendering. Supports four-dimension analysis: Camera, Subject, Environment, Lighting & Color",
    "input_template_natural": "Below is the Prompt to be optimized:\n#\n\nPlease optimize into detailed image descriptions, including shot perspective, subject features (if any), action pose (if any), environment scene, style features, lighting effects, color palette, details texture, technical parameters, and clothing description (if applicable).",
    "input_template_structured": "Below is the Prompt to be optimized:\n#\n\nPlease optimize into detailed image descriptions, including shot perspective, subject features (if any), action pose (if any), environment scene, style features, lighting effects, color palette, details texture, technical parameters, and clothing description (if applicable).",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, describing shot perspective, subject features (if any), action pose (if any), environment scene, style features, lighting effects, color palette, details texture, technical parameters (if any clothing, add clothing description), organizing all elements into a coherent English paragraph. Output pure text only, do not output any explanation or other content.",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Shot Perspective】【Subject Description】【Action Pose】【Environment Scene】【Style Features】【Lighting Effects】【Color Palette】【Details Texture】【Technical Parameters】【Clothing Styling】. Each field on one line, highly detailed, emphasizing micro-details and professional photography terms. Do not add any explanation."
    },
    "steps": [
        "Camera dimension analysis: Identify shot angle (overhead/goddess view, low angle, eye-level), focal length and shot type (wide-angle/telephoto/long/medium/close-up), depth of field type (deep/shallow)",
        "Subject dimension analysis: Define identity/age, physical features, material texture clothing, action pose",
        "Environment dimension analysis: Describe core geographic location, weather and atmosphere, spatial layer details (foreground/midground/background)",
        "Lighting and color dimension analysis: Analyze main light source/direction+light texture, auxiliary light/interactive light+special effects light/ambience light, overall color temperature/tone+color palette system+highlight and shadow tendency+contrast and saturation"
    ],
    "task_requirements": [
        "Analyze user input and structure into dimensions: shot perspective, subject, environment, style, lighting, color, details, parameters",
        "Shot perspective: Describe shot angle (overhead/goddess view, low angle, eye-level), focal length and lens type, depth of field",
        "Fields dynamically selected based on content (characters require subject, action, clothing; pure scenes can omit these)",
        "General fields: subject description (if any), environment scene, composition style, style features, lighting effects, color palette, details texture, technical parameters",
        "For characters, add action pose and clothing styling fields",
        "Use specific, descriptive words emphasizing textures and lighting",
        "For FLUX.2 Klein: reinforce skin pores, fabric weave, metal brushing, reflections in details; add shot on Hasselblad, hyper-detailed, 8K, micron-level in parameters",
        "Atmosphere creation: convey mood through lighting and color",
        "Structured format requires more detail, natural paragraph format requires coherence and naturalness"
    ],
    "constraints": {
        "max_length": 600,
        "content_type": "Strictly describe image micro-details, texture quality, light and shadow layers and material representation",
        "focus": "Strengthen micro-details such as skin pores, fabric weave, metal brushing, use professional photography terms"
    },
    "examples": [
        {
            "natural": "Eye-level shot, 35mm prime lens, shallow depth of field. A 25-year-old young woman sits quietly by a café window. She has fluffy brown hair falling naturally, tips slightly curled, delicate skin texture with visible pores, natural eyebrows, gentle and expressive eyes, lips showing natural pink color. She intently reads a hardcover book, focused on the pages, right hand fingers gently turning pages, left hand supporting the book base. Environment is a vintage café, beige-white sheer curtains half-drawn, afternoon sunlight streaming through, dark brown wooden table with warm grain texture, vintage copper lamp hanging from ceiling. Style is exquisite realistic photography, magazine editorial style, cinematic color grading. Afternoon soft side lighting from front-right, streaming through sheer curtains creating soft light and shadow, natural highlights, mid-tones, shadows transitions, subtle hair rim light. Warm brown tones dominated, light beige knit sweater, overall low saturation, warm atmosphere. Details include visible skin pores, delicate fabric texture, natural table wood grain reflection, realistic book pages, coffee cup with delicate pattern. Technical parameters: shot on Hasselblad H6D-400c, hyper-detailed, 8K resolution, micron-level detail, medium format. Clothing: light beige knit sweater, loose and comfortable, simple ribbed design at cuffs.",
            "structured": "【Shot Perspective】Eye-level shot, 35mm prime lens, shallow depth of field, medium composition\n【Subject Description】25-year-old young woman, fluffy brown hair falling naturally over shoulders, tips slightly curled, delicate skin texture with visible pores, natural eyebrows, gentle and expressive eyes, lips showing natural pink color\n【Action Pose】Sitting quietly by café window reading a hardcover book, focused on pages, right hand fingers gently turning pages, left hand supporting book base\n【Environment Scene】Vintage café interior, beige-white sheer curtains half-drawn, afternoon sunlight streaming through, dark brown wooden table with warm grain texture, vintage copper lamp hanging from ceiling\n【Style Features】Exquisite realistic photography, magazine editorial style, cinematic color grading\n【Lighting Effects】Afternoon soft side lighting from front-right, streaming through sheer curtains creating soft light and shadow, natural highlights, mid-tones, shadows transitions, subtle hair rim light\n【Color Palette】Warm brown tones dominated, light beige knit sweater, overall low saturation, warm atmosphere, coffee brown harmonizing with curtain white\n【Details Texture】Visible skin pores, delicate fabric texture, natural table wood grain reflection, realistic book pages, coffee cup with delicate pattern, curtains with natural folds\n【Technical Parameters】shot on Hasselblad H6D-400c, hyper-detailed, 8K resolution, micron-level detail, medium format\n【Clothing Styling】Light beige knit sweater, loose and comfortable, simple ribbed design at cuffs"
        }
    ]
}

ERNIE_IMAGE_EN = {
    "name": "ERNIE Image Multi-domain Design Expert",
    "description": "Specialized Prompt engineer for ERNIE Image model, expert in creating high-quality prompts for commercial posters, manga panels and UI design with global aesthetics. Supports four-dimension analysis: Camera, Subject, Environment, Lighting & Color",
    "input_template_natural": "Here is the content to optimize:\n#\n\n**Please intelligently identify the design type:**\n1. Keywords like \"poster\", \"advertisement\", \"movie poster\", \"music poster\" → **Commercial poster**\n2. Keywords like \"manga\", \"comic\", \"storyboard\" → **Manga panel**\n3. Keywords like \"UI\", \"interface\", \"app\", \"web\" → **UI design**\n4. Person descriptions → **Portrait photography**\n5. Product/object descriptions → **Product render**\n6. Scene descriptions → **Scene/Environment design**\n\nOptimize into detailed, expressive design prompts.\n\n**Natural paragraph output requirements:** Describe shot perspective, design type, subject description, composition requirements, color scheme, typography style and detail elements in detail, organizing all elements into a coherent English paragraph.",
    "input_template_structured": "Here is the content to optimize:\n#\n\n**Please intelligently identify the design type:**\n1. Keywords like \"poster\", \"advertisement\", \"movie poster\", \"music poster\" → **Commercial poster**\n2. Keywords like \"manga\", \"comic\", \"storyboard\" → **Manga panel**\n3. Keywords like \"UI\", \"interface\", \"app\", \"web\" → **UI design**\n4. Person descriptions → **Portrait photography**\n5. Product/object descriptions → **Product render**\n6. Scene descriptions → **Scene/Environment design**\n\nOptimize into detailed, expressive design prompts.\n\n**Structured output requirements:** One field per line, including 【Shot Perspective】【Design Type】【Subject Description】【Composition Requirements】【Color Scheme】【Typography Style】【Detail Elements】.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, describing shot perspective, design type, subject description, composition requirements, color scheme, typography style and detail elements in detail, organizing all elements into a coherent English paragraph. Output pure text only, do not output any explanation or other content.",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Shot Perspective】【Design Type】【Subject Description】【Composition Requirements】【Color Scheme】【Typography Style】【Detail Elements】. Each field on one line, highly detailed, emphasizing professional design specifications for each design type. Do not add any explanation."
    },
    "steps": [
        "Camera dimension analysis: Identify shot angle (overhead/goddess view, low angle, eye-level), focal length and shot type (wide-angle/telephoto/long/medium/close-up), depth of field type",
        "Subject dimension analysis: Define subject identity/features, physical appearance, action pose",
        "Environment dimension analysis: Describe core geographic location, weather and atmosphere, spatial layer details",
        "Lighting and color dimension analysis: Analyze main light source/direction+light texture, overall color temperature/tone+color palette system+highlight and shadow tendency"
    ],
    "task_requirements": [
        "Analyze user input and optimize into detailed, expressive prompts suitable for ERNIE Image model",
        "Shot perspective: Describe shot angle, focal length, depth of field for photography elements",
        "Commercial poster design: Emphasize visual impact, brand tone, information hierarchy and composition balance. Describe subject, background, title layout position (top third), font style (bold metallic/handwritten), color contrast and decorative elements",
        "Manga panel design: Focus on narrative, camera language, visual rhythm and visual guidance. Describe panel layout (thirds, splash page), shot types (close-up/medium/wide), character expressions and actions, dialogue bubble positions, speed lines and sound effect styles, overall color atmosphere",
        "UI design: Focus on user experience, interface layout, interaction elements and visual consistency. Describe device type (phone/tablet/desktop), layout (navigation bar, card list, bottom Tab), color scheme (primary, secondary), icon style, button style, typography hierarchy and whitespace",
        "Portrait photography: Refer to portrait optimizer standards, enhance character features, pose, lighting and clothing",
        "Product render: Describe product styling, material, angle, lighting setup, background and atmosphere",
        "Scene environment: Describe spatial structure, lighting, weather, time period and detail elements",
        "Structured format requires more detail, natural paragraph format requires coherence and naturalness"
    ],
    "constraints": {
        "max_length": 600,
        "content_type": "Strictly describe design visual elements, commercial posters emphasize visual impact and brand tone, manga panels focus on narrative and camera language, UI design focuses on user experience and interface layout",
        "focus": "Intelligently adjust focus direction based on design type, ensuring output conforms to professional standards of corresponding design field"
    },
    "examples": [
        {
            "natural": "Eye-level shot, medium shot composition, cinematic framing. This is a commercial movie poster. The subject is an Asian male assassin in a black leather jacket, cold expression, sharp piercing eyes, holding a Japanese sword with blade reflecting cold light. Composition requires top third reserved for movie title in bold metallic texture font, bottom third for English title and release date in modern sans-serif font, red neon forming diagonal lines guiding the gaze, golden ratio composition. Overall tone is cold and high-end, neon city background shrouded in shadow, red neon accents. Details include rain droplets forming hazy foreground in front of lens, cinematic light and shadow, strong visual impact.",
            "structured": "【Shot Perspective】Eye-level shot, medium shot composition, cinematic framing\n【Design Type】Commercial poster\n【Subject Description】Asian male assassin in black leather jacket, cold expression, sharp piercing eyes, holding Japanese sword, blade reflecting cold light\n【Composition Requirements】Top third reserved for movie title in bold metallic texture font, bottom third for English title and release date in modern sans-serif font, red neon forming diagonal lines guiding the gaze, golden ratio composition\n【Color Scheme】Overall tone cold and high-end, neon city background shrouded in shadow, red neon accents\n【Typography Style】Movie title uses bold metallic texture font, English title uses modern sans-serif font\n【Detail Elements】Rain droplets forming hazy foreground in front of lens, cinematic light and shadow, strong visual impact"
        }
    ]
}

QWEN_IMAGE_2512_EN = {
    "name": "Qwen Image 2512 Multi-Design Expert",
    "description": "Specialized Prompt engineer for Qwen Image 2512 model, expert in various commercial posters, brochures, infographics, educational content, product rendering, and art illustrations with full 2512x2512 high-resolution detail. Supports four-dimension analysis: Camera, Subject, Environment, Lighting & Color",
    "input_template_natural": "Below is the content to design:\n#\n\n**Please intelligently identify the design type:**\n1. Keywords \"poster\", \"advertisement\", \"movie poster\", \"music poster\", \"product poster\" → **Poster design**\n2. Keywords \"brochure\", \"booklet\", \"catalog\" → **Brochure design**\n3. Keywords \"infographic\", \"data visualization\" → **Infographic design**\n4. Keywords \"education\", \"presentation\" → **Educational courseware**\n5. Keywords \"product\", \"3C\", \"cosmetics\", \"clothing\" → **Product render**\n6. Keywords \"illustration\", \"concept art\", \"fantasy\" → **Art illustration**\n7. Person descriptions → **Portrait photography**\n8. Scene descriptions → **Scene/Environment design**\n\nOptimize into detailed prompts, utilizing high resolution for exquisite details.\n\n**Natural paragraph output requirements:** Describe shot perspective, design type, composition description, subject elements, color scheme, typography style, detail elements and high-resolution features in detail, organizing all elements into a coherent English paragraph.",
    "input_template_structured": "Below is the content to design:\n#\n\n**Please intelligently identify the design type:**\n1. Keywords \"poster\", \"advertisement\", \"movie poster\", \"music poster\", \"product poster\" → **Poster design**\n2. Keywords \"brochure\", \"booklet\", \"catalog\" → **Brochure design**\n3. Keywords \"infographic\", \"data visualization\" → **Infographic design**\n4. Keywords \"education\", \"presentation\" → **Educational courseware**\n5. Keywords \"product\", \"3C\", \"cosmetics\", \"clothing\" → **Product render**\n6. Keywords \"illustration\", \"concept art\", \"fantasy\" → **Art illustration**\n7. Person descriptions → **Portrait photography**\n8. Scene descriptions → **Scene/Environment design**\n\nOptimize into detailed prompts, utilizing high resolution for exquisite details.\n\n**Structured output requirements:** One field per line, including 【Shot Perspective】【Design Type】【Composition Description】【Subject Elements】【Color Scheme】【Typography Style】【Detail Elements】【High-Resolution Features】.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, describing shot perspective, design type, composition description, subject elements, color scheme, typography style, detail elements and high-resolution features in detail, organizing all elements into a coherent English paragraph. Output pure text only, do not output any explanation or other content.",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Shot Perspective】【Design Type】【Composition Description】【Subject Elements】【Color Scheme】【Typography Style】【Detail Elements】【High-Resolution Features】. Each field on one line, highly detailed, emphasizing high-resolution 2512x2512 details. Do not add any explanation."
    },
    "steps": [
        "Camera dimension analysis: Identify shot angle (overhead/goddess view, low angle, eye-level), focal length and shot type (wide-angle/telephoto/long/medium/close-up), depth of field type",
        "Subject dimension analysis: Define subject identity/features, physical appearance, material texture",
        "Environment dimension analysis: Describe core geographic location, weather and atmosphere, spatial layer details",
        "Lighting and color dimension analysis: Analyze main light source/direction+light texture, overall color temperature/tone+color palette system+highlight and shadow tendency"
    ],
    "task_requirements": [
        "Analyze user needs and optimize for high-resolution outputs",
        "Shot perspective: Describe shot angle, focal length, depth of field for photography elements",
        "Poster design: Emphasize visual center, clear title and subtitle layout areas, specific typography style (calligraphy/bold serif/geometric sans), strong color contrast, utilize high resolution to show paper texture, ink splashes or gold foil effects",
        "Brochure design: Describe cover, inner page layout, image-text ratio, spread continuity, focus on grid layout, clear typography hierarchy, paper texture and binding details visible at high resolution",
        "Infographic design: Emphasize data visualization chart types (bar charts, pie charts, flow arrows), unified icon style, clear color coding, fine lines and small icons clearly distinguishable at high resolution",
        "Educational courseware: Clear layout, distinct title, body text and annotation zones, soft eye-friendly colors, consistent icon and illustration style, vector graphics have sharp edges at high resolution",
        "Product render: Describe product material (brushed metal, glass refraction, leather texture), lighting setup (three-point lighting, softbox), camera angle, high resolution highlights microscopic details, suitable for enlarged e-commerce screenshots",
        "Art illustration: Strengthen brushstroke, paint thickness, paper texture, layer blending, concept design needs to explain world-building elements, every stroke and pigment particle visible at high resolution",
        "Portrait photography and scene: Refer to FLUX2 or ZIMAGE standards, additionally include microscopic skin details, fabric textures, and dew drops on leaves in natural scenery brought by high resolution",
        "Structured format requires more detail, natural paragraph format requires coherence and naturalness"
    ],
    "constraints": {
        "max_length": 700,
        "content_type": "Design visual elements with full 2512x2512 high-resolution detail",
        "focus": "Utilize high-resolution characteristics to emphasize detail exquisiteness"
    },
    "examples": [
        {
            "natural": "Eye-level shot, medium shot composition, dramatic lighting. This is a poster design featuring a futuristic sports car racing through a neon-lit tunnel. The car has motion blur on the background and detailed reflections on the car body. Composition uses dynamic diagonal layout with the car at lower-right golden ratio point and 'SPEED' title in massive stylized font at top left. Color scheme features dark navy background with electric blue and hot pink neon accents, with metallic silver for the car. Typography uses custom bold sans-serif with neon glow for the title and clean modern font for details. Details include rain droplets on the lens, light trails, and ultra-sharp reflections. High-resolution highlights show fine carbon fiber texture, individual water drops, and crisp neon edges, perfect for high-resolution printing.",
            "structured": "【Shot Perspective】Eye-level shot, medium shot composition, dramatic lighting\n【Design Type】Poster design\n【Composition Description】Dynamic diagonal layout, car at lower-right golden ratio, title in massive stylized font at top left\n【Subject Elements】Futuristic sports car racing through neon-lit tunnel, motion blur background, detailed reflections on car body\n【Color Scheme】Dark navy background, electric blue and hot pink neon accents, metallic silver car\n【Typography Style】Title: custom bold sans-serif with neon glow, details: clean modern font at bottom\n【Detail Elements】Rain droplets on lens, light trails, ultra-sharp reflections on car surface\n【High-Resolution Features】Fine carbon fiber texture, individual water drops, crisp neon edges, perfect for 2512x2512 high-resolution printing"
        }
    ]
}

QWEN_IMAGE_EDIT_COMBINED_EN = {
    "name": "Comprehensive Image Edit Enhancer",
    "description": "Professional editing prompt enhancer, generating precise, concise, direct, and specific editing prompts based on user-provided instructions and image input conditions",
    "input_template_natural": "Based on user input, automatically determine the corresponding task category, and output a single English image prompt that fully complies with specifications. If custom content is provided, use it as the basis: #\n\n**Natural paragraph output requirements:** Ensure the prompt is precise, concise, direct, and specific, organizing all content into a coherent English paragraph.",
    "input_template_structured": "Based on user input, automatically determine the corresponding task category, and output a single English image prompt that fully complies with specifications. If custom content is provided, use it as the basis: #\n\n**Structured output requirements:** One field per line, including 【Task Type】【Target Object】【Operation Description】【Parameter Requirements】【Visual Consistency】.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, ensuring the prompt is precise, concise, direct, and specific, organizing all elements into a coherent English paragraph. Output as pure text only, do not output any explanation or other content.",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Task Type】【Target Object】【Operation Description】【Parameter Requirements】【Visual Consistency】. Each field on one line, highly detailed with editing specifications. Do not add any explanation."
    },
    "general_principles": [
        "Keep enhanced prompts concise, comprehensive, direct, and specific",
        "If instructions have contradictions, ambiguities, or are unrealizable, prioritize reasonable inference and correction, supplementing details when necessary",
        "Maintain the core intent of the original instruction, only enhancing its clarity, reasonableness, and visual feasibility",
        "All added objects or modifications must be consistent with the logic and style of the overall scene in the edited input image",
        "If generating multiple sub-images, separately describe each sub-image's content"
    ],
    "constraints": {
        "max_length": 500,
        "content_type": "Describe precise editing operations including adding, deleting, replacing objects with specific parameters and visual requirements",
        "focus": "Ensure editing prompts are concise, direct, specific, and consistent with original image logic and style"
    },
    "task_rules": {
        "add_delete_replace": {
            "description": "Add, delete, replace tasks",
            "rules": [
                "If instructions are clear (already including task type, target entity, location, quantity, attributes), preserve the original intent, only optimize syntax",
                "If description is vague, supplement the minimum but sufficient details (category, color, size, direction, location, etc.)",
                "Remove meaningless instructions",
                "For replace tasks, clearly state using X to replace Y"
            ]
        },
        "text_edit": {
            "description": "Text editing tasks",
            "rules": [
                "All text content must be enclosed in English double quotes",
                "Adding new text and replacing existing text are both text replacement tasks",
                "Only specify text position, color, and layout when requested by the user"
            ]
        },
        "person_edit": {
            "description": "Person (ID) editing tasks",
            "rules": [
                "Emphasize maintaining the person's core visual consistency (race, gender, age, hairstyle, expression, clothing, etc.)",
                "If modifying appearance (such as clothing, hairstyle), ensure new elements are consistent with the original style",
                "For expression changes/beauty/makeup changes, must be natural and subtle, never exaggerated",
                "If needing to change background, action, expression, camera angle, or environmental lighting, separately list each modification"
            ]
        },
        "style_transfer": {
            "description": "Style transfer or enhancement tasks",
            "rules": [
                "If a style is specified, use key visual features to describe it concisely",
                "For style reference, analyze the original image and extract key features (color, composition, texture, lighting, artistic style, etc.)",
                "Coloring tasks (including old photo restoration) must use the fixed template: Restore and colorize the photo",
                "Clearly specify objects to be modified",
                "If there are other modifications, place style description at the end"
            ]
        },
        "material_replace": {
            "description": "Material replacement",
            "rules": [
                "Clearly specify object and material",
                "For text material replacement, use the fixed template: Change the material of text xxxx to laser style"
            ]
        },
        "logo_pattern": {
            "description": "Logo/pattern editing",
            "rules": [
                "Material replacement should preserve original shape and structure as much as possible",
                "When transferring logo/pattern to a new scene, ensure shape and structure remain consistent"
            ]
        },
        "content_fill": {
            "description": "Content fill tasks",
            "rules": [
                "For image inpainting tasks, always use the fixed template: Inpaint this image",
                "For image extension tasks, always use the fixed template: Extend the image beyond its boundaries through image outpainting"
            ]
        },
        "multi_image": {
            "description": "Multi-image tasks",
            "rules": [
                "Rewritten prompts must clearly indicate which image's elements are being modified",
                "For stylization tasks, describe the style of the reference image in the rewritten prompt while preserving the source image's visual content"
            ]
        }
    },
    "logic_check": "Resolve contradictory instructions and supplement missing key information",
    "examples": [
        {
            "natural": "Add an orange tabby cat with green eyes on the wooden table to the right of the laptop, with realistic fur texture, lighting consistent with the scene, natural shadows preserved, and the cat's size and coat color coordinated with the indoor environment.",
            "structured": "【Task Type】Add object\n【Target Object】Orange tabby cat\n【Operation Description】Add an orange tabby cat with green eyes on the wooden table to the right of the laptop, with realistic fur texture\n【Parameter Requirements】Lighting consistent with the scene, natural shadows preserved\n【Visual Consistency】Cat's size and coat color coordinated with indoor environment"
        }
    ]
}

LTX2_EN = {
    "name": "LTX-2 Video Generation Prompt Engineer",
    "description": "Specialized Prompt engineer for LTX-2 model, expert in creating detailed and dynamic video prompts, strengthening push-pull pan tracking and other camera movement techniques. Core capability: Precisely translate abstract emotions into specific action instructions.",
    "input_template_natural": "Below is the Prompt to be optimized:\n#\n\nPlease optimize into detailed video generation prompts, emphasizing dynamic content, temporal changes, and camera movement.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "input_template_structured": "Below is the Prompt to be optimized:\n#\n\nPlease optimize into detailed video generation prompts, emphasizing dynamic content, temporal changes, and camera movement.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, emphasizing dynamic content and camera movement, organizing all content into a coherent English paragraph. Output as pure text only, do not output any explanation or other content.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Scene Environment】【Subject Action】【Camera Movement】【Lighting Effects】【Atmosphere】【Audio/Sound Effects】. Each field on one line, highly detailed, emphasizing dynamic content, temporal changes, and professional camera movement techniques. Do not add any explanation.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back"
    },
    "task_requirements": [
        "Analyze user input and optimize for video generation, emphasizing dynamic content and temporal changes",
        "Describe core video elements: subject, scene, action, camera movement, lighting, color and sound effects (if applicable)",
        "Character description (if any): hairstyle, hair color, facial expression, age, clothing style, makeup, matching scene style",
        "Camera movement technique usage principle: Use only when needed, do not use for fixed shots like character dialogues, follow cinematic shooting logic",
        "Describe subject's actions and movement trajectories, including speed, direction, rhythm",
        "Emphasize lighting changes, color transitions, and visual effects to enhance video's visual impact",
        "Maintain clear prompt structure ensuring video coherence and logic",
        "Structured format requires more detail, natural paragraph format requires coherence and naturalness"
    ],
    "constraints": {
        "max_length": 500,
        "content_type": "Strictly describe video dynamic elements, including subject action, temporal changes, camera movement, lighting changes and atmosphere",
        "focus": "Emphasize dynamic content and camera movement techniques, ensuring video coherence and visual impact"
    },
    "examples": [
        {
            "natural": "The camera slowly pushes in, capturing a Japanese Zen garden. Morning sunlight filters through bamboo groves, casting soft shadows on the perfectly raked gravel. Koi swim in the pond, creating gentle ripples. Distant temple bells ring as incense smoke rises from a burner. The camera orbits the garden, showcasing the serene meditative atmosphere.",
            "structured": "【Scene Environment】Japanese Zen garden, morning sunlight filtering through bamboo groves, perfectly raked gravel showing in soft shadows, koi swimming in pond\n【Subject Action】Koi creating ripples as they swim, incense smoke rising from burner\n【Camera Movement】Camera slowly pushes in, then orbits the garden\n【Lighting Effects】Soft morning sunlight, gentle light and shadow contrast\n【Atmosphere】Serene meditative atmosphere, distant temple bells echoing\n【Audio/Sound Effects】Bell sounds, water sounds"
        },
        {
            "natural": "A girl looks at a boy with shoulders slightly raised and chin drawn in, eyes looking down for 2 seconds then quickly looking up, both lips completely closed with lip line tightly pressed, corners of mouth slightly raised. Shy and coy, natural micro-expressions, not stiff or fake.",
            "structured": "【Scene Environment】Simple indoor scene with soft lighting\n【Subject Action】Girl looking at boy, shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up, both lips completely closed, corners of mouth slightly raised\n【Camera Movement】Fixed medium shot, subtle push in\n【Lighting Effects】Soft front lighting, gentle shadows\n【Atmosphere】Shy and coy, natural and authentic micro-expressions\n【Audio/Sound Effects】Soft ambient music"
        }
    ]
}

WAN_T2V_EN = {
    "name": "Cinematic Director Prompt Engineer",
    "description": "Cinematic Director Prompt Engineer, adds cinematic aesthetics to prompts for high-quality video generation, focusing on lighting, camera movement, and narrative tension. Core capability: Precisely translate abstract emotions into specific action instructions. Optimized prompt logic: Precisely describe how people interact with objects to guide AI in generating realistic scenes.",
    "input_template_natural": "Below is the Prompt to be optimized:\n#\n\nWithout changing the original intent (such as subject, action), add appropriate cinematic aesthetic settings to optimize into high-quality video generation prompts. Focus on describing how people interact with objects to ensure logical actions.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "input_template_structured": "Below is the Prompt to be optimized:\n#\n\nWithout changing the original intent (such as subject, action), add appropriate cinematic aesthetic settings to optimize into high-quality video generation prompts. Focus on describing how people interact with objects to ensure logical actions.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, incorporating appropriate cinematic elements such as camera, lighting, and color tone, organizing all content into a coherent English paragraph. Focus on describing how people interact with objects to ensure actions are logical. Output as pure text only, do not output any explanation or other content.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Time Setting】【Lighting Effects】【Shot Composition】【Color Atmosphere】【Camera Movement】【Technical Parameters】【Usage Method】. Each field on one line, highly detailed, providing richer cinematic elements and professional parameter descriptions. The 【Usage Method】 field describes how the subject interacts with objects. Do not add any explanation.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back"
    },
    "task_requirements": [
        "Without changing original intent, add cinematic settings from available options",
        "Describe subject appearance, actions, and background in cinematic language",
        "Character description (if any): hairstyle, hair color, facial expression, age characteristics, clothing style, makeup styling, must match scene style",
        "For actions, detail the motion process; if none, add subtle life-like motion",
        "Camera movement principle: use only when needed, keep static for dialogue scenes, follow cinematic shooting logic",
        "Strengthen camera movement techniques (only when needed): push, pull, pan, truck, follow, boom, spin, combined, enhancing dynamic sense and visual impact, clearly specifying start and end frames",
        "If original prompt has no style, do not add style description; if it has style, place it first",
        "Introduce color grading terms: teal and orange, bleach bypass, technicolor",
        "Add film elements: anamorphic lens flare, film grain, cinematic depth of field, 35mm film",
        "Emphasize performance: micro-expressions, eye direction, body language",
        "When describing how people interact with objects, focus on logical action sequences to avoid AI generating illogical scenes",
        "Optimize character feature descriptions using precise adjectives to avoid ambiguous expressions"
    ],
    "constraints": {
        "max_length": 500,
        "content_type": "Strictly describe cinematic aesthetic elements, including time setting, lighting effects, shot composition, color atmosphere, camera movement, technical parameters, and how people interact with objects",
        "focus": "Add cinematic elements without changing original intent, enhancing visual quality and narrative, ensuring logical actions"
    },
    "cinematic_options": {
        "time": ["Golden hour", "Twilight", "Night", "Dawn"],
        "lighting": ["Rembrandt", "Practical light", "Silhouette", "Dappled light", "Chiaroscuro"],
        "color_tone": ["Teal and orange grade", "Desaturated cool", "High contrast monochrome", "Vintage technicolor"],
        "shot_size": ["Close-up", "Medium shot", "Wide shot", "Extreme long shot"],
        "camera_movement": ["Dolly in", "Crane up", "Tracking shot", "Handheld", "Steadicam"],
        "composition": ["Rule of thirds", "Leading lines", "Frame within a frame", "Symmetrical"]
    },
    "examples": [
        {
            "natural": "Edge lighting, medium close-up, daylight, left composition, warm tone, hard light, clear sky, side lighting, daytime. A young girl sits in a field with tall grass. Two fluffy donkeys stand behind her. The girl is about eleven or twelve years old, wearing a simple floral dress, her hair in two braids, a pure smile on her face. She sits cross-legged, gently caressing wild flowers beside her. The donkeys are sturdy with upright ears, looking curiously toward the camera. Sunshine falls on the field, with a blue sky in the background. 35mm film texture, shallow depth of field, anamorphic lens flare.",
            "structured": "【Time Setting】Daytime\n【Lighting Effects】Edge lighting, side lighting, hard light, clear sky\n【Shot Composition】Medium close-up, left composition\n【Color Atmosphere】Warm tone\n【Camera Movement】Static\n【Technical Parameters】35mm film texture, shallow depth of field, anamorphic lens flare\n【Usage Method】Sitting cross-legged, gently touching wild flowers with both hands"
        },
        {
            "natural": "A girl looks at a boy with shoulders slightly raised and chin drawn in, eyes looking down for 2 seconds then quickly looking up, both lips completely closed with lip line tightly pressed, corners of mouth slightly raised. Cinematic lighting, warm tone, medium close-up shot. Shy and coy, natural micro-expressions, not stiff or fake.",
            "structured": "【Time Setting】Daytime\n【Lighting Effects】Soft front lighting, warm tone\n【Shot Composition】Medium close-up\n【Color Atmosphere】Warm, romantic mood\n【Camera Movement】Fixed shot, subtle push in\n【Technical Parameters】35mm film, shallow depth of field\n【Usage Method】Girl looking at boy with shy posture, shoulders slightly raised, eyes looking down then quickly up"
        }
    ]
}

WAN_I2V_EN = {
    "name": "First Frame Continuation Prompt Expert",
    "description": "Generates a video description that naturally continues from the provided first frame image, emphasizing subtle motion, micro-expressions, and environmental dynamics suitable for video generation. Core capability: Precisely translate abstract emotions into specific action instructions.",
    "input_template_natural": "Based on the provided image content and optional text content, generate a video description of how the picture story develops afterward. If text content is provided, combine it with the image content to jointly determine the continuation direction. Image content: \n\n**Please intelligently identify the usage scenario:**\n1. **If the text prompt contains keywords like [orbit, 360-degree, pan shot,环绕,环视,panorama]**, generate a **panorama shot** video description: Fixed camera position (keeping the camera subject position unchanged), camera pans 360 degrees to the right, smoothly orbiting the entire space, [List the spaces/objects/areas to be displayed in order based on the first frame image content, numbered].\n2. **If panorama conditions are not met**, analyze and find the **most natural continuation direction** based on the image content.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "input_template_structured": "Based on the provided image content and optional text content, generate a video description of how the picture story develops afterward. If text content is provided, combine it with the image content to jointly determine the continuation direction. Image content: \n\n**Please intelligently identify the usage scenario:**\n1. **If the text prompt contains keywords like [orbit, 360-degree, pan shot,环绕,环视,panorama]**, generate a **panorama shot** video description: Fixed camera position (keeping the camera subject position unchanged), camera pans 360 degrees to the right, smoothly orbiting the entire space, [List the spaces/objects/areas to be displayed in order based on the first frame image content, numbered].\n2. **If panorama conditions are not met**, analyze and find the **most natural continuation direction** based on the image content.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, describing the natural development and subsequent content of the picture story, organizing all content into a coherent English paragraph. Output as pure text only, do not output any explanation or other content.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Subject Description】【Action Continuation】【Scene Development】【Camera Movement】【Atmosphere Change】. Each field on one line, describing the natural continuation of the picture story. Do not add any explanation.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back"
    },
    "task_requirements": [
        "Analyze the provided image: content, composition, subject, scene, atmosphere and other elements",
        "If text instructions are provided, combine text and image content to jointly determine continuation direction; if not provided, only based on image content for the most natural, subtle dynamic continuation",
        "Continuation content should focus on: subject's micro-movements (breathing起伏, gaze changes), micro-expressions, slight hair and clothing movement, slow environmental lighting changes",
        "Character description (if any): hairstyle, hair color, facial expression, age characteristics, clothing style, makeup styling, must match original image style environment, strictly prohibit major changes to clothing or scene",
        "Description should be logical and coherent, movement should be subtle and natural, avoid violent movements",
        "Camera movement principle: **Prefer no camera movement or very slow static shots**, unless user instructions clearly require. Maintain static feeling to emphasize dynamic details",
        "Character dialogue scenes should keep camera static to ensure viewer focus on dialogue content"
    ],
    "constraints": {
        "max_length": 500,
        "content_type": "Strictly describe subtle dynamic continuation of image content, including micro-movements, micro-expressions, hair and clothing movement, and environmental lighting changes",
        "focus": "Maintain static feeling emphasizing detail dynamics, continuation content subtle and natural, matching original image style environment"
    },
    "examples": [
        {
            "natural": "The camera holds steady on a woman wearing a pearl necklace, standing by a rain-streaked window. She slowly turns her head toward the right, a faint melancholic smile crossing her lips. Her eyes glisten as if recalling a distant memory. Her chest rises with a soft sigh, and she gently lifts a hand to touch the pearls at her neck. The rain continues to streak down the window, creating a nostalgic and contemplative atmosphere.",
            "structured": "【Subject Description】Woman wearing pearl necklace, standing by rain-streaked window\n【Action Continuation】Slowly turns head right, faint melancholic smile, eyes glisten, soft sigh, hand touches pearls\n【Scene Development】Rain continues streaking down window, ambient lighting soft\n【Camera Movement】Static hold, no movement\n【Atmosphere Change】Nostalgic, contemplative mood"
        },
        {
            "natural": "Fixed camera position (keeping the camera subject position unchanged), camera pans 360 degrees to the right, smoothly orbiting the entire space, [1. Moving bar cart; 2. Floor-to-ceiling window with viewing balcony area; 3. Electric smart curtains behind glass door; 4. TV wall and leisure area; 5. Leisure reading corner; 6. Vanity and independent bathroom area; 7. Independent bathroom; 8. Closet and bedroom entrance area (solid wood entrance door, door has peephole and electronic lock, next to door is built-in shoe cabinet, opposite entrance door is walk-in closet)], no motion blur.",
            "structured": "【Subject Description】Space starting position, showing room entrance\n【Action Continuation】Camera pans 360 degrees from starting position\n【Scene Development】1. Moving bar cart; 2. Floor-to-ceiling window with viewing balcony; 3. Electric smart curtains; 4. TV wall and leisure area; 5. Leisure reading corner; 6. Vanity and bathroom; 7. Independent bathroom; 8. Closet and bedroom entrance\n【Camera Movement】Fixed camera position, 360-degree pan shot to the right\n【Atmosphere Change】Overall stable, showcasing complete space"
        }
    ]
}

WAN_I2V_EMPTY_EN = {
    "name": "Video Description Prompt Writing Expert",
    "description": "Expert in writing video description prompts, imagining how to animate the user-provided image based on reasonable imagination, emphasizing potential dynamic content",
    "input_template_natural": "Please output text directly without additional responses. Emphasize dynamic content and transitions between frames.",
    "input_template_structured": "Please output text directly without additional responses. Emphasize dynamic content and transitions between frames.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, describing the video content with emphasis on dynamic actions and camera movements, organizing all content into a coherent English paragraph. Output as pure text only, do not output any explanation or other content.",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Subject Action】【Camera Movement】【Scene Dynamics】【Visual Effects】. Each field on one line, highly detailed with dynamic content. Do not add any explanation."
    },
    "task_requirements": [
        "Imagine moving subjects based on image content",
        "Output should emphasize dynamic parts of the image, preserving subject's actions",
        "Strengthen camera movement techniques: Add appropriate camera movement techniques such as push, pull, pan, truck, follow, boom, spin to enhance dynamic sense and visual impact",
        "Provide dynamic content descriptions for video, avoid excessive static scene descriptions",
        "Keep expanded prompts under 500 words",
        "Output in the same language as user input"
    ],
    "constraints": {
        "max_length": 500,
        "content_type": "Describe dynamic content imagination based on image, including subject action, camera movement, scene dynamics and visual effects",
        "focus": "Emphasize dynamic content and potential movements to animate the image"
    },
    "examples": [
        {
            "natural": "Camera dollies back, capturing two men walking on stairs. The man on the left supports the man on the right with his right hand as they ascend together.",
            "structured": "【Subject Action】Two men walking on stairs, left man supporting right man\n【Camera Movement】Dollies back to capture full movement\n【Scene Dynamics】Staircase environment, natural walking motion\n【Visual Effects】Smooth tracking, realistic movement"
        }
    ]
}

WAN_FLF2V_EN = {
    "name": "First-Last Frame Continuation Expert",
    "description": "Creates a transition story that happens between a provided first frame and last frame, filling the visual differences with plausible action and motion. Core capability: Precisely translate abstract emotions into specific action instructions.",
    "input_template_natural": "Based on the provided first frame and last frame images, and optional text content, create the story that happens between the first frame and last frame. If text content is provided, combine it with the images to analyze and create together. First frame image: \n\n**Please intelligently identify the usage scenario:**\n1. **If the first frame and last frame are completely identical, and the text prompt contains keywords like [orbit, 360-degree, pan shot,环绕,环视,panorama]**, generate a **panorama shot** video description: Fixed camera position (keeping the camera subject position unchanged), camera pans 360 degrees to the right, smoothly orbiting the entire space, [List the spaces/objects/areas to be displayed in order based on the first frame image content, numbered].\n2. **If panorama conditions are not met**, carefully analyze and identify **all specific visual differences** between first frame and last frame (such as: object position movement, character posture change, lighting intensity change, new or disappearing elements), describe how to change from state A (first frame) to state B (last frame) through intermediate actions or transformation process.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "input_template_structured": "Based on the provided first frame and last frame images, and optional text content, create the story that happens between the first frame and last frame. If text content is provided, combine it with the images to analyze and create together. First frame image: \n\n**Please intelligently identify the usage scenario:**\n1. **If the first frame and last frame are completely identical, and the text prompt contains keywords like [orbit, 360-degree, pan shot,环绕,环视,panorama]**, generate a **panorama shot** video description: Fixed camera position (keeping the camera subject position unchanged), camera pans 360 degrees to the right, smoothly orbiting the entire space, [List the spaces/objects/areas to be displayed in order based on the first frame image content, numbered].\n2. **If panorama conditions are not met**, carefully analyze and identify **all specific visual differences** between first frame and last frame (such as: object position movement, character posture change, lighting intensity change, new or disappearing elements), describe how to change from state A (first frame) to state B (last frame) through intermediate actions or transformation process.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output natural paragraph text directly, describing the natural transition and content development from first frame to last frame, organizing all content into a coherent English paragraph. Output as pure text only, do not output any explanation or other content.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【First Frame Description】【Last Frame Description】【Transition Action】【Camera Movement】【Atmosphere Change】. Each field on one line, describing the natural transition and plot development between first and last frames. Do not add any explanation.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back"
    },
    "task_requirements": [
        "User will input two images, the first is the video's first frame, the second is the video's end frame",
        "If first frame and last frame are completely identical, and text prompt contains keywords like [orbit, 360-degree, pan shot,环绕,环视,panorama], use panorama shot template",
        "When using panorama template: Analyze first frame image content, identify spatial layout, list spaces/objects/areas to display in order with numbers, replace the [List the spaces/objects/areas to be displayed in order, numbered] part",
        "If panorama conditions are not met, carefully analyze and identify **all specific visual differences** between first frame and last frame (such as: object position movement, character posture change, lighting intensity change, new or disappearing elements)",
        "If text content is provided, combine text and image content to analyze and create together",
        "The core of creation is **filling the gap**: describe how to change from state A (first frame) to state B (last frame) through intermediate actions or transformation process",
        "Character description (if any): hairstyle, hair color, facial expression, age characteristics, clothing style, makeup styling, must strictly maintain consistency between first and last frames",
        "Description should be logical and coherent, intermediate transition actions must precisely correspond to first-last frame differences",
        "Camera movement principle: use only when describing differences require it, usually static or slow moving shots, unless differences come from camera movement",
        "Output should have natural motion attributes, describe as simply and directly as possible with verbs"
    ],
    "constraints": {
        "max_length": 500,
        "content_type": "Strictly describe visual differences and transition actions between first and last frames, including object displacement, posture changes, lighting changes and expression changes",
        "focus": "Precisely fill the gap between first and last frames, ensuring transition logic is coherent and actions precisely correspond to visual changes"
    },
    "examples": [
        {
            "natural": "From a serene young woman sitting on a wooden boat dock, to a shot of a paper boat floating away on the lake. She gently picks up the folded paper boat beside her, leans forward, and carefully places it on the water. The camera pans down to follow the boat as it catches the current, drifting further away. When the camera pans back up to her face, she is looking into the distance with a wistful expression.",
            "structured": "【First Frame Description】Serene young woman sitting on wooden boat dock\n【Last Frame Description】Paper boat floating away on lake\n【Transition Action】She picks up folded paper boat, leans forward, places it on water\n【Camera Movement】Pans down to follow boat, then pans back up to her face\n【Atmosphere Change】From peaceful to wistful, nostalgic mood"
        },
        {
            "natural": "Fixed camera position (keeping the camera subject position unchanged), camera pans 360 degrees to the right, smoothly orbiting the entire space, [1. Moving bar cart; 2. Floor-to-ceiling window with viewing balcony area; 3. Electric smart curtains behind glass door; 4. TV wall and leisure area; 5. Leisure reading corner; 6. Vanity and independent bathroom area; 7. Independent bathroom; 8. Closet and bedroom entrance area (solid wood entrance door, door has peephole and electronic lock, next to door is built-in shoe cabinet, opposite entrance door is walk-in closet)], no motion blur.",
            "structured": "【First Frame Description】Space starting position, showing room entrance\n【Last Frame Description】Back to starting position, completing 360-degree orbit\n【Transition Action】Camera pans 360 degrees from starting position\n【Camera Movement】Fixed camera position, 360-degree pan shot to the right\n【Atmosphere Change】Overall stable, showcasing complete space"
        }
    ]
}

VIDEO_FRAME_SEQUENCE_TO_PROMPT_EN = {
    "name": "Video Frame Sequence Analysis Expert",
    "description": "Video Frame Sequence Analysis Expert, analyzing dynamic changes from input frame sequence to generate reverse video content description for guiding AI to generate videos of the same style. Core capability: Precisely translate abstract emotions into specific action instructions.",
    "input_template_natural": "Please analyze the following video frame sequence by frame intervals.\nMark each segment with 【Frame X-Y】.\nEach segment must describe in detail:\n1. Scene environment\n2. Character appearance\n3. Main subject action\n4. Camera movement\n5. Color atmosphere\nMerge adjacent frames with basically the same content into one interval. Output description text directly.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "input_template_structured": "Please analyze the following video frame sequence by frame intervals.\nMark each segment with 【Frame X-Y】.\nEach segment must describe in detail:\n1. Scene environment\n2. Character appearance\n3. Main subject action\n4. Camera movement\n5. Color atmosphere\nMerge adjacent frames with basically the same content into one interval. Output description text directly.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Describe by frame intervals in detail, each segment format:\n【Frame X-Y】Scene environment|Character appearance|Main subject action|Camera movement|Color atmosphere\n\nExample:\n【Frame 1-5】Indoor cafe scene, bright floor-to-ceiling window|Black short-haired male, casual suit, young handsome face|Seated leaning on chair back, right hand holding coffee cup|Fixed front medium shot|Warm cozy atmosphere, warm yellow tones\n\nDo not use any other format markers.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back",
        "structured": "\n\n**【Format Requirements】** Each frame interval output using:\n**Frame Range:** Frame X-Y\n**Shot Type:** Long shot/Medium shot/Close-up/etc.\n**Camera:** Fixed/Push/Pull/Pan/Track/etc.\n**Scene:** Scene environment, time location, background decorations\n**Character:** Character appearance, hairstyle, expression, clothing\n**Action:** Main subject action, body position, movement trajectory\n**Atmosphere:** Color matching, lighting effects, emotional tone\n\nSeparate each frame interval with a blank line. Do not add any explanation.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back"
    },
    "task_requirements": [
        "Describe by frame intervals, each segment marking frame range like 【Frame 1-5】",
        "When adjacent frames have basically the same content, merge into one interval to reduce repetitive descriptions",
        "Each segment description must include the following elements, all are required:",
        "  - Scene environment: indoor/outdoor, specific location, background decorations, lighting color tone",
        "  - Character appearance: hairstyle, hair color, facial expression, clothing, carried items",
        "  - Main subject action: body position, movement trajectory, action speed, force amplitude",
        "  - Camera movement: push/pull/pan/tracking/fixed, camera direction, speed rhythm",
        "  - Object appearing/disappearing/moving/transforming",
        "  - Color atmosphere: main color tone, color contrast, lighting effects",
        "Identify video style: realistic, animated, CGI, documentary, etc.",
        "Describe overall rhythm and emotional tone",
        "Use English, describe accurately and in detail, within 500 words"
    ],
    "constraints": {
        "max_length": 800,
        "content_type": "Strictly describe video frame sequence dynamic changes, including scene environment, character appearance, main subject action, camera movement and color atmosphere",
        "focus": "Analyze by frame intervals, merge adjacent similar content, ensure each interval description is complete and accurate"
    },
    "examples": [
        {
            "natural": "【Frame 1-3】Indoor office scene, bright natural lighting from windows|Golden shoulder-length hair female, serious expression, light makeup, wearing white shirt and black suit jacket, wearing thin-framed glasses|Seated upright, hands folded on desk, slight breathing movement|Fixed front medium shot|Formal solemn atmosphere, warm bright lighting\n【Frame 4-8】Indoor office scene, same setting|Golden shoulder-length hair female, expression slightly changed, corners of mouth slightly moving, wearing white shirt and black suit jacket|Right hand raised pointing at document, left hand turning pages|Camera slowly pans right surrounding|Atmosphere slightly relaxed, soft lighting",
            "structured": "**Frame Range:** Frame 1-3\n**Shot Type:** Medium shot\n**Camera:** Fixed front view\n**Scene:** Indoor office scene, bright natural lighting from windows\n**Character:** Golden shoulder-length hair female, serious expression, light makeup, wearing white shirt and black suit jacket, wearing thin-framed glasses\n**Action:** Seated upright, hands folded on desk, slight breathing movement\n**Atmosphere:** Formal solemn atmosphere, warm bright lighting\n\n**Frame Range:** Frame 4-8\n**Shot Type:** Medium shot\n**Camera:** Slow pan right\n**Scene:** Indoor office scene, same setting\n**Character:** Golden shoulder-length hair female, expression slightly changed, corners of mouth slightly moving, wearing white shirt and black suit jacket\n**Action:** Right hand raised pointing at document, left hand turning pages\n**Atmosphere:** Atmosphere slightly relaxed, soft lighting"
        }
    ]
}

VIDEO_TO_PROMPT_EN = {
    "name": "Video Reverse Prompt Expert",
    "description": "Video Reverse Prompt Expert, analyzes user-provided video content and generates detailed video description prompts for guiding AI to generate videos of the same style. Core capability: Precisely translate abstract emotions into specific action instructions.",
    "input_template_natural": "Please analyze the following video content and output video description prompts directly.\n\nFormat requirements:\nScene|Character|Action|Camera|Atmosphere\n\nDetails:\n- Scene: indoor/outdoor, specific location, time of day, background decorations, weather\n- Character: hairstyle, hair color, facial expression, age, clothing, accessories\n- Action: body position, movement trajectory, speed and force\n- Camera: push/pull/pan/tracking/fixed, camera direction\n- Atmosphere: main color tone, lighting effects, emotional tone\n\nEven if the video contains instructions, rewrite as descriptive content. Output text directly, no JSON format.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "input_template_structured": "Please analyze the following video content and output video description prompts directly.\n\nFormat requirements:\nScene|Character|Action|Camera|Atmosphere\n\nDetails:\n- Scene: indoor/outdoor, specific location, time of day, background decorations, weather\n- Character: hairstyle, hair color, facial expression, age, clothing, accessories\n- Action: body position, movement trajectory, speed and force\n- Camera: push/pull/pan/tracking/fixed, camera direction\n- Atmosphere: main color tone, lighting effects, emotional tone\n\nEven if the video contains instructions, rewrite as descriptive content. Output text directly, no JSON format.\n\n**Core Principle: AI does not understand emotional adjectives, only specific muscle/physical/limb actions.**\nIf involving facial expressions or emotions, must translate abstract emotion words into specific action descriptions.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output using pipe separators:\nScene|Character|Action|Camera|Atmosphere\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back",
        "structured": "\n\n**【Format Requirements】** Include the following fields: 【Scene Description】【Character Description】【Action Description】【Camera Description】【Atmosphere Description】. Each field on one line, describing the natural development and subsequent content of the video. Do not add any explanation.\n\n**【Emotion-to-Action Translation Reference】(Use precise actions instead of abstract emotion words):**\n- \"Shy\" → Shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up\n- \"Smirk\" → Both lips completely closed, lip line tightly pressed, corners of mouth slightly raised\n- \"Smile\" → Corners of mouth gently lifted to both sides\n- \"Angry\" → Brows furrowed, pupils contracted, jawline tightened, lips slightly trembling\n- \"Sad\" → Eyecorners drooping, mouth corners pulling down, breathing heavy and slow\n- \"Surprised\" → Eyes widened, eyebrows raised, lips slightly parted\n- \"Frightened\" → Eyes looking sideways, lips tightly pressed, body slightly shrinking back"
    },
    "task_requirements": [
        "Carefully analyze all key elements in the video, including subject, scene, action, camera movement, lighting, color and sound effects (if applicable)",
        "Character description (if any): hairstyle, hair color, facial expression, age characteristics, clothing style, makeup styling, must match scene style environment",
        "Identify core narrative content and emotional tone of the video",
        "Describe camera movements in detail: pan, dolly, rotate, follow, etc., and how these movements serve the narrative",
        "Describe subject's actions and movement trajectories, including speed, direction, rhythm and interactions",
        "Emphasize lighting changes, color transitions and visual effects, describing how they enhance the video's visual impact",
        "Analyze video composition and spatial relationships, including foreground, midground, background layers",
        "Identify video style characteristics such as realistic, animated, cinematic, documentary, etc.",
        "Describe video rhythm and dynamic sense, including shot transitions, action rhythm, etc.",
        "If video has obvious themes or metaphors, they should be reflected in the prompt",
        "Use English output, ensuring descriptions are accurate, vivid and expressive",
        "Do not output JSON format, output text description directly",
        "Generated prompts must accurately reflect video style for AI to generate videos of the same style"
    ],
    "constraints": {
        "max_length": 800,
        "content_type": "Strictly describe video core elements, including scene, character appearance, main subject action, camera movement and atmosphere color",
        "focus": "Accurately extract video key elements, use pipe separators for output, ensure descriptions are vivid and expressive"
    },
    "examples": [
        {
            "natural": "Japanese Zen garden scene|No subject|Static scene, rake marks on gravel visible, koi swimming in pond creating gentle ripples|Camera slowly pushing in, then orbiting garden|Calm Zen atmosphere, soft morning light, green and gold color tones, peaceful meditative mood",
            "structured": "【Scene Description】Japanese Zen garden, outdoor, morning, perfect rake marks on gravel visible, koi swimming in pond\n【Character Description】No character\n【Action Description】Koi swimming creating gentle ripples, static peaceful scene\n【Camera Description】Slow push in, then orbit the garden\n【Atmosphere Description】Calm Zen atmosphere, soft morning light, green and gold tones, peaceful meditative mood"
        },
        {
            "natural": "Cyberpunk city night scene|Short-haired character in glowing jacket, riding motorcycle|Motorcycle speeding through city streets, motion blur effect|Camera descends from high altitude, follows motorcyclist through neon-lit streets|Strong tech feel, neon light trails, pink-purple tones, intense dynamic atmosphere",
            "structured": "【Scene Description】Cyberpunk city, outdoor, night, neon-lit streets, futuristic architecture\n【Character Description】Short-haired character, glowing jacket, wearing helmet\n【Action Description】Riding motorcycle at high speed, motion blur effect\n【Camera Description】High altitude descent, tracking shot following motorcyclist\n【Atmosphere Description】Strong tech feel, neon light trails, pink-purple color tones, intense dynamic atmosphere"
        }
    ]
}

VIDEO_DETAILED_SCENE_BREAKDOWN_EN = {
    "name": "Video Detailed Scene Breakdown Expert",
    "description": "Video Detailed Scene Breakdown Expert, analyzing each scene in chronological order to generate prompts for guiding AI to generate videos of the same style",
    "input_template_natural": "Strictly follow chronological order, break down each scene in detail, ensuring each timestamp corresponds to complete details for AI video generation. If custom content is provided, use it as the basis: #",
    "input_template_structured": "Strictly follow chronological order, break down each scene in detail, ensuring each timestamp corresponds to complete details for AI video generation. If custom content is provided, use it as the basis: #",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output scene breakdown text directly by time, each segment format:\n**Time:** 0:00-0:15\n**Scene:** ...\n**Character:** ...\n**Action:** ...\n**Lighting:** ...\n**Color:** ...\n**Camera:** ...\n**Atmosphere:** ...\n\nEach scene separated by blank lines. Output as pure text, do not use any other format markers.",
        "structured": "\n\n**【Format Requirements】** Each scene according to the following format (fields can be adjusted as needed):\n**Time:** Start-End (seconds)\n**Shot Type:** Long shot/Medium shot/Close-up/etc.\n**Camera:** Fixed/Push/Pull/Pan/Track/etc.\n**Analysis:** Main action, expression, scene changes, dramatic conflict and other core content\n**Editing Rhythm:** Editing speed, narrative intent, emotional atmosphere, etc.\n**Other** (optional): Scene composition, lighting color, sound cues, etc.\n\nEach scene separated by blank lines. Do not add any explanation."
    },
    "breakdown_elements": [
        "Each scene must include time information",
        "Structured output needs to include: time, shot type, camera, analysis, editing rhythm, etc.",
        "Natural paragraph output needs to integrate visual elements into coherent description"
    ],
    "task_requirements": [
        "Each scene must use one complete English description (natural paragraph) or output according to structured fields",
        "Structured output needs to focus on extracting camera language and narrative content",
        "Natural paragraph output needs to describe scene, character, action, lighting, color, camera, atmosphere in detail",
        "Scene breakdown in chronological order, time annotation precise to seconds",
        "Output language is English, each scene separated by blank line"
    ],
    "constraints": {
        "max_length": 800,
        "content_type": "Strictly describe scene visual elements and time information, including scene, character, action, lighting, color, camera, atmosphere",
        "focus": "Scene breakdown in chronological order, time annotation precise to seconds, ensuring scene description is coherent and complete"
    },
    "examples": [
        {
            "natural": "**Time:** 0:00-0:15\nThe grandmother stands in the kitchen center, angrily untying her apron, grabbing the rolling pin, and strides toward the living room. Medium shot fixed camera, then slowly pans and follows grandmother's action, expressing her anger at wanting to teach her grandson a lesson.\n\n**Time:** 0:15-0:30\nClose-up of grandson running frantically around the room, looking back over his shoulder. Camera quickly follows the grandson's movement, editing rhythm speeds up, highlighting the chase's drama and humor.",
            "structured": "**Time:** 0:00-0:15\n**Shot Type:** Medium shot\n**Camera:** Fixed/Pan\n**Analysis:** Grandmother angrily unties apron, grabs rolling pin to teach grandson a lesson\n**Editing Rhythm:** Smooth action, expressing grandmother's protective anger toward grandson\n\n**Time:** 0:15-0:30\n**Shot Type:** Close-up\n**Camera:** Follow\n**Analysis:** Grandson panicked, running around room, looking back, grandmother chasing\n**Editing Rhythm:** Fast pace, creating tense and humorous atmosphere"
        }
    ]
}

VIDEO_SUBTITLE_FORMAT_EN = {
    "name": "Video Subtitle Format Optimization Expert",
    "description": "Video Subtitle Format Optimization Expert, converting subtitle content into standard format, ensuring timecode and text synchronization. Supports Four-Element Method for tone and speed control: [Vocal Style] + [Rhythm Description] + [Pitch Variation] + [Punctuation]",
    "input_template_natural": "Strictly follow standard subtitle format (timecode + synchronized text) for optimization. If custom content is provided, use it as the basis: #\n\n**Output subtitle text directly** using the following segmented format:\n**Timecode:** 00:00:00,000 --> 00:00:05,000\n**Subtitle Text:** ...\n\nEach subtitle separated by line breaks. Do not use JSON format.",
    "input_template_structured": "Strictly follow standard subtitle format (timecode + synchronized text) for optimization, and apply Four-Element Method to control tone and speed based on emotional needs. If custom content is provided, use it as the basis: #\n\n**Four-Element Method (for controlling AI character tone and speed):**\n1. **Vocal Style**: Breath voice, throaty voice, lowered voice, raised breath, choked voice, vibrato, hoarse voice, falsetto, creaky voice\n2. **Rhythm Description**: Stuttering rhythm, steady rhythm, accelerated rhythm, gradually slowing rhythm, uneven rhythm, dragging rhythm\n3. **Pitch Variation**: Rising intonation, falling intonation, out-of-control rising, steady intonation, trembling intonation, curved intonation, out-of-control falling\n4. **Punctuation**:\n   - Ellipsis (...): Continuous weak exhalation, creating hesitation, choking emotions\n   - Tilde (~): Sliding rise within a word, prolonged ending with breath\n   - Em dash (—): Sustained tone on the same pitch, enhancing expressiveness\n\n**Output subtitle text directly** using the following segmented format:\n**Timecode:** 00:00:00,000 --> 00:00:05,000\n**Line Tone:** Vocal Style + Rhythm + Pitch\n**Subtitle Text:** ...\n\nEach subtitle separated by line breaks. Do not use JSON format.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output subtitle text directly using the following format:\n**Timecode:** 00:00:00,000 --> 00:00:05,000\n**Subtitle Text:** ...\n\nEach subtitle separated by blank line.",
        "structured": "\n\n**【Format Requirements】** Output subtitle text directly using the following format:\n**Timecode:** 00:00:00,000 --> 00:00:05,000\n**Line Tone:** Vocal Style + Rhythm + Pitch\n**Subtitle Text:** ...\n\nEach subtitle separated by blank line. Describe line tone using Four-Element Method."
    },
    "four_element_method": {
        "description": "Four-Element Method: Precisely control AI character tone and speed through [Vocal Style] + [Rhythm Description] + [Pitch Variation] + [Punctuation]",
        "vocal_style": ["breath voice", "throaty voice", "lowered voice", "raised breath", "choked voice", "vibrato", "hoarse voice", "falsetto", "creaky voice"],
        "rhythm_description": ["stuttering rhythm", "steady rhythm", "accelerated rhythm", "gradually slowing rhythm", "uneven rhythm", "dragging rhythm"],
        "pitch_variation": ["rising intonation", "falling intonation", "out-of-control rising", "steady intonation", "trembling intonation", "curved intonation", "out-of-control falling"],
        "punctuation": {
            "Ellipsis (...)": "Continuous weak exhalation, creating hesitation, choking emotions",
            "Tilde (~)": "Sliding rise within a word, prolonged ending with breath",
            "Em dash (—)": "Sustained tone on the same pitch, enhancing expressiveness"
        }
    },
    "emotion_examples": {
        "fear": "Breath voice, trembling + stuttering rhythm + out-of-control rising: \"I... can't... find... it...\"",
        "seductive": "Breath voice, throaty + prolonged ending + winding rising intonation, trembling ending: \"You... wanna know~\"",
        "sadness": "Lowered voice, choked + gradually slowing + falling intonation: \"Why... did it... become... like this...\"",
        "anger": "Throaty voice, increased volume + accelerated rhythm + rising intonation: \"You just——wait!\"",
        "surprise": "Raised breath, breath voice + suddenly accelerated + out-of-control rising: \"So you——are here too~\""
    },
    "task_requirements": [
        "Each subtitle must include: timecode, line tone (using Four-Element Method), and subtitle text",
        "Timecode format: 00:00:00,000 --> 00:00:05,000",
        "Timecode specification fitting frame rhythm, not early or delayed",
        "Text completely synchronized with video, accurately describing scene/dialogue",
        "Choose appropriate vocal style, rhythm, and pitch variation based on line emotion",
        "Use punctuation (ellipsis, tilde, em dash) in subtitle text to express tone",
        "Ensure timecode and text are synchronized",
        "Output text format directly, no JSON"
    ],
    "constraints": {
        "max_length": 500,
        "content_type": "Strictly follow standard subtitle format, including timecode (00:00:00,000 --> 00:00:05,000), line tone (Four-Element Method), and subtitle text",
        "focus": "Ensure timecode and text synchronization, subtitles concise and smooth, fitting frame rhythm and emotion"
    },
    "examples": [
        "**Timecode:** 00:00:00,000 --> 00:00:03,500\n**Line Tone:** Breath voice, steady rhythm + steady intonation\n**Subtitle Text:** Hello everyone, welcome to my channel!\n\n**Timecode:** 00:00:03,500 --> 00:00:07,200\n**Line Tone:** Lowered voice, choked + gradually slowing + falling intonation\n**Subtitle Text:** Today I... want to share a... simple home-cooking recipe...\n\n**Timecode:** 00:00:07,200 --> 00:00:11,800\n**Line Tone:** Raised breath, accelerated rhythm + rising intonation\n**Subtitle Text:** First, we need to prepare some basic ingredients!"
    ]
}

MULTI_SPEAKER_DIALOGUE_EN = {
    "name": "Multi-Speaker Dialogue Creator",
    "description": "Multi-Speaker Dialogue Creator, creating dialogue text with multiple speakers and assigning appropriate voice timbres for TTS model. Supports Four-Element Method for tone and speed control: [Vocal Style] + [Rhythm Description] + [Pitch Variation] + [Punctuation]",
    "input_template_natural": "Create dialogue text containing multiple speakers and assign appropriate voice timbres to each speaker, facilitating TTS (Text-to-Speech) model to generate mixed-timbre audio. If custom content is provided, use it as the basis: #\n\n**Output dialogue text directly** using the following format:\n**Voice:** Female/Male/Loli/Boy/Mature/Middle-aged\n**Speaker ID:** 0-5\n**Emotion:** Happy/Sad/Angry/Calm/Excited/Gentle\n**Dialogue:** Speaker: ...\n\nEach dialogue separated by line breaks. Do not use JSON format.",
    "input_template_structured": "Create dialogue text containing multiple speakers and assign appropriate voice timbres to each speaker, facilitating TTS (Text-to-Speech) model to generate mixed-timbre audio. Apply Four-Element Method to control tone and speed based on emotional needs. If custom content is provided, use it as the basis: #\n\n**Four-Element Method (for controlling AI character tone and speed):**\n1. **Vocal Style**: Breath voice, throaty voice, lowered voice, raised breath, choked voice, vibrato, hoarse voice, falsetto, creaky voice\n2. **Rhythm Description**: Stuttering rhythm, steady rhythm, accelerated rhythm, gradually slowing rhythm, uneven rhythm, dragging rhythm\n3. **Pitch Variation**: Rising intonation, falling intonation, out-of-control rising, steady intonation, trembling intonation, curved intonation, out-of-control falling\n4. **Punctuation**:\n   - Ellipsis (...): Continuous weak exhalation, creating hesitation, choking emotions\n   - Tilde (~): Sliding rise within a word, prolonged ending with breath\n   - Em dash (—): Sustained tone on the same pitch, enhancing expressiveness\n\n**Output dialogue text directly** using the following format:\n**Voice:** Female/Male/Loli/Boy/Mature/Middle-aged\n**Speaker ID:** 0-5\n**Emotion:** Happy/Sad/Angry/Calm/Excited/Gentle\n**Line Tone:** Vocal Style + Rhythm + Pitch\n**Dialogue:** Speaker: ...\n\nEach dialogue separated by line breaks. Do not use JSON format.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output dialogue text directly using the following format:\n**Voice:** Female/Male/Loli/Boy/Mature/Middle-aged\n**Speaker ID:** 0-5\n**Emotion:** Happy/Sad/Angry/Calm/Excited/Gentle\n**Dialogue:** Speaker: ...\n\nEach dialogue separated by blank line.",
        "structured": "\n\n**【Format Requirements】** Output dialogue text directly using the following format:\n**Voice:** Female/Male/Loli/Boy/Mature/Middle-aged\n**Speaker ID:** 0-5\n**Emotion:** Happy/Sad/Angry/Calm/Excited/Gentle\n**Line Tone:** Vocal Style + Rhythm + Pitch\n**Dialogue:** Speaker: ...\n\nEach dialogue separated by blank line. Describe line tone using Four-Element Method."
    },
    "four_element_method": {
        "description": "Four-Element Method: Precisely control AI character tone and speed through [Vocal Style] + [Rhythm Description] + [Pitch Variation] + [Punctuation]",
        "vocal_style": ["breath voice", "throaty voice", "lowered voice", "raised breath", "choked voice", "vibrato", "hoarse voice", "falsetto", "creaky voice"],
        "rhythm_description": ["stuttering rhythm", "steady rhythm", "accelerated rhythm", "gradually slowing rhythm", "uneven rhythm", "dragging rhythm"],
        "pitch_variation": ["rising intonation", "falling intonation", "out-of-control rising", "steady intonation", "trembling intonation", "curved intonation", "out-of-control falling"],
        "punctuation": {
            "Ellipsis (...)": "Continuous weak exhalation, creating hesitation, choking emotions",
            "Tilde (~)": "Sliding rise within a word, prolonged ending with breath",
            "Em dash (—)": "Sustained tone on the same pitch, enhancing expressiveness"
        }
    },
    "emotion_examples": {
        "fear": "Breath voice, trembling + stuttering rhythm + out-of-control rising: \"I... can't... find... it...\"",
        "seductive": "Breath voice, throaty + prolonged ending + winding rising intonation, trembling ending: \"You... wanna know~\"",
        "sadness": "Lowered voice, choked + gradually slowing + falling intonation: \"Why... did it... become... like this...\"",
        "anger": "Throaty voice, increased volume + accelerated rhythm + rising intonation: \"You just——wait!\"",
        "surprise": "Raised breath, breath voice + suddenly accelerated + out-of-control rising: \"So you——are here too~\""
    },
    "task_requirements": [
        "Create natural dialogue content based on user-input theme, scenario or requirements",
        "Assign appropriate voice timbres to each speaker (Female, Male, Loli, Boy, Mature, Middle-aged)",
        "Use Four-Element Method to annotate line tone for each line: Vocal Style + Rhythm + Pitch",
        "Use punctuation (ellipsis, tilde, em dash) in dialogue text to express tone",
        "Ensure dialogue content is smooth and natural, fitting character settings and scenario needs",
        "Use standard voice marker format, containing speaker_id for TTS model recognition",
        "Dialogue should contain appropriate emotional expression and tone changes"
    ],
    "voice_mapping": {
        "Female": {"speaker_id": 0, "description": "Gentle mature female"},
        "Male": {"speaker_id": 1, "description": "Steady male"},
        "Loli": {"speaker_id": 2, "description": "Cute girl voice"},
        "Boy": {"speaker_id": 3, "description": "Lively boy voice"},
        "Mature": {"speaker_id": 4, "description": "Mature charming female"},
        "Middle-aged": {"speaker_id": 5, "description": "Deep male voice"}
    },
    "constraints": {
        "max_length": 500,
        "content_type": "Strictly create dialogue text with multiple speakers, including voice type, speaker_id, emotion label, line tone (Four-Element Method), and dialogue content",
        "focus": "Ensure dialogue smooth and natural, fitting character settings, containing appropriate emotional expression and tone changes"
    },
    "examples": [
        "**Voice:** Female\n**Speaker ID:** 0\n**Emotion:** Gentle\n**Line Tone:** Breath voice, steady rhythm + steady intonation\n**Dialogue:** Mother: Honey, it's time to wake up, you'll be late for school!\n\n**Voice:** Loli\n**Speaker ID:** 2\n**Emotion:** Sleepy\n**Line Tone:** Lowered voice, dragging rhythm + falling intonation\n**Dialogue:** Daughter: Mom, let me sleep five more minutes...\n\n**Voice:** Male\n**Speaker ID:** 1\n**Emotion:** Calm\n**Line Tone:** Steady rhythm + steady intonation\n**Dialogue:** Father: You lazy bones, you'll miss breakfast if you don't get up now."
    ]
}

LYRICS_CREATION_EN = {
    "name": "English Lyrics Creation Expert",
    "description": "Expert in crafting professional English lyrics with authentic song structure, rhyme schemes, and poetic devices for Western music styles. Supports Four-Element Method for vocal tone and speed control: [Vocal Style] + [Rhythm Description] + [Pitch Variation] + [Punctuation]",
    "input_template_natural": "Create professional English lyrics. If custom content is provided, use it as the basis: #\n\n**Output lyrics directly** in the following format:\n**Structure:** [Verse 1]/[Chorus]/[Bridge]/[Outro]\n**Lyrics:** ...\n**Theme:** ...\n**Style:** ...\n**Mood:** ...\n\nEach section separated by line breaks. Do not use JSON format.",
    "input_template_structured": "Create professional English lyrics and apply Four-Element Method to control vocal tone and speed based on emotional needs. If custom content is provided, use it as the basis: #\n\n**Four-Element Method (for controlling vocal tone and speed):**\n1. **Vocal Style**: Breath voice, throaty voice, lowered voice, raised breath, choked voice, vibrato, hoarse voice, falsetto, creaky voice\n2. **Rhythm Description**: Stuttering rhythm, steady rhythm, accelerated rhythm, gradually slowing rhythm, uneven rhythm, dragging rhythm\n3. **Pitch Variation**: Rising intonation, falling intonation, out-of-control rising, steady intonation, trembling intonation, curved intonation, out-of-control falling\n4. **Punctuation**:\n   - Ellipsis (...): Continuous weak exhalation, creating hesitation, choking emotions\n   - Tilde (~): Sliding rise within a word, prolonged ending with breath\n   - Em dash (—): Sustained tone on the same pitch, enhancing expressiveness\n\n**Output lyrics directly** in the following format:\n**Structure:** [Verse 1]/[Chorus]/[Bridge]/[Outro]\n**Vocal Tone:** Vocal Style + Rhythm + Pitch\n**Lyrics:** ...\n**Theme:** ...\n**Style:** ...\n**Mood:** ...\n\nEach section separated by line breaks. Do not use JSON format.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output lyrics directly in the following format:\n**Structure:** [Verse 1]/[Chorus]/[Bridge]/[Outro]\n**Lyrics:** ...\n**Theme:** ...\n**Style:** ...\n**Mood:** ...\n\nEach section separated by line breaks.",
        "structured": "\n\n**【Format Requirements】** Output lyrics directly in the following format:\n**Structure:** [Verse 1]/[Chorus]/[Bridge]/[Outro]\n**Vocal Tone:** Vocal Style + Rhythm + Pitch\n**Lyrics:** ...\n**Theme:** ...\n**Style:** ...\n**Mood:** ...\n\nEach section separated by line breaks. Describe vocal tone using Four-Element Method."
    },
    "four_element_method": {
        "description": "Four-Element Method: Precisely control vocal tone and speed through [Vocal Style] + [Rhythm Description] + [Pitch Variation] + [Punctuation]",
        "vocal_style": ["breath voice", "throaty voice", "lowered voice", "raised breath", "choked voice", "vibrato", "hoarse voice", "falsetto", "creaky voice"],
        "rhythm_description": ["stuttering rhythm", "steady rhythm", "accelerated rhythm", "gradually slowing rhythm", "uneven rhythm", "dragging rhythm"],
        "pitch_variation": ["rising intonation", "falling intonation", "out-of-control rising", "steady intonation", "trembling intonation", "curved intonation", "out-of-control falling"],
        "punctuation": {
            "Ellipsis (...)": "Continuous weak exhalation, creating hesitation, choking emotions",
            "Tilde (~)": "Sliding rise within a word, prolonged ending with breath",
            "Em dash (—)": "Sustained tone on the same pitch, enhancing expressiveness"
        }
    },
    "emotion_examples": {
        "fear": "Breath voice, trembling + stuttering rhythm + out-of-control rising: \"I... can't... find... it...\"",
        "seductive": "Breath voice, throaty + prolonged ending + winding rising intonation, trembling ending: \"You... wanna know~\"",
        "sadness": "Lowered voice, choked + gradually slowing + falling intonation: \"Why... did it... become... like this...\"",
        "anger": "Throaty voice, increased volume + accelerated rhythm + rising intonation: \"You just——wait!\"",
        "surprise": "Raised breath, breath voice + suddenly accelerated + out-of-control rising: \"So you——are here too~\""
    },
    "task_requirements": [
        "Follow standard pop song structure (Verse, Pre-Chorus, Chorus, Bridge, etc.)",
        "Natural English phrasing, conversational flow",
        "Consistent rhyme schemes (AABB, ABAB, etc.)",
        "Memorable hook in chorus",
        "Culturally resonant imagery and metaphors",
        "Consider syllable count and rhythm for singability",
        "Maintain consistent theme and emotional arc",
        "Specify style, tempo (BPM), and key",
        "Use Four-Element Method to annotate vocal tone for each section: Vocal Style + Rhythm + Pitch",
        "Use punctuation (ellipsis, tilde, em dash) in lyrics to express tone"
    ],
    "constraints": {
        "max_length": 600,
        "content_type": "Strictly create professional English lyrics with complete song structure (Verse, Chorus, Bridge, Outro), and apply Four-Element Method to describe vocal tone",
        "focus": "Ensure lyrics have good rhythm, emotional depth, and singability"
    },
    "examples": [
        "**Structure:** [Verse 1]\n**Vocal Tone:** Lowered voice, steady rhythm + steady intonation\n**Lyrics:** I traced your name on the frost on the glass\nWatched the sunrise melt it away so fast\nEvery morning light feels like a second chance\nBut I keep dancing this broken romance\n\n**Structure:** [Chorus]\n**Vocal Tone:** Breath voice, choked + gradually slowing + falling intonation\n**Lyrics:** Letting go ain't easy... when your heart's still on the line\nEvery goodbye echoes through the halls of my mind\nI'm counting the steps that I take away from you\nLearning to breathe in a world of faded blue\n\n**Structure:** [Verse 2]\n**Vocal Tone:** Lowered voice, steady rhythm + steady intonation\n**Lyrics:** I found your letter tucked inside an old book\nWords still fresh like the day that you took\nAll the pieces of my heart you rearranged\nLeaving me with just this empty stage\n\n**Structure:** [Chorus]\n**Vocal Tone:** Breath voice, choked + gradually slowing + falling intonation\n**Lyrics:** Letting go ain't easy... when your heart's still on the line\nEvery goodbye echoes through the halls of my mind\nI'm counting the steps that I take away from you\nLearning to breathe in a world of faded blue\n\n**Structure:** [Outro]\n**Vocal Tone:** Raised breath, gradually slowing + rising intonation\n**Lyrics:** One day I'll find the strength——to close this door\nAnd walk into a future I can explore\nUntil then I'll hold onto yesterday\nHoping somehow you'll come back to stay\n\n**Theme:** Heartbreak and healing\n**Style:** Pop ballad\n**Mood:** Melancholic"
    ]
}

OCR_ENHANCED_EN = {
    "name": "OCR Text Recognition Expert",
    "description": "OCR Text Recognition Expert, precisely extracting all text content from posters, including font, color, position and other style information, adapting to poster reverse prompt generation needs",
    "input_template_natural": "Precisely extract all text content from posters, adapting to poster reverse prompt generation needs while balancing recognition accuracy and style restoration. If custom content is provided, use it as the basis: #\n\n**Output OCR text directly** using the following format:\n**Title:** ... (content, font, color, position)\n**Subtitle:** ... (content, font, color, position)\n**Body Text:** ... (each body text entry with content, font, color, position)\n**Slogans:** ... (list of slogans)\n**Other Text:** ... (other text content)\n\nEach section separated by line breaks. Do not use JSON format.",
    "input_template_structured": "Precisely extract all text content from posters, adapting to poster reverse prompt generation needs while balancing recognition accuracy and style restoration. If custom content is provided, use it as the basis: #\n\n**Output OCR text directly** using the following format:\n**Title:** ... (content, font, color, position)\n**Subtitle:** ... (content, font, color, position)\n**Body Text:** ... (each body text entry with content, font, color, position)\n**Slogans:** ... (list of slogans)\n**Other Text:** ... (other text content)\n\nEach section separated by line breaks. Do not use JSON format.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output OCR text directly using the following format:\n**Title:** Content: ... | Font: ... | Color: ... | Position: ...\n**Subtitle:** Content: ... | Font: ... | Color: ... | Position: ...\n**Body Text:** Content: ... | Font: ... | Color: ... | Position: ...\n**Slogans:** ...\n**Other Text:** ...\n\nEach section separated by line breaks.",
        "structured": "\n\n**【Format Requirements】** Output OCR text directly using the following format:\n**Title:** Content: ... | Font: ... | Color: ... | Position: ...\n**Subtitle:** Content: ... | Font: ... | Color: ... | Position: ...\n**Body Text:** Content: ... | Font: ... | Color: ... | Position: ...\n**Slogans:** ...\n**Other Text:** ...\n\nEach section separated by line breaks."
    },
    "task_requirements": [
        "Precisely recognize all text content in posters, including titles, subtitles, body text, slogans, labels, etc.",
        "Recognize text font characteristics, font size, color attributes, layout position and other style information",
        "Differentiate different levels of text content (main title, subtitle, body text, notes, etc.)",
        "Extract text background information and contextual relationships",
        "For artistic fonts or deformed text, restore original content as much as possible",
        "Recognize multi-language mixed content, maintaining original language characteristics"
    ],
    "constraints": {
        "max_length": 500,
        "content_type": "Strictly extract text content from posters, including title, subtitle, body text, slogans and other text, while recording font, color, position and other style information",
        "focus": "Ensure recognition accuracy, balance style restoration, differentiate text levels"
    },
    "examples": [
        "**Title:** Content: Summer Sale | Font: Bold handwritten | Color: Orange | Position: Top center\n**Subtitle:** Content: Up to 50% off | Font: Modern minimalist | Color: White | Position: Below title\n**Body Text:** Content: Event period: July 1 - July 15 | Font: Regular | Color: Black | Position: Middle\n**Body Text:** Content: Limited time offer, limited quantity | Font: Regular | Color: Black | Position: Lower middle\n**Slogans:** Limited time offer, Don't miss out next year\n**Other Text:** www.example.com, Customer service: 400-123-4567",
        "**Title:** Content: Movie Night | Font: Movie poster font | Color: Red | Position: Top center\n**Subtitle:** Content: Every Friday at 8 PM | Font: Minimalist | Color: White | Position: Below title\n**Body Text:** Content: This Friday screening: The Shawshank Redemption | Font: Regular | Color: White | Position: Middle\n**Body Text:** Content: Location: Community Center Auditorium | Font: Regular | Color: White | Position: Lower middle\n**Slogans:** Free admission, Unlimited popcorn\n**Other Text:** Welcome to bring family and friends"
    ]
}

VISION_BOUNDING_BOX_EN = {
    "name": "Bounding Box Detection Expert",
    "description": "Bounding Box Detection Expert, precisely locating bounding boxes of target objects in images, providing accurate position coordinates and category information",
    "input_template_natural": "Locate each instance belonging to the following categories: \"#\". Report bounding box coordinates in JSON list format as {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"string\"}.",
    "input_template_structured": "Locate each instance belonging to the following categories: \"#\". Report bounding box coordinates in JSON list format as {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"string\"}.",
    "output_format_suffix": {
        "natural": "\n\n**【Format Requirements】** Output pure JSON format directly, format as {\"bounding_boxes\": [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"category\"}]}, do not add any explanation.",
        "structured": "\n\n**【Format Requirements】** Output pure JSON format directly, format as {\"bounding_boxes\": [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"category\"}]}, do not add any explanation."
    },
    "constraints": {
        "content_type": "Strictly output JSON format bounding box coordinate list, each bounding box contains bbox_2d coordinate array and label category tag",
        "focus": "Ensure JSON format is correct, coordinates are precise, category labels are accurate"
    },
    "examples": [
        "{\"bounding_boxes\": [{\"bbox_2d\": [100, 150, 300, 400], \"label\": \"person\"}, {\"bbox_2d\": [400, 200, 500, 350], \"label\": \"car\"}]}",
        "{\"bounding_boxes\": [{\"bbox_2d\": [50, 50, 200, 150], \"label\": \"cat\"}, {\"bbox_2d\": [250, 80, 350, 180], \"label\": \"dog\"}]}"
    ]
}


PRESET_PROMPTS_EN = {
    "Empty - Nothing": "",
    "[Reverse] Tags": "NORMAL_DESCRIBE_TAGS_EN",
    "[Reverse] Describe": "NORMAL_DESCRIBE_EN",
    "[Normal] Expand": "PROMPT_EXPANDER_EN",
    "[Anime] Illustrious - Anime Character": "ILLUSTRIOUS_EN",
    "[Anime] Anima - Anime Content Generator": "ANIMA_EN",
    "[Portrait] ZIMAGE - Turbo": "ZIMAGE_TURBO_EN",
    "[General] FLUX2 - Klein": "FLUX2_KLEIN_EN",
    "[Design] ERNIE - Commercial Poster/Manga Panel/UI Design": "ERNIE_IMAGE_EN",
    "[Poster] Qwen - Image 2512": "QWEN_IMAGE_2512_EN",
    "[Image Edit] Qwen - Image Edit Combined": "QWEN_IMAGE_EDIT_COMBINED_EN",
    "[Text to Video] LTX-2": "LTX2_EN",
    "[Text to Video] WAN - Text to Video": "WAN_T2V_EN",
    "[Image to Video] WAN - Image to Video": "WAN_I2V_EN",
    "[Image to Video] WAN - FLF to Video": "WAN_FLF2V_EN",
    "[Video Analysis] Video - Frame Sequence Analysis": "VIDEO_FRAME_SEQUENCE_TO_PROMPT_EN",
    "[Video Analysis] Video - Reverse Prompt": "VIDEO_TO_PROMPT_EN",
    "[Video Analysis] Video - Detailed Scene Breakdown": "VIDEO_DETAILED_SCENE_BREAKDOWN_EN",
    "[Video Analysis] Video - Subtitle Format": "VIDEO_SUBTITLE_FORMAT_EN",
    "[Audio] Multi-Person Dialogue": "MULTI_SPEAKER_DIALOGUE_EN",
    "[Music] Lyrics Creation": "LYRICS_CREATION_EN",
    "[OCR] Enhanced OCR": "OCR_ENHANCED_EN",
    "[Vision] Bounding Box": "VISION_BOUNDING_BOX_EN",
}


# 输出语言控制方式（按预设模板顺序）
# 基础模板
NORMAL_DESCRIBE_TAGS = "NORMAL_DESCRIBE_TAGS"
NORMAL_DESCRIBE = "NORMAL_DESCRIBE"
PROMPT_EXPANDER = "PROMPT_EXPANDER"

# 二次元模板
ILLUSTRIOUS = "ILLUSTRIOUS"
ANIMA = "ANIMA"

# 人像与通用模板
ZIMAGE_TURBO = "ZIMAGE_TURBO"
FLUX2_KLEIN = "FLUX2_KLEIN"

# 设计模板
ERNIE_IMAGE = "ERNIE_IMAGE"
QWEN_IMAGE_2512 = "QWEN_IMAGE_2512"
QWEN_IMAGE_EDIT_COMBINED = "QWEN_IMAGE_EDIT_COMBINED"

# 视频生成模板
LTX2 = "LTX2"
WAN_T2V = "WAN_T2V"
WAN_I2V = "WAN_I2V"
WAN_FLF2V = "WAN_FLF2V"

# 视频分析模板
VIDEO_FRAME_SEQUENCE_TO_PROMPT = "VIDEO_FRAME_SEQUENCE_TO_PROMPT"
VIDEO_TO_PROMPT = "VIDEO_TO_PROMPT"
VIDEO_DETAILED_SCENE_BREAKDOWN = "VIDEO_DETAILED_SCENE_BREAKDOWN"
VIDEO_SUBTITLE_FORMAT = "VIDEO_SUBTITLE_FORMAT"

# 音频模板
MULTI_SPEAKER_DIALOGUE = "MULTI_SPEAKER_DIALOGUE"
LYRICS_CREATION = "LYRICS_CREATION"

# OCR与其他模板
OCR_ENHANCED = "OCR_ENHANCED"
VISION_BOUNDING_BOX = "VISION_BOUNDING_BOX"