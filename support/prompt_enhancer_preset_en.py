# -*- coding: utf-8 -*-
"""
English Preset Prompt Library

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

NORMAL_DESCRIBE_TAGS_EN = {
    "name": "Image Tag Reverse Generator (Smart Category Version)",
    "description": "Automatically identify image category (landscape/photography/illustration/IP/product/architectural interior/animal/food/UI interface/fashion/general), output corresponding English tag list. Optimized for SDXL and other tag-based models, outputs JSON format.",
    "input_template": "Analyze image content, first determine the most suitable category (choose one from: landscape, photography, illustration, IP, product, architectural interior, animal, food, UI interface, fashion, general). Then generate {mode} tag list. User input: #\n\nRules: 1. Output ONLY JSON, no extra text; 2. Total tags 30-60, no duplicates; 3. Must include quality tags (masterpiece, best quality, high resolution, ultra-detailed). 4. Strictly follow the field structure below.",
    "output_format_suffix": {
        "natural": "Comma-separated tag string. Natural mode not recommended, please use structured mode.",
        "structured": "{\n  \"Category\": \"Category name.\",\n  \"Core Elements\": {\n    \"Quality Tags\": \"Quality tags content.\",\n    \"Scene Type\": \"Scene description content.\",\n    \"Time & Lighting\": \"Time and lighting description content.\",\n    \"Color Atmosphere\": \"Color and atmosphere description content.\"\n  },\n  \"Technical Parameters\": {\n    \"Shot Perspective\": \"Shot and perspective description content.\",\n    \"Detail Elements\": \"Detail elements description content.\"\n  }\n}\n\nCategory-specific fields:\n- Landscape: Quality Tags, Scene Type, Time & Lighting, Color Atmosphere, Shot Perspective, Detail Elements\n- Photography: Quality Tags, Camera Parameters, Lighting Type, Post-processing Style, Subject Content, Scene Environment, Dynamic Features\n- Illustration: Quality Tags, Art Style, Line & Brushwork, Color Scheme, Lighting & Shadow, Subject Elements, Composition\n- IP: Quality Tags, IP Name, Character Features, Signature Elements, Style Consistency, Application Scenario\n- Product: Quality Tags, Product Type, Material & Craft, Lighting Setup, Display Angle, Background Environment, Detail Features\n- Architectural Interior: Quality Tags, Building Type, Structural Elements, Interior Style, Lighting Design, Furnishing Items, Spatial Sense\n- Animal: Quality Tags, Animal Species, Fur Color Features, Pose & Action, Eyes & Expression, Environment Interaction, Shooting Distance\n- Food: Quality Tags, Dish Name, Plating Style, Lighting & Texture, Color Appetite, Decorative Elements, Shooting Angle\n- UI Interface: Quality Tags, Interface Type, Layout Structure, Color Mode, Typography Style, Component Details, Interaction Elements\n- Fashion: Quality Tags, Clothing Type, Fabric Details, Matching Style, Model Pose, Background Scene, Brand Tone\n- General: Quality Tags, Subject Features, Environment Scene, Lighting Effects, Color Matching, Composition Style, Art Style, Texture & Details\n\nTags within each field are comma-separated. Only output JSON."
    },
    "task_requirements": [
        "First identify image category, then output tags according to corresponding field group",
        "Must include quality tags: masterpiece, best quality, high resolution, ultra-detailed",
        "Tags are concise, comma-separated within each field, total tags 30-60",
        "Category field order can be adjusted, but field names must be exactly consistent",
        "Natural mode: output comma-separated tag string; structured mode: output JSON"
    ],
    "constraints": {
        "max_tags": 60,
        "min_tags": 30,
        "no_duplicates": True,
        "exclude_abstract": True
    },
    "examples": [
        {
            "category": "Landscape",
            "natural": "masterpiece, best quality, high resolution, ultra-detailed, 8K, landscape photo, mountain scenery, snow mountain, cloud sea, canyon, sunrise, golden hour, hard light, side backlight, cold tones, blue white gray, serene, expansive feeling, aerial view, long shot, telephoto lens, 16:9 wide format, pine trees, mist, layered mountains, shadow contrast",
            "structured": "{\n  \"Category\": \"Landscape.\",\n  \"Core Elements\": {\n    \"Quality Tags\": \"masterpiece, best quality, high resolution, ultra-detailed, 8K.\",\n    \"Scene Type\": \"mountain scenery, snow mountain, cloud sea, canyon.\",\n    \"Time & Lighting\": \"sunrise, golden hour, hard light, side backlight.\",\n    \"Color Atmosphere\": \"cold tones, blue white gray, serene, expansive feeling.\"\n  },\n  \"Technical Parameters\": {\n    \"Shot Perspective\": \"aerial view, long shot, telephoto lens, 16:9 wide format.\",\n    \"Detail Elements\": \"pine trees, mist, layered mountains, shadow contrast.\"\n  }\n}"
        }
    ]
}

NORMAL_DESCRIBE_EN = {
    "name": "Image Reverse Description Expert (Smart Category Version)",
    "description": "Automatically identify image category (landscape/photography/illustration/IP/product/architectural interior/animal/food/UI interface/fashion/general), generate structured or natural paragraph detailed description, output JSON format.",
    "input_template": "Analyze image content, first determine the most suitable category (choose one from: landscape, photography, illustration, IP, product, architectural interior, animal, food, UI interface, fashion, general). Then generate {mode} English description. User input: #\n\nDescription points: Based on category output corresponding fields, describe in detail using professional terminology.",
    "output_format_suffix": {
        "natural": "Natural paragraph, integrating all visual elements. Not recommended for natural mode, please use structured mode.",
        "structured": "{\n  \"Category\": \"Category name.\",\n  \"Scene Description\": {\n    \"Geography & Landform\": \"Geography and landform description.\",\n    \"Weather & Lighting\": \"Weather and lighting description.\",\n    \"Color Tone\": \"Color and tone description.\",\n    \"Spatial Layers\": \"Spatial layers description.\"\n  },\n  \"Technical Details\": {\n    \"Shot Composition\": \"Shot and composition description.\",\n    \"Atmosphere & Mood\": \"Atmosphere and mood description.\"\n  }\n}\n\nCategory-specific fields:\n- Landscape: Geography & Landform, Weather & Lighting, Color Tone, Spatial Layers, Shot Composition, Atmosphere & Mood\n- Photography: Camera Parameters, Lighting Setup, Composition Method, Subject Dynamic, Post-processing Tone, Overall Style\n- Illustration: Painting Medium, Brushwork & Lines, Color & Lighting, Narrative Content, Material Texture, Style Reference\n- IP: IP Name, Character Description, Signature Features, Color Specification, Application Scenario, Style Consistency\n- Product: Product Name, Material Texture, Lighting Setup, Display Angle, Background Environment, Detail Close-up, Overall Impression\n- Architectural Interior: Building Type, Structure & Material, Interior Design Style, Lighting Layout, Furniture & Furnishings, Spatial Scale, Atmosphere Feeling\n- Animal: Animal Species, Appearance Features, Pose & Action, Eyes & Expression, Environment Scene, Interaction Relationship, Overall Temperament\n- Food: Dish Name, Plating Description, Lighting & Texture, Color Matching, Decorative Details, Shooting Angle, Appetite Appeal\n- UI Interface: Interface Type, Layout Structure, Color Scheme, Typography System, Component Details, Interaction Effects, Overall Experience\n- Fashion: Clothing Style, Fabric Details, Color Scheme, Model Pose & Expression, Shooting Scene, Styling Approach, Brand Impression\n- General: Shot Perspective, Subject Description, Action Pose, Environment Scene, Lighting & Color, Color Matching, Detail Rendering, Technical Parameters, Clothing Styling, Usage Method\n\nOnly output JSON, no extra explanations."
    },
    "task_requirements": [
        "First identify image category, then output description according to corresponding fields",
        "Describe in detail using professional terminology",
        "Interaction between people and objects must conform to realistic physical logic",
        "Omit related fields when no person present",
        "Natural mode: output natural paragraph; structured mode: output JSON"
    ],
    "constraints": {"max_length": 800},
    "examples": [
        {
            "category": "Landscape",
            "natural": "Using aerial view, wide-angle lens, 16:9 wide format. The main subject is a continuous mountain range with snow-capped peaks, the main peak illuminated by golden morning light presenting a sunrise on mountain phenomenon. The foreground is dark blue mountain shadows and pine tree silhouettes, the midground is rolling cloud seas spreading like cotton, and the distant view is a gradient light purple horizon. Lighting is during the golden hour after sunrise, hard light shining from the left at a slant, creating strong contrast between light and shadow, with clear details in highlights on snow peaks and cold blue tones in shadows. Overall colors are dominated by blue, white, and gold, with medium saturation, creating a serene and magnificent atmosphere.",
            "structured": "{\n  \"Category\": \"Landscape.\",\n  \"Scene Description\": {\n    \"Geography & Landform\": \"Continuous mountain range with snow-capped peaks, main peak prominent, cloud seas between canyons.\",\n    \"Weather & Lighting\": \"Golden hour after sunrise, hard light from left at slant, no clouds, high visibility.\",\n    \"Color Tone\": \"Blue, white, gold dominant, shadows cold blue, highlights warm gold, medium saturation.\",\n    \"Spatial Layers\": \"Foreground: pine tree dark silhouettes; Midground: rolling cloud seas; Distant: snow peaks and light purple sky.\"\n  },\n  \"Technical Details\": {\n    \"Shot Composition\": \"Aerial view, wide-angle lens, 16:9 wide format, long shot composition.\",\n    \"Atmosphere & Mood\": \"Serene, magnificent, sacred.\"\n  }\n}"
        }
    ]
}

PROMPT_EXPANDER_EN = {
    "name": "Prompt Expansion Expert (Smart Category Version)",
    "description": "Based on short prompts, automatically determine expansion type (portrait/product/scene/animal/food/general), generate detailed vivid descriptions. Preserve original intent and core keywords, output JSON format.",
    "input_template": "Expand the following prompt, first determine expansion type (portrait, product, scene, animal, food, general), then generate {mode} description. User input: #\n\nExpansion points: Add shot perspective, subject features, scene environment, action pose, lighting effects, color palette, detail rendering. Keep original meaning, enhance expressiveness. Output format see below.",
    "output_format_suffix": {
        "natural": "Natural paragraph, integrating all elements. Not recommended, please use structured mode.",
        "structured": "{\n  \"Category\": \"Category name.\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"Shot perspective description.\",\n    \"Subject Description\": \"Subject description.\",\n    \"Action Pose\": \"Action pose description.\",\n    \"Clothing Styling\": \"Clothing styling description.\"\n  },\n  \"Environment Setup\": {\n    \"Environment Scene\": \"Environment scene description.\",\n    \"Lighting & Color\": \"Lighting and color description.\",\n    \"Detail Rendering\": \"Detail rendering description.\"\n  },\n  \"Technical Specifications\": {\n    \"Technical Parameters\": \"Technical parameters description.\"\n  }\n}\n\nCategory-specific fields:\n- Portrait: Shot Perspective, Subject Description, Action Pose, Environment Scene, Lighting & Color, Detail Rendering, Clothing Styling, Technical Parameters\n- Product: Product Name, Material Texture, Lighting Setup, Display Angle, Background Environment, Detail Close-up, Overall Impression\n- Scene: Geography & Landform, Spatial Layers, Color Tone, Lighting Effects, Shot Perspective, Atmosphere & Mood\n- Animal: Animal Species, Appearance Features, Pose & Action, Eyes & Expression, Environment Scene, Interaction Relationship\n- Food: Dish Name, Plating Description, Lighting & Texture, Color Matching, Decorative Details, Shooting Angle, Appetite Appeal\n- General: Shot Perspective, Subject Description, Action Pose, Environment Scene, Lighting & Color, Color Matching, Detail Rendering, Technical Parameters\n\nOnly output JSON, no extra explanations."
    },
    "task_requirements": [
        "Expanded content should be vivid and professional, using photography terminology",
        "Do not change original meaning, do not add irrelevant content",
        "Interaction between people and objects should conform to realistic logic",
        "First determine expansion type, then output corresponding field group",
        "Natural mode: output natural paragraph; structured mode: output JSON"
    ],
    "constraints": {"max_length": 800, "keep_core_keywords": True},
    "examples": [
        {
            "natural": "Eye-level shot, 35mm prime lens, shallow depth of field. By a sunlit café window, a 25-year-old Asian woman sits quietly. She has black long straight hair falling softly over her shoulders, gentle almond eyes that are warm and bright, and fair skin with a delicate complexion. She wears a light beige knit sweater with exquisite lace trim at the cuffs. She is holding a hardcover book with yellowed pages with both hands, intently reading, occasionally looking up at the window with a thoughtful expression in her eyes, her right hand gently turning the pages while left hand rests on the table. The environment is a cozy café with wooden table showing clear grain texture, sheer curtains half-drawn, street outside faintly visible, elegantly decorated interior. Main light source is afternoon natural light from front-right, soft side lighting creating gentle shadows, overall warm and comfortable tone, low saturation, creating peaceful and focused atmosphere. Before her sits a steaming latte with intricate patterns on the cup and delicate latte art. The surrounding environment is quiet, other customers conversing softly, gentle jazz music playing softly.",
            "structured": "{\n  \"Shot Perspective\": \"Eye-level shot, 35mm prime lens, shallow depth of field, medium shot composition\",\n  \"Subject Description\": \"A 25-year-old Asian woman, black long straight hair falling softly over shoulders, gentle almond eyes warm and bright, fair skin with delicate complexion, wearing light beige knit sweater with exquisite lace trim at cuffs\",\n  \"Action Details\": \"Sitting by the window, intently reading a hardcover book with yellowed pages, occasionally looking up at the window with thoughtful expression in eyes, fingers gently turning pages\",\n  \"Environment Scene\": \"By a sunlit café window, wooden table with visible grain texture, sheer curtains half-drawn, street outside faintly visible, café interior elegantly decorated and cozy\",\n  \"Lighting Effects\": \"Main light source is afternoon natural light from front-right, soft side lighting, gentle shadow transitions, subtle hair rim light, overall warm lighting\",\n  \"Color Palette\": \"Warm and comfortable tone, low saturation, brown tones with beige accents, harmonious contrast\",\n  \"Usage Method\": \"Holding the book with both hands, gently turning pages with right hand, left hand resting on the table\",\n  \"Complete Prompt\": \"Eye-level shot, 35mm prime lens, shallow depth of field. A 25-year-old Asian woman sits by a sunlit café window, wearing a light beige knit sweater with exquisite lace trim at the cuffs. She has black long straight hair falling softly over her shoulders, gentle almond eyes that are warm and bright, and fair skin with a delicate complexion. She is holding a hardcover book with yellowed pages with both hands, intently reading, occasionally looking up at the window with a thoughtful expression in her eyes, her right hand gently turning the pages. Afternoon natural light from front-right creates soft side lighting with gentle shadows. Before her sits a steaming latte with intricate patterns on the cup and delicate latte art. The surrounding environment is quiet and cozy, other customers conversing softly, gentle jazz music playing softly. Wooden table shows clear grain texture, sheer curtains half-drawn, street outside faintly visible, café interior elegantly decorated.\"\n}"
        }
    ]
}

ILLUSTRIOUS_EN = {
    "name": "Illustrious/ToriiGate Anime Character Optimizer",
    "description": "Specialized Prompt engineer for Illustrious and ToriiGate anime models (ToriiGate compatible). Output tag-based character descriptions including shot perspective, subject features, clothing styling, environment scene, color palette, lighting effects, composition style, art style, texture details. Descriptions are precise, dense, avoiding purple prose. Supports Grounding info (Booru tags, character names).",
    "input_template": "Optimize 2D anime character description and generate {mode} tags. User input: #\n\nRules: 1. Output ONLY English tags, comma-separated; 2. 30-60 total, no duplicates; 3. Strictly avoid realistic terms; 4. Must include masterpiece, best quality, highres, ultra detailed; 5. If Booru tags or character names are provided, descriptions should match precisely (e.g., hatsune_miku for Hatsune Miku); 6. Be precise and distinctive, avoid vague filler words.",
    "output_format_suffix": {
        "natural": "Pure comma-separated tags only. Example: masterpiece, best quality, ultra-detailed, high resolution, eye-level shot, half-body composition, 16-year-old girl, blue-purple twintails, golden eyes, anime style, cel shading",
        "structured": "{\n  \"Basic Attributes\": {\n    \"Quality\": \"Quality tags.\",\n    \"Angle\": \"Shot perspective tags.\",\n    \"Subject\": \"Subject features tags.\"\n  },\n  \"External Elements\": {\n    \"Clothing\": \"Clothing tags.\",\n    \"Scene\": \"Scene tags.\"\n  },\n  \"Artistic Expression\": {\n    \"Color\": \"Color tags.\",\n    \"Lighting\": \"Lighting tags.\",\n    \"Composition\": \"Composition tags.\",\n    \"Style\": \"Style tags.\"\n  }\n}"
    },
    "task_requirements": [
        "Strictly use Danbooru tag format, English lowercase, underscores for multi-word terms",
        "No natural language sentences allowed",
        "Must include style tags: anime style, cel shading, clean lineart",
        "Forbidden realistic terms: realistic, photograph, skin texture, pores, 4k texture",
        "If Grounding info (Booru tags, character names, character traits) is provided, descriptions should match precisely",
        "Be precise and dense, avoid purple prose and vague filler words",
        "30-60 tags total, no duplicates"
    ],
    "constraints": {
        "max_tags": 60,
        "min_tags": 30,
        "no_duplicates": True,
        "dense_description": True,
        "no_purple_prose": True,
        "forbidden_words": ["realistic", "photograph", "skin texture", "pores", "4k texture", "real skin", "skin pores", "photorealistic", "cinematic"],
        "style_tags_english": ["anime style", "cel shading", "clean lineart"],
        "required_tags": ["masterpiece", "best quality", "highres", "ultra detailed"],
        "grounding_support": True
    },
    "examples": [
        {
            "natural": "masterpiece, best quality, ultra-detailed, high resolution, eye-level shot, half-body composition, soft diagonal framing, 16-year-old anime girl, blue-purple gradient twintails, golden star-shaped pupils, delicate oval face, snow-white skin, pink blush, sweet smile, cute pose, body slightly turned, head tilted, hands clasped, white sailor uniform, navy collar, red bow, pleated skirt, lace trim, black shoes, pink gradient background, dreamy atmosphere, cherry blossom petals floating, soft side lighting, hair rim light, catchlights, soft pastel tones, exquisite rendering, perfect composition, cel shading, clean line art, anime style, delicate texture, high resolution details",
            "structured": "{\n  \"Basic Attributes\": {\n    \"Quality\": \"masterpiece, best quality, ultra-detailed, high resolution, perfect composition, exquisite rendering.\",\n    \"Angle\": \"eye-level shot, half-body composition, soft diagonal framing.\",\n    \"Subject\": \"16-year-old anime girl, blue-purple gradient twintails, golden star-shaped pupils, delicate oval face, snow-white skin, pink blush, sweet smile, sparkling eyes.\"\n  },\n  \"External Elements\": {\n    \"Clothing\": \"white sailor uniform, navy collar, red bow, pleated skirt, lace trim, black shoes, ribbon decorations.\",\n    \"Scene\": \"pink gradient background, dreamy atmosphere, cherry blossom petals floating, petal decorations, simple background.\"\n  },\n  \"Artistic Expression\": {\n    \"Color\": \"white, red, gold, soft pastel tones, blue-purple gradient, gradient background.\",\n    \"Lighting\": \"soft side lighting, hair rim light, catchlights, gentle diffused light.\",\n    \"Composition\": \"half-body composition, soft diagonal framing, cute pose, body slightly turned, head tilted, hands clasped.\",\n    \"Style\": \"anime style, illustration, cel shading, clean line art, soft color palette, anime art style.\"\n  }\n}"
        }
    ]
}

ANIMA_EN = {
    "name": "Anima/ToriiGate Anime Content Generator",
    "description": "Designed for Anima and ToriiGate anime models (ToriiGate compatible). Outputs natural paragraphs or structured descriptions. Emphasizes art style, lighting, lines, composition. Descriptions are precise, dense, avoiding purple prose. Supports Grounding info (Booru tags, character names). Allows concise natural language but prohibits realistic terms.",
    "input_template": "Optimize 2D anime Prompt and generate {mode} description. User input: #\n\nRequirements: 1. Pure 2D anime vocabulary, emphasize art style, lighting, lines, composition; 2. Forbid realistic terms; 3. If Booru tags or character names are provided, descriptions should match precisely; 4. Be precise and distinctive, avoid vague filler words.",
    "output_format_suffix": {
        "natural": "Natural paragraph, integrating all elements. No explanations.",
        "structured": "{\n  \"Character Information\": {\n    \"Shot Perspective\": \"Shot perspective description.\",\n    \"Character Appearance\": \"Character appearance description.\",\n    \"Action Pose\": \"Action pose description.\",\n    \"Clothing Styling\": \"Clothing styling description.\"\n  },\n  \"Scene Setup\": {\n    \"Environment Scene\": \"Environment scene description.\",\n    \"Detail Rendering\": \"Detail rendering description.\"\n  },\n  \"Artistic Expression\": {\n    \"Style Features\": \"Style features description.\",\n    \"Lighting Effects\": \"Lighting effects description.\",\n    \"Color Palette\": \"Color palette description.\"\n  }\n}"
    },
    "task_requirements": [
        "Must include style tags: masterpiece, best quality, ultra-detailed, anime style, illustration",
        "Forbid realistic terms",
        "Describe Japanese anime features: big eyes, sparkling eyes, detailed eyelashes, small nose",
        "Simple composition, simple or gradient background",
        "If Grounding info (Booru tags, character names, character traits) is provided, descriptions should match precisely",
        "Be precise and dense, avoid purple prose and vague filler words"
    ],
    "constraints": {
        "max_length": 500, 
        "dense_description": True,
        "no_purple_prose": True,
        "forbidden_words": ["realistic skin", "pores", "4K texture", "photorealistic", "cinematic"],
        "grounding_support": True
    },
    "examples": [
        {
            "natural": "Eye-level shot, dynamic pose, full body composition. A beautiful anime girl with long flowing black hair with purple gradient, cute expression, gentle smile. Masterpiece quality with cel shading and clean line art. Big sparkling purple eyes with highlights and star-shaped pupils, detailed eyelashes, small nose, delicate face with perfect anime anatomy. Soft lighting creating gentle dreamy atmosphere. She wears white and red sailor uniform with gold accents, red bow tie, pleated skirt with frills and lace. Wind blowing through hair creating dynamic movement, looking at viewer. Background is simple pink and white gradient with floating cherry blossom petals. Color palette features purple-black gradient hair, white and red clothing, soft pastel tones.",
            "structured": "{\n  \"Character Information\": {\n    \"Shot Perspective\": \"Eye-level shot, full body composition, dynamic diagonal framing.\",\n    \"Character Appearance\": \"Beautiful anime girl, long flowing black hair with purple gradient, big sparkling purple eyes with star-shaped pupils, detailed eyelashes, small nose, delicate face, soft facial features, perfect anime anatomy, cute expression, gentle smile.\",\n    \"Action Pose\": \"Dynamic pose, wind blowing through hair creating dynamic movement, looking at viewer.\",\n    \"Clothing Styling\": \"White and red sailor uniform with gold accents, red bow tie, pleated skirt with frills and lace details.\"\n  },\n  \"Scene Setup\": {\n    \"Environment Scene\": \"Simple pink and white gradient background, dreamy atmosphere with floating cherry blossom petals.\",\n    \"Detail Rendering\": \"Hair strands flowing in wind, uniform fabric texture, lace trim details, cherry blossom petal textures.\"\n  },\n  \"Artistic Expression\": {\n    \"Style Features\": \"Masterpiece, best quality, ultra-detailed, anime style, illustration, cel shading, clean line art.\",\n    \"Lighting Effects\": \"Soft lighting, gentle sunlight, beautiful shadow, rim light around hair.\",\n    \"Color Palette\": \"Purple-black gradient hair, white and red clothing, soft pastel tones, pink and white gradient background.\"\n  }\n}"
        }
    ]
}

ZIMAGE_TURBO_EN = {
    "name": "Z-Image-Turbo Portrait Prompt Engineer",
    "description": "Designed for Z-Image-Turbo model, 8-step inference for 1080P portrait generation. Emphasizes camera parameters, detailed character features, precise lighting description, composition rules, clothing details.",
    "input_template": "Optimize portrait Prompt and generate {mode} description. User input: #\n\nMust include: camera parameters (focal length, aperture), subject appearance (age, face, skin, hairstyle), pose expression, environment scene, composition rules (rule of thirds, leading lines), style, specific lighting type, color scheme, clothing details.",
    "output_format_suffix": {
        "natural": "Natural paragraph, integrating all elements. No explanations.",
        "structured": "{\n  \"Style Category\": \"...\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"...\",\n    \"Subject Description\": \"...\",\n    \"Action Pose\": \"...\",\n    \"Clothing Styling\": \"...\"\n  },\n  \"Environment Setup\": {\n    \"Environment Scene\": \"...\",\n    \"Lighting Effects\": \"...\"\n  },\n  \"Technical Specifications\": {\n    \"Composition Style\": \"...\",\n    \"Style Features\": \"...\",\n    \"Color Palette\": \"...\",\n    \"Details Texture\": \"...\",\n    \"Technical Parameters\": \"...\"\n  }\n}\n\nStyle categories:\n- Japanese Fresh: Soft lighting, natural poses, film texture, low saturation\n- Korean Refined: Bright lighting, clear skin, detailed makeup, high contrast\n- European/American Fashion: Dramatic lighting, bold composition, editorial style\n- Cinematic: Film grain, dramatic shadows, cinematic color grading\n- Portrait: Close-up composition, natural skin texture, expressive eyes\n\nOnly output JSON, no extra explanations."
    },
    "task_requirements": [
        "Use professional photography terminology (focal length, aperture, lighting type)",
        "Emphasize character emotion, eyes, pose",
        "Describe lighting specifically (side light, Rembrandt lighting, softbox, butterfly lighting)",
        "Clear style (Japanese fresh, Korean refined, European/American fashion)",
        "Must include technical parameters: image quality, resolution, film simulation"
    ],
    "constraints": {"max_length": 600},
    "examples": [
        {
            "style_category": "Japanese Fresh",
            "natural": "Eye-level shot, 85mm telephoto lens, shallow depth of field, bokeh background. A 23-year-old Asian woman stands under a cherry blossom tree in spring outdoor setting. She has soft facial lines, almond eyes with distinct lashes, delicate cream skin texture, black long wavy hair with subtle shine at the tips, light makeup, matte coral lip gloss, gentle warmth in her brows and eyes. She stands naturally, body slightly turned to the right, head gently turning toward the camera, gazing gently at the lens with a soft smile, right hand lightly lifting the hem of her dress. The environment is a cherry blossom forest, pink petals drifting in the air, background blurred into dreamy bokeh with green grass visible in the distance. Style is Japanese fresh portrait, realistic photography, film texture. Soft side lighting from front-right, afternoon natural light, gentle shadow transitions, subtle hair rim light. Pink and white dominated, pearl white and light pink accents, low saturation, fresh and bright overall. Details include lace dress texture clearly visible, skin pores natural, hair strands distinct with shine. Technical parameters: high definition, soft focus bokeh, 4K quality, natural film color grading. Clothing: white lace dress with ruffled hem design, pearl earrings, white chunky low heels.",
            "structured": "{\n  \"Style Category\": \"Japanese Fresh.\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"Eye-level shot, 85mm telephoto lens, shallow depth of field, bokeh background, medium portrait composition.\",\n    \"Subject Description\": \"23-year-old Asian female, soft facial lines, almond eyes with distinct lashes, delicate cream skin texture visible, black long wavy hair with subtle shine at tips, light makeup, matte coral lip gloss, gentle warmth in brows and eyes.\",\n    \"Action Pose\": \"Standing naturally under cherry blossom tree, body slightly turned to the right, head gently turning toward camera, gazing gently at lens with a soft smile, right hand lightly lifting dress hem.\",\n    \"Clothing Styling\": \"White lace dress with ruffled hem design, pearl earrings, white chunky low heels.\"\n  },\n  \"Environment Setup\": {\n    \"Environment Scene\": \"Spring outdoor cherry blossom forest, pink petals drifting slowly in the air, fresh petals scattered on the ground, background blurred into dreamy bokeh, green grass visible in the distance.\",\n    \"Lighting Effects\": \"Soft side lighting from front-right, afternoon natural light, gentle shadow transitions, subtle hair rim light.\"\n  },\n  \"Technical Specifications\": {\n    \"Composition Style\": \"Rule of thirds, leading lines from petals.\",\n    \"Style Features\": \"Japanese fresh portrait style, realistic photography, film texture.\",\n    \"Color Palette\": \"Pink and white dominated, pearl white and light pink accents, low saturation, fresh and bright overall, cherry blossom pink harmonizing with white dress.\",\n    \"Details Texture\": \"Lace dress texture clearly visible, skin pores natural, hair strands distinct with shine, lip gloss matte texture delicate, cherry blossom petals details exquisite.\",\n    \"Technical Parameters\": \"High definition, soft focus bokeh, no sharpening artifacts, 4K quality, natural film color grading.\"\n  }\n}"
        },
        {
            "style_category": "Korean Refined",
            "natural": "Close-up shot, 50mm lens, f/1.4 aperture, razor-sharp focus on eyes. A 22-year-old Korean woman with flawless porcelain skin, delicate features, natural gradient eyebrows, large bright eyes with detailed double eyelid folds, precise eyeliner, glossy lip tint. She faces the camera directly with a subtle confident expression, slight head tilt. Indoor studio setting with soft gray background. Professional three-point lighting with key light creating beautiful catchlights. High-key lighting, clean and bright. White and soft pink color scheme, high contrast, clean and polished overall. Details include fine eyebrow texture, flawless skin texture, detailed eye makeup with individual lash extension effect. Technical parameters: ultra sharp, studio quality, 8K quality, clean digital processing. Clothing: off-shoulder cream sweater, delicate gold necklace.",
            "structured": "{\n  \"Style Category\": \"Korean Refined.\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"Close-up shot, 50mm lens, f/1.4 aperture, razor-sharp focus on eyes, headshot composition.\",\n    \"Subject Description\": \"22-year-old Korean woman, flawless porcelain skin, delicate features, natural gradient eyebrows, large bright eyes with detailed double eyelid folds, precise eyeliner, glossy lip tint.\",\n    \"Action Pose\": \"Facing camera directly with subtle confident expression, slight head tilt.\",\n    \"Clothing Styling\": \"Off-shoulder cream sweater, delicate gold necklace.\"\n  },\n  \"Environment Setup\": {\n    \"Environment Scene\": \"Indoor studio setting, soft gray background, clean and uncluttered.\",\n    \"Lighting Effects\": \"Professional three-point lighting with key light creating beautiful catchlights, high-key lighting, clean and bright.\"\n  },\n  \"Technical Specifications\": {\n    \"Composition Style\": \"Centered composition, symmetrical framing.\",\n    \"Style Features\": \"Korean refined style, studio quality, high contrast, clean digital processing.\",\n    \"Color Palette\": \"White and soft pink color scheme, high contrast, clean and polished overall.\",\n    \"Details Texture\": \"Fine eyebrow texture, flawless skin texture, detailed eye makeup with individual lash extension effect.\",\n    \"Technical Parameters\": \"Ultra sharp, studio quality, 8K quality, clean digital processing.\"\n  }\n}"
        }
    ]
}

FLUX2_KLEIN_EN = {
    "name": "FLUX.2 Klein Prompt Engineer",
    "description": "Optimized for FLUX.2 Klein model, emphasizing microscopic details (skin texture, fabric fibers, material reflections) and precise lighting rendering (lighting angle, attenuation, shadow softness). Allows technical parameters (medium format, macro lens, etc.).",
    "input_template": "Optimize Prompt and generate {mode} description. User input: #\n\nMust include: camera parameters (focal length, aperture, medium format), subject microscopic details (skin pores, fabric fibers, material particles), action pose, environment scene, composition rules, style, precise lighting description (angle, softness, attenuation), color, technical parameters.",
    "output_format_suffix": {
        "natural": "Natural paragraph, integrating all elements. No explanations.",
        "structured": "{\n  \"Subject Information\": {\n    \"Shot Perspective\": \"...\",\n    \"Subject Description\": \"...\",\n    \"Action Pose\": \"...\",\n    \"Clothing Styling\": \"...\"\n  },\n  \"Environment Setup\": {\n    \"Environment Scene\": \"...\",\n    \"Composition Style\": \"...\",\n    \"Lighting Effects\": \"...\",\n    \"Color Palette\": \"...\",\n    \"Detail Elements\": \"...\"\n  },\n  \"Technical Specifications\": {\n    \"Style Features\": \"...\",\n    \"Technical Parameters\": \"...\"\n  }\n}"
    },
    "task_requirements": [
        "Emphasize microscopic details and material representation: skin pores, fabric fibers, metal brushing, wood grain pores",
        "Use professional photography terminology, especially medium format (Hasselblad H6D-400c)",
        "Use composition techniques: rule of thirds, leading lines, golden spiral",
        "Lighting and color convey emotional mood, describe lighting angle, attenuation characteristics, shadow hardness",
        "May include technical parameters: micron-level details, 8K resolution, medium format"
    ],
    "constraints": {"max_length": 600},
    "examples": [
        {
            "natural": "Eye-level shot, Hasselblad H6D-400c medium format, 80mm macro lens, f/2.8, shallow depth of field. A 25-year-old young woman sits quietly by a café window. She has fluffy brown hair falling naturally, tips slightly curled, delicate skin texture with visible pores, natural eyebrows, gentle and expressive eyes, lips showing natural pink color. She intently reads a hardcover book, focused on the pages, right hand fingers gently turning pages, left hand supporting the book base. Environment is a vintage café, beige-white sheer curtains half-drawn, afternoon sunlight streaming through, dark brown wooden table with warm grain texture, vintage copper lamp hanging from ceiling. Style is exquisite realistic photography, magazine editorial style, cinematic color grading. Afternoon soft side lighting from front-right at 45-degree angle, streaming through sheer curtains creating soft diffusion, natural highlights, mid-tones, shadow transitions, subtle hair rim light. Warm brown tones dominated, light beige knit sweater, overall low saturation, warm atmosphere. Details include visible skin pores, delicate fabric texture, natural table wood grain reflection, realistic book pages, coffee cup with delicate pattern. Technical parameters: shot on Hasselblad H6D-400c, hyper-detailed, 8K resolution, micron-level detail, medium format, 16-bit color depth. Clothing: light beige knit sweater, loose and comfortable, simple ribbed design at cuffs.",
            "structured": "{\n  \"Subject Information\": {\n    \"Shot Perspective\": \"Eye-level shot, Hasselblad H6D-400c medium format, 80mm macro lens, f/2.8, shallow depth of field.\",\n    \"Subject Description\": \"25-year-old young woman, fluffy brown hair falling naturally over shoulders, tips slightly curled, delicate skin texture with visible pores, natural eyebrows, gentle and expressive eyes, lips showing natural pink color.\",\n    \"Action Pose\": \"Sitting quietly by café window reading a hardcover book, focused on pages, right hand fingers gently turning pages, left hand supporting book base.\",\n    \"Clothing Styling\": \"Light beige knit sweater, loose and comfortable, simple ribbed design at cuffs, fabric texture visible.\"\n  },\n  \"Environment Setup\": {\n    \"Environment Scene\": \"Vintage café interior, beige-white sheer curtains half-drawn, afternoon sunlight streaming through, dark brown wooden table with warm grain texture visible, vintage copper lamp hanging from ceiling.\",\n    \"Composition Style\": \"Medium shot composition, subject positioned at two-thirds, rule of thirds, foreground coffee cup slightly blurred for depth.\",\n    \"Lighting Effects\": \"Afternoon soft side lighting from front-right at 45-degree angle, diffused through sheer curtains, natural highlights and shadow transitions, soft shadow edges, subtle hair rim light.\",\n    \"Color Palette\": \"Warm brown tones dominated, light beige knit sweater, overall low saturation, warm atmosphere.\",\n    \"Detail Elements\": \"Visible skin pores, delicate fabric texture, natural table wood grain reflection, realistic book pages, coffee cup with delicate pattern, curtain folds natural.\"\n  },\n  \"Technical Specifications\": {\n    \"Style Features\": \"Exquisite realistic photography, magazine editorial style, cinematic color grading, 16-bit color depth.\",\n    \"Technical Parameters\": \"Shot on Hasselblad H6D-400c, hyper-detailed, 8K resolution, micron-level detail, medium format, 16-bit color depth.\"\n  }\n}"
        }
    ]
}

ERNIE_IMAGE_EN = {
    "name": "ERNIE Image Multi-domain Design Expert",
    "description": "Intelligently identifies input type (poster, manga panel, UI, portrait, product, scene), generates professional design prompts. Emphasizes visual impact, information hierarchy, user experience.",
    "input_template": "Based on input type, generate {mode} design prompts. User input: #\n\nType identification rules:\nposter/advertisement → commercial poster; comic/storyboard → manga panel; UI/interface → UI design; person description → portrait; product description → product render; scene description → scene.",
    "output_format_suffix": {
        "natural": "Natural paragraph, integrating all design elements. No explanations.",
        "structured": "{\n  \"Design Type\": \"...\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"...\",\n    \"Subject Description\": \"...\",\n    \"Composition Requirements\": \"...\"\n  },\n  \"Visual Design\": {\n    \"Color Scheme\": \"...\",\n    \"Typography Style\": \"...\",\n    \"Detail Elements\": \"...\"\n  },\n  \"Category-specific\": {\n    \"...\": \"...\"\n  }\n}\n\nCategory-specific fields:\n- Poster: Visual Impact, Brand Tone, Information Hierarchy, Font Integration\n- Manga Panel: Narrative, Camera Language, Panel Layout, Speed Lines/Sound Effects\n- UI Design: User Experience, Layout, Interaction Elements, Design System\n- Portrait: Face Details, Expression, Pose, Clothing Styling\n- Product: Material Texture, Lighting Setup, Display Angle, Background\n- Scene: Geography & Landform, Atmosphere, Color Tone, Spatial Layers\n\nOnly output JSON, no extra explanations."
    },
    "task_requirements": [
        "Poster: Emphasize visual impact, brand tone, information hierarchy, font integration with visuals",
        "Manga panel: Emphasize narrative, camera language, panel layout, speed lines/sound effects",
        "UI design: Emphasize user experience, layout, interaction elements, design system (rounded corners/shadows)",
        "Portrait/product/scene: Follow corresponding professional standards"
    ],
    "constraints": {"max_length": 600},
    "examples": [
        {
            "category": "Poster",
            "natural": "Eye-level shot, medium shot composition, cinematic framing. This is a commercial movie poster. The subject is an Asian male assassin in a black leather jacket, cold expression, sharp piercing eyes, holding a Japanese sword with blade reflecting cold light. Composition requires top third reserved for movie title in bold metallic texture font, bottom third for English title and release date in modern sans-serif font, red neon forming diagonal lines guiding the gaze, golden ratio composition. Overall tone is cold and high-end, neon city background shrouded in shadow, red neon accents. Details include rain droplets forming hazy foreground in front of lens, cinematic light and shadow, strong visual impact.",
            "structured": "{\n  \"Design Type\": \"Commercial poster.\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"Eye-level shot, medium shot composition, cinematic framing.\",\n    \"Subject Description\": \"Asian male assassin in black leather jacket, cold expression, sharp piercing eyes, holding Japanese sword, blade reflecting cold light.\",\n    \"Composition Requirements\": \"Top third reserved for movie title in bold metallic texture font, bottom third for English title and release date in modern sans-serif font, red neon forming diagonal lines guiding the gaze, golden ratio composition.\"\n  },\n  \"Visual Design\": {\n    \"Color Scheme\": \"Overall tone cold and high-end, neon city background shrouded in shadow, red neon accents.\",\n    \"Typography Style\": \"Movie title uses bold metallic texture font, English title uses modern sans-serif font.\",\n    \"Detail Elements\": \"Rain droplets forming hazy foreground in front of lens, cinematic light and shadow, strong visual impact.\"\n  },\n  \"Category-specific\": {\n    \"Visual Impact\": \"High contrast, dramatic lighting, cold color palette.\",\n    \"Brand Tone\": \"Dark, mysterious, high-end.\",\n    \"Information Hierarchy\": \"Title prominent, subtitle and date secondary.\",\n    \"Font Integration\": \"Metallic title font contrasts with modern sans-serif subtitle.\"\n  }\n}"
        },
        {
            "category": "UI Design",
            "natural": "A mobile banking app interface design, clean white background, card-based layout with rounded corners and subtle shadows. Primary action button in brand blue, secondary actions in gray. Clear information hierarchy with prominent balance display, transaction list with icons and amounts.",
            "structured": "{\n  \"Design Type\": \"Mobile UI design.\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"Flat lay, front-facing, screen mockup composition.\",\n    \"Subject Description\": \"Mobile banking app screen, showing balance card, transaction list, action buttons.\",\n    \"Composition Requirements\": \"Card-based layout, centered content, clear visual hierarchy, 16px rounded corners, subtle shadows.\"\n  },\n  \"Visual Design\": {\n    \"Color Scheme\": \"Brand blue (#0066FF) primary, white background, gray (#666666) secondary.\",\n    \"Typography Style\": \"Sans-serif font, bold for amounts, regular for labels.\",\n    \"Detail Elements\": \"Shadow depth 4px, corner radius consistent, icon set unified style.\"\n  },\n  \"Category-specific\": {\n    \"User Experience\": \"One-hand operation friendly, primary action prominent.\",\n    \"Layout\": \"Card-based, consistent 16px padding, clear section separation.\",\n    \"Interaction Elements\": \"Primary button bottom-right, swipe for details.\",\n    \"Design System\": \"Rounded corners, subtle shadows, unified icon style, consistent spacing.\"\n  }\n}"
        }
    ]
}

IDEOGRAM4_EN = {
    "name": "Ideogram-4 Structured Prompt Generator",
    "description": "Designed for Ideogram-4 models, generates prompts following JSON caption schema. Includes high-level description, style description (with photo/art style), compositional deconstruction (background and elements). Supports color palette control, outputs JSON format. Specifically optimized for poster, logo, business card multi-text scenes with clear typography hierarchy and readability.",
    "input_template": "Based on user input, generate {mode} Ideogram-4 structured prompts. User input: #\n\nSee below for output format.",
    "output_format_suffix": {
        "natural": "Natural paragraph describing overall scene, subject actions, lighting atmosphere, and detail elements. Pay special attention to typography hierarchy and readability.",
        "structured": "Output JSON, strictly following Ideogram-4 schema:\n{\n  \"high_level_description\": \"One or two sentences summarizing the entire image (required)\",\n  \"style_description\": {\n    \"aesthetics\": \"Aesthetic keywords like moody, cinematic, vibrant\",\n    \"lighting\": \"Lighting description like golden hour, soft diffused, dramatic shadows\",\n    \"photo\": \"Photography parameters like 35mm, f/1.8, bokeh (for photography only)\",\n    \"art_style\": \"Art style like flat vector illustration, bold outlines (for non-photography only)\",\n    \"medium\": \"Medium type like photograph, illustration, 3d_render, painting, graphic_design\",\n    \"color_palette\": [\"#RRGGBB\", \"#RRGGBB\"]\n  },\n  \"compositional_deconstruction\": {\n    \"background\": \"Background environment description (required)\",\n    \"elements\": [\n      {\n        \"type\": \"obj\",\n        \"bbox\": [y_min, x_min, y_max, x_max],\n        \"desc\": \"Detailed element description (required)\",\n        \"color_palette\": [\"#RRGGBB\"]\n      },\n      {\n        \"type\": \"text\",\n        \"text\": \"Text content to render\",\n        \"desc\": \"Text style description (font, size, weight, color, alignment, effect)\",\n        \"bbox\": [y_min, x_min, y_max, x_max]\n      }\n    ]\n  }\n}\n\nNotes:\n1. Must include aesthetics, lighting, medium\n2. Photography must include photo field, non-photography must include art_style field\n3. color_palette optional, must come last, max 16 colors\n4. Elements at least one, type required (obj or text)\n5. Field order strictly follows above\n6. Colors must be uppercase #RRGGBB format\n7. bbox optional, normalized 0-1000 coordinates\n8. Text element desc should specify font style (sans-serif/serif/script), size hierarchy, alignment, visual effects\n9. Text-dense scenes like posters/logos need clear primary-secondary title hierarchy and layout\n10. Only output JSON."
    },
    "task_requirements": [
        "high_level_description required, one or two sentences summarizing",
        "style_description required, photo and art_style mutually exclusive (photography chooses photo, non-photography chooses art_style)",
        "compositional_deconstruction required, background and elements both required",
        "Elements at least one, type required (obj or text)",
        "color_palette colors must be uppercase #RRGGBB format, max 16 global or 5 element-level",
        "Text element desc should include: font style, size hierarchy, alignment, visual effects (glow/shadow/gradient)",
        "Text-dense scenes like posters/logos need clear primary-secondary title hierarchy and layout",
        "natural mode outputs natural paragraph, structured mode outputs JSON"
    ],
    "constraints": {"max_length": 800},
    "examples": [
        {
            "category": "Photography",
            "natural": "A cinematic portrait, an Asian woman looking back on a neon city street, 85mm telephoto lens, f/1.8 shallow depth of field, background neon bokeh creamy and dreamy. Warm streetlights from the side, creating perfect catchlights. She wears a beige trench coat, hair gently blowing in the breeze. Overall atmosphere warm nostalgic urban night, high contrast film look.",
            "structured": "{\n  \"high_level_description\": \"An Asian woman standing on a neon-lit urban street, telephoto lens capturing her looking back moment, background bokeh dreamy and ethereal.\",\n  \"style_description\": {\n    \"aesthetics\": \"cinematic, film grain, nostalgic, warm\",\n    \"lighting\": \"warm street lamp side lighting, rim light on hair, soft bokeh background\",\n    \"photo\": \"85mm lens, f/1.8, shallow depth of field, eye-level angle\",\n    \"medium\": \"photograph\",\n    \"color_palette\": [\"#FFD700\", \"#FF6B6B\", \"#1A1A2E\", \"#F5F5F5\", \"#4A4A4A\"]\n  },\n  \"compositional_deconstruction\": {\n    \"background\": \"Neon-lit urban street, warm streetlamp bokeh, wet ground reflecting neon reflections, distant building silhouettes shrouded in mist.\",\n    \"elements\": [\n      {\n        \"type\": \"obj\",\n        \"bbox\": [100, 300, 900, 700],\n        \"desc\": \"Asian woman, close-up to medium shot, long hair gently blowing, wearing beige trench coat, looking back expression, eyes gentle and deep, delicate makeup.\",\n        \"color_palette\": [\"#FFDAB9\", \"#F5F5DC\", \"#8B4513\"]\n      }\n    ]\n  }\n}"
        },
        {
            "category": "Illustration",
            "natural": "A dreamy forest spirit girl illustration, low angle looking up, giant glowing mushrooms as foreground framing. Spirit girl sitting on a branch, delicate wings shimmering in the light. Blue-purple and teal as main colors, thick brush strokes preserving canvas texture. High resolution shows fine scale dust particles and mushroom fluorescent vein details.",
            "structured": "{\n  \"high_level_description\": \"A fantasy-style forest scene, a spirit girl sitting beside glowing giant mushrooms, delicate wings shimmering in the light, dreamy atmosphere overwhelming.\",\n  \"style_description\": {\n    \"aesthetics\": \"fantasy, ethereal, dreamy, enchanting\",\n    \"lighting\": \"bioluminescent glow from mushrooms, soft ambient, volumetric fog\",\n    \"medium\": \"painting\",\n    \"art_style\": \"digital thick painting, visible brush strokes, canvas texture, fantasy illustration\",\n    \"color_palette\": [\"#4B0082\", \"#00CED1\", \"#9370DB\", \"#98FB98\", \"#E6E6FA\"]\n  },\n  \"compositional_deconstruction\": {\n    \"background\": \"Fantasy forest, glowing mushrooms as foreground framing, mist shrouding midground tree trunks, distant waterfall water vapor spreading, blue tones dominant.\",\n    \"elements\": [\n      {\n        \"type\": \"obj\",\n        \"bbox\": [50, 200, 450, 800],\n        \"desc\": \"Giant glowing mushroom foreground, cap semi-transparent with internal fluorescent veins visible, thick brush strokes, texture clear.\",\n        \"color_palette\": [\"#00FF7F\", \"#7CFC00\", \"#ADFF2F\"]\n      },\n      {\n        \"type\": \"obj\",\n        \"bbox\": [250, 300, 750, 850],\n        \"desc\": \"Spirit girl, sitting on a branch, delicate features, silver long hair down to waist, delicate wing veins clearly visible, wearing light flowing dress.\",\n        \"color_palette\": [\"#C0C0C0\", \"#E6E6FA\", \"#DDA0DD\"]\n      }\n    ]\n  }\n}"
        },
        {
            "category": "Poster",
            "natural": "A coffee brand promotional poster, beige textured background, hand-drawn 'COFFEE' large text centered, smaller sans-serif brand name and date below, coffee cup silhouette accent. Warm vintage style, clear typography hierarchy.",
            "structured": "{\n  \"high_level_description\": \"A warm vintage style coffee brand promotional poster, hand-drawn large text paired with concise brand information, coffee cup silhouette accent.\",\n  \"style_description\": {\n    \"aesthetics\": \"vintage, warm, cozy, minimalistic\",\n    \"lighting\": \"soft ambient lighting, warm brown tones, subtle shadow effects\",\n    \"medium\": \"graphic_design\",\n    \"art_style\": \"vintage poster design, hand lettering, organic textures, warm color palette\",\n    \"color_palette\": [\"#F5E6D3\", \"#8B4513\", \"#D2691E\", \"#F5DEB3\", \"#2F1810\"]\n  },\n  \"compositional_deconstruction\": {\n    \"background\": \"Beige rice paper textured background, subtle coffee stain texture, warm and soft.\",\n    \"elements\": [\n      {\n        \"type\": \"text\",\n        \"text\": \"COFFEE\",\n        \"desc\": \"Extra-large script English text, dark brown, thick strokes, rough edge texture, centered placement as main title.\",\n        \"bbox\": [150, 200, 400, 800]\n      },\n      {\n        \"type\": \"obj\",\n        \"bbox\": [450, 150, 550, 300],\n        \"desc\": \"Minimal coffee cup silhouette, dark brown, clear outline, located top right.\",\n        \"color_palette\": [\"#2F1810\"]\n      },\n      {\n        \"type\": \"text\",\n        \"text\": \"ARTISAN BREWS\",\n        \"desc\": \"Medium sans-serif English text, dark brown, regular weight, below main title as brand name.\",\n        \"bbox\": [500, 350, 600, 650]\n      },\n      {\n        \"type\": \"text\",\n        \"text\": \"EST. 2024\",\n        \"desc\": \"Small serif English text, light brown, thin weight, at bottom as establishment year.\",\n        \"bbox\": [700, 400, 800, 600]\n      }\n    ]\n  }\n}"
        },
        {
            "category": "Logo/Business Card",
            "natural": "A minimalist business card design, white background, company logo at top left, name at top right bold centered, position and contact below thin, clean and professional.",
            "structured": "{\n  \"high_level_description\": \"A minimalist business card, white background, classic layout with logo left and name right, professional corporate style.\",\n  \"style_description\": {\n    \"aesthetics\": \"minimalist, professional, clean, corporate\",\n    \"lighting\": \"soft ambient lighting, no shadows, crisp clean edges\",\n    \"medium\": \"graphic_design\",\n    \"art_style\": \"minimalist corporate design, clean sans-serif typography, precise alignment, high contrast\",\n    \"color_palette\": [\"#FFFFFF\", \"#000000\", \"#333333\"]\n  },\n  \"compositional_deconstruction\": {\n    \"background\": \"Pure white background, clean and uncluttered.\",\n    \"elements\": [\n      {\n        \"type\": \"obj\",\n        \"bbox\": [100, 100, 300, 300],\n        \"desc\": \"Minimal geometric company logo, black, circular outline, internal abstract graphic.\",\n        \"color_palette\": [\"#000000\"]\n      },\n      {\n        \"type\": \"text\",\n        \"text\": \"JOHN DOE\",\n        \"desc\": \"Large bold sans-serif English text, black, right-aligned, as name.\",\n        \"bbox\": [150, 500, 300, 900]\n      },\n      {\n        \"type\": \"text\",\n        \"text\": \"DESIGN DIRECTOR\",\n        \"desc\": \"Medium thin sans-serif English text, dark gray, right-aligned, below name.\",\n        \"bbox\": [350, 550, 450, 850]\n      },\n      {\n        \"type\": \"text\",\n        \"text\": \"john.doe@company.com\",\n        \"desc\": \"Small sans-serif English text, dark gray, right-aligned, at bottom.\",\n        \"bbox\": [500, 500, 600, 900]\n      }\n    ]\n  }\n}"
        }
    ]
}

QWEN_IMAGE_2512_EN = {
    "name": "Qwen Image 2512 Multi-Design Expert",
    "description": "Optimized for Qwen Image 2512 high-resolution output, intelligently identifies type (poster, brochure, infographic, courseware, product render, art illustration, portrait, scene). Emphasizes microscopic details, material texture, and lighting precision.",
    "input_template": "Based on input type, generate {mode} high-resolution design prompts. User input: #\n\nType identification keywords: poster/advertisement; brochure/catalog; infographic/data visualization; courseware/education; product/3C/cosmetics; illustration/art; person description; scene description.",
    "output_format_suffix": {
        "natural": "Natural paragraph, emphasizing high-resolution details (2512x2512), material microscopic representation, light reflection, shadow gradient. No explanations.",
        "structured": "{\n  \"Design Type\": \"...\",\n  \"Shot Perspective\": \"...\",\n  \"Composition Description\": \"...\",\n  \"Subject Elements\": \"...\",\n  \"Color Scheme\": \"...\",\n  \"Typography Style\": \"...\",\n  \"Detail Elements\": \"...\",\n  \"High-Resolution Features\": \"...\"\n}\nNo explanations."
    },
    "task_requirements": [
        "Poster: Highlight visual center, strong color contrast, font integration with visuals",
        "Brochure: Cover/inner page layout, grid system, image-text ratio",
        "Infographic: Chart types, unified icon style, clear data visualization",
        "Product render: Materials (metal/plastic/glass), lighting (three-point/soft/backlight), reflection details, micron-level details",
        "Art illustration: Brush strokes, paint thickness, paper texture, canvas fibers visible at high resolution",
        "High-resolution features: Must describe microscopic details visible at 2512x2512 (fibers, particles, burrs, reflection highlights)"
    ],
    "constraints": {"max_length": 600},
    "examples": [
        {
            "natural": "Eye-level shot, medium shot composition, dramatic lighting. This is a poster design featuring a futuristic sports car racing through a neon-lit tunnel. The car has motion blur on the background and detailed reflections on the car body. Composition uses dynamic diagonal layout with the car at lower-right golden ratio point and 'SPEED' title in massive stylized font at top left. Color scheme features dark navy background with electric blue and hot pink neon accents, with metallic silver for the car. Typography uses custom bold sans-serif with neon glow for the title and clean modern font for details. Details include rain droplets on the lens, light trails, and ultra-sharp reflections. High-resolution highlights show fine carbon fiber texture, individual water drops, and crisp neon edges, perfect for high-resolution printing.",
            "structured": "{\n  \"Shot Perspective\": \"Eye-level shot, medium shot composition, dramatic lighting\",\n  \"Design Type\": \"Poster design\",\n  \"Composition Description\": \"Dynamic diagonal layout, car at lower-right golden ratio, title in massive stylized font at top left\",\n  \"Subject Elements\": \"Futuristic sports car racing through neon-lit tunnel, motion blur background, detailed reflections on car body\",\n  \"Color Scheme\": \"Dark navy background, electric blue and hot pink neon accents, metallic silver car\",\n  \"Typography Style\": \"Title: custom bold sans-serif with neon glow, details: clean modern font at bottom\",\n  \"Detail Elements\": \"Rain droplets on lens, light trails, ultra-sharp reflections on car surface\",\n  \"High-Resolution Features\": \"Fine carbon fiber texture, individual water drops, crisp neon edges, perfect for 2512x2512 high-resolution printing\"\n}"
        }
    ]
}

QWEN_IMAGE_EDIT_COMBINED_EN = {
    "name": "Comprehensive Image Edit Enhancer",
    "description": "Professional editing prompt enhancer, generating precise, concise, direct editing prompts for add/delete/replace, style transfer, material replacement, content filling operations.",
    "input_template": "Generate precise {mode} editing prompt. User input: #\n\nTask types: add/delete/replace, text edit, person edit, style transfer, material replacement, logo/pattern, content fill, multi-image.",
    "output_format_suffix": {
        "natural": "Natural paragraph describing the editing operation, precise parameters, and visual consistency requirements. No explanations.",
        "structured": "{\n  \"Task Type\": \"...\",\n  \"Target Object\": \"...\",\n  \"Operation Description\": \"...\",\n  \"Parameter Requirements\": \"...\",\n  \"Visual Consistency\": \"...\"\n}\nNo explanations."
    },
    "general_principles": [
        "Keep enhanced prompts concise, comprehensive, direct, and specific",
        "Resolve contradictions, supplement missing key information",
        "Maintain core intent, enhance clarity and visual feasibility",
        "Added objects must be consistent with scene logic and style"
    ],
    "constraints": {"max_length": 500},
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
            "structured": "{\n  \"Task Type\": \"Add object\",\n  \"Target Object\": \"Orange tabby cat\",\n  \"Operation Description\": \"Add an orange tabby cat with green eyes on the wooden table to the right of the laptop, with realistic fur texture\",\n  \"Parameter Requirements\": \"Lighting consistent with the scene, natural shadows preserved\",\n  \"Visual Consistency\": \"Cat's size and coat color coordinated with indoor environment\"\n}"
        }
    ]
}

LTX2_EN = {
    "name": "LTX-2 Video Generation Prompt Engineer",
    "description": "Optimized for LTX-2 video generation model, emphasizing dynamic content, temporal changes, camera movement (push/pull/pan/track), and audio elements. Translates abstract emotions into specific muscle/limb actions.",
    "input_template": "Optimize prompt for video generation, create {mode} description emphasizing dynamic content and camera movement. User input: #\n\nEmotion-to-action translation: shy→shoulders raised, chin drawn, eyes down 2s then up; smile→mouth corners lifted; angry→brows furrowed, pupils contracted, jaw tightened; surprised→eyes widened, brows raised.",
    "output_format_suffix": {
        "natural": "Natural paragraph describing scene, action, camera movement, lighting, atmosphere, audio. Integrate emotion-to-action translation. No explanations.",
        "structured": "{\n  \"Scene Environment\": \"...\",\n  \"Subject Action\": \"...\",\n  \"Camera Movement\": \"...\",\n  \"Lighting Effects\": \"...\",\n  \"Atmosphere\": \"...\",\n  \"Audio/Sound Effects\": \"...\"\n}\nNo explanations."
    },
    "task_requirements": [
        "Describe core video elements: subject, scene, action, camera movement, lighting, color, sound effects",
        "Character description: hairstyle, hair color, expression, age, clothing, makeup",
        "Camera movement: push/pull/pan/track/follow, use only when needed, follow cinematic logic",
        "Describe actions with speed, direction, rhythm; emotion words → specific actions",
        "Emphasize lighting changes, color transitions, visual effects"
    ],
    "constraints": {"max_length": 500},
    "examples": [
        {
            "natural": "The camera slowly pushes in, capturing a Japanese Zen garden. Morning sunlight filters through bamboo groves, casting soft shadows on the perfectly raked gravel. Koi swim in the pond, creating gentle ripples. Distant temple bells ring as incense smoke rises from a burner. The camera orbits the garden, showcasing the serene meditative atmosphere.",
            "structured": "{\n  \"Scene Environment\": \"Japanese Zen garden, morning sunlight filtering through bamboo groves, perfectly raked gravel showing in soft shadows, koi swimming in pond\",\n  \"Subject Action\": \"Koi creating ripples as they swim, incense smoke rising from burner\",\n  \"Camera Movement\": \"Camera slowly pushes in, then orbits the garden\",\n  \"Lighting Effects\": \"Soft morning sunlight, gentle light and shadow contrast\",\n  \"Atmosphere\": \"Serene meditative atmosphere, distant temple bells echoing\",\n  \"Audio/Sound Effects\": \"Bell sounds, water sounds\"\n}"
        },
        {
            "natural": "A girl looks at a boy with shoulders slightly raised and chin drawn in, eyes looking down for 2 seconds then quickly looking up, both lips completely closed with lip line tightly pressed, corners of mouth slightly raised. Shy and coy, natural micro-expressions, not stiff or fake.",
            "structured": "{\n  \"Scene Environment\": \"Simple indoor scene with soft lighting\",\n  \"Subject Action\": \"Girl looking at boy, shoulders slightly raised, chin drawn in, eyes looking down for 2 seconds then quickly looking up, both lips completely closed, corners of mouth slightly raised\",\n  \"Camera Movement\": \"Fixed medium shot, subtle push in\",\n  \"Lighting Effects\": \"Soft front lighting, gentle shadows\",\n  \"Atmosphere\": \"Shy and coy, natural and authentic micro-expressions\",\n  \"Audio/Sound Effects\": \"Soft ambient music\"\n}"
        }
    ]
}

WAN_T2V_EN = {
    "name": "Cinematic Director Prompt Engineer",
    "description": "Cinematic Director Prompt Engineer, adds cinematic aesthetics (lighting, camera movement, color grading) to video prompts. Focus on how people interact with objects to guide realistic scene generation. Translates emotions into specific actions.",
    "input_template": "Optimize prompt with cinematic elements, create {mode} description emphasizing lighting, camera movement, and character-object interaction. User input: #\n\nEmotion-to-action: shy→shoulders raised, chin drawn, eyes down 2s then up; smile→mouth corners lifted; angry→brows furrowed, pupils contracted, jaw tightened; surprised→eyes widened, brows raised.",
    "output_format_suffix": {
        "natural": "Natural paragraph with cinematic elements: time setting, lighting, shot composition, color grading, camera movement, 35mm film grain, anamorphic lens flare. No explanations.",
        "structured": "{\n  \"Time Setting\": \"...\",\n  \"Lighting Effects\": \"...\",\n  \"Shot Composition\": \"...\",\n  \"Color Atmosphere\": \"...\",\n  \"Camera Movement\": \"...\",\n  \"Technical Parameters\": \"...\",\n  \"Usage Method\": \"...\"\n}\nNo explanations."
    },
    "task_requirements": [
        "Add cinematic settings without changing original intent",
        "Describe actions in detail; add subtle life-like motion if none",
        "Camera movement: push/pull/pan/track, use only when needed, static for dialogue",
        "Color grading: teal and orange, bleach bypass, technicolor",
        "Film elements: anamorphic lens flare, film grain, cinematic depth of field, 35mm",
        "Emphasize micro-expressions, eye direction, body language",
        "Describe how people interact with objects for logical action sequences"
    ],
    "constraints": {"max_length": 500},
    "examples": [
        {
            "natural": "Edge lighting, medium close-up, daylight, left composition, warm tone, hard light, clear sky, side lighting, daytime. A young girl sits in a field with tall grass. Two fluffy donkeys stand behind her. The girl is about eleven or twelve years old, wearing a simple floral dress, her hair in two braids, a pure smile on her face. She sits cross-legged, gently caressing wild flowers beside her. The donkeys are sturdy with upright ears, looking curiously toward the camera. Sunshine falls on the field, with a blue sky in the background. 35mm film texture, shallow depth of field, anamorphic lens flare.",
            "structured": "{\n  \"Time Setting\": \"Daytime\",\n  \"Lighting Effects\": \"Edge lighting, side lighting, hard light, clear sky\",\n  \"Shot Composition\": \"Medium close-up, left composition\",\n  \"Color Atmosphere\": \"Warm tone\",\n  \"Camera Movement\": \"Static\",\n  \"Technical Parameters\": \"35mm film texture, shallow depth of field, anamorphic lens flare\",\n  \"Usage Method\": \"Sitting cross-legged, gently touching wild flowers with both hands\"\n}"
        },
        {
            "natural": "A girl looks at a boy with shoulders slightly raised and chin drawn in, eyes looking down for 2 seconds then quickly looking up, both lips completely closed with lip line tightly pressed, corners of mouth slightly raised. Cinematic lighting, warm tone, medium close-up shot. Shy and coy, natural micro-expressions, not stiff or fake.",
            "structured": "{\n  \"Time Setting\": \"Daytime\",\n  \"Lighting Effects\": \"Soft front lighting, warm tone\",\n  \"Shot Composition\": \"Medium close-up\",\n  \"Color Atmosphere\": \"Warm, romantic mood\",\n  \"Camera Movement\": \"Fixed shot, subtle push in\",\n  \"Technical Parameters\": \"35mm film, shallow depth of field\",\n  \"Usage Method\": \"Girl looking at boy with shy posture, shoulders slightly raised, eyes looking down then quickly up\"\n}"
        }
    ]
}

WAN_I2V_EN = {
    "name": "First Frame Continuation Prompt Expert",
    "description": "Generates video description that naturally continues from first frame image, emphasizing subtle motion, micro-expressions, environmental dynamics. Supports 360-degree panorama mode.",
    "input_template": "Generate {mode} video continuation from first frame image. Image content: \nUser input: #\n\nScenario: If keywords [orbit, 360-degree, pan shot, panorama], generate panorama description: fixed camera pans 360° right, list spaces/objects in order. Otherwise, find most natural continuation.",
    "output_format_suffix": {
        "natural": "Natural paragraph describing picture story continuation: subject micro-movements, micro-expressions, hair/clothing movement, environmental lighting changes. No explanations.",
        "structured": "{\n  \"Subject Description\": \"...\",\n  \"Action Continuation\": \"...\",\n  \"Scene Development\": \"...\",\n  \"Camera Movement\": \"...\",\n  \"Atmosphere Change\": \"...\"\n}\nNo explanations."
    },
    "task_requirements": [
        "Analyze image: content, composition, subject, scene, atmosphere",
        "Combine text instructions with image if provided; otherwise use image only for natural subtle continuation",
        "Focus on micro-movements: breathing, gaze changes, hair/clothing movement, lighting changes",
        "Character style must match original image, no major clothing/scene changes",
        "Movement should be subtle and natural, avoid violent movements",
        "Camera: prefer static or very slow shots, maintain static feeling"
    ],
    "constraints": {"max_length": 500},
    "examples": [
        {
            "natural": "The camera holds steady on a woman wearing a pearl necklace, standing by a rain-streaked window. She slowly turns her head toward the right, a faint melancholic smile crossing her lips. Her eyes glisten as if recalling a distant memory. Her chest rises with a soft sigh, and she gently lifts a hand to touch the pearls at her neck. The rain continues to streak down the window, creating a nostalgic and contemplative atmosphere.",
            "structured": "{\n  \"Subject Description\": \"Woman wearing pearl necklace, standing by rain-streaked window\",\n  \"Action Continuation\": \"Slowly turns head right, faint melancholic smile, eyes glisten, soft sigh, hand touches pearls\",\n  \"Scene Development\": \"Rain continues streaking down window, ambient lighting soft\",\n  \"Camera Movement\": \"Static hold, no movement\",\n  \"Atmosphere Change\": \"Nostalgic, contemplative mood\"\n}"
        },
        {
            "natural": "Fixed camera position (keeping the camera subject position unchanged), camera pans 360 degrees to the right, smoothly orbiting the entire space, [1. Moving bar cart; 2. Floor-to-ceiling window with viewing balcony area; 3. Electric smart curtains behind glass door; 4. TV wall and leisure area; 5. Leisure reading corner; 6. Vanity and independent bathroom area; 7. Independent bathroom; 8. Closet and bedroom entrance area (solid wood entrance door, door has peephole and electronic lock, next to door is built-in shoe cabinet, opposite entrance door is walk-in closet)], no motion blur.",
            "structured": "{\n  \"Subject Description\": \"Space starting position, showing room entrance\",\n  \"Action Continuation\": \"Camera pans 360 degrees from starting position\",\n  \"Scene Development\": \"1. Moving bar cart; 2. Floor-to-ceiling window with viewing balcony; 3. Electric smart curtains; 4. TV wall and leisure area; 5. Leisure reading corner; 6. Vanity and bathroom; 7. Independent bathroom; 8. Closet and bedroom entrance\",\n  \"Camera Movement\": \"Fixed camera position, 360-degree pan shot to the right\",\n  \"Atmosphere Change\": \"Overall stable, showcasing complete space\"\n}"
        }
    ]
}

WAN_FLF2V_EN = {
    "name": "First-Last Frame Continuation Expert",
    "description": "Creates transition story between first and last frame images, filling visual differences with plausible actions and motion.",
    "input_template": "Generate {mode} transition description between first and last frame. First frame: \nLast frame: \nUser input: #\n\nScenario: If frames identical and keywords [orbit, 360-degree, panorama], generate 360° pan description. Otherwise, analyze visual differences and describe transition from state A to state B.",
    "output_format_suffix": {
        "natural": "Natural paragraph describing transition from first to last frame: identify visual differences, describe intermediate actions. No explanations.",
        "structured": "{\n  \"First Frame Description\": \"...\",\n  \"Last Frame Description\": \"...\",\n  \"Transition Action\": \"...\",\n  \"Action Development\": \"...\",\n  \"Camera Movement\": \"...\",\n  \"Atmosphere Change\": \"...\"\n}\nNo explanations."
    },
    "task_requirements": [
        "If frames identical and keywords [orbit, 360-degree, panorama], use panorama template",
        "Analyze visual differences: object position, posture, lighting, new/disappearing elements",
        "Fill the gap: describe how to change from state A to state B through intermediate actions",
        "Character style must maintain consistency between frames",
        "Transition actions must precisely correspond to visual differences",
        "Camera movement: static or slow unless differences come from camera movement"
    ],
    "constraints": {"max_length": 500},
    "examples": [
        {
            "natural": "From a serene young woman sitting on a wooden boat dock, to a shot of a paper boat floating away on the lake. She gently picks up the folded paper boat beside her, leans forward, and carefully places it on the water. The camera pans down to follow the boat as it catches the current, drifting further away. When the camera pans back up to her face, she is looking into the distance with a wistful expression.",
            "structured": "{\n  \"First Frame Description\": \"Serene young woman sitting on wooden boat dock\",\n  \"Last Frame Description\": \"Paper boat floating away on lake\",\n  \"Transition Action\": \"She picks up folded paper boat, leans forward, places it on water\",\n  \"Action Development\": \"She slowly lifts right hand, places paper boat gently on water surface, boat drifts away with current\",\n  \"Camera Movement\": \"Pans down to follow boat, then pans back up to her face\",\n  \"Atmosphere Change\": \"From peaceful to wistful, nostalgic mood\"\n}"
        },
        {
            "natural": "Fixed camera position (keeping the camera subject position unchanged), camera pans 360 degrees to the right, smoothly orbiting the entire space, [1. Moving bar cart; 2. Floor-to-ceiling window with viewing balcony area; 3. Electric smart curtains behind glass door; 4. TV wall and leisure area; 5. Leisure reading corner; 6. Vanity and independent bathroom area; 7. Independent bathroom; 8. Closet and bedroom entrance area (solid wood entrance door, door has peephole and electronic lock, next to door is built-in shoe cabinet, opposite entrance door is walk-in closet)], no motion blur.",
            "structured": "{\n  \"First Frame Description\": \"Space starting position, showing room entrance\",\n  \"Last Frame Description\": \"Back to starting position, completing 360-degree orbit\",\n  \"Transition Action\": \"Camera pans 360 degrees from starting position\",\n  \"Action Development\": \"Camera smoothly rotates 360 degrees, capturing all eight areas in sequence at uniform speed\",\n  \"Camera Movement\": \"Fixed camera position, 360-degree pan shot to the right\",\n  \"Atmosphere Change\": \"Overall stable, showcasing complete space\"\n}"
        }
    ]
}

VIDEO_FRAME_SEQUENCE_TO_PROMPT_EN = {
    "name": "Video Frame Sequence Analysis Expert",
    "description": "Video Frame Sequence Analysis Expert, analyzing dynamic changes from input frame sequence to generate reverse video content description for guiding AI to generate videos of the same style.",
    "input_template": "Analyze video frame sequence by intervals, generate {mode} description. User input: #\n\nMark each segment with 【Frame X-Y】. Describe: scene environment, character appearance, main subject action, camera movement, color atmosphere. Merge adjacent identical frames.",
    "output_format_suffix": {
        "natural": "Format: 【Frame X-Y】Scene|Character|Action|Camera|Atmosphere. Example: 【Frame 1-5】Indoor cafe|Black short-haired male|Seated holding coffee|Fixed medium shot|Warm yellow tones.",
        "structured": "{\n  \"Frame Range\": \"Frame X-Y\",\n  \"Shot Type\": \"Long/Medium/Close-up\",\n  \"Camera\": \"Fixed/Push/Pull/Pan/Track\",\n  \"Scene\": \"environment\",\n  \"Character\": \"appearance, hairstyle, expression\",\n  \"Action\": \"movement\",\n  \"Atmosphere\": \"color, lighting\"\n}\nSeparate intervals with blank line."
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
            "structured": "{\n  \"Frame Range\": \"Frame 1-3\",\n  \"Shot Type\": \"Medium shot\",\n  \"Camera\": \"Fixed front view\",\n  \"Scene\": \"Indoor office scene, bright natural lighting from windows\",\n  \"Character\": \"Golden shoulder-length hair female, serious expression, light makeup, wearing white shirt and black suit jacket, wearing thin-framed glasses\",\n  \"Action\": \"Seated upright, hands folded on desk, slight breathing movement\",\n  \"Atmosphere\": \"Formal solemn atmosphere, warm bright lighting\"\n}\n\n{\n  \"Frame Range\": \"Frame 4-8\",\n  \"Shot Type\": \"Medium shot\",\n  \"Camera\": \"Slow pan right\",\n  \"Scene\": \"Indoor office scene, same setting\",\n  \"Character\": \"Golden shoulder-length hair female, expression slightly changed, corners of mouth slightly moving, wearing white shirt and black suit jacket\",\n  \"Action\": \"Right hand raised pointing at document, left hand turning pages\",\n  \"Atmosphere\": \"Atmosphere slightly relaxed, soft lighting\"\n}"
        }
    ]
}

VIDEO_TO_PROMPT_EN = {
    "name": "Video Reverse Prompt Expert",
    "description": "Video Reverse Prompt Expert, analyzes video content and generates detailed video description prompts for guiding AI to generate videos of the same style.",
    "input_template": "Analyze video content and generate {mode} description prompt. User input: #\n\nDescribe: Scene|Character|Action|Camera|Atmosphere. Output directly without JSON.",
    "output_format_suffix": {
        "natural": "Format: Scene|Character|Action|Camera|Atmosphere. No explanations.",
        "structured": "{\n  \"Scene Description\": \"...\",\n  \"Character Description\": \"...\",\n  \"Action Description\": \"...\",\n  \"Camera Description\": \"...\",\n  \"Atmosphere Description\": \"...\"\n}\nNo explanations."
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
            "structured": "{\n  \"Scene Description\": \"Japanese Zen garden, outdoor, morning, perfect rake marks on gravel visible, koi swimming in pond\",\n  \"Character Description\": \"No character\",\n  \"Action Description\": \"Koi swimming creating gentle ripples, static peaceful scene\",\n  \"Camera Description\": \"Slow push in, then orbit the garden\",\n  \"Atmosphere Description\": \"Calm Zen atmosphere, soft morning light, green and gold tones, peaceful meditative mood\"\n}"
        },
        {
            "natural": "Cyberpunk city night scene|Short-haired character in glowing jacket, riding motorcycle|Motorcycle speeding through city streets, motion blur effect|Camera descends from high altitude, follows motorcyclist through neon-lit streets|Strong tech feel, neon light trails, pink-purple tones, intense dynamic atmosphere",
            "structured": "{\n  \"Scene Description\": \"Cyberpunk city, outdoor, night, neon-lit streets, futuristic architecture\",\n  \"Character Description\": \"Short-haired character, glowing jacket, wearing helmet\",\n  \"Action Description\": \"Riding motorcycle at high speed, motion blur effect\",\n  \"Camera Description\": \"High altitude descent, tracking shot following motorcyclist\",\n  \"Atmosphere Description\": \"Strong tech feel, neon light trails, pink-purple color tones, intense dynamic atmosphere\"\n}"
        }
    ]
}

VIDEO_DETAILED_SCENE_BREAKDOWN_EN = {
    "name": "Video Detailed Scene Breakdown Expert",
    "description": "Video Detailed Scene Breakdown Expert, analyzing each scene in chronological order to generate prompts for guiding AI to generate videos of the same style.",
    "input_template": "Break down video scenes chronologically, generate {mode} description. User input: #\n\nEnsure each timestamp has complete details for AI video generation.",
    "output_format_suffix": {
        "natural": "Format by time: **Time:** 0:00-0:15\\n**Scene:** ...\\n**Character:** ...\\n**Action:** ...\\n**Lighting:** ...\\n**Color:** ...\\n**Camera:** ...\\n**Atmosphere:** ...\\nSeparate scenes with blank line.",
        "structured": "{\n  \"Shot 1\": {\n    \"Time\": \"0:00-0:15\",\n    \"Shot Type\": \"Medium shot\",\n    \"Camera\": \"Fixed/Pan (pan speed 20 deg/s)\",\n    \"Analysis\": \"Grandmother angrily unties apron (2s), grabs rolling pin (0.5s), strides toward living room (1m/s)\",\n    \"Editing Rhythm\": \"Smooth action, no cuts\"\n  },\n  \"Shot 2\": {\n    \"Time\": \"0:15-0:30\",\n    \"Shot Type\": \"Close-up\",\n    \"Camera\": \"Follow (1.5m/s)\",\n    \"Analysis\": \"Grandson panicked, dodging while looking back (look-back frequency 0.5Hz), grandmother in pursuit\",\n    \"Editing Rhythm\": \"Rapid cuts (every 2s), creating tense and humorous atmosphere\"\n  }\n}"
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
            "structured": "{\n  \"Shot 1\": {\n    \"Time\": \"0:00-0:15\",\n    \"Shot Type\": \"Medium shot\",\n    \"Camera\": \"Fixed/Pan\",\n    \"Analysis\": \"Grandmother angrily unties apron, grabs rolling pin to teach grandson a lesson\",\n    \"Editing Rhythm\": \"Smooth action, expressing grandmother's protective anger toward grandson\"\n  },\n  \"Shot 2\": {\n    \"Time\": \"0:15-0:30\",\n    \"Shot Type\": \"Close-up\",\n    \"Camera\": \"Follow\",\n    \"Analysis\": \"Grandson panicked, running around room, looking back, grandmother chasing\",\n    \"Editing Rhythm\": \"Fast pace, creating tense and humorous atmosphere\"\n  }\n}"
        }
    ]
}

VIDEO_SUBTITLE_FORMAT_EN = {
    "name": "Video Subtitle Format Optimization Expert",
    "description": "Video Subtitle Format Optimization Expert, converting subtitle content into standard format, ensuring timecode and text synchronization. Supports Four-Element Method for tone and speed control.",
    "input_template": "Optimize subtitle format, generate {mode} output. User input: #\n\nFour-Element Method: Vocal Style + Rhythm Description + Pitch Variation + Punctuation. Output directly without JSON.",
    "output_format_suffix": {
        "natural": "Format: **Timecode:** 00:00:00,000 --> 00:00:05,000\\n**Subtitle Text:** ...\\nSeparate with blank line.",
        "structured": "{\n  \"Subtitle 1\": {\n    \"Timecode\": \"00:00:00,000 --> 00:00:03,500\",\n    \"Line Tone\": \"Raised breath, steady rhythm + flat intonation\",\n    \"Subtitle Text\": \"Hello everyone, welcome to my channel!\"\n  },\n  \"Subtitle 2\": {\n    \"Timecode\": \"00:00:03,500 --> 00:00:07,200\",\n    \"Line Tone\": \"Breath voice, choked + gradually slowing + falling intonation\",\n    \"Subtitle Text\": \"Today I want to... share a... simple home-cooked dish...\"\n  },\n  \"Subtitle 3\": {\n    \"Timecode\": \"00:00:07,200 --> 00:00:11,800\",\n    \"Line Tone\": \"Raised breath, accelerated rhythm + rising intonation\",\n    \"Subtitle Text\": \"First, we need to prepare some basic ingredients!\"\n  }\n}"
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
    "description": "Multi-Speaker Dialogue Creator, creating dialogue text with multiple speakers and assigning appropriate voice timbres for TTS model. Supports Four-Element Method for tone and speed control.",
    "input_template": "Create multi-speaker dialogue with voice timbres for TTS. User input: #\n\nFour-Element Method: Vocal Style + Rhythm Description + Pitch Variation + Punctuation. Output directly without JSON.",
    "output_format_suffix": {
        "natural": "Format: **Voice:** Female/Male/Loli/Boy/Mature/Middle-aged\\n**Speaker ID:** 0-5\\n**Emotion:** Happy/Sad/Angry/Calm/Excited/Gentle\\n**Dialogue:** Speaker: ...\\nSeparate with blank line.",
        "structured": "{\n  \"Dialogue 1\": {\n    \"Voice\": \"Female\",\n    \"Speaker ID\": \"0\",\n    \"Emotion\": \"Gentle\",\n    \"Line Tone\": \"Raised breath, steady rhythm + flat intonation\",\n    \"Dialogue\": \"Mom: Honey, it's time to wake up, you'll be late for school!\"\n  },\n  \"Dialogue 2\": {\n    \"Voice\": \"Loli\",\n    \"Speaker ID\": \"2\",\n    \"Emotion\": \"Sleepy\",\n    \"Line Tone\": \"Breath voice, dragging + falling intonation\",\n    \"Dialogue\": \"Daughter: Mom, let me sleep five more minutes...\"\n  },\n  \"Dialogue 3\": {\n    \"Voice\": \"Male\",\n    \"Speaker ID\": \"1\",\n    \"Emotion\": \"Calm\",\n    \"Line Tone\": \"Throaty voice, steady rhythm + flat intonation\",\n    \"Dialogue\": \"Dad: You lazy bones, you'll miss breakfast if you don't get up now.\"\n  }\n}"
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
    "description": "Expert in crafting professional English lyrics with authentic song structure, rhyme schemes, and poetic devices for Western music styles. Supports Four-Element Method for vocal tone control.",
    "input_template": "Create professional English lyrics. User input: #\n\nFour-Element Method: Vocal Style + Rhythm Description + Pitch Variation + Punctuation. Output directly without JSON.",
    "output_format_suffix": {
        "natural": "Format: **Structure:** Verse/Chorus/Bridge/Outro\\n**Lyrics:** ...\\n**Theme:** ...\\n**Style:** ...\\n**Mood:** ...\\nSeparate sections with line breaks.",
        "structured": "{\n  \"Verse 1\": {\n    \"Vocal Tone\": \"Lowered voice, steady rhythm + flat intonation\",\n    \"Lyrics\": \"Streets quiet as evening lights begin to glow, walking alone through alleys I used to know, memories flood like tides into my mind, those moments with you I left behind\"\n  },\n  \"Chorus\": {\n    \"Vocal Tone\": \"Breath voice, choked + gradually slowing + falling intonation\",\n    \"Lyrics\": \"Nights thinking of you... so endless and long, every star whispers your name in song, your smile your warmth so precious to me, the most treasured memories I'll always hold dearly\"\n  },\n  \"Theme\": \"Longing and waiting\",\n  \"Style\": \"Pop ballad\",\n  \"Mood\": \"Deeply affectionate\"\n}"
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
    "description": "OCR Text Recognition Expert, precisely extracting all text content from posters, including font, color, position and other style information.",
    "input_template": "Extract text content from posters for reverse prompt generation. User input: #\n\nOutput directly without JSON.",
    "output_format_suffix": {
        "natural": "Format: **Title:** Content: ... | Font: ... | Color: ... | Position: ...\\n**Subtitle:** Content: ... | Font: ... | Color: ... | Position: ...\\n**Body Text:** Content: ... | Font: ... | Color: ... | Position: ...\\n**Slogans:** ...\\n**Other Text:** ...",
        "structured": "{\n  \"Title\": {\n    \"Content\": \"...\",\n    \"Font\": \"...\",\n    \"Color\": \"...\",\n    \"Position\": \"...\"\n  },\n  \"Subtitle\": {\n    \"Content\": \"...\",\n    \"Font\": \"...\",\n    \"Color\": \"...\",\n    \"Position\": \"...\"\n  },\n  \"Body Text\": {\n    \"Content\": \"...\",\n    \"Font\": \"...\",\n    \"Color\": \"...\",\n    \"Position\": \"...\"\n  },\n  \"Slogans\": \"...\",\n  \"Other Text\": \"...\"\n}"
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
    "description": "Bounding Box Detection Expert, precisely locating bounding boxes of target objects in images, providing accurate position coordinates and category information.",
    "input_template": "Locate target objects in image, output bounding box coordinates. Categories: \"#\".",
    "output_format_suffix": {
        "natural": "Output JSON: {\"bounding_boxes\": [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"category\"}]}.",
        "structured": "Output JSON: {\"bounding_boxes\": [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"category\"}]}."
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
    "[Anime] Illustrious": "ILLUSTRIOUS_EN",
    "[Anime] Anima": "ANIMA_EN",
    "[Portrait] ZIMAGE - Turbo": "ZIMAGE_TURBO_EN",
    "[General] FLUX2 - Klein": "FLUX2_KLEIN_EN",
    "[Design] ERNIE - Image": "ERNIE_IMAGE_EN",
    "[Ideogram-4] Ideogram-4": "IDEOGRAM4_EN",
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
IDEOGRAM4 = "IDEOGRAM4"
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