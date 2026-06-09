# -*- coding: utf-8 -*-
"""
English Preset Prompt Library

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

NORMAL_DESCRIBE_TAGS_EN = {
    "name": "Image Tag Reverse Generator (Smart Category Version)",
    "description": "Automatically identify image category (landscape/photography/illustration/IP/cosplay/game character/product/architectural interior/animal/food/UI interface/fashion/general), output corresponding English tag list. Optimized for SDXL and other tag-based models, outputs JSON format.",
    "input_template": "Analyze image content, first determine the most suitable category (choose one from: landscape, photography, illustration, IP, cosplay, game character, product, architectural interior, animal, food, UI interface, fashion, general). If no category matches, generate appropriate category name based on image content and output using the same field structure. Then generate {mode} tag list. User input: #\n\nRules: 1. Output ONLY JSON, no extra text; 2. Total tags 30-60, no duplicates; 3. Must include quality tags (masterpiece, best quality, high resolution, ultra-detailed). 4. Strictly follow the field structure below.",
    "output_format_suffix": {
        "natural": "Comma-separated tag string. Natural mode not recommended, please use structured mode.",
        "structured": "【Category】Category name\n【Core Elements】\n  - Quality Tags: Quality tags content\n  - Scene Type: Scene description content\n  - Time & Lighting: Time and lighting description content\n  - Color Atmosphere: Color and atmosphere description content\n【Technical Parameters】\n  - Shot Perspective: Shot and perspective description content\n  - Detail Elements: Detail elements description content\n\nCategory-specific fields:\n- Landscape: Quality Tags, Scene Type, Time & Lighting, Color Atmosphere, Shot Perspective, Detail Elements\n- Photography: Quality Tags, Camera Parameters, Lighting Type, Post-processing Style, Subject Content, Scene Environment, Dynamic Features\n- Illustration: Quality Tags, Art Style, Line & Brushwork, Color Scheme, Lighting & Shadow, Subject Elements, Composition\n- IP: Quality Tags, IP Name, Character Features, Signature Elements, Style Consistency, Application Scenario\n- Cosplay: Quality Tags, Character Name, Costume Details, Props & Accessories, Makeup Style, Pose Expression, Background Setting, Photography Style\n- Game Character: Quality Tags, Game Title, Character Name, Class/Role, Equipment Details, Skill Effects, Scene Environment, Art Style\n- Product: Quality Tags, Product Type, Material & Craft, Lighting Setup, Display Angle, Background Environment, Detail Features\n- Architectural Interior: Quality Tags, Building Type, Structural Elements, Interior Style, Lighting Design, Furnishing Items, Spatial Sense\n- Animal: Quality Tags, Animal Species, Fur Color Features, Pose & Action, Eyes & Expression, Environment Interaction, Shooting Distance\n- Food: Quality Tags, Dish Name, Plating Style, Lighting & Texture, Color Appetite, Decorative Elements, Shooting Angle\n- UI Interface: Quality Tags, Interface Type, Layout Structure, Color Mode, Typography Style, Component Details, Interaction Elements\n- Fashion: Quality Tags, Clothing Type, Fabric Details, Matching Style, Model Pose, Background Scene, Brand Tone\n- General: Quality Tags, Subject Features, Environment Scene, Lighting Effects, Color Matching, Composition Style, Art Style, Texture & Details\n\nTags within each field are comma-separated."
    },
    "task_requirements": [
        "First identify image category, then output tags according to corresponding field group",
        "Must include quality tags: masterpiece, best quality, high resolution, ultra-detailed",
        "Tags are concise, comma-separated within each field, total tags 30-60",
        "Category field order can be adjusted, but field names must be exactly consistent",
        "Natural mode: output comma-separated tag string; structured mode: output structured prompt"
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
            "natural": "masterpiece, best quality, high resolution, ultra-detailed, 8K, landscape photo, mountain scenery, snow mountain, cloud sea, canyon, sunrise, golden hour, hard light, side backlight, cold tones, blue white gray, serene, expansive feeling, aerial view, long shot, telephoto lens, 16:9 wide format, pine trees, mist, layered mountains, shadow contrast, blue sky white clouds, sky gradient, atmospheric perspective, aerial perspective, near far light, air humidity, extremely high visibility, morning mist swirling, dawn light penetration, layered depth, strong perspective, atmospheric mood",
            "structured": "【Category】Landscape\n【Quality Tags】masterpiece, best quality, high resolution, ultra-detailed, 8K\n【Core Elements】\n  - Scene Type: mountain scenery, snow mountain, cloud sea, canyon\n  - Time & Lighting: sunrise, golden hour, hard light, side backlight\n  - Color Atmosphere: cold tones, blue white gray, serene, expansive feeling\n【Sky & Atmosphere】\n  - Sky Condition: blue sky with white clouds, sky gradient, sunrise dawn light\n  - Atmospheric Conditions: atmospheric perspective, moderate air humidity, extremely high visibility, morning mist swirling in valleys\n  - Light Layers: near far light attenuation, dawn light penetrating clouds, layered light rays\n【Technical Parameters】\n  - Shot Perspective: aerial view, long shot, telephoto lens, 16:9 wide format\n  - Detail Elements: pine trees, mist, layered mountains, shadow contrast"
        }
    ]
}

NORMAL_DESCRIBE_EN = {
    "name": "Image Reverse Description Expert (Smart Category Version)",
    "description": "Automatically identify image category (landscape/photography/illustration/IP/cosplay/game character/product/architectural interior/animal/food/UI interface/fashion/general), generate structured or natural paragraph detailed description, output structured prompt format.",
    "input_template": "Analyze image content, first determine the most suitable category (choose one from: landscape, photography, illustration, IP, cosplay, game character, product, architectural interior, animal, food, UI interface, fashion, general). If no category matches, generate appropriate category name based on image content and output using the same field structure. Then generate {mode} English description. User input: #\n\nDescription points: Based on category output corresponding fields, describe in detail using professional terminology.",
    "output_format_suffix": {
        "natural": "Natural paragraph, integrating all visual elements. Not recommended for natural mode, please use structured mode.",
        "structured": "【Category】Category name\n【Scene Description】\n  - Geography & Landform: Geography and landform description\n  - Weather & Lighting: Weather and lighting description\n  - Color Tone: Color and tone description\n  - Spatial Layers: Spatial layers description\n【Technical Details】\n  - Shot Composition: Shot and composition description\n  - Atmosphere & Mood: Atmosphere and mood description\n\nCategory-specific fields:\n- Landscape: Geography & Landform, Weather & Lighting, Color Tone, Spatial Layers, Shot Composition, Atmosphere & Mood\n- Photography: Camera Parameters, Lighting Setup, Composition Method, Subject Dynamic, Post-processing Tone, Overall Style\n- Illustration: Painting Medium, Brushwork & Lines, Color & Lighting, Narrative Content, Material Texture, Style Reference\n- IP: IP Name, Character Description, Signature Features, Color Specification, Application Scenario, Style Consistency\n- Cosplay: Character Name, Costume Details, Props & Accessories, Makeup Style, Pose Expression, Background Setting, Photography Style, Overall Impression\n- Game Character: Game Title, Character Name, Class/Role, Equipment Details, Skill Effects, Scene Environment, Art Style, Visual Effects\n- Product: Product Name, Material Texture, Lighting Setup, Display Angle, Background Environment, Detail Close-up, Overall Impression\n- Architectural Interior: Building Type, Structure & Material, Interior Design Style, Lighting Layout, Furniture & Furnishings, Spatial Scale, Atmosphere Feeling\n- Animal: Animal Species, Appearance Features, Pose & Action, Eyes & Expression, Environment Scene, Interaction Relationship, Overall Temperament\n- Food: Dish Name, Plating Description, Lighting & Texture, Color Matching, Decorative Details, Shooting Angle, Appetite Appeal\n- UI Interface: Interface Type, Layout Structure, Color Scheme, Typography System, Component Details, Interaction Effects, Overall Experience\n- Fashion: Clothing Style, Fabric Details, Color Scheme, Model Pose & Expression, Shooting Scene, Styling Approach, Brand Impression\n- General: Shot Perspective, Subject Description, Action Pose, Environment Scene, Lighting & Color, Color Matching, Detail Rendering, Technical Parameters, Clothing Styling, Usage Method"
    },
    "task_requirements": [
        "First identify image category, then output description according to corresponding fields",
        "Describe in detail using professional terminology",
        "Interaction between people and objects must conform to realistic physical logic",
        "Omit related fields when no person present",
        "Natural mode: output natural paragraph; structured mode: output structured prompt"
    ],
    "constraints": {"max_length": 800},
    "examples": [
        {
            "category": "Landscape",
            "natural": "Using aerial view, wide-angle lens, 16:9 wide format. The main subject is a continuous mountain range with snow-capped peaks, the main peak illuminated by golden morning light presenting a sunrise on mountain phenomenon. The foreground is dark blue mountain shadows and pine tree silhouettes, the midground is rolling cloud seas spreading like cotton, and the distant view is a gradient light purple horizon. The sky transitions from blue to light purple in the distance, with dawn light radiating from behind the main peak creating golden rim lighting. Light on the horizon shows warm orange-yellow gradually transitioning upward to cold blue. Atmospheric perspective is evident with distant snow mountains showing light purple blending into the sky. Clear visibility allows distant mountain details to be faintly visible. The air is pure and dry with morning mist swirling through the valleys creating natural layered transitions. Lighting is during the golden hour after sunrise, hard light shining from the left at a slant, creating strong contrast between light and shadow, with clear details in highlights on snow peaks and cold blue tones in shadows. Overall colors are dominated by blue, white, and gold, with medium saturation, creating a serene and magnificent atmosphere.",
            "structured": "【Category】Landscape\n【Scene Description】\n  - Geography & Landform: Continuous mountain range with snow-capped peaks, main peak prominent, cloud seas between canyons\n  - Weather & Lighting: Golden hour after sunrise, hard light from left at slant, no clouds, high visibility\n  - Color Tone: Blue, white, gold dominant, shadows cold blue, highlights warm gold, medium saturation\n  - Spatial Layers: Foreground: pine tree dark silhouettes; Midground: rolling cloud seas; Distant: snow peaks and light purple sky\n【Sky & Atmosphere】\n  - Sky Condition: Blue sky transitioning to light purple, dawn light radiating from behind main peak, sunrise dawn\n  - Atmospheric Conditions: Strong atmospheric perspective, pure dry air, extremely high visibility, morning mist swirling through valleys\n  - Perspective Relations: Near-far light attenuation, distant mountains blend with sky colors, faint distant details visible\n  - Light Layers: Horizon light orange-yellow transitioning upward to cold blue, golden dawn light penetrating clouds creating rim lighting\n【Technical Details】\n  - Shot Composition: Aerial view, wide-angle lens, 16:9 wide format, long shot composition\n  - Atmosphere & Mood: Serene, magnificent, sacred"
        }
    ]
}

PROMPT_EXPANDER_EN = {
    "name": "Prompt Expansion Expert (Smart Category Version)",
    "description": "Based on short prompts, automatically determine expansion type (portrait/cosplay/game character/product/scene/animal/food/general), generate detailed vivid descriptions. Preserve original intent and core keywords, output structured prompt format.",
    "input_template": "Expand the following prompt, first determine expansion type (portrait, cosplay, game character, product, scene, animal, food, general). If no type matches, generate appropriate category name based on prompt content and output using the same field structure. Then generate {mode} description. User input: #\n\nExpansion points: Add shot perspective, subject features, scene environment, action pose, lighting effects, color palette, detail rendering. Keep original meaning, enhance expressiveness. Output format see below.",
    "output_format_suffix": {
        "natural": "Natural paragraph, integrating all elements. Not recommended, please use structured mode.",
        "structured": "【Category】Category name\n【Subject Information】\n  - Shot Perspective: Shot perspective description\n  - Subject Description: Subject description\n  - Action Pose: Action pose description\n  - Clothing Styling: Clothing styling description\n【Environment Setup】\n  - Environment Scene: Environment scene description\n  - Composition Style: Composition style description\n  - Lighting Effects: Lighting effects description\n  - Color Palette: Color palette description\n  - Detail Rendering: Detail rendering description\n  - Material Wear State: Material wear degree, stain traces, usage traces description\n【Technical Specifications】\n  - Technical Parameters: Technical parameters description\n\nCategory-specific fields:\n- Portrait: Shot Perspective, Subject Description, Action Pose, Environment Scene, Composition Style, Lighting Effects, Color Palette, Detail Rendering, Clothing Styling, Material Wear State, Technical Parameters\n- Cosplay: Character Name, Costume Details, Props & Accessories, Makeup Style, Pose Expression, Background Setting, Photography Style, Material Wear State, Technical Parameters\n- Game Character: Game Title, Character Name, Class/Role, Equipment Details, Skill Effects, Scene Environment, Art Style, Visual Effects, Technical Parameters\n- Product: Product Name, Material Texture, Lighting Setup, Display Angle, Background Environment, Detail Close-up, Material Wear State, Overall Impression\n- Scene: Geography & Landform, Spatial Layers, Color Tone, Lighting Effects, Shot Perspective, Atmosphere & Mood\n- Animal: Animal Species, Appearance Features, Pose & Action, Eyes & Expression, Environment Scene, Interaction Relationship\n- Food: Dish Name, Plating Description, Lighting & Texture, Color Matching, Decorative Details, Shooting Angle, Appetite Appeal\n- General: Shot Perspective, Subject Description, Action Pose, Environment Scene, Composition Style, Lighting Effects, Color Palette, Detail Rendering, Material Wear State, Technical Parameters"
    },
    "task_requirements": [
        "Expanded content should be vivid and professional, using photography terminology",
        "Do not change original meaning, do not add irrelevant content",
        "Interaction between people and objects should conform to realistic logic",
        "First determine expansion type, then output corresponding field group",
        "Natural mode: output natural paragraph; structured mode: output structured prompt"
    ],
    "constraints": {"max_length": 800, "keep_core_keywords": True},
    "examples": [
        {
            "natural": "Eye-level shot, 35mm prime lens, shallow depth of field. By a sunlit café window, a 25-year-old Asian woman sits quietly. She has black long straight hair falling softly over her shoulders, gentle almond eyes that are warm and bright, and fair skin with a delicate complexion. She wears a light beige knit sweater with exquisite lace trim at the cuffs. She is holding a hardcover book with yellowed pages with both hands, intently reading, occasionally looking up at the window with a thoughtful expression in her eyes, her right hand gently turning the pages while left hand rests on the table. The environment is a cozy café with wooden table showing clear grain texture, sheer curtains half-drawn, street outside faintly visible, elegantly decorated interior. Main light source is afternoon natural light from front-right, soft side lighting creating gentle shadows, overall warm and comfortable tone, low saturation, creating peaceful and focused atmosphere. Before her sits a steaming latte with intricate patterns on the cup and delicate latte art. The surrounding environment is quiet, other customers conversing softly, gentle jazz music playing softly.",
            "structured": "【Category】Portrait\n【Subject Information】\n  - Shot Perspective: Eye-level shot, 35mm prime lens, shallow depth of field, medium shot composition\n  - Subject Description: 25-year-old Asian woman, black long straight hair falling softly over shoulders, gentle almond eyes warm and bright, fair skin with delicate complexion, light beige knit sweater with lace trim\n  - Action Pose: Sitting by window, reading hardcover book, right hand turning pages, left hand resting on table\n  - Clothing Styling: Light beige knit sweater with exquisite lace trim at cuffs\n【Environment Setup】\n  - Environment Scene: Cozy café, wooden table with grain texture, sheer curtains half-drawn, street outside faintly visible\n  - Composition Style: Rule of thirds, leading lines from window light\n  - Lighting Effects: Afternoon natural light from front-right, soft side lighting, gentle shadows, subtle hair rim light\n  - Color Palette: Warm brown tones, beige accents, low saturation\n  - Detail Rendering: Visible skin pores, book page texture, latte art, curtain folds\n【Technical Specifications】\n  - Technical Parameters: 35mm prime lens, f/1.8, shallow DOF, 4K quality"
        }
    ]
}

ILLUSTRIOUS_EN = {
    "name": "Illustrious/ToriiGate Anime Character Optimizer",
    "description": "Specialized Prompt engineer for Illustrious and ToriiGate anime models (ToriiGate compatible). Output tag-based character descriptions including shot perspective, subject features, clothing styling, environment scene, color palette, lighting effects, composition style, art style, texture details. Descriptions are precise, dense, avoiding purple prose. Supports Grounding info (Booru tags, character names).",
    "input_template": "Optimize 2D anime character description and generate {mode} tags. User input: #\n\nRules: 1. Output ONLY English tags, comma-separated; 2. 30-60 total, no duplicates; 3. Strictly avoid realistic terms; 4. Must include masterpiece, best quality, highres, ultra detailed; 5. If Booru tags or character names are provided, descriptions should match precisely (e.g., hatsune_miku for Hatsune Miku); 6. Be precise and distinctive, avoid vague filler words.",
    "output_format_suffix": {
        "natural": "Pure comma-separated tags only. Example: masterpiece, best quality, ultra-detailed, high resolution, eye-level shot, half-body composition, 16-year-old girl, blue-purple twintails, golden eyes, anime style, cel shading",
        "structured": "【Basic Attributes】\n  - Quality: Quality tags\n  - Angle: Shot perspective tags\n  - Subject: Subject features tags\n【External Elements】\n  - Clothing: Clothing tags\n  - Scene: Scene tags\n【Artistic Expression】\n  - Color: Color tags\n  - Lighting: Lighting tags\n  - Composition: Composition tags\n  - Style: Style tags"
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
            "structured": "【Basic Attributes】\n  - Quality: masterpiece, best quality, ultra-detailed, high resolution, perfect composition, exquisite rendering\n  - Angle: eye-level shot, half-body composition, soft diagonal framing\n  - Subject: 16-year-old anime girl, blue-purple gradient twintails, golden star-shaped pupils, delicate oval face, snow-white skin, pink blush, sweet smile, sparkling eyes\n【External Elements】\n  - Clothing: white sailor uniform, navy collar, red bow, pleated skirt, lace trim, black shoes, ribbon decorations\n  - Scene: pink gradient background, dreamy atmosphere, cherry blossom petals floating, petal decorations, simple background\n【Artistic Expression】\n  - Color: white, red, gold, soft pastel tones, blue-purple gradient, gradient background\n  - Lighting: soft side lighting, hair rim light, catchlights, gentle diffused light\n  - Composition: half-body composition, soft diagonal framing, cute pose, body slightly turned, head tilted, hands clasped\n  - Style: anime style, illustration, cel shading, clean line art, soft color palette, anime art style"
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
    "description": "Designed for Z-Image-Turbo model, 8-step inference for 1080P portrait generation. Automatically identifies clothing style (ancient Chinese/Korean/Japanese school/workwear/sports/wedding/street fashion/Lolita/swimwear/lingerie/cosplay/game character/general), outputs detailed descriptions for corresponding dimensions.",
    "input_template": "Optimize portrait Prompt and generate {mode} description. User input: #\n\nStep 1: Determine the clothing style category from the image/text (choose one from: Ancient Chinese/Hanfu, Korean, Japanese/School, Office/Professional, Sports/Fitness, Wedding/Formal, Street/Hip-hop, Lolita, Swimwear/Lingerie, Cosplay/Game Character, General).\nStep 2: If a clear category matches, output according to corresponding field group; if no category matches, generate appropriate style category name based on character clothing features and content (e.g., Western/European, French/Vintage, Cyberpunk, etc.) and output using the same field structure, DO NOT force classification to similar styles.\nStep 3: Must include: camera parameters, subject appearance, pose expression, environment scene, composition rules, lighting type, color scheme.\nCategory-specific fields see format description below.",
    "output_format_suffix": {
        "natural": "Natural paragraph, integrating all elements, highlighting style characteristics.",
        "structured": "【Style Category】...\n【Subject Information】\n  - Shot Perspective: ...\n  - Subject Description: ...\n  - Action Pose: ...\n  - Clothing Styling: ...\n【Environment Setup】\n  - Environment Scene: ...\n  - Composition Style: ...\n  - Lighting Effects: ...\n  - Color Palette: ...\n  - Detail Elements: ...\n  - Material Wear State: Material wear degree, stain traces, usage traces description\n【Technical Specifications】\n  - Style Features: ...\n  - Technical Parameters: ...\n\nCategory-specific fields (additions to general fields):\n- Ancient Chinese/Hanfu: Add \"Hairstyle & Accessories\", \"Flowing Robe Dynamics\", \"Traditional Patterns\", \"Vintage Tones\"\n- Japanese/School: Add \"Uniform Details\", \"Youthful Vibe\", \"Campus Elements\"\n- Office/Professional: Add \"Fabric Texture\", \"Professional Demeanor\", \"Clean Background\"\n- Sports/Fitness: Add \"Action Dynamics\", \"Muscle Definition\", \"Stretchy Fabric\", \"Sweat/Moisture\"\n- Wedding/Formal: Add \"Dress Dynamics\", \"Veil Movement\", \"Romantic Lighting\", \"Luxury Details\"\n- Street/Hip-hop: Add \"Loose Fit\", \"Street Elements\", \"High Contrast/Saturation\"\n- Lolita: Add \"Layered Dress\", \"Accessory Details\", \"Dreamy Tones\"\n- Swimwear/Lingerie: Add \"Body Curves\", \"Wet Skin\", \"Intimacy\"\n- Cosplay/Game Character: Add \"Character Name\", \"Costume Details\", \"Props & Accessories\", \"Makeup Style\", \"Background Setting\", \"Photography Style\", \"Material Wear State\"\n- General: No additions.\n\nOnly output structured format, no extra explanations."
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
            "natural": "Eye-level shot, 85mm telephoto lens, shallow depth of field, bokeh background. A 23-year-old Asian woman stands under a cherry blossom tree in spring outdoor setting. She has soft facial lines, almond eyes with distinct lashes, delicate cream skin texture, black long wavy hair with subtle shine at the tips, light makeup, matte coral lip gloss, gentle warmth in her brows and eyes. She stands naturally, body slightly turned to the right, head gently turning toward the camera, gazing gently at the lens with a soft smile, right hand lightly lifting the hem of her dress. The environment is a cherry blossom forest, pink petals drifting in the air, background blurred into dreamy bokeh with green grass visible in the distance. Style is Japanese fresh portrait, realistic photography, film texture. Soft side lighting from front-right, afternoon natural light, gentle shadow transitions, subtle hair rim light. Pink and white dominated, pearl white and light pink accents, low saturation, fresh and bright overall. Details include lace dress texture clearly visible, skin pores naturally visible with slight size variation on cheeks, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, fine wrinkles at outer eye corners and under eyes, subcutaneous capillary color variations, pores呈现真实3D凹坑结构带微阴影 under 45-degree side-top lighting, uneven natural sebum glow on cheekbone and nose bridge, eye iris deep brown color, tear trough area natural fine lines, hair strands distinct with shine. Technical parameters: high definition, soft focus bokeh, 4K quality, natural film color grading, fine organic film grain throughout, zero digital sharpening, all sharpness from optics, shadow details preserved, highlights soft without overexposure. Clothing: white lace dress with ruffled hem design, pearl earrings, white chunky low heels.",
            "structured": "【Style Category】Japanese Fresh\n【Subject Information】\n  - Shot Perspective: Eye-level shot, 85mm telephoto lens, shallow depth of field, bokeh background, medium portrait composition\n  - Subject Description: 23-year-old Asian female, soft facial lines, almond eyes with distinct lashes, delicate cream skin texture visible, black long wavy hair with subtle shine at tips, light makeup, matte coral lip gloss, gentle warmth in brows and eyes\n  - Action Pose: Standing naturally under cherry blossom tree, body slightly turned to the right, head gently turning toward camera, gazing gently at lens with a soft smile, right hand lightly lifting dress hem\n  - Clothing Styling: White lace dress with ruffled hem design, pearl earrings, white chunky low heels\n【Environment Setup】\n  - Environment Scene: Spring outdoor cherry blossom forest, pink petals drifting slowly in the air, fresh petals scattered on the ground, background blurred into dreamy bokeh, green grass visible in the distance\n  - Lighting Effects: Soft side lighting from front-right, afternoon natural light, gentle shadow transitions, subtle hair rim light\n【Technical Specifications】\n  - Composition Style: Rule of thirds, leading lines from petals\n  - Style Features: Japanese fresh portrait style, realistic photography, film texture\n  - Color Palette: Pink and white dominated, pearl white and light pink accents, low saturation, fresh and bright overall, cherry blossom pink harmonizing with white dress\n  - Details Texture: Lace dress texture clearly visible, skin pores naturally visible with slight size variation on cheeks, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, fine wrinkles at outer eye corners and under eyes, subcutaneous capillary color variations, pores呈现真实3D凹坑结构带微阴影 under 45-degree side-top lighting, uneven natural sebum glow on cheekbone and nose bridge, eye iris deep brown color, tear trough area natural fine lines, hair strands distinct with shine, lip gloss matte texture delicate, cherry blossom petals details exquisite\n  - Technical Parameters: High definition, soft focus bokeh, no sharpening artifacts, 4K quality, natural film color grading, fine organic film grain throughout, zero digital sharpening, all sharpness from optics, shadow details preserved, highlights soft without overexposure"
        },
        {
            "style_category": "Korean Refined",
            "natural": "Close-up shot, 50mm lens, f/1.4 aperture, razor-sharp focus on eyes. A 22-year-old Korean woman with healthy natural warm-toned fair skin with subtle subcutaneous pinkish undertone, dewy luminous foundation texture, pores naturally visible with slight size variation on cheeks, delicate features, natural gradient eyebrows, large bright eyes with detailed double eyelid folds, precise eyeliner, glossy lip tint. She faces the camera directly with a subtle confident expression, slight head tilt. Indoor studio setting with soft gray background. Professional three-point lighting with key light creating beautiful catchlights. High-key lighting, clean and bright. White and soft pink color scheme, high contrast, clean and polished overall. Details include fine eyebrow texture, flawless skin texture, detailed eye makeup with individual lash extension effect. Technical parameters: ultra sharp, studio quality, 8K quality, clean digital processing. Clothing: off-shoulder cream sweater, delicate gold necklace.",
            "structured": "【Style Category】Korean Refined\n【Subject Information】\n  - Shot Perspective: Close-up shot, 50mm lens, f/1.4 aperture, razor-sharp focus on eyes, headshot composition\n  - Subject Description: 22-year-old Korean woman, healthy natural warm-toned fair skin with subtle subcutaneous pinkish undertone, dewy luminous foundation texture, pores naturally visible with slight size variation on cheeks, delicate features, natural gradient eyebrows, large bright eyes with detailed double eyelid folds, precise eyeliner, glossy lip tint\n  - Action Pose: Facing camera directly with subtle confident expression, slight head tilt\n  - Clothing Styling: Off-shoulder cream sweater, delicate gold necklace\n【Environment Setup】\n  - Environment Scene: Indoor studio setting, soft gray background, clean and uncluttered\n  - Lighting Effects: Professional three-point lighting with key light creating beautiful catchlights, high-key lighting, clean and bright\n【Technical Specifications】\n  - Composition Style: Centered composition, symmetrical framing\n  - Style Features: Korean refined style, studio quality, high contrast, clean digital processing\n  - Color Palette: White and soft pink color scheme, high contrast, clean and polished overall\n  - Details Texture: Fine eyebrow texture, flawless skin texture, detailed eye makeup with individual lash extension effect\n  - Technical Parameters: Ultra sharp, studio quality, 8K quality, clean digital processing"
        },
        {
            "style_category": "European Elegance",
            "natural": "Eye-level shot, 85mm portrait lens, f/1.8 aperture, shallow depth of field. A 28-year-old European woman with striking facial features: defined cheekbones, strong jawline, deep-set blue eyes with intense gaze, naturally arched brows, slightly freckled fair skin with visible texture, healthy natural warm-toned fair skin with subtle subcutaneous pinkish undertone, dewy luminous foundation texture, pores naturally visible with slight size variation on cheeks, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, fine wrinkles at outer eye corners and under eyes, subcutaneous capillary color variations, pores呈现真实3D凹坑结构带微阴影 under 45-degree side-top lighting, uneven natural sebum glow on cheekbone and nose bridge, eye iris deep blue color, tear trough area natural fine lines, wavy chestnut hair cascading over shoulders with natural volume. She stands in a Parisian street setting, early evening golden hour light. Her expression is poised and confident, subtle enigmatic smile, head slightly turned, looking past camera with contemplative gaze. She wears a tailored navy blazer over cream silk blouse, vintage gold pendant necklace, minimal elegant makeup with defined brows and nude lipstick. Environment is cobblestone street with vintage lampposts, soft bokeh background showing Parisian architecture, warm evening atmosphere. Soft golden hour side lighting from left, creating gentle Rembrandt lighting pattern on face, natural skin texture visible, subtle shadow under cheekbone defining facial structure. Warm golden and cool navy contrast, sophisticated muted tones, elegant refined atmosphere. Details include visible skin pores and light freckles, blazer fabric texture, silk blouse sheen, hair natural wave pattern, gold necklace antique finish. Technical parameters: editorial photography, Vogue magazine style, natural film grain, 8K quality, cinematic color grading.",
            "structured": "【Style Category】European Elegance\n【Subject Information】\n  - Shot Perspective: Eye-level shot, 85mm portrait lens, f/1.8 aperture, shallow depth of field, medium portrait composition\n  - Subject Description: 28-year-old European woman, defined cheekbones, strong jawline, deep-set blue eyes with intense gaze, naturally arched brows, slightly freckled fair skin with visible texture, healthy natural warm-toned fair skin with subtle subcutaneous pinkish undertone, dewy luminous foundation texture, pores naturally visible with slight size variation on cheeks, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, fine wrinkles at outer eye corners and under eyes, subcutaneous capillary color variations, pores呈现真实3D凹坑结构带微阴影 under 45-degree side-top lighting, uneven natural sebum glow on cheekbone and nose bridge, eye iris deep blue color, tear trough area natural fine lines, wavy chestnut hair cascading over shoulders with natural volume\n  - Action Pose: Standing in Parisian street, poised and confident expression, subtle enigmatic smile, head slightly turned, looking past camera with contemplative gaze\n  - Clothing Styling: Tailored navy blazer over cream silk blouse, vintage gold pendant necklace, minimal elegant makeup with defined brows and nude lipstick\n【Environment Setup】\n  - Environment Scene: Parisian cobblestone street with vintage lampposts, soft bokeh background showing Parisian architecture, warm evening atmosphere\n  - Composition Style: Subject positioned at one-third, rule of thirds, background architecture providing context\n  - Lighting Effects: Soft golden hour side lighting from left, creating gentle Rembrandt lighting pattern on face, natural skin texture visible, subtle shadow under cheekbone defining facial structure\n  - Color Palette: Warm golden and cool navy contrast, sophisticated muted tones, elegant refined atmosphere\n  - Detail Elements: Visible skin pores and light freckles, blazer fabric texture, silk blouse sheen, hair natural wave pattern, gold necklace antique finish\n【Technical Specifications】\n  - Style Features: European elegance style, editorial photography, Vogue magazine style, cinematic color grading\n  - Technical Parameters: Natural film grain, 8K quality, medium format quality, sophisticated color grading"
        },
        {
            "style_category": "Western Fashion",
            "natural": "Low angle shot, 50mm lens, f/2.0 aperture. A 24-year-old Western model with high fashion features: sharp facial contours, prominent brow bone, piercing green eyes with smoky eye makeup, defined lips with matte red lipstick, porcelain skin with subtle luminosity, healthy natural warm-toned fair skin with subtle subcutaneous pinkish undertone, dewy luminous foundation texture, pores naturally visible with slight size variation on cheeks, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, subcutaneous capillary color variations, pores呈现真实3D凹坑结构带微阴影 under 45-degree side-top lighting, uneven natural sebum glow on cheekbone and nose bridge, blonde hair styled in sleek straight bob with precision cut ends. She poses dynamically against industrial loft backdrop, body angled 45 degrees, one hand on hip, chin lifted slightly, confident runway-ready expression. Raw brick walls, large windows with diffused city light, minimalist industrial aesthetic. Dramatic side lighting from large window creating strong contrast, highlight on cheekbone, shadow defining jawline, fashion editorial lighting setup. Black and white with red accent, high contrast fashion aesthetic, editorial mood. Details include sharp facial definition, sleek hair precision cut, bold lip color, designer fabric texture. Technical parameters: fashion editorial quality, high contrast, 8K resolution, professional model photography. Clothing: black structured blazer dress, architectural silhouette, statement earrings.",
            "structured": "【Style Category】Western Fashion\n【Subject Information】\n  - Shot Perspective: Low angle shot, 50mm lens, f/2.0 aperture, dynamic fashion composition\n  - Subject Description: 24-year-old Western model, sharp facial contours, prominent brow bone, piercing green eyes with smoky eye makeup, defined lips with matte red lipstick, porcelain skin with subtle luminosity, healthy natural warm-toned fair skin with subtle subcutaneous pinkish undertone, dewy luminous foundation texture, pores naturally visible with slight size variation on cheeks, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, subcutaneous capillary color variations, pores呈现真实3D凹坑结构带微阴影 under 45-degree side-top lighting, uneven natural sebum glow on cheekbone and nose bridge, blonde hair styled in sleek straight bob with precision cut ends\n  - Action Pose: Posing dynamically against industrial backdrop, body angled 45 degrees, one hand on hip, chin lifted slightly, confident runway-ready expression\n  - Clothing Styling: Black structured blazer dress, architectural silhouette, statement earrings\n【Environment Setup】\n  - Environment Scene: Industrial loft backdrop, raw brick walls, large windows with diffused city light, minimalist industrial aesthetic\n  - Composition Style: Dynamic diagonal composition, subject dominates frame, fashion editorial framing\n  - Lighting Effects: Dramatic side lighting from large window creating strong contrast, highlight on cheekbone, shadow defining jawline, fashion editorial lighting setup\n  - Color Palette: Black and white with red accent, high contrast fashion aesthetic, editorial mood\n  - Detail Elements: Sharp facial definition, sleek hair precision cut, bold lip color, designer fabric texture\n【Technical Specifications】\n  - Style Features: Western high fashion style, editorial photography, runway aesthetic, dramatic composition\n  - Technical Parameters: Fashion editorial quality, high contrast, 8K resolution, professional model photography"
        }
    ]
}

FLUX2_KLEIN_EN = {
    "name": "FLUX.2 Klein Prompt Engineer",
    "description": "Optimized for FLUX.2 Klein model, emphasizing microscopic details (skin texture, fabric fibers, material reflections) and precise lighting rendering (lighting angle, attenuation, shadow softness). Allows technical parameters (medium format, macro lens, etc.).",
    "input_template": "Optimize Prompt and generate {mode} description. User input: #\n\nMust include: camera parameters (focal length, aperture, medium format), subject microscopic details (skin pores, fabric fibers, material particles), action pose, environment scene, composition rules, style, precise lighting description (angle, softness, attenuation), color, technical parameters.",
    "output_format_suffix": {
        "natural": "Natural paragraph, integrating all elements. No explanations.",
        "structured": "【Subject Information】\n  - Shot Perspective: ...\n  - Subject Description: ...\n  - Action Pose: ...\n  - Clothing Styling: ...\n【Environment Setup】\n  - Environment Scene: ...\n  - Composition Style: ...\n  - Lighting Effects: ...\n  - Color Palette: ...\n  - Detail Elements: ...\n【Technical Specifications】\n  - Style Features: ...\n  - Technical Parameters: ..."
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
            "natural": "Eye-level shot, Hasselblad H6D-400c medium format, 80mm macro lens, f/2.8, shallow depth of field. A 25-year-old young woman sits quietly by a café window. She has fluffy brown hair falling naturally, tips slightly curled, delicate skin texture with pores naturally visible with slight size variation, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, fine wrinkles at outer eye corners and under eyes, subcutaneous capillary color variations, pores呈现真实3D凹坑结构带微阴影 under 45-degree side-top lighting, uneven natural sebum glow on cheekbone and nose bridge, natural eyebrows, gentle and expressive eyes, lips showing natural pink color. She intently reads a hardcover book, focused on the pages, right hand fingers gently turning pages, left hand supporting the book base. Environment is a vintage café, beige-white sheer curtains half-drawn, afternoon sunlight streaming through, dark brown wooden table with warm grain texture, vintage copper lamp hanging from ceiling. Style is exquisite realistic photography, magazine editorial style, cinematic color grading. Afternoon soft side lighting from front-right at 45-degree angle, streaming through sheer curtains creating soft diffusion, natural highlights, mid-tones, shadow transitions, subtle hair rim light. Warm brown tones dominated, light beige knit sweater, overall low saturation, warm atmosphere. Details include visible skin pores, delicate fabric texture, natural table wood grain reflection, realistic book pages, coffee cup with delicate pattern. Technical parameters: shot on Hasselblad H6D-400c, hyper-detailed, 8K resolution, micron-level detail, medium format, 16-bit color depth. Clothing: light beige knit sweater, loose and comfortable, simple ribbed design at cuffs.",
            "structured": "【Subject Information】\n  - Shot Perspective: Eye-level shot, Hasselblad H6D-400c medium format, 80mm macro lens, f/2.8, shallow depth of field\n  - Subject Description: 25-year-old young woman, fluffy brown hair falling naturally over shoulders, tips slightly curled, delicate skin texture with pores naturally visible with slight size variation, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, fine wrinkles at outer eye corners and under eyes, subcutaneous capillary color variations, pores呈现真实3D凹坑结构带微阴影 under 45-degree side-top lighting, uneven natural sebum glow on cheekbone and nose bridge, natural eyebrows, gentle and expressive eyes, lips showing natural pink color\n  - Action Pose: Sitting quietly by café window reading a hardcover book, focused on pages, right hand fingers gently turning pages, left hand supporting book base\n  - Clothing Styling: Light beige knit sweater, loose and comfortable, simple ribbed design at cuffs, fabric texture visible\n【Environment Setup】\n  - Environment Scene: Vintage café interior, beige-white sheer curtains half-drawn, afternoon sunlight streaming through, dark brown wooden table with warm grain texture visible, vintage copper lamp hanging from ceiling\n  - Composition Style: Medium shot composition, subject positioned at two-thirds, rule of thirds, foreground coffee cup slightly blurred for depth\n  - Lighting Effects: Afternoon soft side lighting from front-right at 45-degree angle, diffused through sheer curtains, natural highlights and shadow transitions, soft shadow edges, subtle hair rim light\n  - Color Palette: Warm brown tones dominated, light beige knit sweater, overall low saturation, warm atmosphere\n  - Detail Elements: Visible skin pores, delicate fabric texture, natural table wood grain reflection, realistic book pages, coffee cup with delicate pattern, curtain folds natural\n【Technical Specifications】\n  - Style Features: Exquisite realistic photography, magazine editorial style, cinematic color grading, 16-bit color depth\n  - Technical Parameters: Shot on Hasselblad H6D-400c, hyper-detailed, 8K resolution, micron-level detail, medium format, 16-bit color depth, fine organic film grain throughout, zero digital sharpening, all sharpness from optics, shadow details preserved, highlights soft without overexposure"
        },
        {
            "natural": "Eye-level shot, Hasselblad H6D-400c medium format, 100mm portrait lens, f/2.2, shallow depth of field. A 30-year-old European woman with striking facial architecture: high cheekbones creating natural shadow definition, strong jawline with visible bone structure, deep-set hazel eyes with intense penetrating gaze, naturally arched brows with individual hair strands visible, slightly freckled fair skin with natural texture and visible pores, wavy auburn hair cascading over shoulders with natural volume and shine. She stands in an Italian Renaissance courtyard, marble columns and terracotta tiles, early morning diffused light. Her expression is contemplative and poised, subtle knowing smile, head slightly tilted, looking past camera with thoughtful gaze. She wears a tailored charcoal wool coat over cream cashmere sweater, vintage leather handbag, minimal sophisticated makeup with defined brows and matte berry lipstick. Environment is Florentine courtyard with marble arches, terracotta planters with olive trees, soft bokeh background showing Renaissance architecture, warm Mediterranean atmosphere. Soft diffused morning light from overhead skylight, creating gentle butterfly lighting pattern on face, natural skin texture visible with light freckles, subtle shadow under cheekbone defining facial structure, soft ambient fill. Warm terracotta and cool charcoal contrast, sophisticated muted earth tones, elegant refined Mediterranean atmosphere. Details include visible skin pores and light freckles across cheeks and nose, wool coat fabric texture with visible fibers, cashmere sweater soft sheen, hair natural wave pattern with individual strands, leather handbag grain texture, marble column surface detail. Technical parameters: editorial photography, Harper's Bazaar magazine style, natural film grain, 8K quality, cinematic Mediterranean color grading, micron-level skin detail.",
            "structured": "【Subject Information】\n  - Shot Perspective: Eye-level shot, Hasselblad H6D-400c medium format, 100mm portrait lens, f/2.2, shallow depth of field, medium portrait composition\n  - Subject Description: 30-year-old European woman, high cheekbones creating natural shadow definition, strong jawline with visible bone structure, deep-set hazel eyes with intense penetrating gaze, naturally arched brows with individual hair strands visible, slightly freckled fair skin with natural texture and visible pores, wavy auburn hair cascading over shoulders with natural volume and shine\n  - Action Pose: Standing in Florentine courtyard, contemplative and poised expression, subtle knowing smile, head slightly tilted, looking past camera with thoughtful gaze\n  - Clothing Styling: Tailored charcoal wool coat over cream cashmere sweater, vintage leather handbag, minimal sophisticated makeup with defined brows and matte berry lipstick\n【Environment Setup】\n  - Environment Scene: Italian Renaissance courtyard, marble columns and terracotta tiles, terracotta planters with olive trees, soft bokeh background showing Renaissance architecture, warm Mediterranean atmosphere\n  - Composition Style: Subject positioned at one-third, rule of thirds, marble columns providing vertical leading lines, background architecture providing context\n  - Lighting Effects: Soft diffused morning light from overhead skylight, creating gentle butterfly lighting pattern on face, natural skin texture visible with light freckles, subtle shadow under cheekbone defining facial structure, soft ambient fill\n  - Color Palette: Warm terracotta and cool charcoal contrast, sophisticated muted earth tones, elegant refined Mediterranean atmosphere\n  - Detail Elements: Visible skin pores and light freckles across cheeks and nose, wool coat fabric texture with visible fibers, cashmere sweater soft sheen, hair natural wave pattern with individual strands, leather handbag grain texture, marble column surface detail\n【Technical Specifications】\n  - Style Features: European elegance style, editorial photography, Harper's Bazaar magazine style, cinematic Mediterranean color grading\n  - Technical Parameters: Shot on Hasselblad H6D-400c, natural film grain, 8K quality, medium format quality, micron-level skin detail, sophisticated color grading"
        }
    ]
}

ERNIE_IMAGE_EN = {
    "name": "ERNIE Image Multi-domain Design Expert",
    "description": "Intelligently identifies input type (poster, manga panel, UI, portrait, product, scene), generates professional design prompts. Emphasizes visual impact, information hierarchy, user experience.",
    "input_template": "Based on input type, generate {mode} design prompts. User input: #\n\nType identification rules:\nposter/advertisement → commercial poster; comic/storyboard → manga panel; UI/interface → UI design; person description → portrait; product description → product render; scene description → scene; if no type matches, generate appropriate design type name based on content and output using the same field structure, DO NOT force classification to similar types.",
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
            "category": "Manga Panel",
            "natural": "A manga page layout with four panels. Top-left close-up shows character's surprised expression with pupils dilated and sweat drop on forehead, speed lines converging in background. Top-right medium shot shows him suddenly turning head towards window with motion blur. Bottom-left bird's-eye view shows a mecha passing by outside with shattered glass. Bottom-right full-body panning shot shows character clenching fist with determined expression and speech bubble: \"I can't let it escape\". Black and white manga style, rough pen lines, screentone grayscale shading, tense fast-paced narrative.",
            "structured": "{\n  \"Design Type\": \"Manga panel.\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"Top-left close-up, top-right medium shot, bottom-left bird's-eye view, bottom-right full-body panning.\",\n    \"Subject Description\": \"Character showing surprised expression → sudden head turn → mecha passing by → clenched fist with determined expression.\",\n    \"Composition Requirements\": \"Four-panel layout, tight panel spacing, visual flow left-to-right, top-to-bottom.\"\n  },\n  \"Visual Design\": {\n    \"Color Scheme\": \"Black and white manga, grayscale screentone shading, pure black speed lines and white highlights.\",\n    \"Typography Style\": \"Handwritten font for speech bubbles, explosion-shaped art font for sound effects.\",\n    \"Detail Elements\": \"Speed lines, motion blur, shattered glass fragments, screentone texture.\"\n  },\n  \"Category-specific\": {\n    \"Narrative\": \"Tense fast-paced storytelling.\",\n    \"Camera Language\": \"Close-up for emotion, medium shot for action, bird's-eye for context.\",\n    \"Panel Layout\": \"Four-panel grid, dynamic diagonal compositions.\",\n    \"Speed Lines/Sound Effects\": \"Converging speed lines, onomatopoeia 'BOOM'.\"\n  }\n}"
        },
        {
            "category": "UI Design",
            "natural": "A music player app UI interface design. Dark background with top search bar featuring rounded rectangle semi-transparent frosted glass effect. Middle section shows large album cover card with 18px rounded corners and soft shadow. Bottom section has playback controls including previous, play/pause, next buttons with metallic shimmer effect. Bottom navigation bar with three icons: Home, Discover, My. Color scheme uses purple to blue gradient as accent, sans-serif modern font with bold title and regular artist name. Layout relaxed with ample white space and breathing room.",
            "structured": "{\n  \"Design Type\": \"Mobile UI design.\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"Flat lay, front-facing, screen mockup composition.\",\n    \"Subject Description\": \"Mobile banking app screen, showing balance card, transaction list, action buttons.\",\n    \"Composition Requirements\": \"Card-based layout, centered content, clear visual hierarchy, 16px rounded corners, subtle shadows.\"\n  },\n  \"Visual Design\": {\n    \"Color Scheme\": \"Brand blue (#0066FF) primary, white background, gray (#666666) secondary.\",\n    \"Typography Style\": \"Sans-serif font, bold for amounts, regular for labels.\",\n    \"Detail Elements\": \"Shadow depth 4px, corner radius consistent, icon set unified style.\"\n  },\n  \"Category-specific\": {\n    \"User Experience\": \"One-hand operation friendly, primary action prominent.\",\n    \"Layout\": \"Card-based, consistent 16px padding, clear section separation.\",\n    \"Interaction Elements\": \"Primary button bottom-right, swipe for details.\",\n    \"Design System\": \"Rounded corners, subtle shadows, unified icon style, consistent spacing.\"\n  }\n}"
        },
        {
            "category": "Scene",
            "natural": "Using aerial view, wide-angle lens, 16:9 wide format. The main subject is a continuous mountain range with snow-capped peaks, the main peak illuminated by golden morning light presenting a sunrise on mountain phenomenon. The foreground is dark blue mountain shadows and pine tree silhouettes, the midground is rolling cloud seas spreading like cotton, and the distant view is a gradient light purple horizon. The sky transitions from blue to light purple in the distance, with dawn light radiating from behind the main peak creating golden rim lighting. Light on the horizon shows warm orange-yellow gradually transitioning upward to cold blue. Atmospheric perspective is evident with distant snow mountains showing light purple blending into the sky. Clear visibility allows distant mountain details to be faintly visible. The air is pure and dry with morning mist swirling through the valleys creating natural layered transitions. Lighting is during the golden hour after sunrise, hard light shining from the left at a slant, creating strong contrast between light and shadow, with clear details in highlights on snow peaks and cold blue tones in shadows. Overall colors are dominated by blue, white, and gold, with medium saturation, creating a serene and magnificent atmosphere.",
            "structured": "{\n  \"Design Type\": \"Landscape photography.\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"Aerial view, wide-angle lens, 16:9 wide format, long shot composition.\",\n    \"Subject Description\": \"Continuous mountain range with snow-capped peaks, main peak prominent, cloud seas between canyons, pine tree foreground silhouettes.\"\n  },\n  \"Scene Description\": {\n    \"Geography & Landform\": \"Continuous mountain range with snow-capped peaks, main peak illuminated by golden morning light.\",\n    \"Weather & Lighting\": \"Golden hour after sunrise, hard light from left at slant, no clouds, high visibility.\",\n    \"Color Tone\": \"Blue, white, gold dominant, shadows cold blue, highlights warm gold, medium saturation.\",\n    \"Spatial Layers\": \"Foreground: pine tree dark silhouettes; Midground: rolling cloud seas; Distant: snow peaks and light purple sky.\"\n  },\n  \"Sky & Atmosphere\": {\n    \"Sky Condition\": \"Blue sky transitioning to light purple, dawn light radiating from behind main peak, sunrise dawn.\",\n    \"Atmospheric Conditions\": \"Strong atmospheric perspective, pure dry air, extremely high visibility, morning mist swirling through valleys.\",\n    \"Perspective Relations\": \"Near-far light attenuation, distant mountains blend with sky colors, faint distant details visible.\",\n    \"Light Layers\": \"Horizon light orange-yellow transitioning upward to cold blue, golden dawn light penetrating clouds creating rim lighting.\"\n  },\n  \"Visual Design\": {\n    \"Color Scheme\": \"Blue, white, gold dominant, cold tones, serene and expansive feeling.\",\n    \"Typography Style\": \"N/A\",\n    \"Detail Elements\": \"Pine trees, mist, layered mountains, shadow contrast, cloud sea texture.\"\n  },\n  \"Category-specific\": {\n    \"Atmosphere & Mood\": \"Serene, magnificent, sacred.\",\n    \"Technical Parameters\": \"Aerial view, wide-angle lens, 16:9 wide format, long shot composition.\"\n  }\n}"
        },
        {
            "category": "Portrait",
            "natural": "Close-up shot, 85mm portrait lens, f/1.4 aperture, razor-sharp focus on eyes. A 26-year-old woman with delicate natural beauty: soft facial features with visible pores on cheeks and nose with slight size variation, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, fine wrinkles at outer eye corners and under eyes, subcutaneous capillary color variations, pores呈现真实3D凹坑结构带微阴影 under 45-degree side-top lighting, uneven natural sebum glow on cheekbone and nose bridge, natural skin texture with slight unevenness, fine hairs on face visible at this proximity, natural gradient eyebrows with individual hair strands, warm gentle eyes with clear iris detail and pupil, natural lip color with subtle lip line definition. She faces the camera directly with a subtle confident expression, slight head tilt. Indoor studio setting with soft gray background. Professional three-point lighting with key light creating beautiful catchlights, butterfly lighting pattern, natural skin texture clearly visible. High-key lighting, clean and bright. White and warm beige color scheme, natural realistic overall. Details include visible skin pores on cheeks and T-zone, natural skin texture with micro-imperfections, fine facial hairs, natural skin luminosity, realistic eye moisture in pupils, subtle skin oil sheen on forehead and nose. Technical parameters: ultra sharp, studio quality, 8K quality, medium format, natural skin texture rendering, fine organic film grain throughout, zero digital sharpening, all sharpness from optics, shadow details preserved, highlights soft without overexposure. Clothing: cream cashmere sweater, minimal natural makeup.",
            "structured": "{\n  \"Design Type\": \"Portrait photography.\",\n  \"Subject Information\": {\n    \"Shot Perspective\": \"Close-up shot, 85mm portrait lens, f/1.4 aperture, razor-sharp focus on eyes, headshot composition.\",\n    \"Subject Description\": \"26-year-old woman with delicate natural beauty, soft facial features with visible pores on cheeks and nose with slight size variation, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, fine wrinkles at outer eye corners and under eyes, subcutaneous capillary color variations, pores呈现真实3D凹坑结构带微阴影 under 45-degree side-top lighting, uneven natural sebum glow on cheekbone and nose bridge, natural skin texture with slight unevenness, fine hairs on face visible at this proximity.\",\n    \"Composition Requirements\": \"Centered composition, symmetrical framing, eyes as main focal point.\"\n  },\n  \"Face Details\": {\n    \"Skin Texture\": \"Visible pores on cheeks and T-zone, natural skin texture with micro-imperfections, slight unevenness, fine facial hairs visible at close proximity.\",\n    \"Eye Details\": \"Warm gentle eyes with clear iris detail and pupil, natural eye moisture visible, realistic catchlights.\",\n    \"Feature Details\": \"Natural gradient eyebrows with individual hair strands, natural lip color with subtle lip line definition.\"\n  },\n  \"Environment Setup\": {\n    \"Environment Scene\": \"Indoor studio setting, soft gray background, clean and uncluttered.\",\n    \"Lighting Effects\": \"Professional three-point lighting, key light creating butterfly lighting pattern, beautiful catchlights, high-key lighting, clean and bright.\"\n  },\n  \"Visual Design\": {\n    \"Color Scheme\": \"White and warm beige color scheme, natural realistic overall.\",\n    \"Typography Style\": \"N/A\",\n    \"Detail Elements\": \"Visible skin pores, natural skin texture, fine facial hairs, natural skin luminosity, subtle skin oil sheen on forehead and nose.\"\n  },\n  \"Category-specific\": {\n    \"Expression\": \"Subtle confident expression, slight head tilt.\",\n    \"Pose\": \"Facing camera directly.\",\n    \"Clothing Styling\": \"Cream cashmere sweater, minimal natural makeup.\",\n    \"Technical Parameters\": \"Ultra sharp, studio quality, 8K quality, medium format, natural skin texture rendering, fine organic film grain throughout, zero digital sharpening, all sharpness from optics, shadow details preserved, highlights soft without overexposure.\"\n  }\n}"
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
            "natural": "A cinematic portrait, an Asian woman looking back on a neon city street, 85mm telephoto lens, f/1.8 shallow depth of field, background neon bokeh creamy and dreamy. She has healthy natural warm-toned fair skin with subtle subcutaneous pinkish undertone, dewy luminous foundation texture, pores naturally visible with slight size variation on cheeks, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, fine wrinkles at outer eye corners and under eyes, eye iris deep brown color, tear trough area natural fine lines, lip texture with complete physical precision, lip color gradient from center to edge, upper lip fine vellus hair delicate visible. Warm streetlights from the side, creating perfect catchlights. She wears a beige trench coat, hair gently blowing in the breeze. Overall atmosphere warm nostalgic urban night, high contrast film look, fine organic film grain throughout, zero digital sharpening, all sharpness from optics. Technical parameters: ultra detailed, 8K quality, shadow details preserved, highlights soft without overexposure.",
            "structured": "{\n  \"high_level_description\": \"An Asian woman standing on a neon-lit urban street, telephoto lens capturing her looking back moment, background bokeh dreamy and ethereal.\",\n  \"style_description\": {\n    \"aesthetics\": \"cinematic, film grain, nostalgic, warm, fine organic film grain throughout, zero digital sharpening, all sharpness from optics\",\n    \"lighting\": \"warm street lamp side lighting, rim light on hair, soft bokeh background, 45-degree side-top lighting creating true 3D pore structure with micro-shadows\",\n    \"photo\": \"85mm lens, f/1.8, shallow depth of field, eye-level angle, ultra detailed, 8K quality\",\n    \"medium\": \"photograph\",\n    \"color_palette\": [\"#FFD700\", \"#FF6B6B\", \"#1A1A2E\", \"#F5F5F5\", \"#4A4A4A\"]\n  },\n  \"compositional_deconstruction\": {\n    \"background\": \"Neon-lit urban street, warm streetlamp bokeh, wet ground reflecting neon reflections, distant building silhouettes shrouded in mist.\",\n    \"elements\": [\n      {\n        \"type\": \"obj\",\n        \"bbox\": [100, 300, 900, 700],\n        \"desc\": \"Asian woman, close-up to medium shot, long hair gently blowing, healthy natural warm-toned fair skin with subtle subcutaneous pinkish undertone, pores naturally visible with slight size variation on cheeks, capillary flush near alae of nose, natural melanin gradient in mid-cheek area, fine wrinkles at outer eye corners and under eyes, subcutaneous capillary color variations, eye iris deep brown color, tear trough area natural fine lines, lip texture with complete physical precision, lip color gradient from center to edge, upper lip fine vellus hair delicate visible, wearing beige trench coat, looking back expression, eyes gentle and deep, delicate makeup.\",\n        \"color_palette\": [\"#FFDAB9\", \"#F5F5DC\", \"#8B4513\"]\n      }\n    ]\n  }\n}"
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
    "input_template": "Based on input type, generate {mode} high-resolution design prompts. User input: #\n\nType identification keywords: poster/advertisement; brochure/catalog; infographic/data visualization; courseware/education; product/3C/cosmetics; illustration/art; person description; scene description; if no type matches, generate appropriate design type name based on content and output using the same field structure, DO NOT force classification to similar types.",
    "output_format_suffix": {
        "natural": "Natural paragraph, emphasizing high-resolution details (2512x2512), material microscopic representation, light reflection, shadow gradient. No explanations.",
        "structured": "【Category】Category name\n【Basic Information】\n  - Design Type：Design type description\n  - Shot Perspective：Shot perspective description\n  - Composition Description：Composition description\n  - Subject Elements：Subject elements description\n【Visual Design】\n  - Color Scheme：Color scheme description\n  - Typography Style：Typography style description\n  - Detail Elements：Detail elements description\n【Technical Parameters】High-resolution features description"
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
            "category": "Poster",
            "natural": "Eye-level shot, medium shot composition, dramatic lighting. This is a poster design featuring a futuristic sports car racing through a neon-lit tunnel. The car has motion blur on the background and detailed reflections on the car body. Composition uses dynamic diagonal layout with the car at lower-right golden ratio point and 'SPEED' title in massive stylized font at top left. Color scheme features dark navy background with electric blue and hot pink neon accents, with metallic silver for the car. Typography uses custom bold sans-serif with neon glow for the title and clean modern font for details. Details include rain droplets on the lens, light trails, and ultra-sharp reflections. High-resolution highlights show fine carbon fiber texture, individual water drops, and crisp neon edges, perfect for high-resolution printing.",
            "structured": "【Shot Perspective】Eye-level shot, medium shot composition, dramatic lighting\n【Design Type】Poster design\n【Composition Description】Dynamic diagonal layout, car at lower-right golden ratio, title in massive stylized font at top left\n【Subject Elements】Futuristic sports car racing through neon-lit tunnel, motion blur background, detailed reflections on car body\n【Color Scheme】Dark navy background, electric blue and hot pink neon accents, metallic silver car\n【Typography Style】Title: custom bold sans-serif with neon glow, details: clean modern font at bottom\n【Detail Elements】Rain droplets on lens, light trails, ultra-sharp reflections on car surface\n【High-Resolution Features】Fine carbon fiber texture, individual water drops, crisp neon edges, perfect for 2512x2512 high-resolution printing"
        },
        {
            "category": "Product Render",
            "natural": "A 45-degree angle view of wireless noise-cancelling headphones floating on a gradient deep blue background. The headphones feature matte black plastic earcups with memory foam ear pads and brushed metal headband. Three-point lighting setup creates soft highlights from front-left, fill light from right to eliminate shadows, and backlight for contour. Mirror reflection beneath. High-resolution details reveal fine matte grain on earcups, visible air vents in foam padding, precise scale markings on headband extension, and consistent brushed metal texture direction.",
            "structured": "【Shot Perspective】45-degree angle view, floating display, mirror reflection below\n【Design Type】E-commerce product rendering\n【Composition Description】Centered product placement, minimalist background, negative space framing\n【Subject Elements】Wireless noise-cancelling headphones, matte black plastic earcups, memory foam ear pads, brushed metal headband\n【Color Scheme】Gradient deep blue background, charcoal gray body, silver metal accents\n【Typography Style】Clean modern sans-serif for product name and specifications\n【Detail Elements】Three-point lighting, soft front highlights, contour backlight, subtle reflection\n【High-Resolution Features】Matte grain visible at 2512×2512, individual foam pores, sharp scale markings, brush stroke direction consistent"
        },
        {
            "category": "Art Illustration",
            "natural": "A fantasy forest scene viewed from a low-angle perspective, featuring glowing giant mushrooms as foreground framing. A spirit girl sits on a branch with delicate translucent wings shimmering in bioluminescent light. Color palette dominated by blue-purple and teal tones with thick brush strokes preserving canvas texture. High-resolution details reveal individual scale particles on wings, intricate fluorescent vein patterns within mushroom caps, water droplet highlights on leaves, and volumetric light scattering through mist layers.",
            "structured": "【Shot Perspective】Low-angle upward view, foreground mushroom framing, depth layering\n【Design Type】Digital thick-paint illustration\n【Composition Description】Rule of thirds placement for spirit girl, foreground mushrooms create frame, waterfall in background\n【Subject Elements】Giant glowing mushrooms with translucent caps, spirit girl with silver hair and delicate wings, cascading waterfall\n【Color Scheme】Blue-purple primary, teal complementary, fluorescent green accents, ethereal white highlights\n【Typography Style】None, pure visual art\n【Detail Elements】Visible brush strokes, canvas texture, bioluminescent glow, volumetric fog\n【High-Resolution Features】Individual scale particles on wings visible, fluorescent veins thin as hair, brush bristle texture, water droplet edge definition, light scattering layers smooth without banding"
        }
    ]
}

QWEN_IMAGE_EDIT_COMBINED_EN = {
    "name": "Comprehensive Image Edit Enhancer",
    "description": "Professional editing prompt enhancer, generating precise, concise, direct editing prompts for add/delete/replace, style transfer, material replacement, content filling operations.",
    "input_template": "Generate precise {mode} editing prompt. User input: #\n\nTask types: add/delete/replace, text edit, person edit, style transfer, material replacement, logo/pattern, content fill, multi-image.",
    "output_format_suffix": {
        "natural": "Natural paragraph describing the editing operation, precise parameters, and visual consistency requirements. No explanations.",
        "structured": "【Task Type】...\n【Target Object】...\n【Operation Description】...\n【Parameter Requirements】...\n【Visual Consistency】..."
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
            "structured": "【Task Type】Add object\n【Target Object】Orange tabby cat\n【Operation Description】Add an orange tabby cat with green eyes on the wooden table to the right of the laptop, with realistic fur texture\n【Parameter Requirements】Lighting consistent with the scene, natural shadows preserved\n【Visual Consistency】Cat's size and coat color coordinated with indoor environment"
        }
    ]
}

LTX2_EN = {
    "name": "LTX-2 Video Generation Prompt Engineer",
    "description": "Optimized for LTX-2 video generation model, emphasizing dynamic content, temporal changes, camera movement (push/pull/pan/track), and audio elements. Translates abstract emotions into specific muscle/limb actions.",
    "input_template": "Optimize prompt for video generation, create {mode} description emphasizing dynamic content and camera movement. User input: #\n\nEmotion-to-action translation: shy→shoulders raised, chin drawn, eyes down 2s then up; smile→mouth corners lifted; angry→brows furrowed, pupils contracted, jaw tightened; surprised→eyes widened, brows raised.",
    "output_format_suffix": {
        "natural": "Natural paragraph describing scene, action, camera movement, lighting, atmosphere, audio. Integrate emotion-to-action translation. No explanations.",
        "structured": "【Scene Environment】...\n【Subject Action】...\n【Camera Movement】...\n【Lighting Effects】...\n【Atmosphere】...\n【Audio/Sound Effects】..."
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
    "description": "Adds cinematic aesthetics to text-to-video prompts. Intelligently identifies style (Sci-Fi/Action/Romance/Horror/Art House/Epic/Mystery/Documentary/Anime Film). Includes time setting, lighting, shot composition, color grading, camera movement, technical parameters.",
    "input_template": "Optimize video Prompt, first determine film style (choose one from: Sci-Fi, Action, Romance, Horror, Art House, Epic, Mystery, Documentary, Anime Film). If no style matches, generate appropriate film style name based on content and output using the same field structure, DO NOT force classification to similar styles. Then generate {mode} description. User input: #\n\nDo not change original meaning, add cinematic aesthetics. Must include technical parameters.",
    "output_format_suffix": {
        "natural": "Natural paragraph, incorporating cinematic elements, highlighting style characteristics.",
        "structured": "【Category】Film style\n【Scene Setup】\n  - Time Setting: ...\n  - Lighting Effects: ...\n  - Shot Composition: ...\n  - Color Atmosphere: ...\n【Technical Specifications】\n  - Camera Movement: ...\n  - Technical Parameters: ...\n  - Usage Method: ...\n\nStyle-specific fields:\n- Sci-Fi: Add \"Futuristic Elements\", \"Glowing/Holographic Effects\", \"Sci-Fi Sound Effects\"\n- Action: Add \"Explosion/Gunfight Elements\", \"Fast Cutting Rhythm\", \"Intense Music Style\"\n- Romance: Add \"Romantic Elements\", \"Soft Focus/Lens Flare\", \"Emotional Music\"\n- Horror: Add \"Dark/Shadow Elements\", \"Jump Scare Timing\", \"Eerie Sound Effects\"\n- Art House: Add \"Poetic Composition\", \"Natural/Realistic Lighting\", \"Negative Space/Slow Pace\"\n- Epic: Add \"Grand Scenes\", \"Golden Hour/Twilight\", \"Orchestral Score\"\n- Mystery: Add \"Asymmetrical Composition\", \"Low Key Lighting/Shadows\", \"Enhanced Ambient Sound\"\n- Documentary: Add \"Handheld Camera\", \"Natural Lighting\", \"Real Environmental Sound\"\n- Anime Film: Add \"Animation Style\", \"Line Treatment\", \"Flat Color\", \"Timing Chart Rhythm\"\n\nOnly output structured format, no extra explanations."
    },
    "cinematic_options": {
        "Time": ["Day", "Night", "Dawn", "Sunrise", "Golden Hour (30 minutes before sunset)"],
        "Light Source": ["Daylight", "Artificial light (tungsten)", "Moonlight", "Practical light (prop lights)", "Firelight", "Fluorescent"],
        "Lighting Angle": ["Top light", "Side light", "Bottom light", "Rim light", "Butterfly lighting", "Loop lighting"],
        "Color Tone": ["Warm tone (2700K)", "Cool tone (5600K)", "Mixed tones", "Teal & Orange contrast"],
        "Shot Size": ["Medium shot", "Medium close-up", "Wide shot", "Close-up", "Extreme close-up", "Extreme wide shot"],
        "Camera Angle": ["Over-the-shoulder", "Low angle", "High angle", "Dutch angle", "Aerial", "POV"],
        "Camera Movement": ["Dolly in", "Dolly out", "Pan", "Truck", "Follow", "Crane", "Roll", "Steadicam"],
        "Composition": ["Centered composition", "Balanced composition", "Symmetrical composition", "Diagonal composition", "Frame composition", "Rule of thirds"],
        "Special Effects": ["Anamorphic lens flare", "Film grain", "Shallow depth of field cinematic look", "HDR lighting", "Chromatic aberration"]
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
            "category": "Sci-Fi",
            "natural": "Nighttime, neon lights, cool tones, city skyline. A young girl stands on a transparent observation deck of a skyscraper, overlooking a futuristic city. Holographic advertisements float in the air, flying cars zip by. Camera slowly pushes in from behind her (5 seconds), then orbits around her (180 degrees, 4 seconds). A massive holographic Earth projection looms in the background. Lighting consists of cool practical lights (blue-purple) with rim lighting. Color grading features teal and magenta contrast with purple highlights. Shot on 35mm film with anamorphic lens flare, film grain, and cyberpunk color grading.",
            "structured": "【Category】Sci-Fi\n【Scene Setting】\n  - Time Setting：Nighttime\n  - Lighting Effects：Neon lights, cool practical lights (blue-purple), rim lighting\n  - Shot Composition：Medium shot, transparent observation deck, city background, shallow depth of field\n  - Color Atmosphere：Teal and magenta contrast, purple highlights, cyberpunk grading\n【Technical Parameters】35mm film, anamorphic lens flare, film grain；Camera Movement：Dolly in from behind (5 seconds), orbit 180 degrees (4 seconds)；Usage Method：Girl standing and overlooking, holographic ads floating, flying cars passing by\n【Futuristic Elements】Holographic projections, flying cars, transparent observation deck\n【Glowing/Holographic Effects】Glowing holographic ads, Earth hologram projection, lens flare effects\n【Sci-Fi Sound Effects】Deep electronic tones, vehicle engine sounds"
        },
        {
            "category": "Anime Film",
            "natural": "Golden hour, soft lighting, wide shot with deep depth of field. A girl runs across a flower-filled meadow with pink sunset sky and falling petals in the background. Camera slowly tracks her movement. Highly saturated colors with pink-purple gradient and lens flare. Digital animation with cel shading, no outlines, smooth lines. Two-frame timing for dreamy effect.",
            "structured": "【Category】Anime Film\n【Scene Setting】\n  - Time Setting：Golden hour\n  - Lighting Effects：Soft lighting, rim lighting\n  - Shot Composition：Wide shot, deep depth of field\n  - Color Atmosphere：Highly saturated, pink-purple gradient\n【Technical Parameters】Digital animation, no film grain；Camera Movement：Slow tracking shot；Usage Method：Girl running\n【Animation Style】Cel shading, 3D background\n【Line Treatment】No outlines, smooth\n【Flat Color】Gradient fills\n【Timing Chart Rhythm】Two-frame animation"
        }
    ]
}

WAN_I2V_EN = {
    "name": "First Frame Continuation Prompt Expert",
    "description": "Generates video description that naturally continues from first frame image, emphasizing subtle motion, micro-expressions, environmental dynamics. Supports 360-degree panorama mode.",
    "input_template": "Generate {mode} video continuation from first frame image. Image content: \nUser input: #\n\nScenario: If keywords [orbit, 360-degree, pan shot, panorama], generate panorama description: fixed camera pans 360° right, list spaces/objects in order. Otherwise, find most natural continuation.",
    "output_format_suffix": {
        "natural": "Natural paragraph describing picture story continuation: subject micro-movements, micro-expressions, hair/clothing movement, environmental lighting changes. No explanations.",
        "structured": "【Category】Realistic/Animated\n【Subject Information】\n  - Subject：Subject description\n  - Action：Action continuation description\n【Scene Dynamics】\n  - Scene Development：Scene development description\n  - Camera Movement：Camera movement description\n  - Atmosphere Change：Atmosphere change description"
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
            "structured": "【Subject Description】Woman wearing pearl necklace, standing by rain-streaked window\n【Action Continuation】Slowly turns head right, faint melancholic smile, eyes glisten, soft sigh, hand touches pearls\n【Scene Development】Rain continues streaking down window, ambient lighting soft\n【Camera Movement】Static hold, no movement\n【Atmosphere Change】Nostalgic, contemplative mood"
        },
        {
            "natural": "Fixed camera position (keeping the camera subject position unchanged), camera pans 360 degrees to the right, smoothly orbiting the entire space, [1. Moving bar cart; 2. Floor-to-ceiling window with viewing balcony area; 3. Electric smart curtains behind glass door; 4. TV wall and leisure area; 5. Leisure reading corner; 6. Vanity and independent bathroom area; 7. Independent bathroom; 8. Closet and bedroom entrance area (solid wood entrance door, door has peephole and electronic lock, next to door is built-in shoe cabinet, opposite entrance door is walk-in closet)], no motion blur.",
            "structured": "【Subject Description】Space starting position, showing room entrance\n【Action Continuation】Camera pans 360 degrees from starting position\n【Scene Development】1. Moving bar cart; 2. Floor-to-ceiling window with viewing balcony; 3. Electric smart curtains; 4. TV wall and leisure area; 5. Leisure reading corner; 6. Vanity and bathroom; 7. Independent bathroom; 8. Closet and bedroom entrance\n【Camera Movement】Fixed camera position, 360-degree pan shot to the right\n【Atmosphere Change】Overall stable, showcasing complete space"
        }
    ]
}

WAN_FLF2V_EN = {
    "name": "First-Last Frame Continuation Expert",
    "description": "Creates transition story between first and last frame images, filling visual differences with plausible actions and motion.",
    "input_template": "Generate {mode} transition description between first and last frame. First frame: \nLast frame: \nUser input: #\n\nScenario: If frames identical and keywords [orbit, 360-degree, panorama], generate 360° pan description. Otherwise, analyze visual differences and describe transition from state A to state B.",
    "output_format_suffix": {
        "natural": "Natural paragraph describing transition from first to last frame: identify visual differences, describe intermediate actions. No explanations.",
        "structured": "【Category】Realistic/Animated\n【Frame Information】\n  - First Frame：First frame description\n  - Last Frame：Last frame description\n【Transition Content】\n  - Transition：Transition action description\n  - Action Development：Action development description\n【Technical Parameters】Camera movement description；Atmosphere change description"
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
            "structured": "【First Frame Description】Serene young woman sitting on wooden boat dock\n【Last Frame Description】Paper boat floating away on lake\n【Transition Action】She picks up folded paper boat, leans forward, places it on water\n【Action Development】She slowly lifts right hand, places paper boat gently on water surface, boat drifts away with current\n【Camera Movement】Pans down to follow boat, then pans back up to her face\n【Atmosphere Change】From peaceful to wistful, nostalgic mood"
        },
        {
            "natural": "Fixed camera position (keeping the camera subject position unchanged), camera pans 360 degrees to the right, smoothly orbiting the entire space, [1. Moving bar cart; 2. Floor-to-ceiling window with viewing balcony area; 3. Electric smart curtains behind glass door; 4. TV wall and leisure area; 5. Leisure reading corner; 6. Vanity and independent bathroom area; 7. Independent bathroom; 8. Closet and bedroom entrance area (solid wood entrance door, door has peephole and electronic lock, next to door is built-in shoe cabinet, opposite entrance door is walk-in closet)], no motion blur.",
            "structured": "【First Frame Description】Space starting position, showing room entrance\n【Last Frame Description】Back to starting position, completing 360-degree orbit\n【Transition Action】Camera pans 360 degrees from starting position\n【Action Development】Camera smoothly rotates 360 degrees, capturing all eight areas in sequence at uniform speed\n【Camera Movement】Fixed camera position, 360-degree pan shot to the right\n【Atmosphere Change】Overall stable, showcasing complete space"
        }
    ]
}

VIDEO_FRAME_SEQUENCE_TO_PROMPT_EN = {
    "name": "Video Frame Sequence Analysis Expert (Smart Category Version)",
    "description": "Analyzes dynamic changes by frame intervals, automatically identifies realistic/animated categories, merges adjacent similar frames, outputs JSON format. Emotions translated into actions.",
    "input_template": "Analyze video frame sequence by intervals, first determine overall category (realistic/animated). Mark each segment with 【Frame X-Y】. Describe each segment: scene, character, action, camera movement, color atmosphere. Merge adjacent similar frames. Output {mode}.",
    "output_format_suffix": {
        "natural": "One complete paragraph per interval, start with 【Frame X-Y】, naturally integrate all information. No explanations.",
        "structured": "【Category】Realistic/Animated\n【Basic Info】\n  - Frame Range: Frame X-Y\n  - Shot Type: Shot type description\n  - Camera Movement: Camera movement description\n【Content Description】\n  - Scene: Scene description\n  - Character: Character/Role description\n  - Action: Action description\n  - Atmosphere: Atmosphere/Color description"
    },
    "task_requirements": [
        "Segment by frame intervals, mark frame ranges",
        "Merge adjacent frames with similar content (less than 10% pixel change)",
        "Each segment must include: scene, character, action, camera movement, color atmosphere",
        "Action descriptions must be dynamic, avoid static descriptions",
        "First determine if realistic or animated, use corresponding terminology",
        "Natural mode: output natural paragraph; structured mode: output structured prompt"
    ],
    "constraints": {"max_length": 800},
    "examples": [
        {
            "category": "Realistic",
            "natural": "【Frame 1-30】Indoor office scene, golden shoulder-length hair woman, serious expression, light makeup, wearing white shirt and black suit jacket, wearing thin-framed glasses. She sits upright, hands folded on desk. Camera fixed front medium shot, formal solemn atmosphere, warm bright lighting. No dynamic changes.",
            "structured": "【Frame Sequence】\n  - 【Frame 1-30】\n    - Category：Realistic\n    - Shot Type：Medium shot\n    - Camera Movement：Fixed\n    - Scene：Office\n    - Character：Blonde woman, white shirt\n    - Action：Sitting, hands folded\n    - Atmosphere：Warm tones, bright"
        },
        {
            "category": "Animated",
            "natural": "【Frame 1-24】Classroom scene, close-up shot, push-pull camera movement. A black-haired boy in school uniform sits at desk, suddenly turns head, showing surprised expression, pupils dilated. Highly saturated colors, cel shading lighting effects.",
            "structured": "【Frame Sequence】\n  - 【Frame 1-24】\n    - Category：Animated\n    - Shot Type：Close-up\n    - Camera Movement：Push-pull\n    - Scene：Classroom\n    - Character：Boy, school uniform, black hair\n    - Action：Turning head, surprised expression\n    - Atmosphere：Highly saturated, cel shading"
        }
    ]
}

VIDEO_TO_PROMPT_EN = {
    "name": "Video Reverse Prompt Expert (Smart Category Version)",
    "description": "Analyzes video content, automatically identifies category (realistic/animated), generates detailed video description prompts. Outputs JSON format, emotions translated into actions.",
    "input_template": "Analyze video, first determine category (realistic, animated). Then generate {mode} description. User input: #\n\nUse photography/cinema terminology for realistic videos, use anime terminology (cel shading, lines, flat color) for animated videos. Output format see below.",
    "output_format_suffix": {
        "natural": "Natural paragraph description.",
        "structured": "【Category】Realistic/Animated\n【Description】\n  - Scene Description: ...\n  - Character Description: ...\n  - Action Description: ...\n  - Camera Description: ...\n  - Atmosphere Description: ...\n  - Technical Parameters: ..."
    },
    "task_requirements": [
        "First determine if video is realistic or animated",
        "Use corresponding professional terminology",
        "Action descriptions must be dynamic (speed, direction, time)",
        "Omit character field when no characters present",
        "Natural mode: output natural paragraph; structured mode: output structured prompt"
    ],
    "constraints": {"max_length": 800},
    "examples": [
        {
            "category": "Realistic",
            "natural": "Japanese Zen garden scene, first morning sunlight filters through bamboo groves, light and shadow slowly moving across gravel. Koi swim in pond, tails swaying left and right at 0.5Hz frequency, creating ripples that spread outward. Incense smoke rises from burner, drifting in wind. Camera slowly pushes in (dolly in, speed 1cm/s), then orbits garden (pan speed 30 degrees/s). Serene Zen atmosphere, soft morning light, color temperature 5500K, dominated by green and gold tones.",
            "structured": "【Category】Realistic\n【Description】\n  - Scene Description：Japanese Zen garden, morning sunlight filtering through bamboo\n  - Character Description：No characters\n  - Action Description：Koi swimming (tail movement 0.5Hz), smoke drifting\n  - Camera Description：Dolly in at 1cm/s, then pan at 30 degrees/s\n  - Atmosphere Description：Serene, color temperature 5500K, green and gold tones\n  - Technical Parameters：35mm film, natural lighting"
        },
        {
            "category": "Animated",
            "natural": "Cel shading animation style, clean line art. Black-haired girl in sailor uniform, large eyes with star-shaped highlights. She slowly turns her head (1 second duration), black hair flowing in wind (approx 5cm amplitude). Close-up shot, slightly low angle, speed lines in background. Highly saturated colors, soft lens flare, pink gradient background, cherry blossom petals falling. No outlines, uniform line thickness.",
            "structured": "【Category】Animated\n【Description】\n  - Art Style：Cel shading, clean line art, anime style\n  - Character Description：Black-haired girl, sailor uniform, large eyes with star highlights\n  - Action Description：Head turn (1 second), hair flowing (5cm amplitude)\n  - Camera Description：Close-up, low angle, speed lines background\n  - Color & Lighting：Highly saturated, soft lens flare, pink gradient\n  - Line Style：No outlines, uniform thickness\n  - Background Details：Cherry blossom petals falling, simple gradient"
        }
    ]
}

VIDEO_DETAILED_SCENE_BREAKDOWN_EN = {
    "name": "Video Detailed Scene Breakdown Expert (Smart Category Version)",
    "description": "Analyze each scene in chronological order, automatically identify realistic/animated categories, generate shot analysis format. Supports time slicing format. Emotions translated into actions.",
    "input_template": "Generate {mode} description for each scene in chronological order. First determine overall category (realistic/animated). User input: #\n\nCore principle: Emotion adjectives must be translated into specific actions.",
    "output_format_suffix": {
        "natural": "One complete paragraph per scene, start with Time (0:00-0:15), naturally integrate all information. No explanations.\nEmotion translation examples: shy→shoulders raised, chin drawn, eyes down 2s then up; smile→mouth corners lifted.",
        "structured": "【Shot Number】\n  - Category: Realistic/Animated\n  - Time: Start-End\n  - Shot Type: Shot type description\n  - Camera Movement: Camera movement description\n  - Content: Analysis content description\n  - Rhythm: Editing rhythm description"
    },
    "task_requirements": [
        "Each scene one complete English description (natural paragraph) or output by structured fields",
        "Structured output focuses on extracting camera language and narrative content",
        "Natural paragraph describes scene, character, action, lighting, color, camera movement, atmosphere in detail",
        "Time precise to seconds or milliseconds",
        "First determine if realistic or animated, use corresponding terminology",
        "Natural mode: output natural paragraph; structured mode: output structured prompt"
    ],
    "constraints": {"max_length": 800},
    "examples": [
        {
            "category": "Realistic",
            "natural": "**Time:** 0:00-0:15\nGrandmother stands in kitchen center, angrily unties apron (2s), grabs rolling pin (0.5s), strides toward living room (speed 1m/s). Medium shot fixed camera, then slowly pans following grandmother's action (pan speed 20 deg/s), expressing her protective anger.\n\n**Time:** 0:15-0:30\nClose-up, grandson panicked running around room (speed 1.5m/s), looking back frequently (0.5Hz). Camera quickly follows grandson (tracking speed 1.5m/s), editing rhythm speeds up (cuts every 2s), creating tense and humorous atmosphere.",
            "structured": "【Shot 1】\n  - Category：Realistic\n  - Time：0:00-0:15\n  - Shot Type：Medium shot\n  - Camera Movement：Fixed/Pan (20 deg/s)\n  - Content：Grandmother angrily unties apron (2s), grabs rolling pin (0.5s), strides toward living room (1m/s)\n  - Rhythm：Smooth action, no cuts\n【Shot 2】\n  - Category：Realistic\n  - Time：0:15-0:30\n  - Shot Type：Close-up\n  - Camera Movement：Follow (1.5m/s)\n  - Content：Grandson panicked, dodging while looking back (0.5Hz), grandmother in pursuit\n  - Rhythm：Fast cuts (every 2s), creating tense and humorous atmosphere"
        },
        {
            "category": "Animated",
            "natural": "**Time:** 0:00-0:05\nClose-up shot, fixed. Female protagonist's pupils dilate, a tear slides down cheek (speed 0.5cm/s). Slow motion, lens flare in background. Sad atmosphere.",
            "structured": "【Shot 1】\n  - Category：Animated\n  - Time：0:00-0:05\n  - Shot Type：Close-up\n  - Camera Movement：Fixed\n  - Content：Female protagonist's pupils dilate, tear sliding down\n  - Rhythm：Slow motion"
        }
    ]
}

VIDEO_SUBTITLE_FORMAT_EN = {
    "name": "Video Subtitle Format Optimization Expert",
    "description": "Video Subtitle Format Optimization Expert, converting subtitle content into standard format, ensuring timecode and text synchronization. Supports Four-Element Method for tone and speed control.",
    "input_template": "Optimize subtitle format, generate {mode} output. User input: #\n\nFour-Element Method: Vocal Style + Rhythm Description + Pitch Variation + Punctuation. Output directly without JSON.",
    "output_format_suffix": {
        "natural": "Format: **Timecode:** 00:00:00,000 --> 00:00:05,000\\n**Subtitle Text:** ...\\nSeparate with blank line.",
        "structured": "【Subtitle 1】\n  - Timecode: 00:00:00,000 --> 00:00:03,500\n  - Line Tone: Raised breath, steady rhythm + flat intonation\n  - Subtitle Text: Hello everyone, welcome to my channel!\n【Subtitle 2】\n  - Timecode: 00:00:03,500 --> 00:00:07,200\n  - Line Tone: Breath voice, choked + gradually slowing + falling intonation\n  - Subtitle Text: Today I want to... share a... simple home-cooked dish...\n【Subtitle 3】\n  - Timecode: 00:00:07,200 --> 00:00:11,800\n  - Line Tone: Raised breath, accelerated rhythm + rising intonation\n  - Subtitle Text: First, we need to prepare some basic ingredients!"
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
        "structured": "【Dialogue 1】\n  - Voice: Female\n  - Speaker ID: 0\n  - Emotion: Gentle\n  - Line Tone: Raised breath, steady rhythm + flat intonation\n  - Dialogue: Mom: Honey, it's time to wake up, you'll be late for school!\n【Dialogue 2】\n  - Voice: Loli\n  - Speaker ID: 2\n  - Emotion: Sleepy\n  - Line Tone: Breath voice, dragging + falling intonation\n  - Dialogue: Daughter: Mom, let me sleep five more minutes...\n【Dialogue 3】\n  - Voice: Male\n  - Speaker ID: 1\n  - Emotion: Calm\n  - Line Tone: Throaty voice, steady rhythm + flat intonation\n  - Dialogue: Dad: You lazy bones, you'll miss breakfast if you don't get up now."
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
        "structured": "【Verse 1】\n  - Vocal Tone: Lowered voice, steady rhythm + flat intonation\n  - Lyrics: Streets quiet as evening lights begin to glow, walking alone through alleys I used to know, memories flood like tides into my mind, those moments with you I left behind\n【Chorus】\n  - Vocal Tone: Breath voice, choked + gradually slowing + falling intonation\n  - Lyrics: Nights thinking of you... so endless and long, every star whispers your name in song, your smile your warmth so precious to me, the most treasured memories I'll always hold dearly\n【Theme】Longing and waiting\n【Style】Pop ballad\n【Mood】Deeply affectionate"
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
    "description": "Precisely extract all text content from images, including font, color, position and other style information, output structured prompt format.",
    "input_template": "Extract all text from image. User input: image#\n\nOutput structured prompt format, structure as follows:\n【Title】\n  - Content: ...\n  - Font: ...\n  - Color: ...\n  - Position: ...\n【Subtitle】\n  - Content: ...\n  - Font: ...\n  - Color: ...\n  - Position: ...\n【Body Text】\n  - Content: ...\n  - Font: ...\n  - Color: ...\n  - Position: ...\n【Slogans】...\n【Other Text】...\nNo extra explanations.",
    "output_format_suffix": {
        "natural": "Output in structured prompt format.",
        "structured": "【Title】\n  - Content: Summer Sale\n  - Font: Bold handwritten (Brush Script)\n  - Color: #FF6600 (Orange)\n  - Position: Top center\n【Subtitle】\n  - Content: Up to 50% off\n  - Font: Minimalist sans-serif\n  - Color: White\n  - Position: Below title\n【Body Text】\n  - Content: Event period: July 1 - July 15\n  - Font: Regular sans-serif\n  - Color: #333333\n  - Position: Middle\n【Body Text】\n  - Content: Limited time offer, limited quantity\n  - Font: Regular sans-serif\n  - Color: #333333\n  - Position: Lower middle\n【Slogans】Limited time offer, Don't miss out next year\n【Other Text】www.example.com, Customer service: 400-123-4567"
    },
    "task_requirements": [
        "Precisely recognize all text: title, subtitle, body text, slogans, labels",
        "Recognize font characteristics (serif/sans-serif/handwritten/bold etc.), color (HEX or description), layout position",
        "Differentiate text hierarchy, restore artistic font descriptions as much as possible",
        "Both natural mode and structured mode output structured prompt format"
    ],
    "constraints": {"max_length": 500},
    "examples": [
        {
            "natural": "【Title】\n  - Content: Summer Sale\n  - Font: Bold handwritten (Brush Script)\n  - Color: #FF6600 (Orange)\n  - Position: Top center\n【Subtitle】\n  - Content: Up to 50% off\n  - Font: Minimalist sans-serif\n  - Color: White\n  - Position: Below title\n【Body Text】\n  - Content: Event period: July 1 - July 15\n  - Font: Regular sans-serif\n  - Color: #333333\n  - Position: Middle\n【Body Text】\n  - Content: Limited time offer, limited quantity\n  - Font: Regular sans-serif\n  - Color: #333333\n  - Position: Lower middle\n【Slogans】Limited time offer, Don't miss out next year\n【Other Text】www.example.com, Customer service: 400-123-4567"
        }
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
    "[Design] Ideogram-4": "IDEOGRAM4_EN",
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