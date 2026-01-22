# ComfyUI-llama-cpp-vlmforQo  
Run LLM/VLM models natively in ComfyUI based on llama.cpp  
**[[📃中文版](./README_zh.md)]**

## Changelog  
#### 2026-01-17  
- Added support for llama-joycaption reverse model, personal recommendation: Qwen3VL unrestricted model
- Added mmproj model switch to support pure text generation
- Added clean session node (releases resources occupied by current conversation, reduces cases of no results)
- Added unload model node (reduces VRAM usage)
- Added hardware optimization module to adapt to different performance hardware, improve inference speed, and ensure smooth usage on different hardware
- Rewrote Prompt Style preset information

## Model Downloads  
- **llama-joycaption model**: [Hugging Face Link](https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF/tree/main)  
  - Select a quantization model package suitable for your computer's GPU (e.g., llama-joycaption-beta-one-hf-llava.Q4_K_M.gguf)
- **mmproj model**: [Hugging Face Link](https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf/tree/main)  
  - Download: llama-joycaption-beta-one-llava-mmproj-model-f16.gguf
- **Qwen3VL unrestricted model**: [Hugging Face Link](https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF/tree/main)

## Preview  
![](./img/preview.jpg)

## Installation  

#### Install the node:  
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp-vlmforQo.git
python -m pip install -r ComfyUI-llama-cpp-vlmforQo/requirements.txt
```

#### Download models:  
- Place your model files in the `ComfyUI/models/LLM` folder.  

> If you need a VLM model to process image input, don't forget to download the `mmproj` weights.

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
