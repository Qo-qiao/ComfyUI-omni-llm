# ComfyUI-llama-cpp-vlmforQo

在ComfyUI中基于llama.cpp原生运行LLM/VLM模型。  
**\[**[**📃English**](./README.md)**]**

## 更新日志

#### 2026-01-17
- 添加了对llama-joycaption反推模型的支持类型，个人推荐使用Qwen3VL解禁版模型
- 添加了mmproj模型开关，为了支持纯文本生成
- 添加了清理会话节点（释放当前对话占用的资源，减少不出结果的情况）
- 添加了卸载模型节点（减少显存占用）
- 添加了硬件优化模块，适配高低不同性能硬件，提高推理速度，确保不同硬件都能流畅使用
- 重写了Prompt Style预设信息


llama-joycaption模型下载地址：https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF/tree/main
【选择适合自己电脑显卡的量化模型包，只下一个就可例如：llama-joycaption-beta-one-hf-llava.Q4_K_M.gguf】
mmproj模型下载地址：https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf/tree/main
【下载这个模型：llama-joycaption-beta-one-llava-mmproj-model-f16.gguf】

Qwen3VL解禁版模型下载地址：
https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF/tree/main

## 致谢
* [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) @lihaoyun6
* [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
* [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
* [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
