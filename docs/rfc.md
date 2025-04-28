## 2. RFC ——《段落-配图-配音同步技术方案》

### 2.1 概述
提出一种 **“分段-并行生成-同步合成”** 的流水线，解决多段图像与 TTS 音频精准对齐难题，同时保持高吞吐与低成本。

### 2.2 架构总览
```
User Script
   │
   ├─► ① Segmenter (GPT-4)
   │       └── segments.json  # [{id,text}]
   │
   ├─► ② Prompt Generator (GPT-4)
   │       └── prompts.json   # [{id,prompt}]
   │
   ├─► ③ Image Renderer (SDXL)
   │       └── img_<id>.png
   │
   ├─► ④ TTS Engine (Azure/Polly)
   │       └── audio_<id>.mp3 + <duration>
   │
   ├─► ⑤ Clip Assembler (moviepy)
   │       └── seg_<id>.mp4  # image held for duration
   │
   └─► ⑥ Concatenator (FFmpeg)
           └── final.mp4
```

### 2.3 关键模块

| 模块 | 输入 | 处理 | 输出 | 技术要点 |
| ---- | ---- | ---- | ---- | -------- |
| Segmenter | `script.txt` | GPT-4 分段 + 正则 fallback | `segments.json` | 限 200 字，情感标注 |
| PromptGenerator | 段落文本 | GPT-4 `system+user` prompt | `prompts.json` | 加入 `style: Ghibli` token |
| ImageRenderer | prompt | SDXL, 1024×576, CFG=7 | `img_n.png` | FP16，8 G GPU < 6 s |
| TTSEngine | 段落文本 | Azure Neural TTS (viseme off) | `audio_n.mp3`, dur | Pitch + Style 根据情感 |
| ClipAssembler | image+audio | moviepy `ImageClip.set_audio()` | `seg_n.mp4` | img duration=audio.duration |
| Concatenator | N 段 mp4 | FFmpeg concat demuxer | `final.mp4` | 无转码；保持 H.264 CBR |

#### 2.3.1 Segmenter模块设计


1. 使用python语言实现
2. 输入为用户给的txt文件，内容是口播稿的文案
3. 实现逻辑：调用大模型api对口播稿文案进行分段处理，这里大模型的提示词如下：
   ```
    请将以下文案拆分成多个简短的段落，分段原则如下：

    1. 每个段落应围绕一个独立的主题、场景或故事，避免一个段落包含多个主题。
    2. 每个段落的字数不超过200字，确保段落简洁明了。
    3. 每当场景、主题或情感发生明显切换时，应开始新的段落。
    4. 段落之间的过渡要自然流畅，避免突兀的切换。

    请确保不修改原文内容，只进行段落切分。

    以下是要拆分的文本：
    [插入长文案]

    返回结果时，请按照以下格式提供拆分后的段落：
    1. 段落1...
    2. 段落2...
    3. 段落3...
    ...
   ```
4. 使用哪个大模，以及对应的key均使用配置文件配置，不在代码里面硬编码
5. 调用大模型使用openai的兼容的接口协议的方式，可参考如下代码
    ```python

    # Please install OpenAI SDK first: `pip3 install openai`

    from openai import OpenAI

    client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )

    print(response.choices[0].message.content)

    ```
#### 2.3.2 PromptGenerator模块设计

1. 继续使用python语言实现
2. 调用大语言模型对2.3.1输出的分段文本进行如下处理

   2.1 一次调用只读取1个分段文本，按照顺序读取，比如第一次读取第1个分段，第2次读取第2个分段

   2.2 对本次读取的分段文本，使用如下提示词调用大模型： 
   ```
    你是一个图像提示词生成专家，请根据我提供的文字稿生成对应的图像提示词。请遵循以下要求：

    1. 提取文字稿的核心主题、场景、情感、人物、时间等要素，并确保这些要素清晰地反映在图像提示词中。
    2. 生成一个具体且详细的图像提示词，确保其内容与文字稿紧密相关，能够准确传达段落的意图和氛围。
    3. 每个图像提示词要具备足够的描述信息，以确保图像生成模型能够生成符合文字稿内容的准确图像。
    4. 确保每个段落的图像提示词符合该文字稿的独特背景和情感基调，优先使用吉卜力风格（Ghibli style）进行图像风格设计。

    以下是文字稿内容：
    [插入分段文本]

    请根据上述要求生成图像提示词。
   ```
3. 将每次处理返回的结果保存一个字符串数组，所有的分段文本处理完了之后，统一存到结果路径。
4. 使用哪个大模型，以及对应的key均使用配置文件配置，不在代码里面硬编码
5. 使用openai sdk调用大模型

#### 2.3.2 ImageRenderer模块设计

1. 使用python语言实现
2. 一次调用只读取2.3.2生成结果文件中的一个提示词，且按顺序读取
3. 读取了图像生成提示词，请参考如下代码生成图像，生成的图像放到输出目录下，命令以2.3.2提示词id来命名：
    
    ```python
        from __future__ import print_function

        from volcengine.visual.VisualService import VisualService

        if __name__ == '__main__':
            visual_service = VisualService()

            # call below method if you don't set ak and sk in $HOME/.volc/config
            visual_service.set_ak('your ak')
            visual_service.set_sk('your sk')
            
            # 请求Body(查看接口文档请求参数-请求示例，将请求参数内容复制到此)
            form = {
                "req_key": "xxx",
                # ...
            }

            resp = visual_service.cv_process(form)
            print(resp)

    ```
4. 通过配置文件配置ak、sk



### 2.4 同步机制
1. **权威时长**：以 TTS 输出的 `audio.duration` 作为时间轴。  
2. `ImageClip(duration=audio.duration)` 保证图像帧数与音频帧数一致。  
3. 统一 FPS=30，避免不同段落因帧率差异出现黑帧。  
4. 可选 **字幕 burn-in**：使用 `moviepy.SubtitlesClip` 按段内逐词时间戳渲染，进一步增强可视-听对齐。

### 2.5 自动化与容错
- **调度**：Celery + Redis；任务原子化到段落级，失败只重跑失败段。  
- **缓存**：对相同脚本文本 + 风格的提示词与图像做哈希缓存，节省 GPU。  
- **监控**：Prometheus 采集 GPU/队列长度；Grafana 告警。  

### 2.6 性能估算（3 分钟脚本，6 段）
| 步骤 | 单段时长 | 并行度=6 | 总用时 |
| ---- | -------- | -------- | ------ |
| GPT-4 分段 | 2 s | 串行 | 2 s |
| GPT-4 提示词 | 2 s | 并行 | 2 s |
| SDXL 生图 | 6 s | 并行 | 6 s |
| TTS | 3 s | 并行 | 3 s |
| 片段合成 | 1 s | 并行 | 1 s |
| 合片 | 2 s | - | 2 s |
| **总计** | | | **≈ 16 s**（不含队列等待） |

### 2.7 开放议题
| # | 议题 | 影响 | 待定方案 |
| - | ---- | ---- | -------- |
| O-1 | SDXL 授权模型缓存策略 | 成本 | 私有 checkpoint vs. API |
| O-2 | 支持视频风格动态切换 | 产品 | 多 prompt 模板 or ControlNet |
| O-3 | BGM 自动匹配 | 体验 | beat-tracking 混音 |

### 2.8 版本演进
- **v1.0**：静态图 + 旁白  
- **v1.1**：字幕 / BGM / 竖屏  
- **v2.0**：多图滚动、Ken-Burns 动效、口型合成

---

### 使用说明（最小可运行 Demo）
```bash
# 1. 安装依赖
pip install openai moviepy diffusers torch
# 2. 运行 pipeline
python run_pipeline.py --script script.txt --style ghibli
# 3. 输出
dist/final.mp4
```

如需深度集成，可拆分各子模块为微服务，通过 gRPC 或 REST 互联。

---


**完毕。如有补充或变更，请在 RFC 讨论区留言！**