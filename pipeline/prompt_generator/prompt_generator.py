# -*- coding: utf-8 -*-
"""
PromptGenerator 图像提示词生成工具

【命令行用法】

    python pipeline\prompt_generator\prompt_generator.py 分段文本输入.txt --output 图像提示词输出.txt

【参数说明】
- input_path：必填，分段文本输入文件路径（每行为一个分段）
- --output：必填，图像提示词输出文件路径

配置文件路径：
- 默认读取 codebase根目录\config\prompt_generator.yaml
- 可通过设置环境变量 $env:PROMPT_GENERATOR_CONFIG_PATH 指定自定义配置文件路径
"""
from openai import OpenAI
import os
import yaml
import concurrent.futures

# 读取配置文件
CONFIG_PATH = os.environ.get(
    'PROMPT_GENERATOR_CONFIG_PATH',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'prompt_generator.yaml')
)
print(f"[INFO] 正在读取配置文件: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
print("[INFO] 配置文件读取成功")

API_BASE = config.get('api_base')  # openai兼容API地址
API_KEY = config.get('api_key')    # 大模型API密钥
MODEL = config.get('model')        # 使用的大模型名称
THREAD_NUM = config.get('thread_num', 5)
STYLE = config.get('style', 'Ghibli style')

missing = []
if not API_BASE:
    missing.append('api_base')
if not API_KEY:
    missing.append('api_key')
if not MODEL:
    missing.append('model')
if missing:
    print(f"[ERROR] 配置文件缺少以下必要项：{', '.join(missing)}，请检查config/prompt_generator.yaml配置后重试！")
    import sys
    sys.exit(1)

# 构造图像提示词生成的提示模板
PROMPT_TEMPLATE = '''
你是一个图像提示词生成专家，请根据我提供的文字稿生成对应的图像提示词。请遵循以下要求：

1. 提取文字稿的核心主题、场景、情感、人物、时间等要素，并确保这些要素清晰地反映在图像提示词中。
2. 生成一个具体且详细的图像提示词，确保其内容与文字稿紧密相关，能够准确传达段落的意图和氛围。
3. 每个图像提示词要具备足够的描述信息，以确保图像生成模型能够生成符合文字稿内容的准确图像。
4. 确保每个段落的图像提示词符合该文字稿的独特背景和情感基调。

以下是文字稿内容：
{content}

请根据上述要求生成{style}风格的图像提示词。

返回结果时，请用中文回复，只需要返回图像提示词，不要返回任何其他内容。
'''

def read_segment_file(input_path):
    """读取分段文本文件，每行为一个分段，返回分段列表"""
    print(f"[INFO] 正在读取分段文本文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        segments = [line.strip() for line in f if line.strip()]
    print(f"[INFO] 共读取到{len(segments)}个分段")
    return segments

def call_openai_api(prompt):
    """调用openai兼容API生成图像提示词，使用openai库"""
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content

def generate_prompts(input_path, output_path):
    """主流程：读取分段，使用多线程并行生成图像提示词，写入输出文件"""
    segments = read_segment_file(input_path)
    results = [None] * len(segments)
    def process(idx_seg):
        idx, seg = idx_seg
        print(f"[INFO] 正在处理第{idx+1}个分段...")
        prompt = PROMPT_TEMPLATE.format(content=seg, style=STYLE)
        result = call_openai_api(prompt)
        print(f"[INFO] 第{idx+1}个分段处理完成")
        return idx, result
    print(f"[INFO] 并发线程数设置为: {THREAD_NUM}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_NUM) as executor:
        futures = [executor.submit(process, (idx, seg)) for idx, seg in enumerate(segments)]
        for future in concurrent.futures.as_completed(futures):
            idx, result = future.result()
            results[idx] = result
    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(results, 1):
            f.write(f"{idx}. {item.strip()}\n")
    print(f"[INFO] 所有分段已处理完毕，结果已写入: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PromptGenerator：图像提示词生成工具')
    parser.add_argument('input_path', help='分段文本输入文件路径（每行为一个分段）')
    parser.add_argument('--output', required=True, help='图像提示词输出文件路径')
    args = parser.parse_args()
    generate_prompts(args.input_path, args.output) 