# -*- coding: utf-8 -*-
"""
Segmenter 口播稿分段工具

【命令行用法】
    # 仅打印分段结果到控制台
    python pipeline\segment\segmenter.py 输入文案.txt

    # 分段结果输出到指定文件
    python pipeline\segment\segmenter.py 输入文案.txt --output 输出结果.txt

【参数说明】
- txt_path：必填，输入的txt文案文件路径
- --output：可选，分段结果输出文件路径

配置文件路径：
- 默认读取 codebase根目录\config\segmenter.yaml
- 可通过设置环境变量 $env:SEGMENTER_CONFIG_PATH 指定自定义配置文件路径

"""
from openai import OpenAI
import os
import yaml

# 读取配置文件
CONFIG_PATH = os.environ.get(
    'SEGMENTER_CONFIG_PATH',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'segmenter.yaml')
)
print(f"[INFO] 正在读取配置文件: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
print("[INFO] 配置文件读取成功")

API_BASE = config.get('api_base')  # openai兼容API地址
API_KEY = config.get('api_key')    # 大模型API密钥
MODEL = config.get('model')        # 使用的大模型名称

missing = []
if not API_BASE:
    missing.append('api_base')
if not API_KEY:
    missing.append('api_key')
if not MODEL:
    missing.append('model')
if missing:
    print(f"[ERROR] 配置文件缺少以下必要项：{', '.join(missing)}，请检查config/segmenter.yaml配置后重试！")
    import sys
    sys.exit(1)

# 构造分段提示词
SEGMENT_PROMPT = '''
请将以下文案拆分成多个简短的段落，分段原则如下：

1. 每个段落应围绕一个独立的主题、场景或故事，避免一个段落包含多个主题。
2. 每个段落的字数不超过200字，确保段落简洁明了。
3. 每当场景、主题或情感发生明显切换时，应开始新的段落。
4. 段落之间的过渡要自然流畅，避免突兀的切换。

请确保不修改原文内容，只进行段落切分。

以下是要拆分的文本：
{content}

返回结果时，请按照以下格式提供拆分后的段落：
1. 段落1...
2. 段落2...
3. 段落3...
...'''

def read_txt_file(txt_path):
    """读取txt文件内容"""
    print(f"[INFO] 正在读取输入文案文件: {txt_path}")
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print("[INFO] 输入文案读取成功")
    return content

def call_openai_api(prompt):
    """调用openai兼容API进行分段，使用openai库"""
    print("[INFO] 正在调用大模型API进行分段...")
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    print("[INFO] 大模型API调用完成")
    return response.choices[0].message.content

def segment_txt(txt_path, output_path=None):
    """主流程：读取txt，分段，输出结果"""
    content = read_txt_file(txt_path)
    prompt = SEGMENT_PROMPT.format(content=content)
    result = call_openai_api(prompt)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"[INFO] 分段结果已写入文件: {output_path}")
    else:
        print("[INFO] 分段结果如下：")
        print(result)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segmenter：口播稿分段工具')
    parser.add_argument('txt_path', help='输入的txt文案文件路径')
    parser.add_argument('--output', help='分段结果输出文件路径，可选')
    args = parser.parse_args()
    segment_txt(args.txt_path, args.output) 