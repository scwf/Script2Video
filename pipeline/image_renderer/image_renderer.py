# -*- coding: utf-8 -*-
"""
ImageRenderer 图像生成工具

【命令行用法】

    python pipeline\image_renderer\image_renderer.py 图像提示词输入.txt --output_dir 输出图片目录

【参数说明】
- input_path：必填，图像提示词输入文件路径（每行为一个提示词）
- --output_dir：必填，生成图片的输出目录

配置文件路径：
- 默认读取 codebase根目录\config\image_renderer.yaml
- 可通过设置环境变量 $env:IMAGE_RENDERER_CONFIG_PATH 指定自定义配置文件路径
"""
import os
import yaml
from volcengine.visual.VisualService import VisualService
import concurrent.futures
import base64
import time

# 读取配置文件
CONFIG_PATH = os.environ.get(
    'IMAGE_RENDERER_CONFIG_PATH',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'image_renderer.yaml')
)
print(f"[INFO] 正在读取配置文件: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
print("[INFO] 配置文件读取成功")

AK = config.get('ak')
SK = config.get('sk')
WIDTH = config.get('width')
HEIGHT = config.get('height')
MODEL_NAME = config.get('model_name')
THREAD_NUM = config.get('thread_num', 5)

missing = []
if not AK:
    missing.append('ak')
if not SK:
    missing.append('sk')
if missing:
    print(f"[ERROR] 配置文件缺少以下必要项：{', '.join(missing)}，请检查config/image_renderer.yaml配置后重试！")
    import sys
    sys.exit(1)

def read_prompt_file(input_path):
    """读取图像提示词文件，每行为一个提示词，返回列表"""
    print(f"[INFO] 正在读取图像提示词文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"[INFO] 共读取到{len(prompts)}个提示词")
    return prompts

def render_images(input_path, output_dir):
    """主流程：读取提示词，多线程并发生成图像，保存到输出目录"""
    prompts = read_prompt_file(input_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    visual_service = VisualService()
    visual_service.set_ak(AK)
    visual_service.set_sk(SK)
    def process(idx_prompt):
        idx, prompt = idx_prompt
        print(f"[INFO] 正在生成第{idx+1}张图像...")
        form = {
            "req_key": MODEL_NAME,
            "prompt": prompt,
        }
        if WIDTH:
            form["width"] = WIDTH
        if HEIGHT:
            form["height"] = HEIGHT
        resp = visual_service.cv_process(form)
        # 只保存图片
        try:
            img_base64 = resp['data']['binary_data_base64'][0]
            img_bytes = base64.b64decode(img_base64)
            # 生成带时间戳的文件名
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            img_file = os.path.join(output_dir, f"image_{idx+1}_{timestamp}.png")
            with open(img_file, 'wb') as imgf:
                imgf.write(img_bytes)
            print(f"[INFO] 第{idx+1}张图像已保存为: {img_file}")
        except Exception as e:
            print(f"[WARN] 第{idx+1}张图像base64保存失败: {e}")
        print(f"[INFO] 第{idx+1}张图像生成完成")
        return idx
    print(f"[INFO] 并发线程数设置为: {THREAD_NUM}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_NUM) as executor:
        futures = [executor.submit(process, (idx, prompt)) for idx, prompt in enumerate(prompts)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print(f"[INFO] 所有图像已生成，输出目录: {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ImageRenderer：图像生成工具')
    parser.add_argument('input_path', help='图像提示词输入文件路径（每行为一个提示词）')
    parser.add_argument('--output_dir', required=True, help='生成图片的输出目录')
    args = parser.parse_args()
    render_images(args.input_path, args.output_dir) 