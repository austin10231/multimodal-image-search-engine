import os  # 导入 os，用于设置离线环境变量
import json  # 导入 json，用于保存查询向量结果
from pathlib import Path  # 导入 Path，用于更稳健地处理文件路径

# 在导入 torch 前设置线程相关环境变量，降低 OpenMP 冲突概率
os.environ.setdefault("OMP_NUM_THREADS", "1")  # 限制 OpenMP 线程数
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 允许重复 KMP 运行时（兼容性兜底）

import torch  # 导入 PyTorch，用于设备选择与推理

from transformers import CLIPModel, AutoTokenizer  # 导入 CLIP 模型和文本分词器

torch.set_num_threads(1)  # 限制 PyTorch 线程数，降低与其他 OpenMP 库冲突概率


def load_clip_for_text(model_name: str = "openai/clip-vit-base-patch32"):  # 加载文本侧 CLIP 组件
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 优先使用 GPU，否则使用 CPU
    try:  # 先尝试仅从本地缓存加载，启动更快
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)  # 从本地缓存加载 tokenizer
        model = CLIPModel.from_pretrained(model_name, local_files_only=True, use_safetensors=False).to(device)  # 从本地缓存加载 CLIP 模型并放到设备上
    except Exception:  # 若本地无缓存，则在线下载模型
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)  # 在线加载 tokenizer
        model = CLIPModel.from_pretrained(model_name, local_files_only=False, use_safetensors=False).to(device)  # 在线加载模型并放到设备
    model.eval()  # 切换到推理模式
    return model, tokenizer, device  # 返回模型、tokenizer 和设备


def prepare_text_inputs(text: str, tokenizer, device: str):  # 把原始文本转成模型输入张量
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")  # 对单条文本做分词并返回 PyTorch 张量
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 把所有输入张量移动到与模型相同设备
    return inputs  # 返回可直接送入模型的输入字典


def encode_one_text(model, inputs):  # 对单条文本做编码
    with torch.no_grad():  # 关闭梯度计算，节省内存并提升推理速度
        text_outputs = model.get_text_features(**inputs)  # 调用 CLIP 文本特征接口
    return text_outputs.pooler_output  # 返回真正的文本向量张量


def postprocess_text_embedding(text_features):  # 对文本向量做后处理
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 对向量做 L2 归一化
    embedding = text_features[0].cpu().tolist()  # 取 batch 第 1 条，移到 CPU，并转成 Python 列表
    return embedding  # 返回可保存的向量列表


def test_one_text(query: str):  # 最小测试函数：验证单条文本编码流程
    model, tokenizer, device = load_clip_for_text()  # 加载模型组件
    inputs = prepare_text_inputs(query, tokenizer, device)  # 准备模型输入
    text_features = encode_one_text(model, inputs)  # 得到文本特征张量
    embedding = postprocess_text_embedding(text_features)  # 转成最终可读向量
    print("text embedding dim:", len(embedding))  # 打印向量维度
    print("text embedding[:5]:", embedding[:5])  # 打印前 5 个值做快速检查


def save_query_embedding(query: str, embedding: list[float], output_file: Path):  # 保存查询向量到 JSON 文件
    output_file.parent.mkdir(parents=True, exist_ok=True)  # 若输出目录不存在则自动创建
    record = {  # 构建输出记录
        "query": query,  # 保存原始查询文本
        "embedding": embedding,  # 保存查询对应的向量
    }  # 记录字典构建完成
    with output_file.open("w", encoding="utf-8") as f:  # 以 UTF-8 写模式打开输出文件
        json.dump(record, f, ensure_ascii=False)  # 写入 JSON，保留中文字符


def encode_query(query: str) -> list[float]:  # 一站式函数：输入查询文本，输出查询向量
    model, tokenizer, device = load_clip_for_text()  # 加载模型组件
    inputs = prepare_text_inputs(query, tokenizer, device)  # 文本转模型输入
    text_features = encode_one_text(model, inputs)  # 计算文本向量张量
    embedding = postprocess_text_embedding(text_features)  # 后处理成 list[float]
    return embedding  # 返回最终查询向量


def main():  # 脚本入口：示例生成并保存一条查询向量
    query = input("请输入查询文本: ").strip()  # 从终端读取用户查询文本并去除首尾空格
    if query == "":  # 判断是否输入了空文本
        print("查询文本不能为空")  # 提示用户输入不能为空
        return  # 空输入时直接结束
    embedding = encode_query(query)  # 计算查询向量
    project_root = Path(__file__).resolve().parents[1]  # 定位项目根目录
    output_file = project_root / "data" / "query_embedding.json"  # 设定输出文件路径
    save_query_embedding(query, embedding, output_file)  # 保存查询向量结果
    print(f"saved query embedding to {output_file}")  # 打印保存路径


if __name__ == "__main__":  # 仅在直接运行当前脚本时执行
    main()  # 调用主函数
