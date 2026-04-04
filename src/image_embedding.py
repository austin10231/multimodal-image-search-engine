import torch  # 导入 PyTorch，用于张量和模型推理
from transformers import CLIPImageProcessor, CLIPModel  # 导入 CLIP 图像处理器和模型
from PIL import Image  # 导入 PIL，用于读取图片文件
import json  # 导入 json，用于写入 JSONL
import csv  # 导入 csv，用于读取处理后的样本表
import os  # 导入 os，用于设置离线环境变量


def load_clip(model_name: str = "openai/clip-vit-base-patch32"):  # 加载 CLIP 模型和处理器
    os.environ["HF_HUB_OFFLINE"] = "1"  # 强制 HuggingFace 离线模式
    os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制 transformers 离线模式
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 优先使用 GPU，否则使用 CPU
    processor = CLIPImageProcessor.from_pretrained(model_name, local_files_only=True)  # 从本地缓存加载图像处理器
    model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)  # 从本地缓存加载模型并移动到设备
    model.eval()  # 切换到推理模式
    return model, processor, device  # 返回模型、处理器和设备


def prepare_image_inputs(image_path: str, processor, device: str):  # 读取并预处理单张图片
    image = Image.open(image_path).convert("RGB")  # 打开图片并统一为 RGB 三通道
    inputs = processor(images=image, return_tensors="pt")  # 按 CLIP 规则把图片转为张量
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 把输入张量移动到同一设备
    return inputs  # 返回模型可直接使用的输入字典


def encode_one_image(model, inputs):  # 用 CLIP 编码一张图片得到向量
    with torch.no_grad():  # 关闭梯度计算，节省显存并加速推理
        vision_outputs = model.get_image_features(**inputs)  # 调用图像特征提取接口
    return vision_outputs.pooler_output  # 返回真正的图像向量张量


def postprocess_embedding(image_features):  # 对向量做后处理
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 做 L2 归一化
    embedding = image_features[0].cpu().tolist()  # 取 batch 第 1 条，移到 CPU 并转为 list
    return embedding  # 返回可保存的 Python 列表向量


def write_embedding_record(f, image_id: str, image_path: str, embedding: list[float]):  # 写入一条 JSONL 记录
    record = {  # 构建单条输出记录
        "image_id": image_id,  # 保存图片 id
        "image_path": image_path,  # 保存图片路径
        "embedding": embedding,  # 保存图片向量
    }  # 记录字典构建结束
    f.write(json.dumps(record, ensure_ascii=False) + "\n")  # 写入一行 JSON


def embed_from_processed_csv(processed_csv: str, output_jsonl: str, limit: int | None = None):  # 批量生成图片向量
    model, processor, device = load_clip()  # 加载模型相关组件
    count = 0  # 初始化已处理计数
    with open(processed_csv, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:  # 打开输入 CSV 和输出 JSONL
        reader = csv.DictReader(fin)  # 按列名读取 CSV 每一行
        for row in reader:  # 遍历每条样本记录
            inputs = prepare_image_inputs(row["image_path"], processor, device)  # 预处理当前图片
            image_features = encode_one_image(model, inputs)  # 计算当前图片向量
            embedding = postprocess_embedding(image_features)  # 归一化并转换为 list
            write_embedding_record(fout, row["image_id"], row["image_path"], embedding)  # 写入输出文件
            count += 1  # 处理计数加一
            if limit is not None and count >= limit:  # 如果设置了 limit 且达到上限则停止
                break  # 提前结束循环
    print(f"saved {count} embeddings to {output_jsonl}")  # 打印保存结果


if __name__ == "__main__":  # 仅在直接运行脚本时执行以下调用
    embed_from_processed_csv(  # 执行批量 embedding 任务
        processed_csv="data/processed_samples.csv",  # 输入清洗后的样本 CSV
        output_jsonl="data/image_embeddings.jsonl",  # 输出 embedding 的 JSONL 文件
        limit=None,  # 不限制条数，默认全量处理
    )  # 调用结束
