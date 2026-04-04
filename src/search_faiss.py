import argparse  # 导入 argparse，用于解析命令行参数
import json  # 导入 json，用于读取和保存 JSON 数据
from pathlib import Path  # 导入 Path，用于处理路径

import faiss  # 导入 faiss，用于高效向量检索
import numpy as np  # 导入 numpy，用于向量数组计算

faiss.omp_set_num_threads(1)  # 限制 FAISS 使用的 OpenMP 线程，降低崩溃概率


def load_image_embeddings(image_file: Path) -> tuple[np.ndarray, list[dict]]:  # 读取图片向量和元信息
    vectors = []  # 存放图片向量
    metadata = []  # 存放图片 id/path 元信息
    with image_file.open("r", encoding="utf-8") as f:  # 打开图片向量 JSONL 文件
        for line in f:  # 逐行读取每一条记录
            line = line.strip()  # 去掉首尾空白字符
            if line == "":  # 跳过空行
                continue  # 进入下一行
            record = json.loads(line)  # 把当前行解析为字典
            vectors.append(np.asarray(record["embedding"], dtype="float32"))  # 记录向量并转为 float32
            metadata.append(  # 保存和向量对应的图片信息
                {
                    "image_id": record["image_id"],  # 保存图片 id
                    "image_path": record["image_path"],  # 保存图片路径
                }
            )  # 当前元信息字典构建完成
    matrix = np.vstack(vectors).astype("float32")  # 把所有向量拼成二维矩阵 [N, D]
    return matrix, metadata  # 返回向量矩阵和元信息列表


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:  # 构建内积索引（配合归一化可等价余弦相似度）
    vectors = vectors.copy()  # 复制一份，避免原地修改影响外部变量
    faiss.normalize_L2(vectors)  # 对每个向量做 L2 归一化
    dim = vectors.shape[1]  # 获取向量维度 D
    index = faiss.IndexFlatIP(dim)  # 创建内积检索索引
    index.add(vectors)  # 把全部图片向量加入索引
    return index  # 返回构建完成的索引


def load_query_embedding(query_file: Path) -> np.ndarray:  # 读取查询向量
    with query_file.open("r", encoding="utf-8") as f:  # 打开查询向量 JSON 文件
        record = json.load(f)  # 读取 JSON 对象
    query = np.asarray(record["embedding"], dtype="float32")  # 转成 float32 向量
    return query  # 返回查询向量


def search_index(  # 在 FAISS 索引中搜索 Top-K 结果
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    query_embedding: np.ndarray,
    k: int = 5,
) -> list[dict]:
    query = query_embedding.reshape(1, -1).astype("float32")  # 把查询向量整理成 [1, D]
    faiss.normalize_L2(query)  # 对查询向量做归一化
    scores, indices = index.search(query, k)  # 执行 Top-K 检索
    results = []  # 准备结果列表
    for score, idx in zip(scores[0], indices[0]):  # 遍历当前查询的每一个命中结果
        if idx < 0:  # FAISS 在无结果时可能返回 -1
            continue  # 跳过无效结果
        item = metadata[idx]  # 取出该向量对应的图片元信息
        results.append(  # 组装一条结果记录
            {
                "image_id": item["image_id"],  # 保存图片 id
                "image_path": item["image_path"],  # 保存图片路径
                "score": float(score),  # 保存相似度分数
            }
        )  # 当前结果记录构建完成
    return results  # 返回 Top-K 结果列表


def save_results(results: list[dict], output_file: Path):  # 保存检索结果到 JSON 文件
    output_file.parent.mkdir(parents=True, exist_ok=True)  # 若目录不存在则自动创建
    with output_file.open("w", encoding="utf-8") as f:  # 以 utf-8 写模式打开文件
        json.dump(results, f, ensure_ascii=False, indent=2)  # 写入格式化 JSON


class FaissSearcher:  # 封装一个可复用的 FAISS 检索器
    def __init__(self, image_file: Path):  # 初始化时加载向量并建立索引
        self.vectors, self.metadata = load_image_embeddings(image_file)  # 读取图片向量和元信息
        self.index = build_index(self.vectors)  # 构建 FAISS 索引

    def search(self, query_embedding: list[float] | np.ndarray, k: int = 5) -> list[dict]:  # 对外提供搜索接口
        query = np.asarray(query_embedding, dtype="float32")  # 统一把查询向量转换成 float32
        return search_index(self.index, self.metadata, query, k=k)  # 返回 Top-K 检索结果


def parse_args():  # 解析命令行参数
    parser = argparse.ArgumentParser(description="Search images with FAISS index")  # 创建参数解析器
    parser.add_argument("--image-file", type=str, default="data/image_embeddings.jsonl", help="path to image embeddings jsonl")  # 图片向量文件
    parser.add_argument("--query-file", type=str, default="data/query_embedding.json", help="path to query embedding json")  # 查询向量文件
    parser.add_argument("--output-file", type=str, default="data/search_results_faiss.json", help="path to save search results json")  # 输出结果文件
    parser.add_argument("--k", type=int, default=5, help="number of top results")  # Top-K 参数
    return parser.parse_args()  # 返回解析结果


def main():  # 命令行入口：执行一次 FAISS 检索
    args = parse_args()  # 读取命令行参数
    project_root = Path(__file__).resolve().parents[1]  # 定位项目根目录
    image_file = Path(args.image_file)  # 读取图片向量文件参数
    query_file = Path(args.query_file)  # 读取查询向量文件参数
    output_file = Path(args.output_file)  # 读取输出文件参数
    if not image_file.is_absolute():  # 若图片向量文件是相对路径
        image_file = project_root / image_file  # 转为项目根目录下的绝对路径
    if not query_file.is_absolute():  # 若查询向量文件是相对路径
        query_file = project_root / query_file  # 转为项目根目录下的绝对路径
    if not output_file.is_absolute():  # 若输出文件是相对路径
        output_file = project_root / output_file  # 转为项目根目录下的绝对路径

    vectors, metadata = load_image_embeddings(image_file)  # 加载图片向量和元信息
    index = build_index(vectors)  # 建立 FAISS 索引
    query_embedding = load_query_embedding(query_file)  # 加载查询向量
    results = search_index(index, metadata, query_embedding, k=max(1, args.k))  # 执行 Top-K 检索
    save_results(results, output_file)  # 保存结果到 JSON 文件

    print(f"Top-{len(results)} Results (FAISS):")  # 打印结果标题
    for i, item in enumerate(results, start=1):  # 遍历并打印每条检索结果
        print(f"{i}. score={item['score']:.4f} image_id={item['image_id']} path={item['image_path']}")  # 打印单条结果
    print(f"saved results to {output_file}")  # 打印结果保存路径


if __name__ == "__main__":  # 仅当脚本被直接运行时执行
    main()  # 调用主函数
