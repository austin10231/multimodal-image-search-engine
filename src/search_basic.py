import argparse  # 导入 argparse，用于解析命令行参数
import json  # 导入 json，用于读取 JSON 和 JSONL 数据
from pathlib import Path  # 导入 Path，用于处理文件路径


def load_query_embedding(query_file: Path) -> list[float]:  # 读取单条查询向量
    with query_file.open("r", encoding="utf-8") as f:  # 以 utf-8 打开查询向量文件
        data = json.load(f)  # 读取整个 JSON 对象
    return data["embedding"]  # 返回查询向量列表


def load_image_embeddings(image_file: Path) -> list[dict]:  # 读取所有图片向量记录
    records = []  # 准备列表存放每一条图片记录
    with image_file.open("r", encoding="utf-8") as f:  # 以 utf-8 打开图片向量 JSONL 文件
        for line in f:  # 逐行读取（每行是一个 JSON 对象）
            line = line.strip()  # 去掉每行首尾空白字符
            if line == "":  # 跳过空行，避免解析报错
                continue  # 空行直接进入下一行
            records.append(json.loads(line))  # 把当前行 JSON 解析为字典并加入列表
    return records  # 返回所有图片向量记录


def dot_similarity(vec_a: list[float], vec_b: list[float]) -> float:  # 计算两个向量的点积相似度
    score = 0.0  # 初始化相似度分数
    for a, b in zip(vec_a, vec_b):  # 按维度成对遍历两个向量
        score += a * b  # 累加每一维乘积
    return score  # 返回最终点积分数


def score_images(query_embedding: list[float], image_records: list[dict]) -> list[dict]:  # 为每张图计算与查询的相似度
    scored = []  # 准备列表存放带分数的结果
    for record in image_records:  # 遍历每一条图片向量记录
        image_embedding = record["embedding"]  # 取出当前图片向量
        score = dot_similarity(query_embedding, image_embedding)  # 计算当前图片与查询的点积分数
        scored.append(  # 组装一条带分数的结果
            {
                "image_id": record["image_id"],  # 保存图片 id
                "image_path": record["image_path"],  # 保存图片路径
                "score": score,  # 保存相似度分数
            }
        )  # 当前结果字典构建结束
    return scored  # 返回全部打分结果


def top_k_results(scored_records: list[dict], k: int = 5) -> list[dict]:  # 选出分数最高的前 k 条结果
    sorted_records = sorted(  # 对全部结果按分数降序排序
        scored_records,  # 输入待排序的打分结果列表
        key=lambda x: x["score"],  # 指定按 score 字段排序
        reverse=True,  # 设置为降序（从大到小）
    )  # 排序完成
    return sorted_records[:k]  # 截取并返回前 k 条结果


def save_results(results: list[dict], output_file: Path):  # 把检索结果保存成 JSON 文件
    output_file.parent.mkdir(parents=True, exist_ok=True)  # 若目录不存在则自动创建
    with output_file.open("w", encoding="utf-8") as f:  # 以 utf-8 写模式打开输出文件
        json.dump(results, f, ensure_ascii=False, indent=2)  # 把结果列表格式化写入 JSON


def parse_args():  # 解析命令行参数
    parser = argparse.ArgumentParser(description="Basic image search with CLIP embeddings")  # 创建参数解析器
    parser.add_argument("--k", type=int, default=5, help="number of top results")  # 支持从命令行设置 Top-K
    parser.add_argument("--query-file", type=str, default="data/query_embedding.json", help="path to query embedding json")  # 支持从命令行指定查询向量文件
    parser.add_argument("--output-file", type=str, default="data/search_results.json", help="path to save search results json")  # 支持从命令行指定结果输出文件
    return parser.parse_args()  # 返回解析后的参数对象


def main():  # 主函数：串联读取、打分、排序并打印结果
    args = parse_args()  # 读取命令行参数
    project_root = Path(__file__).resolve().parents[1]  # 定位项目根目录
    query_file = Path(args.query_file)  # 从命令行参数读取查询向量文件路径
    if not query_file.is_absolute():  # 如果给的是相对路径
        query_file = project_root / query_file  # 则按项目根目录拼接成绝对路径
    image_file = project_root / "data" / "image_embeddings.jsonl"  # 图片向量文件路径
    output_file = Path(args.output_file)  # 从命令行参数读取结果输出文件路径
    if not output_file.is_absolute():  # 如果给的是相对路径
        output_file = project_root / output_file  # 则按项目根目录拼接成绝对路径
    query_embedding = load_query_embedding(query_file)  # 读取查询向量
    image_records = load_image_embeddings(image_file)  # 读取全部图片向量记录
    scored = score_images(query_embedding, image_records)  # 对所有图片进行相似度打分
    top_results = top_k_results(scored, k=args.k)  # 按命令行参数取分数最高的前 k 条结果
    save_results(top_results, output_file)  # 把 Top-K 结果保存到 JSON 文件
    print(f"Top-{args.k} Results:")  # 打印结果标题
    for i, item in enumerate(top_results, start=1):  # 遍历前 k 条结果并编号
        print(f"{i}. score={item['score']:.4f} image_id={item['image_id']} path={item['image_path']}")  # 打印单条结果
    print(f"saved results to {output_file}")  # 打印结果文件保存路径


if __name__ == "__main__":  # 当脚本被直接运行时执行主函数
    main()  # 调用主函数开始检索
