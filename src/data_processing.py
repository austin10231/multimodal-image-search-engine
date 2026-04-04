from load_data import build_dataset  # 从加载模块导入构建数据集函数
import csv  # 导入 csv 模块用于写出表格文件
from pathlib import Path  # 导入 Path 用于路径拼接与管理


def process_samples(samples: list[dict]) -> list[dict]:  # 定义样本处理函数
    processed = []  # 创建列表存放处理后的样本
    seen = set()  # 创建集合记录已保留过的 image_id
    for sample in samples:  # 遍历每一条原始样本
        caption = sample["caption"].strip()  # 读取并清理 caption 两端空白
        if caption == "":  # 判断 caption 是否为空字符串
            continue  # 如果为空就跳过这条样本
        image_id = sample["image_id"]  # 取出当前样本的图片 id
        if image_id in seen:  # 判断这张图是否已经保留过
            continue  # 如果已保留过就跳过
        seen.add(image_id)  # 记录这张图已经被保留
        new_sample = {  # 构建清洗后的新样本字典
            "image_id": sample["image_id"],  # 保存图片 id 字段
            "image_path": sample["image_path"],  # 保存图片路径字段
            "caption": caption,  # 保存清理后的 caption 字段
        }  # 新样本字典构建结束
        processed.append(new_sample)  # 把新样本加入结果列表
    return processed  # 返回处理完成的样本列表


def save_processed_samples(processed: list[dict], output_file: Path) -> None:  # 定义保存函数
    output_file.parent.mkdir(parents=True, exist_ok=True)  # 若目录不存在则自动创建
    fieldnames = ["image_id", "image_path", "caption"]  # 指定 CSV 列顺序
    with output_file.open("w", encoding="utf-8", newline="") as f:  # 以写模式打开输出文件
        writer = csv.DictWriter(f, fieldnames=fieldnames)  # 创建字典写入器
        writer.writeheader()  # 写入 CSV 表头
        writer.writerows(processed)  # 写入全部处理后样本


def main():  # 定义脚本主函数用于本地测试
    samples = build_dataset()  # 加载原始样本数据
    processed = process_samples(samples)  # 执行样本清洗与去重
    project_root = Path(__file__).resolve().parents[1]  # 定位到项目根目录
    output_file = project_root / "data" / "processed_samples.csv"  # 设置输出文件路径
    save_processed_samples(processed, output_file)  # 将处理结果保存到 CSV
    print(f"saved to: {output_file}")  # 打印输出文件位置
    print(f"raw samples: {len(samples)}")  # 打印原始样本数量
    print(f"processed samples: {len(processed)}")  # 打印处理后样本数量


if __name__ == "__main__":  # 判断是否直接运行当前脚本
    main()  # 直接运行时执行主函数
    
