from pathlib import Path  # 导入 Path，用于更安全地处理文件路径
import csv  # 导入 csv 模块，用于读取 captions.txt


def row_to_sample(row: dict, images_dir: Path) -> dict:  # 把一行原始数据转换成统一样本格式
    image_name = row['image'].strip()  # 读取图片文件名，并去掉首尾空格
    caption = row['caption'].strip()  # 读取 caption 文本，并去掉首尾空格

    sample = {  # 创建一条标准样本字典
        "image_id": image_name,  # 使用图片文件名作为样本 id
        "image_path": str(images_dir / image_name),  # 拼接图片完整路径并转成字符串
        "caption": caption  # 保存图片对应的文本描述
    }  # 样本字典构建结束
    return sample  # 返回这条样本


def load_sample(captions_file: Path, images_dir: Path) -> list:  # 从 captions 文件加载所有样本
    samples = []  # 准备一个列表，用来存放所有样本
    with open(captions_file, 'r', encoding='utf-8') as f:  # 以 utf-8 编码打开 captions 文件
        reader = csv.DictReader(f)  # 按表头读取，每一行会变成一个字典
        for row in reader:  # 遍历文件中的每一行
            sample = row_to_sample(row, images_dir)  # 把当前行转换成标准样本
            samples.append(sample)  # 把样本加入列表
    return samples  # 返回所有样本


def count_missing_images(samples: list[dict]) -> int:  # 统计样本中图片路径不存在的数量
    missing = 0  # 初始化缺失计数器
    for sample in samples:  # 遍历每一条样本
        if not Path(sample["image_path"]).exists():  # 如果图片路径不存在
            missing += 1  # 缺失数量加一
    return missing  # 返回缺失图片总数


def build_dataset() -> list[dict]:  # 封装统一入口，构建完整数据集
    project_root = Path(__file__).resolve().parents[1]  # 定位到项目根目录
    captions_file = project_root / "data" / "captions.txt"  # 定位 captions 文件路径
    images_dir = project_root / "data" / "Images"  # 定位图片目录路径
    return load_sample(captions_file, images_dir)  # 调用加载函数并返回样本列表


def main():  # 主函数：用于本地快速测试加载结果
    samples = build_dataset()  # 构建并加载整个数据集

    print(f"loaded samples: {len(samples)}")  # 打印加载到的样本总数

    if samples:  # 如果至少有一条样本
        print("sample 0:", samples[0])  # 打印第 1 条样本
    if len(samples) > 1:  # 如果至少有两条样本
        print("sample 1:", samples[1])  # 打印第 2 条样本

    missing_count = count_missing_images(samples)  # 统计缺失图片数量
    print(f"missing image files: {missing_count}")  # 打印缺失图片统计结果


if __name__ == "__main__":  # 只有直接运行当前脚本时才执行下面一行
    main()  # 调用主函数开始测试
