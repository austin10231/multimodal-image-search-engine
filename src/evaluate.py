import argparse  # 导入 argparse，用于解析命令行参数
import csv  # 导入 csv，用于读取 captions 文件
import json  # 导入 json，用于保存评估结果
import os  # 导入 os，用于设置线程环境变量
import statistics  # 导入 statistics，用于计算中位数
import time  # 导入 time，用于统计评估耗时
from pathlib import Path  # 导入 Path，用于更稳健地处理路径

# 在导入 torch/faiss 相关逻辑前先限制线程，降低 OpenMP 冲突风险
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.search_faiss import FaissSearcher  # 复用现有 FAISS 检索器
from src.text_embedding import (  # 复用现有文本编码流程，保持逻辑一致
    encode_one_text,
    load_clip_for_text,
    postprocess_text_embedding,
    prepare_text_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 项目根目录
DATA_DIR = PROJECT_ROOT / "data"  # 全量数据目录
IMAGES_DIR = DATA_DIR / "Images"  # 全量图片目录
FULL_IMAGE_FILE = DATA_DIR / "image_embeddings.jsonl"  # 全量向量文件
DEMO_DIR = PROJECT_ROOT / "demo_data"  # demo 数据目录
DEMO_IMAGES_DIR = DEMO_DIR / "images"  # demo 图片目录
DEMO_IMAGE_FILE = DEMO_DIR / "image_embeddings_demo.jsonl"  # demo 向量文件


def choose_runtime_assets() -> tuple[Path | None, Path | None, str]:  # 与 web_app/streamlit_app 保持一致
    if FULL_IMAGE_FILE.exists() and IMAGES_DIR.exists():
        return FULL_IMAGE_FILE, IMAGES_DIR, "full"
    if DEMO_IMAGE_FILE.exists() and DEMO_IMAGES_DIR.exists():
        return DEMO_IMAGE_FILE, DEMO_IMAGES_DIR, "demo"
    return None, None, "missing"


def parse_k_values(raw: str) -> list[int]:  # 解析并清洗 k 列表
    ks: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if token == "":
            continue
        ks.add(max(1, int(token)))
    if not ks:
        raise ValueError("k-values cannot be empty")
    return sorted(ks)


def resolve_path(path_str: str) -> Path:  # 把相对路径转成项目根目录下绝对路径
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_eval_queries(captions_file: Path, limit: int | None = None, one_caption_per_image: bool = False) -> list[dict]:
    if not captions_file.exists():
        raise FileNotFoundError(f"captions file not found: {captions_file}")

    queries: list[dict] = []
    seen_image_ids: set[str] = set()

    with captions_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "caption" not in fields:
            raise ValueError("captions file must contain a 'caption' column")

        if "image" in fields:
            id_field = "image"
        elif "image_id" in fields:
            id_field = "image_id"
        else:
            raise ValueError("captions file must contain 'image' or 'image_id' column")

        for row in reader:
            image_id = str(row.get(id_field, "")).strip()
            caption = str(row.get("caption", "")).strip()
            if image_id == "" or caption == "":
                continue

            if one_caption_per_image and image_id in seen_image_ids:
                continue

            queries.append({"image_id": image_id, "caption": caption})
            seen_image_ids.add(image_id)

            if limit is not None and len(queries) >= limit:
                break

    return queries


def encode_query_live(query: str, model, tokenizer, device) -> list[float]:  # 与在线检索逻辑一致
    inputs = prepare_text_inputs(query, tokenizer, device)
    text_features = encode_one_text(model, inputs)
    return postprocess_text_embedding(text_features)


def evaluate_recall_at_k(
    queries: list[dict],
    searcher: FaissSearcher,
    model,
    tokenizer,
    device,
    k_values: list[int],
    search_k: int,
) -> dict:
    total = len(queries)
    if total == 0:
        raise ValueError("no valid evaluation queries found")

    search_k = max(1, min(search_k, len(searcher.metadata)))
    k_values = [k for k in k_values if k <= search_k]
    if not k_values:
        raise ValueError(f"all k-values are larger than search_k={search_k}")

    ranks: list[int | None] = []
    start = time.perf_counter()

    for i, item in enumerate(queries, start=1):
        query_embedding = encode_query_live(item["caption"], model, tokenizer, device)
        results = searcher.search(query_embedding, k=search_k)

        rank = None
        for pos, res in enumerate(results, start=1):
            if res["image_id"] == item["image_id"]:
                rank = pos
                break
        ranks.append(rank)

        if i % 100 == 0 or i == total:
            print(f"[{i}/{total}] evaluated")

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    found_ranks = [r for r in ranks if r is not None]
    not_found = total - len(found_ranks)

    metrics: dict[str, float | int | None] = {
        "num_queries": total,
        "search_k": search_k,
        "not_found_within_search_k": not_found,
        "hit_rate_at_search_k": len(found_ranks) / total,
        "mrr_at_search_k": sum((1.0 / r) if r is not None else 0.0 for r in ranks) / total,
        "median_rank_found_only": float(statistics.median(found_ranks)) if found_ranks else None,
        "mean_rank_found_only": (sum(found_ranks) / len(found_ranks)) if found_ranks else None,
        "elapsed_ms": elapsed_ms,
    }

    for k in k_values:
        hit = sum(1 for r in ranks if r is not None and r <= k)
        metrics[f"recall@{k}"] = hit / total

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality (Recall@K) for multimodal image search")
    parser.add_argument("--captions-file", type=str, default="data/captions.txt", help="captions CSV/TXT file with columns: image(or image_id), caption")
    parser.add_argument("--embeddings-file", type=str, default="", help="optional image embeddings jsonl path; empty uses runtime auto-selection")
    parser.add_argument("--k-values", type=str, default="1,5,10", help="comma-separated k values, e.g. 1,5,10")
    parser.add_argument("--search-k", type=int, default=10, help="top-k depth used during retrieval in evaluation")
    parser.add_argument("--limit", type=int, default=0, help="optional max number of caption queries to evaluate; 0 means all")
    parser.add_argument("--one-caption-per-image", action="store_true", help="if set, keep only the first caption for each image")
    parser.add_argument("--output-file", type=str, default="data/eval_metrics.json", help="path to save evaluation metrics json")
    return parser.parse_args()


def main():
    args = parse_args()

    captions_file = resolve_path(args.captions_file)
    output_file = resolve_path(args.output_file)
    limit = args.limit if args.limit > 0 else None

    k_values = parse_k_values(args.k_values)

    if args.embeddings_file.strip() == "":
        embeddings_file, _, data_mode = choose_runtime_assets()
        if embeddings_file is None:
            raise FileNotFoundError("No embedding file found. Expected full data or demo data assets.")
    else:
        embeddings_file = resolve_path(args.embeddings_file)
        data_mode = "custom"
        if not embeddings_file.exists():
            raise FileNotFoundError(f"embeddings file not found: {embeddings_file}")

    print(f"data mode: {data_mode}")
    print(f"embeddings file: {embeddings_file}")
    print(f"captions file: {captions_file}")

    queries = load_eval_queries(
        captions_file=captions_file,
        limit=limit,
        one_caption_per_image=args.one_caption_per_image,
    )
    print(f"num evaluation queries: {len(queries)}")

    model, tokenizer, device = load_clip_for_text()
    searcher = FaissSearcher(embeddings_file)

    metrics = evaluate_recall_at_k(
        queries=queries,
        searcher=searcher,
        model=model,
        tokenizer=tokenizer,
        device=device,
        k_values=k_values,
        search_k=args.search_k,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "captions_file": str(captions_file),
        "embeddings_file": str(embeddings_file),
        "k_values": k_values,
        "one_caption_per_image": bool(args.one_caption_per_image),
        "metrics": metrics,
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")
    print(f"\nsaved metrics to {output_file}")


if __name__ == "__main__":
    main()
