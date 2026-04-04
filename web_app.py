import time  # 导入 time，用于统计搜索耗时
from pathlib import Path  # 导入 Path，用于路径处理
import os  # 导入 os，用于设置运行时环境变量

# 先限制并行线程，减少 faiss + torch 在 macOS 上的 OpenMP 冲突风险
os.environ["OMP_NUM_THREADS"] = "1"  # 限制 OpenMP 线程数
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # 限制 OpenBLAS 线程数
os.environ["MKL_NUM_THREADS"] = "1"  # 限制 MKL 线程数
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # 限制 Accelerate 线程数
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载 KMP 以规避部分崩溃场景

from flask import Flask, jsonify, render_template_string, request, send_from_directory  # 导入 Flask 相关组件

from src.search_faiss import FaissSearcher  # 从 src 包导入 FAISS 检索器
from src.text_embedding import (  # 从 src 包导入文本向量编码相关函数
    encode_one_text,
    load_clip_for_text,
    postprocess_text_embedding,
    prepare_text_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parent  # 当前文件就在项目根目录
DATA_DIR = PROJECT_ROOT / "data"  # 数据目录路径
IMAGES_DIR = DATA_DIR / "Images"  # 图片目录路径
IMAGE_EMBEDDINGS_FILE = DATA_DIR / "image_embeddings.jsonl"  # 图片向量文件路径
DEMO_DIR = PROJECT_ROOT / "demo_data"  # Demo 数据目录路径
DEMO_IMAGES_DIR = DEMO_DIR / "images"  # Demo 图片目录路径
DEMO_IMAGE_EMBEDDINGS_FILE = DEMO_DIR / "image_embeddings_demo.jsonl"  # Demo 图片向量文件路径


def choose_runtime_assets() -> tuple[Path | None, Path | None, str]:  # 选择运行时使用的向量文件与图片目录
    if IMAGE_EMBEDDINGS_FILE.exists() and IMAGES_DIR.exists():  # 若本地全量数据存在
        return IMAGE_EMBEDDINGS_FILE, IMAGES_DIR, "full"  # 优先使用全量数据
    if DEMO_IMAGE_EMBEDDINGS_FILE.exists() and DEMO_IMAGES_DIR.exists():  # 若只有 demo 数据可用
        return DEMO_IMAGE_EMBEDDINGS_FILE, DEMO_IMAGES_DIR, "demo"  # 回退到 demo 数据
    return None, None, "missing"  # 都不存在时标记缺失


ACTIVE_EMBEDDINGS_FILE, ACTIVE_IMAGES_DIR, DATA_MODE = choose_runtime_assets()  # 记录当前生效的数据来源模式

app = Flask(__name__)  # 创建 Flask 应用实例
searcher = None  # 全局 FAISS 检索器（延迟初始化）
text_model = None  # 全局文本模型（延迟初始化）
text_tokenizer = None  # 全局文本 tokenizer（延迟初始化）
text_device = None  # 全局文本推理设备（延迟初始化）


def get_searcher() -> FaissSearcher:  # 获取 FAISS 检索器（首次调用时构建）
    global searcher  # 声明使用全局变量
    if searcher is None:  # 若尚未初始化
        if ACTIVE_EMBEDDINGS_FILE is None:  # 若不存在可用向量文件
            raise FileNotFoundError("No embedding file found. Expected data/image_embeddings.jsonl or demo_data/image_embeddings_demo.jsonl")  # 给出明确报错
        searcher = FaissSearcher(ACTIVE_EMBEDDINGS_FILE)  # 加载向量并构建索引
    return searcher  # 返回可用检索器


def get_text_components():  # 获取文本编码组件（首次调用时加载）
    global text_model, text_tokenizer, text_device  # 声明使用全局变量
    if text_model is None or text_tokenizer is None or text_device is None:  # 若尚未初始化
        text_model, text_tokenizer, text_device = load_clip_for_text()  # 加载模型、tokenizer 和设备
    return text_model, text_tokenizer, text_device  # 返回可用组件


def encode_query_live(query: str) -> list[float]:  # 把实时查询文本编码为向量
    text_model, text_tokenizer, text_device = get_text_components()  # 获取文本编码组件
    inputs = prepare_text_inputs(query, text_tokenizer, text_device)  # 文本转模型输入
    text_features = encode_one_text(text_model, inputs)  # 计算文本特征
    return postprocess_text_embedding(text_features)  # 归一化并转为列表向量


HTML_PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>LensSeek</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --ink: #1f2a37;
      --muted: #6a7789;
      --line: #d5dde8;
      --brand: #3165f6;
      --brand-soft: #e8efff;
      --card: #ffffff;
      --shadow: 0 10px 30px rgba(23, 38, 74, 0.08);
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: "Avenir Next", "Trebuchet MS", sans-serif;
    }
    .wrap {
      width: min(1120px, 94vw);
      margin: 0 auto;
      padding: 36px 0 56px;
    }
    .hero {
      min-height: 45vh;
      display: grid;
      place-items: center;
      position: relative;
      margin-bottom: 22px;
    }
    .orb {
      position: absolute;
      border-radius: 999px;
      filter: blur(8px);
      opacity: 0.28;
      pointer-events: none;
    }
    .orb.a {
      width: 280px;
      height: 280px;
      background: #d9e6ff;
      top: -40px;
      left: 4%;
    }
    .orb.b {
      width: 220px;
      height: 220px;
      background: #ffe6d3;
      bottom: -10px;
      right: 8%;
    }
    .center {
      width: min(860px, 100%);
      text-align: center;
      z-index: 1;
    }
    .brand {
      margin: 0;
      font-family: "Baskerville", "Book Antiqua", serif;
      font-size: clamp(44px, 7vw, 70px);
      letter-spacing: 0.6px;
      line-height: 1.02;
    }
    .brand-accent {
      color: var(--brand);
    }
    .sub {
      margin: 10px auto 22px;
      max-width: 680px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.6;
    }
    .search-shell {
      text-align: left;
      background: #fff;
      border: 1px solid #dbe3ee;
      border-radius: 26px;
      box-shadow: var(--shadow);
      padding: 16px;
    }
    .search-row {
      display: grid;
      grid-template-columns: 1fr 165px 120px;
      gap: 10px;
      align-items: end;
    }
    .label {
      display: block;
      font-size: 12px;
      letter-spacing: 0.03em;
      color: var(--muted);
      margin: 0 0 6px 2px;
    }
    .field,
    .k,
    .btn {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: 16px;
      padding: 13px 16px;
      background: #fff;
      color: var(--ink);
      outline: none;
      transition: border-color 0.18s ease, box-shadow 0.18s ease;
    }
    .field:focus,
    .k:focus {
      border-color: var(--brand);
      box-shadow: 0 0 0 4px var(--brand-soft);
    }
    .btn {
      border: none;
      background: linear-gradient(135deg, #2f66ff, #2e8bff);
      color: #fff;
      font-weight: 650;
      cursor: pointer;
    }
    .btn:hover {
      filter: brightness(1.04);
    }
    .helper {
      margin: 10px 2px 2px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .status {
      margin: 10px 2px 0;
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
    }
    .results-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin: 12px 2px 10px;
      gap: 10px;
      flex-wrap: wrap;
    }
    .results-title {
      font-size: 15px;
      font-weight: 650;
      margin: 0;
    }
    .results-note {
      color: var(--muted);
      font-size: 12px;
      margin: 0;
    }
    .results {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--card);
      border: 1px solid #d8e2f0;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 4px 14px rgba(22, 43, 76, 0.06);
      transition: transform 0.16s ease, box-shadow 0.16s ease;
    }
    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 26px rgba(22, 43, 76, 0.14);
    }
    .thumb {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      display: block;
      background: #eef2f8;
    }
    .meta {
      padding: 10px 12px 12px;
      font-size: 13px;
      line-height: 1.5;
    }
    .score {
      color: #1f4dd6;
      font-weight: 700;
    }
    .id {
      color: #53657a;
      font-family: "Menlo", "Monaco", "Courier New", monospace;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    @media (max-width: 860px) {
      .hero {
        min-height: 40vh;
      }
      .search-row {
        grid-template-columns: 1fr;
      }
      .btn {
        border-radius: 14px;
      }
      .search-shell {
        border-radius: 20px;
      }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="orb a"></div>
      <div class="orb b"></div>
      <div class="center">
        <h1 class="brand">Lens<span class="brand-accent">Seek</span></h1>
        <p class="sub">Search images with natural language. Type what you want to see, and the engine returns semantically similar photos.</p>

        <section class="search-shell">
          <form id="search-form" class="search-row">
            <div>
              <label class="label" for="query">Search Query (describe the image you want)</label>
              <input id="query" class="field" placeholder="e.g. a dog running on grass" required />
            </div>
            <div>
              <label class="label" for="k">Top-K (how many results)</label>
              <input id="k" class="k" type="number" min="1" max="20" value="8" />
            </div>
            <div>
              <label class="label" for="search-btn">Action</label>
              <button id="search-btn" class="btn" type="submit">Search</button>
            </div>
          </form>
          <p class="helper">Tip: Use subject + action + scene, for example: "two children playing football in a park".</p>
          <div class="status" id="status">Ready. Enter a query and click Search.</div>
        </section>
      </div>
    </section>

    <section class="results-head">
      <h2 class="results-title">Visual Matches</h2>
      <p class="results-note">Sorted by vector similarity score.</p>
    </section>

    <section class="results" id="results"></section>
  </main>

  <script>
    const form = document.getElementById("search-form");
    const statusEl = document.getElementById("status");
    const resultsEl = document.getElementById("results");

    function renderCards(items) {
      resultsEl.innerHTML = "";
      if (!items.length) {
        resultsEl.innerHTML = "<p>No results.</p>";
        return;
      }
      for (const item of items) {
        const card = document.createElement("article");
        card.className = "card";
        card.innerHTML = `
          <img class="thumb" src="${item.image_url}" alt="${item.image_id}" loading="lazy" />
          <div class="meta">
            <div class="score">score: ${item.score.toFixed(4)}</div>
            <div class="id">${item.image_id}</div>
          </div>
        `;
        resultsEl.appendChild(card);
      }
    }

    document.addEventListener("keydown", (e) => {
      if (e.key === "/" && document.activeElement !== document.getElementById("query")) {
        e.preventDefault();
        document.getElementById("query").focus();
      }
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const query = document.getElementById("query").value.trim();
      const k = Number(document.getElementById("k").value || 8);
      if (!query) {
        statusEl.textContent = "Query cannot be empty.";
        return;
      }

      statusEl.textContent = "Searching...";
      resultsEl.innerHTML = "";
      try {
        const resp = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, k })
        });
        const data = await resp.json();
        if (!resp.ok) {
          statusEl.textContent = data.error || "Search failed.";
          return;
        }
        statusEl.textContent = `Query: "${data.query}" · ${data.count} results · ${data.elapsed_ms} ms`;
        renderCards(data.results);
      } catch (err) {
        statusEl.textContent = "Network error. Please retry.";
      }
    });
  </script>
</body>
</html>
"""


@app.get("/")  # 首页：返回搜索页面
def home():
    return render_template_string(HTML_PAGE)  # 渲染并返回内嵌 HTML


@app.get("/images/<path:filename>")  # 图片服务：按文件名返回本地图片
def serve_image(filename: str):
    if ACTIVE_IMAGES_DIR is None:  # 若没有可用图片目录
        return jsonify({"error": "image directory not available"}), 404  # 返回 404 错误
    return send_from_directory(ACTIVE_IMAGES_DIR, filename)  # 从当前生效图片目录中返回图片文件


@app.get("/healthz")  # 健康检查接口：便于 Render 判断服务是否正常
def healthz():
    return jsonify(  # 返回当前运行模式与关键路径状态
        {
            "ok": True,  # 服务可用
            "data_mode": DATA_MODE,  # 当前数据模式：full/demo/missing
            "embeddings_file": str(ACTIVE_EMBEDDINGS_FILE) if ACTIVE_EMBEDDINGS_FILE else None,  # 当前向量文件路径
            "images_dir": str(ACTIVE_IMAGES_DIR) if ACTIVE_IMAGES_DIR else None,  # 当前图片目录路径
        }
    )


@app.post("/api/search")  # 搜索接口：接收 query 并返回 Top-K 结果
def api_search():
    payload = request.get_json(silent=True) or {}  # 读取 JSON 请求体
    query = str(payload.get("query", "")).strip()  # 读取并清理查询文本
    k_raw = payload.get("k", 8)  # 读取 Top-K 参数
    try:
        k = int(k_raw)  # 尝试转换为整数
    except (TypeError, ValueError):
        return jsonify({"error": "k must be an integer"}), 400  # 参数非法时返回 400

    if query == "":  # 校验空查询
        return jsonify({"error": "query cannot be empty"}), 400  # 空查询返回 400
    k = max(1, min(20, k))  # 把 k 限制在 1 到 20 区间

    try:
        start = time.perf_counter()  # 记录开始时间
        query_embedding = encode_query_live(query)  # 计算查询文本向量
        results = get_searcher().search(query_embedding, k=k)  # 执行 FAISS 检索
    except FileNotFoundError as e:  # 数据文件缺失时返回明确错误
        return jsonify({"error": str(e)}), 500
    for item in results:  # 为每条结果补充图片访问 URL
        filename = Path(item["image_path"]).name  # 从绝对路径提取图片文件名
        item["image_url"] = f"/images/{filename}"  # 生成前端可直接访问的图片链接
    elapsed_ms = int((time.perf_counter() - start) * 1000)  # 计算检索耗时（毫秒）

    return jsonify(  # 返回 JSON 响应
        {
            "query": query,  # 回传查询文本
            "count": len(results),  # 回传结果数量
            "elapsed_ms": elapsed_ms,  # 回传检索耗时
            "results": results,  # 回传检索结果列表
        }
    )


if __name__ == "__main__":  # 仅当脚本被直接运行时执行
    port = int(os.environ.get("PORT", "8000"))  # Render 会通过 PORT 环境变量注入端口
    app.run(host="0.0.0.0", port=port, debug=False)  # 启动服务并监听所有网卡
