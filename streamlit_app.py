import base64  # 导入 base64，用于把本地图片转成 data URI
import html  # 导入 html，用于转义文本，避免注入
import time  # 导入 time，用于统计检索耗时
from pathlib import Path  # 导入 Path，用于路径处理
import os  # 导入 os，用于设置运行时环境变量

import streamlit as st  # 导入 Streamlit，用于构建页面

# 设置线程相关环境变量，降低 faiss + torch 在部分环境下冲突概率
os.environ.setdefault("OMP_NUM_THREADS", "1")  # 限制 OpenMP 线程数
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 允许重复 KMP 运行时（兼容性兜底）

from src.search_faiss import FaissSearcher  # 导入 FAISS 检索器
from src.text_embedding import (  # 导入文本向量编码相关函数
    encode_one_text,
    load_clip_for_text,
    postprocess_text_embedding,
    prepare_text_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parent  # 项目根目录
DEMO_IMAGE_FILE = PROJECT_ROOT / "demo_data" / "image_embeddings_demo.jsonl"  # Demo 向量文件路径
FULL_IMAGE_FILE = PROJECT_ROOT / "data" / "image_embeddings.jsonl"  # 全量向量文件路径


@st.cache_resource(show_spinner=False)  # 缓存文本模型资源，避免重复加载
def get_text_components():
    return load_clip_for_text()  # 返回文本模型、tokenizer 和设备


@st.cache_resource(show_spinner=False)  # 缓存 FAISS 索引资源，避免重复建索引
def get_searcher(image_file: str):
    return FaissSearcher(Path(image_file))  # 根据向量文件构建并返回检索器


def encode_query_live(query: str) -> list[float]:  # 实时把查询文本编码为向量
    model, tokenizer, device = get_text_components()  # 获取缓存的文本编码组件
    inputs = prepare_text_inputs(query, tokenizer, device)  # 文本转模型输入张量
    text_features = encode_one_text(model, inputs)  # 计算文本特征
    return postprocess_text_embedding(text_features)  # 归一化并转成列表向量


def resolve_image_path(image_path: str) -> Path:  # 解析图片路径，兼容相对路径和绝对路径
    p = Path(image_path)  # 把字符串路径转为 Path 对象
    if p.is_absolute() and p.exists():  # 如果是存在的绝对路径
        return p  # 直接返回该路径
    p2 = PROJECT_ROOT / image_path  # 否则按项目根目录拼接相对路径
    if p2.exists():  # 若拼接后路径存在
        return p2  # 返回拼接后的路径
    return p  # 若都不存在则返回原路径（供后续错误提示）


@st.cache_data(show_spinner=False)  # 缓存图片编码结果，避免重复读盘
def image_to_data_uri(path_str: str) -> str | None:  # 把图片文件转成 data URI
    img_path = resolve_image_path(path_str)  # 先解析出图片真实路径
    if not img_path.exists():  # 路径不存在时返回空
        return None

    suffix = img_path.suffix.lower()  # 读取后缀名用于推断 MIME
    if suffix in {".jpg", ".jpeg"}:  # jpg/jpeg 类型
        mime = "image/jpeg"
    elif suffix == ".png":  # png 类型
        mime = "image/png"
    elif suffix == ".webp":  # webp 类型
        mime = "image/webp"
    else:  # 其他类型默认按二进制流处理
        mime = "application/octet-stream"

    data = base64.b64encode(img_path.read_bytes()).decode("ascii")  # 编码成 base64 字符串
    return f"data:{mime};base64,{data}"  # 返回可放到 img src 的 data URI


def render_results_html(results: list[dict]) -> str:  # 用和本地版一致的 HTML 结构渲染结果卡片
    if not results:  # 没有结果时返回提示
        return "<p>No results.</p>"

    cards: list[str] = []  # 存放每张卡片 HTML
    for item in results:  # 遍历每条检索结果
        image_id = html.escape(str(item.get("image_id", "")))  # 转义 image_id 防止 HTML 注入
        score = float(item.get("score", 0.0))  # 读取分数
        image_uri = image_to_data_uri(str(item.get("image_path", "")))  # 转成可显示图片地址

        if image_uri is None:  # 图片缺失时使用占位块
            thumb_html = '<div class="thumb thumb-missing">Image not found</div>'
        else:  # 图片存在时渲染 img
            thumb_html = f'<img class="thumb" src="{image_uri}" alt="{image_id}" loading="lazy" />'

        cards.append(
            f'<article class="card">{thumb_html}<div class="meta"><div class="score">score: {score:.4f}</div><div class="id">{image_id}</div></div></article>'
        )  # 拼接单张卡片，类名与本地 Flask 页面保持一致

    return f'<section class="results">{"".join(cards)}</section>'  # 返回完整结果网格


def get_query_params() -> tuple[str, int, bool]:  # 读取并规范化 URL 查询参数
    params = st.query_params  # 获取当前 URL 参数
    has_query = "query" in params  # 判断是否显式携带 query 参数

    query_val = params.get("query", "")  # 读取 query 参数
    if isinstance(query_val, list):  # 兼容列表形式
        query_val = query_val[0] if query_val else ""
    query = str(query_val)  # 确保是字符串

    k_val = params.get("k", "8")  # 读取 k 参数（默认 8）
    if isinstance(k_val, list):  # 兼容列表形式
        k_val = k_val[0] if k_val else "8"
    try:
        k = int(str(k_val))  # 尝试转整数
    except ValueError:
        k = 8  # 失败时回退默认值
    k = max(1, min(20, k))  # 把 k 限制在 1~20

    return query, k, has_query  # 返回查询文本、k、是否显式查询


def main():  # Streamlit 页面主函数
    st.set_page_config(page_title="LensSeek", layout="wide", initial_sidebar_state="collapsed")  # 设置页面配置

    st.markdown(  # 注入与本地 Flask 页同款 CSS
        """
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
        .stApp {
          background: var(--bg);
          color: var(--ink);
          font-family: "Avenir Next", "Trebuchet MS", sans-serif;
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        footer {
          visibility: hidden;
          height: 0;
          position: fixed;
        }
        .block-container {
          padding: 0 !important;
          max-width: none !important;
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
        .thumb-missing {
          display: grid;
          place-items: center;
          color: #53657a;
          font-size: 12px;
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
        """,
        unsafe_allow_html=True,
    )

    image_file = DEMO_IMAGE_FILE if DEMO_IMAGE_FILE.exists() else FULL_IMAGE_FILE  # 云端优先使用 demo 数据
    if not image_file.exists():  # 若向量文件不存在
        st.error(f"Embedding file not found: {image_file}")  # 提示缺失
        st.stop()  # 中断执行

    query, k, has_query = get_query_params()  # 获取 URL 查询参数
    query_clean = query.strip()  # 清理查询文本

    status_text = "Ready. Enter a query and click Search."  # 默认状态文案
    results: list[dict] = []  # 默认结果为空

    if has_query:  # 仅当用户显式提交 query 时执行检索
        if query_clean == "":  # 空查询校验
            status_text = "Query cannot be empty."  # 空查询提示
        else:
            start = time.perf_counter()  # 记录开始时间
            query_embedding = encode_query_live(query_clean)  # 计算查询文本向量
            results = get_searcher(str(image_file)).search(query_embedding, k=k)  # 执行检索
            elapsed_ms = int((time.perf_counter() - start) * 1000)  # 计算耗时
            status_text = f'Query: "{query_clean}" · {len(results)} results · {elapsed_ms} ms'  # 成功状态文案

    page_html = f"""
    <main class="wrap">
      <section class="hero">
        <div class="orb a"></div>
        <div class="orb b"></div>
        <div class="center">
          <h1 class="brand">Lens<span class="brand-accent">Seek</span></h1>
          <p class="sub">Search images with natural language. Type what you want to see, and the engine returns semantically similar photos.</p>

          <section class="search-shell">
            <form id="search-form" class="search-row" method="get">
              <div>
                <label class="label" for="query">Search Query (describe the image you want)</label>
                <input id="query" name="query" class="field" value="{html.escape(query, quote=True)}" placeholder="e.g. a dog running on grass" required />
              </div>
              <div>
                <label class="label" for="k">Top-K (how many results)</label>
                <input id="k" name="k" class="k" type="number" min="1" max="20" value="{k}" />
              </div>
              <div>
                <label class="label" for="search-btn">Action</label>
                <button id="search-btn" class="btn" type="submit">Search</button>
              </div>
            </form>
            <p class="helper">Tip: Use subject + action + scene, for example: "two children playing football in a park".</p>
            <div class="status" id="status">{html.escape(status_text)}</div>
          </section>
        </div>
      </section>

      <section class="results-head">
        <h2 class="results-title">Visual Matches</h2>
        <p class="results-note">Sorted by vector similarity score.</p>
      </section>

      {render_results_html(results)}
    </main>
    """

    st.markdown(page_html, unsafe_allow_html=True)  # 输出与本地版同结构页面


if __name__ == "__main__":  # 仅当脚本被直接运行时执行
    main()  # 调用主函数启动 Streamlit 页面
