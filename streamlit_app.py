import base64  # 导入 base64，用于把本地图片转成 data URI 便于 HTML 直接显示
import html  # 导入 html，用于安全转义文本内容
import time  # 导入 time，用于统计检索耗时
from pathlib import Path  # 导入 Path，用于处理文件路径
import os  # 导入 os，用于设置运行时环境变量

import streamlit as st  # 导入 Streamlit，用于构建网页应用

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


PROJECT_ROOT = Path(__file__).resolve().parent  # 定位项目根目录
DEMO_IMAGE_FILE = PROJECT_ROOT / "demo_data" / "image_embeddings_demo.jsonl"  # Demo 图片向量文件路径
FULL_IMAGE_FILE = PROJECT_ROOT / "data" / "image_embeddings.jsonl"  # 全量图片向量文件路径


@st.cache_resource(show_spinner=False)  # 缓存文本模型资源，避免每次交互重复加载
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

        cards.append(  # 拼接单张卡片，类名与本地 Flask 页面保持一致
            ""
            f'<article class="card">'
            f"{thumb_html}"
            f'<div class="meta">'
            f'<div class="score">score: {score:.4f}</div>'
            f'<div class="id">{image_id}</div>'
            f"</div>"
            f"</article>"
        )

    return f'<section class="results">{"".join(cards)}</section>'  # 返回完整结果网格


def main():  # Streamlit 页面主函数
    st.set_page_config(page_title="LensSeek", layout="wide", initial_sidebar_state="collapsed")  # 设置页面配置

    st.markdown(  # 注入与本地 web_app 完全同款色系、结构与卡片样式
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
        html, body, [class*="css"]  {
          font-family: "Avenir Next", "Trebuchet MS", sans-serif;
        }
        .stApp {
          background: var(--bg);
          color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 0.7rem; }
        .block-container {
          width: min(1120px, 94vw);
          max-width: 1120px;
          padding-top: 36px;
          padding-bottom: 56px;
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
        .brand-accent { color: var(--brand); }
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
        .label {
          display: block;
          font-size: 12px;
          letter-spacing: 0.03em;
          color: var(--muted);
          margin: 0 0 6px 2px;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
          width: 100%;
          border: 1px solid var(--line) !important;
          border-radius: 999px !important;
          font-size: 16px !important;
          padding: 13px 16px !important;
          background: #fff !important;
          color: var(--ink) !important;
          outline: none !important;
          transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
        }
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus {
          border-color: var(--brand) !important;
          box-shadow: 0 0 0 4px var(--brand-soft) !important;
        }
        div[data-testid="stFormSubmitButton"] button {
          width: 100%;
          border: none !important;
          border-radius: 999px !important;
          font-size: 16px !important;
          padding: 13px 16px !important;
          background: linear-gradient(135deg, #2f66ff, #2e8bff) !important;
          color: #fff !important;
          font-weight: 650 !important;
          cursor: pointer;
          margin-top: 1.68rem !important;
        }
        div[data-testid="stFormSubmitButton"] button:hover {
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
          div[data-testid="stHorizontalBlock"] {
            display: grid !important;
            grid-template-columns: 1fr !important;
            gap: 0.65rem !important;
          }
          div[data-testid="stFormSubmitButton"] button {
            border-radius: 14px !important;
            margin-top: 0.2rem !important;
          }
          .search-shell {
            border-radius: 20px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(  # 顶部品牌与副标题（加入同款光晕）
        """
        <section class="hero">
          <div class="orb a"></div>
          <div class="orb b"></div>
          <div class="center">
            <h1 class="brand">Lens<span class="brand-accent">Seek</span></h1>
            <p class="sub">Search images with natural language. Type what you want to see, and the engine returns semantically similar photos.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if "last_results" not in st.session_state:  # 初始化结果缓存
        st.session_state["last_results"] = []  # 默认无结果
    if "last_elapsed_ms" not in st.session_state:  # 初始化耗时缓存
        st.session_state["last_elapsed_ms"] = None  # 默认无耗时
    if "last_query" not in st.session_state:  # 初始化查询缓存
        st.session_state["last_query"] = ""  # 默认空查询
    if "last_k" not in st.session_state:  # 初始化 K 缓存
        st.session_state["last_k"] = 8  # 默认 Top-K=8

    image_file = DEMO_IMAGE_FILE if DEMO_IMAGE_FILE.exists() else FULL_IMAGE_FILE  # Streamlit 云端默认使用 demo 数据
    if not image_file.exists():  # 若向量文件不存在
        st.error(f"Embedding file not found: {image_file}")  # 提示缺失
        st.stop()  # 中断执行

    st.markdown("<section class='search-shell'>", unsafe_allow_html=True)  # 搜索框容器开始
    with st.form("search_form", clear_on_submit=False):  # 搜索表单
        c1, c2, c3 = st.columns([7, 2, 2])  # 与本地版一致的三列布局
        with c1:
            st.markdown("<label class='label' for='query_hidden'>Search Query (describe the image you want)</label>", unsafe_allow_html=True)  # 查询标签
            query = st.text_input(  # 查询输入框
                "query_hidden",
                value=st.session_state.get("last_query", ""),
                placeholder="e.g. a dog running on grass",
                label_visibility="collapsed",
            )
        with c2:
            st.markdown("<label class='label' for='k_hidden'>Top-K (how many results)</label>", unsafe_allow_html=True)  # K 值标签
            k = st.number_input(  # K 输入框
                "k_hidden",
                min_value=1,
                max_value=20,
                value=int(st.session_state.get("last_k", 8)),
                step=1,
                label_visibility="collapsed",
            )
        with c3:
            st.markdown("<label class='label' for='search_btn'>Action</label>", unsafe_allow_html=True)  # 按钮标签
            submitted = st.form_submit_button("Search", use_container_width=True)  # 搜索按钮
        st.markdown(  # 提示文案（与本地版一致）
            "<p class='helper'>Tip: Use subject + action + scene, for example: \"two children playing football in a park\".</p>",
            unsafe_allow_html=True,
        )
    st.markdown("</section>", unsafe_allow_html=True)  # 搜索框容器结束

    status_text = "Ready. Enter a query and click Search."  # 默认状态文案
    if submitted:  # 点击搜索后执行检索
        if query.strip() == "":  # 空查询校验
            status_text = "Query cannot be empty."  # 空查询状态提示
        else:
            with st.spinner("Searching..."):  # 检索中提示
                start = time.perf_counter()  # 记录开始时间
                query_embedding = encode_query_live(query.strip())  # 计算查询向量
                results = get_searcher(str(image_file)).search(query_embedding, k=int(k))  # 执行检索
                elapsed_ms = int((time.perf_counter() - start) * 1000)  # 计算耗时
            st.session_state["last_results"] = results  # 写入结果缓存
            st.session_state["last_elapsed_ms"] = elapsed_ms  # 写入耗时缓存
            st.session_state["last_query"] = query.strip()  # 写入查询缓存
            st.session_state["last_k"] = int(k)  # 写入 K 值缓存
            status_text = f'Query: "{query.strip()}" · {len(results)} results · {elapsed_ms} ms'  # 成功状态文案

    st.markdown(f"<div class='status'>{html.escape(status_text)}</div>", unsafe_allow_html=True)  # 展示状态行

    st.markdown(  # 结果区标题（与本地版一致）
        "<section class='results-head'><h2 class='results-title'>Visual Matches</h2><p class='results-note'>Sorted by vector similarity score.</p></section>",
        unsafe_allow_html=True,
    )

    results = st.session_state.get("last_results", [])  # 读取结果缓存
    st.markdown(render_results_html(results), unsafe_allow_html=True)  # 以本地同款 HTML 卡片渲染结果


if __name__ == "__main__":  # 仅当直接运行脚本时执行
    main()  # 调用主函数启动 Streamlit 页面
