import base64  # 导入 base64，用于把图片转成 data URI 方便网页展示
import html  # 导入 html，用于安全转义文本
import os  # 导入 os，用于设置运行时环境变量
import time  # 导入 time，用于统计检索耗时
from pathlib import Path  # 导入 Path，用于路径处理

import streamlit as st  # 导入 Streamlit

# 线程和 OpenMP 兜底，降低 faiss + torch 在部分环境下冲突概率
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.search_faiss import FaissSearcher  # 导入 FAISS 检索器
from src.text_embedding import (  # 导入文本编码函数
    encode_one_text,
    load_clip_for_text,
    postprocess_text_embedding,
    prepare_text_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parent  # 项目根目录
DEMO_IMAGE_FILE = PROJECT_ROOT / "demo_data" / "image_embeddings_demo.jsonl"  # Demo 向量路径
FULL_IMAGE_FILE = PROJECT_ROOT / "data" / "image_embeddings.jsonl"  # 全量向量路径


def choose_embeddings_file() -> Path | None:  # 自动选择可用向量文件
    if DEMO_IMAGE_FILE.exists():  # 云端优先使用 demo，内存更友好
        return DEMO_IMAGE_FILE
    if FULL_IMAGE_FILE.exists():  # 本地有全量数据时可回退
        return FULL_IMAGE_FILE
    return None


@st.cache_resource(show_spinner=False)  # 缓存文本模型，避免重复加载

def get_text_components():
    return load_clip_for_text()


@st.cache_resource(show_spinner=False)  # 缓存 FAISS 检索器

def get_searcher(image_file: str):
    return FaissSearcher(Path(image_file))


def encode_query_live(query: str) -> list[float]:  # 把查询文本编码为向量
    model, tokenizer, device = get_text_components()
    inputs = prepare_text_inputs(query, tokenizer, device)
    text_features = encode_one_text(model, inputs)
    return postprocess_text_embedding(text_features)


def resolve_image_path(image_path: str) -> Path:  # 解析图片路径
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return p
    p2 = PROJECT_ROOT / image_path
    if p2.exists():
        return p2
    return p


@st.cache_data(show_spinner=False)  # 缓存图片编码结果，减少重复读盘

def image_to_data_uri(path_str: str) -> str | None:
    img_path = resolve_image_path(path_str)
    if not img_path.exists():
        return None

    suffix = img_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"

    data = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def render_results_html(results: list[dict]) -> str:  # 渲染结果卡片 HTML
    if not results:
        return "<p class='empty'>No results.</p>"

    cards: list[str] = []
    for item in results:
        image_id = html.escape(str(item.get("image_id", "")))
        score = float(item.get("score", 0.0))
        image_uri = image_to_data_uri(str(item.get("image_path", "")))

        if image_uri is None:
            thumb_html = '<div class="thumb thumb-missing">Image not found</div>'
        else:
            thumb_html = f'<img class="thumb" src="{image_uri}" alt="{image_id}" loading="lazy" />'

        cards.append(
            f'<article class="card">{thumb_html}<div class="meta"><div class="score">score: {score:.4f}</div><div class="id">{image_id}</div></div></article>'
        )

    return f'<section class="results">{"".join(cards)}</section>'


def main():
    st.set_page_config(page_title="LensSeek", page_icon="🔎", layout="wide")

    st.markdown(
        """
        <style>
        :root {
          --bg: #ffffff;
          --ink: #202124;
          --muted: #5f6368;
          --line: #dfe1e5;
          --brand: #1a73e8;
          --card: #ffffff;
        }
        .stApp {
          background: var(--bg);
          color: var(--ink);
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
        div[data-testid="InputInstructions"] {
          display: none !important;
        }
        .block-container {
          width: min(1220px, 96vw);
          margin: 0 auto;
          padding-top: 24px;
          padding-bottom: 64px;
        }
        .hero {
          min-height: 24vh;
          display: grid;
          place-items: center;
          text-align: center;
        }
        .logo {
          margin: 0;
          color: #202124;
          font-family: "Arial", "Helvetica", sans-serif;
          font-size: clamp(76px, 8.4vw, 110px);
          font-weight: 700;
          letter-spacing: -1.8px;
          line-height: 1;
        }
        .logo-accent { color: #1a73e8; }
        .subtitle {
          margin: 12px auto 18px;
          max-width: 760px;
          color: var(--muted);
          font-size: 15px;
          line-height: 1.45;
        }

        div[data-testid="stTextInput"] {
          margin-bottom: 8px !important;
        }
        div[data-testid="stTextInputRootElement"] {
          background: #fff !important;
          border: 1px solid var(--line) !important;
          border-radius: 999px !important;
          box-shadow: 0 4px 16px rgba(60,64,67,.14) !important;
          min-height: 66px !important;
          padding: 2px 12px !important;
          transition: box-shadow 0.18s ease, border-color 0.18s ease, transform 0.18s ease !important;
        }
        div[data-testid="stTextInputRootElement"]:focus-within {
          border-color: #1a73e8 !important;
          box-shadow: 0 0 0 4px rgba(26,115,232,.14), 0 10px 22px rgba(60,64,67,.18) !important;
          transform: translateY(-1px) !important;
        }
        div[data-testid="stTextInputRootElement"] > div,
        div[data-testid="stTextInputRootElement"] [data-baseweb="base-input"],
        div[data-testid="stTextInputRootElement"] [data-baseweb="input"] {
          background: transparent !important;
          border: 0 !important;
          box-shadow: none !important;
          min-height: 0 !important;
          padding: 0 !important;
        }
        div[data-testid="stTextInputRootElement"] input {
          background: transparent !important;
          color: var(--ink) !important;
          font-size: 20px !important;
          font-weight: 450 !important;
          padding-left: 22px !important;
          padding-right: 22px !important;
          line-height: 1.2 !important;
          caret-color: #1a73e8 !important;
        }
        div[data-testid="stTextInputRootElement"] input::placeholder {
          color: #80868b !important;
          opacity: 1 !important;
          font-size: 20px !important;
        }
        div[data-testid="stTextInputRootElement"] input:focus::placeholder {
          color: #9aa0a6 !important;
        }
        div[data-testid="stButton"] {
          text-align: center;
          margin-top: 10px;
        }
        div[data-testid="stButton"] button {
          border: 0 !important;
          border-radius: 999px !important;
          background: linear-gradient(135deg, #3f83f8 0%, #2463eb 100%) !important;
          color: #ffffff !important;
          font-size: 20px !important;
          font-weight: 650 !important;
          height: 54px !important;
          min-width: 220px !important;
          padding: 0 30px !important;
          margin-top: 0 !important;
          white-space: nowrap !important;
          display: inline-flex !important;
          align-items: center !important;
          justify-content: center !important;
          letter-spacing: 0.2px !important;
          box-shadow: 0 6px 14px rgba(36,99,235,.24) !important;
          transition: transform 0.14s ease, box-shadow 0.14s ease !important;
        }
        div[data-testid="stButton"] button:hover {
          box-shadow: 0 10px 18px rgba(36,99,235,.3) !important;
          transform: translateY(-1px) !important;
        }
        div[data-testid="stButton"] button:active {
          transform: translateY(0);
          box-shadow: 0 4px 10px rgba(36,99,235,.22) !important;
        }
        .helper {
          text-align: center;
          margin: 18px auto 0;
          color: var(--muted);
          font-size: 14px;
          max-width: 980px;
          line-height: 1.45;
        }
        .status {
          text-align: center;
          margin: 12px auto 30px;
          color: var(--muted);
          font-size: 14px;
          min-height: 20px;
        }
        .results-head {
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          margin: 16px auto 16px;
          gap: 4px;
          flex-direction: column;
          flex-wrap: wrap;
        }
        .results-title {
          margin: 0;
          color: #202124;
          font-size: 26px;
          font-weight: 650;
        }
        .results-note {
          margin: 0;
          color: var(--muted);
          font-size: 13px;
        }
        .results {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(240px, 320px));
          justify-content: center;
          gap: 14px;
        }
        .card {
          background: var(--card);
          border: 1px solid #e8eaed;
          border-radius: 14px;
          overflow: hidden;
          box-shadow: 0 1px 3px rgba(60,64,67,.12);
          transition: transform 0.15s ease;
        }
        .card:hover { transform: translateY(-2px); }
        .thumb {
          width: 100%;
          aspect-ratio: 4 / 3;
          object-fit: cover;
          display: block;
          background: #f1f3f4;
        }
        .thumb-missing {
          display: grid;
          place-items: center;
          color: var(--muted);
          font-size: 12px;
        }
        .meta { padding: 10px 12px 12px; font-size: 13px; line-height: 1.5; }
        .score { color: var(--brand); font-weight: 700; }
        .id {
          color: var(--muted);
          font-family: "Menlo", "Monaco", "Courier New", monospace;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .empty { color: var(--muted); margin-top: 6px; text-align: center; }
        @media (max-width: 900px) {
          .hero { min-height: 22vh; }
          .logo { font-size: clamp(62px, 13vw, 90px); }
          .subtitle { font-size: 15px; }
          div[data-testid="stTextInputRootElement"] { min-height: 60px !important; }
          div[data-testid="stTextInputRootElement"] input,
          div[data-testid="stTextInputRootElement"] input::placeholder {
            font-size: 18px !important;
          }
          div[data-testid="stButton"] button {
            min-width: 170px !important;
            height: 48px !important;
            font-size: 17px !important;
          }
          .helper { font-size: 13px; }
          .status { font-size: 13px; min-height: 22px; }
          .results-title { font-size: 24px; }
          .results-note { font-size: 12px; }
          .results {
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            justify-content: stretch;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="hero">
          <div>
            <h1 class="logo">Lens<span class="logo-accent">Seek</span></h1>
            <p class="subtitle">Search images in natural language.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if "last_results" not in st.session_state:
        st.session_state["last_results"] = []
    if "last_elapsed_ms" not in st.session_state:
        st.session_state["last_elapsed_ms"] = None
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""

    image_file = choose_embeddings_file()
    if image_file is None:
        st.error("Embedding file not found. Expected demo_data/image_embeddings_demo.jsonl or data/image_embeddings.jsonl")
        st.stop()

    outer_l, outer_c, outer_r = st.columns([0.8, 7.4, 0.8])  # 中间列控制搜索框宽度
    with outer_c:
        query = st.text_input(
            "searchbox_hidden",
            value=st.session_state.get("last_query", ""),
            placeholder="Search images...",
            label_visibility="collapsed",
            autocomplete="off",  # 关闭浏览器自动补全，避免浮层重叠
        )
        c1, c2, c3 = st.columns([3, 2, 3])
        with c2:
            submitted = st.button("Search", use_container_width=False)

    st.markdown(
        "<p class='helper'>Try: \"a dog running on grass\" · \"people hiking in mountains\" · \"city street at night\"</p>",
        unsafe_allow_html=True,
    )

    status_text = "Ready. Enter a query and click Search."

    if submitted:
        query_clean = query.strip()
        if query_clean == "":
            status_text = "Query cannot be empty."
        else:
            try:
                with st.spinner("Searching..."):
                    start = time.perf_counter()
                    query_embedding = encode_query_live(query_clean)
                    results = get_searcher(str(image_file)).search(query_embedding, k=8)
                    elapsed_ms = int((time.perf_counter() - start) * 1000)

                st.session_state["last_results"] = results
                st.session_state["last_elapsed_ms"] = elapsed_ms
                st.session_state["last_query"] = query_clean
                status_text = f'Query: "{query_clean}" · {len(results)} results · {elapsed_ms} ms'
            except Exception as e:
                status_text = "Search failed. Please retry."
                st.error(f"Runtime error: {e}")

    elif st.session_state.get("last_results"):
        status_text = (
            f'Query: "{st.session_state.get("last_query", "")}" · '
            f'{len(st.session_state.get("last_results", []))} results · '
            f'{st.session_state.get("last_elapsed_ms", 0)} ms'
        )

    st.markdown(f"<div class='status'>{html.escape(status_text)}</div>", unsafe_allow_html=True)

    st.markdown(
        "<section class='results-head'><h2 class='results-title'>Image Results</h2><p class='results-note'>Sorted by vector similarity score.</p></section>",
        unsafe_allow_html=True,
    )

    results = st.session_state.get("last_results", [])
    st.markdown(render_results_html(results), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
