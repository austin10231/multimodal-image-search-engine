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
DATA_DIR = PROJECT_ROOT / "data"  # 全量数据目录
IMAGES_DIR = DATA_DIR / "Images"  # 全量图片目录
DEMO_IMAGE_FILE = PROJECT_ROOT / "demo_data" / "image_embeddings_demo.jsonl"  # Demo 向量路径
FULL_IMAGE_FILE = PROJECT_ROOT / "data" / "image_embeddings.jsonl"  # 全量向量路径
DEMO_IMAGES_DIR = PROJECT_ROOT / "demo_data" / "images"  # Demo 图片目录


def choose_runtime_assets() -> tuple[Path | None, Path | None, str]:  # 与本地 Flask 版保持一致的数据选择逻辑
    if FULL_IMAGE_FILE.exists() and IMAGES_DIR.exists():  # full 模式：要求向量与图片目录都存在
        return FULL_IMAGE_FILE, IMAGES_DIR, "full"
    if DEMO_IMAGE_FILE.exists() and DEMO_IMAGES_DIR.exists():  # demo 模式：同样要求向量与图片目录都存在
        return DEMO_IMAGE_FILE, DEMO_IMAGES_DIR, "demo"
    return None, None, "missing"  # 都不存在时标记为 missing


ACTIVE_EMBEDDINGS_FILE, ACTIVE_IMAGES_DIR, DATA_MODE = choose_runtime_assets()  # 与本地版一致：提前确定当前运行模式


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


def resolve_image_path(image_path: str, active_images_dir: Path | None = None) -> Path:  # 解析图片路径
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return p
    p2 = PROJECT_ROOT / image_path
    if p2.exists():
        return p2
    if active_images_dir is not None:  # 对齐本地版逻辑：可用图片目录下按文件名兜底
        p3 = active_images_dir / p.name
        if p3.exists():
            return p3
    return p


@st.cache_data(show_spinner=False)  # 缓存图片编码结果，减少重复读盘

def image_to_data_uri(path_str: str, active_images_dir_str: str | None = None) -> str | None:
    active_images_dir = Path(active_images_dir_str) if active_images_dir_str else None
    img_path = resolve_image_path(path_str, active_images_dir=active_images_dir)
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


def render_results_html(results: list[dict], active_images_dir: Path | None = None) -> str:  # 渲染结果卡片 HTML
    if not results:
        return "<p class='empty'>No results.</p>"

    cards: list[str] = []
    for item in results:
        image_id = html.escape(str(item.get("image_id", "")))
        score = float(item.get("score", 0.0))
        image_uri = image_to_data_uri(str(item.get("image_path", "")), str(active_images_dir) if active_images_dir else None)

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
          --bg: #f6f8fb;
          --ink: #1f2a37;
          --muted: #6a7789;
          --line: #d5dde8;
          --brand: #3165f6;
          --brand-soft: #e8efff;
          --card: #ffffff;
          --shadow: 0 10px 30px rgba(23, 38, 74, 0.08);
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
        div[data-testid="InputInstructions"] {
          display: none !important;
        }
        .block-container {
          width: min(1120px, 94vw);
          margin: 0 auto;
          padding-top: 24px;
          padding-bottom: 56px;
        }
        .hero {
          min-height: 22vh;
          display: grid;
          place-items: center;
          position: relative;
          text-align: center;
          margin-bottom: 10px;
        }
        .orb {
          position: absolute;
          border-radius: 999px;
          filter: blur(8px);
          opacity: 0.28;
          pointer-events: none;
          z-index: 0;
        }
        .orb-a {
          width: 280px;
          height: 280px;
          background: #d9e6ff;
          top: -40px;
          left: 4%;
        }
        .orb-b {
          width: 220px;
          height: 220px;
          background: #ffe6d3;
          bottom: -10px;
          right: 8%;
        }
        .center {
          width: min(940px, 100%);
          text-align: center;
          z-index: 1;
          position: relative;
        }
        .logo {
          margin: 0;
          color: var(--ink);
          font-family: "Baskerville", "Times New Roman", "Georgia", serif;
          font-size: clamp(62px, 8.6vw, 104px);
          font-weight: 700;
          letter-spacing: 0.6px;
          line-height: 1.02;
        }
        .logo-accent { color: var(--brand); }
        .subtitle {
          margin: 10px auto 14px;
          max-width: 100%;
          color: var(--muted);
          font-size: 14px;
          line-height: 1.6;
          text-align: center;
          white-space: nowrap;
        }
        div[data-testid="stForm"] {
          background: #fff;
          border: 1px solid #dbe3ee;
          border-radius: 26px;
          box-shadow: var(--shadow);
          padding: 20px;
          max-width: 940px;
          margin: 0 auto;
        }
        div[data-testid="stForm"] > div {
          border: 0;
        }
        .label {
          display: block;
          font-size: 12px;
          letter-spacing: 0.03em;
          color: var(--muted);
          margin: 0 0 6px 2px;
        }
        div[data-testid="stForm"] div[data-testid="stTextInput"] {
          margin-bottom: 0 !important;
        }
        div[data-testid="stForm"] div[data-testid="stNumberInput"] {
          margin-bottom: 0 !important;
        }
        div[data-testid="stForm"] div[data-testid="stTextInputRootElement"] {
          background: #fff !important;
          border: 1px solid var(--line) !important;
          border-radius: 999px !important;
          min-height: 50px !important;
          padding: 0 12px !important;
          transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
          box-shadow: none !important;
        }
        div[data-testid="stForm"] div[data-testid="stTextInputRootElement"]:focus-within {
          border-color: var(--brand) !important;
          box-shadow: 0 0 0 4px var(--brand-soft) !important;
        }
        div[data-testid="stForm"] div[data-testid="stTextInputRootElement"] > div,
        div[data-testid="stForm"] div[data-testid="stTextInputRootElement"] [data-baseweb="base-input"],
        div[data-testid="stForm"] div[data-testid="stTextInputRootElement"] [data-baseweb="input"] {
          background: transparent !important;
          border: 0 !important;
          box-shadow: none !important;
          min-height: 0 !important;
          padding: 0 !important;
        }
        div[data-testid="stForm"] div[data-testid="stTextInputRootElement"] input {
          background: transparent !important;
          color: var(--ink) !important;
          font-size: 16px !important;
          padding-left: 4px !important;
          padding-right: 4px !important;
          caret-color: var(--brand) !important;
        }
        div[data-testid="stForm"] div[data-testid="stTextInputRootElement"] input::placeholder {
          color: #8693a5 !important;
          opacity: 1 !important;
        }
        div[data-testid="stForm"] div[data-testid="stNumberInput"] [data-baseweb="input"] {
          background: #fff !important;
          border: 1px solid var(--line) !important;
          border-radius: 999px !important;
          min-height: 50px !important;
          box-shadow: none !important;
          transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
        }
        div[data-testid="stForm"] div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
          border-color: var(--brand) !important;
          box-shadow: 0 0 0 4px var(--brand-soft) !important;
        }
        div[data-testid="stForm"] div[data-testid="stNumberInput"] input {
          color: var(--ink) !important;
          font-size: 16px !important;
          font-weight: 600 !important;
        }
        div[data-testid="stForm"] div[data-testid="stNumberInput"] button {
          background: #fff !important;
          color: #4c5f78 !important;
          border: 0 !important;
          box-shadow: none !important;
        }
        div[data-testid="stForm"] div[data-testid="stNumberInput"] button:hover {
          background: #eef2f8 !important;
          color: var(--ink) !important;
        }
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] {
          margin-top: 0 !important;
        }
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] button {
          width: 100%;
          border: 0 !important;
          border-radius: 999px !important;
          background: linear-gradient(135deg, #2f66ff, #2e8bff) !important;
          color: #fff !important;
          font-size: 16px !important;
          font-weight: 650 !important;
          min-height: 44px !important;
          box-shadow: 0 6px 14px rgba(49,101,246,.24) !important;
        }
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] button:hover {
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
          margin: 0;
          color: var(--ink);
          font-size: 15px;
          font-weight: 650;
        }
        .results-note {
          margin: 0;
          color: var(--muted);
          font-size: 12px;
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
          color: var(--muted);
          font-size: 12px;
        }
        .meta { padding: 10px 12px 12px; font-size: 13px; line-height: 1.5; }
        .score { color: #1f4dd6; font-weight: 700; }
        .id {
          color: #53657a;
          font-family: "Menlo", "Monaco", "Courier New", monospace;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .empty {
          color: var(--muted);
          margin-top: 6px;
        }
        @media (max-width: 860px) {
          .hero { min-height: 18vh; }
          .logo { font-size: clamp(44px, 11vw, 74px); }
          .subtitle {
            font-size: 14px;
            white-space: normal;
          }
          div[data-testid="stForm"] {
            border-radius: 20px;
            padding: 14px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="hero">
          <div class="orb orb-a"></div>
          <div class="orb orb-b"></div>
          <div class="center">
            <h1 class="logo">Lens<span class="logo-accent">Seek</span></h1>
            <p class="subtitle">Search images with natural language. Type what you want to see, and the engine returns semantically similar photos.</p>
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
    if "last_k" not in st.session_state:
        st.session_state["last_k"] = 8
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = st.session_state.get("last_query", "")
    if "k_input" not in st.session_state:
        st.session_state["k_input"] = int(st.session_state.get("last_k", 8))

    image_file = ACTIVE_EMBEDDINGS_FILE
    active_images_dir = ACTIVE_IMAGES_DIR
    if image_file is None:
        st.error("Embedding file not found. Expected full data or demo data assets.")
        st.stop()

    status_text = "Ready. Enter a query and click Search."
    if st.session_state.get("last_results"):
        status_text = (
            f'Query: "{st.session_state.get("last_query", "")}" · '
            f'{len(st.session_state.get("last_results", []))} results · '
            f'{st.session_state.get("last_elapsed_ms", 0)} ms'
        )

    outer_l, outer_c, outer_r = st.columns([0.3, 11.4, 0.3])  # 中间列控制整体搜索区宽度
    with outer_c:
        with st.form("search_form", clear_on_submit=False):
            query_col, k_col, action_col = st.columns([1.0, 0.32, 0.22], gap="medium")
            with query_col:
                st.markdown("<label class='label'>Search Query (describe the image you want)</label>", unsafe_allow_html=True)
                query = st.text_input(
                    "search_query",
                    key="query_input",
                    placeholder="e.g. a dog running on grass",
                    label_visibility="collapsed",
                    autocomplete="off",
                )
            with k_col:
                st.markdown("<label class='label'>Top-K (how many results)</label>", unsafe_allow_html=True)
                k_value = st.number_input(
                    "top_k",
                    min_value=1,
                    max_value=20,
                    step=1,
                    key="k_input",
                    label_visibility="collapsed",
                )
            with action_col:
                st.markdown("<label class='label'>Action</label>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Search", use_container_width=True)

            st.markdown(
                "<p class='helper'>Tip: Use subject + action + scene, for example: \"two children playing football in a park\".</p>",
                unsafe_allow_html=True,
            )
            status_placeholder = st.empty()

    if submitted:
        query_clean = str(query).strip()
        if query_clean == "":
            status_text = "Query cannot be empty."
        else:
            k_int = max(1, min(20, int(k_value)))
            try:
                with st.spinner("Searching..."):
                    start = time.perf_counter()
                    query_embedding = encode_query_live(query_clean)
                    results = get_searcher(str(image_file)).search(query_embedding, k=k_int)
                    elapsed_ms = int((time.perf_counter() - start) * 1000)

                st.session_state["last_results"] = results
                st.session_state["last_elapsed_ms"] = elapsed_ms
                st.session_state["last_query"] = query_clean
                st.session_state["last_k"] = k_int
                st.session_state["k_input"] = k_int
                status_text = f'Query: "{query_clean}" · {len(results)} results · {elapsed_ms} ms'
            except Exception as e:
                status_text = "Search failed. Please retry."
                st.error(f"Runtime error: {e}")

    status_placeholder.markdown(f"<div class='status'>{html.escape(status_text)}</div>", unsafe_allow_html=True)

    st.markdown(
        "<section class='results-head'><h2 class='results-title'>Visual Matches</h2><p class='results-note'>Sorted by vector similarity score.</p></section>",
        unsafe_allow_html=True,
    )

    results = st.session_state.get("last_results", [])
    st.markdown(render_results_html(results, active_images_dir=active_images_dir), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
