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


def main():  # Streamlit 页面主函数
    st.set_page_config(page_title="LensSeek", page_icon="🔎", layout="wide", initial_sidebar_state="collapsed")  # 设置页面基础配置

    st.markdown(  # 注入与本地 Flask 版一致的页面风格
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
          background:
            radial-gradient(900px 500px at 6% -8%, #d9e6ff 0%, transparent 60%),
            radial-gradient(700px 420px at 102% 8%, #ffe6d3 0%, transparent 58%),
            var(--bg);
          color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1120px; }
        .hero { min-height: 34vh; display: grid; place-items: center; margin-bottom: 8px; }
        .center { width: min(860px, 100%); text-align: center; }
        .brand {
          margin: 0;
          font-family: "Baskerville", "Book Antiqua", serif;
          font-size: clamp(44px, 7vw, 70px);
          letter-spacing: 0.6px;
          line-height: 1.02;
          color: #1f2a37;
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
          padding: 12px;
        }
        .helper {
          margin: 10px 6px 2px;
          color: var(--muted);
          font-size: 12px;
          line-height: 1.45;
        }
        .results-head { display: flex; justify-content: space-between; align-items: baseline; margin: 14px 4px 10px; }
        .results-title { font-size: 15px; font-weight: 650; margin: 0; color: #334155; }
        .results-note { color: var(--muted); font-size: 12px; margin: 0; }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
          border-radius: 999px !important;
          border: 1px solid var(--line) !important;
          padding: 0.62rem 0.95rem !important;
        }
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus {
          border-color: var(--brand) !important;
          box-shadow: 0 0 0 4px var(--brand-soft) !important;
        }
        div[data-testid="stFormSubmitButton"] button {
          border-radius: 999px !important;
          border: none !important;
          height: 42px !important;
          margin-top: 1.72rem !important;
          color: #fff !important;
          background: linear-gradient(135deg, #2f66ff, #2e8bff) !important;
          font-weight: 650 !important;
        }
        div[data-testid="stImage"] img {
          border-radius: 14px !important;
          border: 1px solid #d8e2f0 !important;
          box-shadow: 0 4px 14px rgba(22, 43, 76, 0.08) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(  # 顶部品牌区（与本地版一致）
        """
        <section class="hero">
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
        st.session_state["last_query"] = "a dog running on grass"  # 默认示例查询
    if "last_k" not in st.session_state:  # 初始化 K 值缓存
        st.session_state["last_k"] = 8  # 默认 Top-K=8

    image_file = DEMO_IMAGE_FILE if DEMO_IMAGE_FILE.exists() else FULL_IMAGE_FILE  # Streamlit 默认使用可部署的 demo 数据
    if not image_file.exists():  # 若数据文件不存在
        st.error(f"Embedding file not found: {image_file}")  # 提示缺失向量文件
        st.stop()  # 中断后续执行

    st.markdown("<section class='search-shell'>", unsafe_allow_html=True)  # 搜索面板开始
    with st.form("search_form", clear_on_submit=False):  # 搜索表单
        c1, c2, c3 = st.columns([7, 2, 2])  # 三列布局：query、k、button
        with c1:
            query = st.text_input(  # 查询输入框
                "Search Query (describe the image you want)",
                value=st.session_state.get("last_query", "a dog running on grass"),
                placeholder="e.g. a dog running on grass",
            )
        with c2:
            k = st.number_input(  # Top-K 输入框
                "Top-K (how many results)",
                min_value=1,
                max_value=20,
                value=int(st.session_state.get("last_k", 8)),
                step=1,
            )
        with c3:
            submitted = st.form_submit_button("Search")  # 搜索按钮
        st.markdown(  # 提示语（与本地风格一致）
            "<p class='helper'>Tip: Use subject + action + scene, for example: \"two children playing football in a park\".</p>",
            unsafe_allow_html=True,
        )
    st.markdown("</section>", unsafe_allow_html=True)  # 搜索面板结束

    if submitted:  # 提交后执行检索
        if query.strip() == "":  # 校验空查询
            st.warning("Query cannot be empty.")  # 提示输入不能为空
        else:
            with st.spinner("Searching..."):  # 显示检索提示
                start = time.perf_counter()  # 记录起始时间
                query_embedding = encode_query_live(query.strip())  # 文本编码为向量
                results = get_searcher(str(image_file)).search(query_embedding, k=int(k))  # FAISS 检索 Top-K
                elapsed_ms = int((time.perf_counter() - start) * 1000)  # 计算耗时毫秒
            st.session_state["last_results"] = results  # 存储结果
            st.session_state["last_elapsed_ms"] = elapsed_ms  # 存储耗时
            st.session_state["last_query"] = query.strip()  # 存储查询
            st.session_state["last_k"] = int(k)  # 存储 K 值

    results = st.session_state.get("last_results", [])  # 读取缓存结果
    elapsed_ms = st.session_state.get("last_elapsed_ms")  # 读取缓存耗时
    last_query = st.session_state.get("last_query", "")  # 读取缓存查询
    last_k = st.session_state.get("last_k", 8)  # 读取缓存 K 值

    if results:  # 若有检索结果则渲染
        st.info(f'Query: "{last_query}" · Top-{last_k} · {len(results)} results · {elapsed_ms} ms')  # 结果摘要
        st.markdown(  # 结果区标题
            "<div class='results-head'><p class='results-title'>Visual Matches</p><p class='results-note'>Sorted by vector similarity score.</p></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(4)  # 四列结果网格
        for i, item in enumerate(results):  # 遍历结果
            with cols[i % 4]:  # 分配到对应列
                img_path = resolve_image_path(item["image_path"])  # 解析图片路径
                if img_path.exists():  # 图片存在则显示
                    st.image(str(img_path), use_container_width=True)  # 展示图片
                else:  # 图片不存在则提示
                    st.warning("Image not found.")  # 缺失图片提示
                st.caption(f"score: {item['score']:.4f}")  # 显示分数
                st.code(item["image_id"])  # 显示图片 id


if __name__ == "__main__":  # 仅当直接运行脚本时执行
    main()  # 调用主函数启动 Streamlit 页面
