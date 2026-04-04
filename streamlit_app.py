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
    st.set_page_config(  # 设置页面基础信息
        page_title="LensSeek",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(  # 注入页面样式，尽量贴近 Chrome 搜索首页视觉
        """
        <style>
        .stApp {
          background:
            radial-gradient(900px 500px at 8% -10%, #e8efff 0%, transparent 58%),
            radial-gradient(700px 420px at 100% 6%, #ffeedd 0%, transparent 55%),
            #f6f8fb;
        }
        .hero { text-align: center; margin-top: 8px; margin-bottom: 14px; }
        .brand { font-family: Georgia, "Times New Roman", serif; font-size: 68px; line-height: 1; margin: 0; color: #1f2a37; }
        .brand span { color: #3165f6; }
        .sub { color: #6b7280; margin-top: 8px; margin-bottom: 18px; font-size: 15px; }
        .panel {
          background: rgba(255,255,255,0.96);
          border: 1px solid #d9e2ef;
          border-radius: 24px;
          box-shadow: 0 12px 30px rgba(20, 40, 74, 0.10);
          padding: 10px 12px 6px 12px;
          margin-bottom: 12px;
        }
        .helper {
          border: 1px solid #dbe4f0;
          background: #f8fbff;
          border-radius: 12px;
          padding: 10px 12px;
          color: #556377;
          font-size: 13px;
          margin-top: 4px;
          margin-bottom: 10px;
        }
        .result-head { margin-top: 6px; margin-bottom: 10px; color: #334155; font-weight: 600; font-size: 15px; }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div {
          border-radius: 999px !important;
        }
        div[data-testid="stFormSubmitButton"] button {
          border-radius: 999px !important;
          height: 42px;
          border: none !important;
          color: white !important;
          background: linear-gradient(135deg, #2f66ff, #2e8bff) !important;
          font-weight: 650 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(  # 顶部品牌区
        """
        <div class="hero">
          <h1 class="brand">Lens<span>Seek</span></h1>
          <div class="sub">Chrome-like multimodal image search with CLIP + FAISS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "last_results" not in st.session_state:  # 初始化会话中的结果缓存
        st.session_state["last_results"] = []  # 默认无结果
    if "last_elapsed_ms" not in st.session_state:  # 初始化会话中的耗时缓存
        st.session_state["last_elapsed_ms"] = None  # 默认无耗时
    if "last_query" not in st.session_state:  # 初始化会话中的查询缓存
        st.session_state["last_query"] = ""  # 默认空查询

    st.markdown("<div class='panel'>", unsafe_allow_html=True)  # 搜索面板开始
    with st.form("search_form", clear_on_submit=False):  # 使用 form 保证按钮触发一次完整检索
        c1, c2, c3 = st.columns([6, 2, 3])  # 三列布局：查询、Top-K、数据源
        with c1:
            query = st.text_input(  # 查询输入框
                "Search Query (用途: 描述你想找的图片)",
                value=st.session_state.get("last_query", "a dog running on grass"),
                placeholder="e.g. two children playing football in a park",
            )
        with c2:
            k = st.number_input(  # Top-K 输入框
                "Top-K (用途: 返回多少结果)",
                min_value=1,
                max_value=20,
                value=8,
                step=1,
            )
        with c3:
            mode = st.selectbox(  # 数据源选择框
                "Data Source (用途: 选择检索库)",
                options=["Demo (recommended for cloud)", "Full (local only)"],
                index=0,
            )
        st.markdown(  # 流程说明框
            "<div class='helper'>流程: 文本查询 -> CLIP 文本向量 -> FAISS 相似度检索 -> 返回 Top-K 图片</div>",
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button("Search")  # 提交按钮
    st.markdown("</div>", unsafe_allow_html=True)  # 搜索面板结束

    if mode.startswith("Full") and FULL_IMAGE_FILE.exists():  # 若用户选择全量且全量文件存在
        image_file = FULL_IMAGE_FILE  # 使用全量向量库
    else:  # 否则默认使用 Demo 向量库
        image_file = DEMO_IMAGE_FILE  # 使用 demo 向量库

    if not image_file.exists():  # 若目标向量文件不存在
        st.error(f"Embedding file not found: {image_file}")  # 提示缺失文件
        st.stop()  # 中断执行

    if submitted:  # 用户点击 Search 后执行检索
        if query.strip() == "":  # 空查询校验
            st.warning("Query cannot be empty.")  # 反馈错误信息
        else:
            with st.spinner("Loading model and searching..."):  # 显示检索中提示
                start = time.perf_counter()  # 记录开始时间
                query_embedding = encode_query_live(query.strip())  # 计算查询向量
                results = get_searcher(str(image_file)).search(query_embedding, k=int(k))  # 执行 FAISS 检索
                elapsed_ms = int((time.perf_counter() - start) * 1000)  # 计算耗时（毫秒）
            st.session_state["last_results"] = results  # 保存检索结果到会话状态
            st.session_state["last_elapsed_ms"] = elapsed_ms  # 保存耗时到会话状态
            st.session_state["last_query"] = query.strip()  # 保存查询文本到会话状态

    results = st.session_state.get("last_results", [])  # 读取会话中的最近结果
    elapsed_ms = st.session_state.get("last_elapsed_ms")  # 读取会话中的最近耗时
    last_query = st.session_state.get("last_query", "")  # 读取会话中的最近查询

    if results:  # 若存在可展示的结果
        st.success(f'Query: "{last_query}" · {len(results)} results · {elapsed_ms} ms')  # 展示摘要信息
        st.markdown("<div class='result-head'>Visual Matches (sorted by similarity)</div>", unsafe_allow_html=True)  # 结果区标题
        cols = st.columns(4)  # 使用四列网格展示结果卡片
        for i, item in enumerate(results):  # 遍历每条检索结果
            with cols[i % 4]:  # 轮流分配到四列
                img_path = resolve_image_path(item["image_path"])  # 解析图片路径
                if img_path.exists():  # 若图片文件存在
                    st.image(str(img_path), use_container_width=True)  # 展示图片
                else:  # 若图片文件不存在
                    st.info("Image not found in repo.")  # 提示缺图
                st.caption(f"score: {item['score']:.4f}")  # 显示相似度分数
                st.code(item["image_id"])  # 显示图片标识


if __name__ == "__main__":  # 仅当直接运行脚本时执行
    main()  # 调用主函数启动 Streamlit 页面
