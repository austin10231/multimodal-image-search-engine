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
    st.set_page_config(page_title="LensSeek", page_icon="🔎", layout="wide")  # 设置页面标题、图标和布局

    st.markdown(  # 注入自定义样式，做成简洁搜索引擎风格
        """
        <style>
        .big-title { font-size: 3rem; margin-bottom: 0.2rem; font-family: Georgia, serif; }
        .sub-title { color: #6b7280; margin-bottom: 1.2rem; }
        .hint-box { border: 1px solid #dbe4f0; border-radius: 14px; padding: 12px 14px; background: #f8fbff; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='big-title'>LensSeek</div>", unsafe_allow_html=True)  # 显示主标题
    st.markdown(  # 显示副标题
        "<div class='sub-title'>Chrome-like multimodal search with CLIP + FAISS</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:  # 侧边栏配置区
        st.header("Settings")  # 侧边栏标题
        mode = st.selectbox(  # 数据源选择控件
            "Data Source (用途: 选择检索库)",
            options=["Demo (recommended for cloud)", "Full (local only)"],
            index=0,
        )
        k = st.slider("Top-K (用途: 返回多少结果)", min_value=1, max_value=20, value=8, step=1)  # Top-K 滑条
        st.caption("首次查询会加载模型，耗时会更长。")  # 说明首次加载成本

    if mode.startswith("Full") and FULL_IMAGE_FILE.exists():  # 若用户选择全量且文件存在
        image_file = FULL_IMAGE_FILE  # 使用全量向量文件
    else:  # 否则默认使用 demo
        image_file = DEMO_IMAGE_FILE  # 使用 demo 向量文件

    if not image_file.exists():  # 如果向量文件不存在
        st.error(f"Embedding file not found: {image_file}")  # 提示缺失文件
        st.stop()  # 停止后续执行

    query = st.text_input(  # 查询输入框
        "Search Query (用途: 输入你想找的图片描述)",
        value="a dog running on grass",
        help="例如: two kids playing football in a park",
    )

    st.markdown(  # 功能说明框
        "<div class='hint-box'>"
        "流程: 文本查询 -> CLIP 文本向量 -> FAISS 相似度检索 -> 返回 Top-K 图片"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("Search", type="primary", use_container_width=True):  # 点击搜索按钮时执行检索
        if query.strip() == "":  # 空查询校验
            st.warning("Query cannot be empty.")  # 提示用户输入不能为空
            st.stop()  # 停止执行

        with st.spinner("Loading model and searching..."):  # 显示加载提示
            start = time.perf_counter()  # 记录开始时间
            query_embedding = encode_query_live(query.strip())  # 计算查询向量
            searcher = get_searcher(str(image_file))  # 获取 FAISS 检索器
            results = searcher.search(query_embedding, k=k)  # 执行 Top-K 检索
            elapsed_ms = int((time.perf_counter() - start) * 1000)  # 计算耗时毫秒数

        st.success(f"Found {len(results)} results in {elapsed_ms} ms")  # 显示检索结果摘要

        cols = st.columns(4)  # 创建 4 列网格展示结果
        for i, item in enumerate(results):  # 遍历每条检索结果
            col = cols[i % 4]  # 轮流把结果放入不同列
            with col:
                img_path = resolve_image_path(item["image_path"])  # 解析当前图片路径
                if img_path.exists():  # 若图片存在
                    st.image(str(img_path), use_container_width=True)  # 展示图片
                else:  # 若图片文件不存在
                    st.info("Image not found in repo.")  # 提示图片缺失
                st.caption(f"score: {item['score']:.4f}")  # 显示相似度分数
                st.code(item["image_id"])  # 显示图片 id


if __name__ == "__main__":  # 仅当直接运行脚本时执行
    main()  # 调用主函数启动 Streamlit 页面
