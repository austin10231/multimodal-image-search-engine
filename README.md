# Multimodal Image Search (CLIP + FAISS)

A lightweight multimodal image search project with:
- `streamlit_app.py` (Streamlit, Chrome-like search UI)
- `web_app.py` (Flask)

## Quick Start (Streamlit)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Open: `http://localhost:8501`

## Streamlit Cloud Deploy
1. Push latest code to GitHub.
2. In Streamlit Community Cloud, click **New app**.
3. Select:
   - Repository: `austin10231/multimodal-image-search-engine`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Click **Deploy**.

## Notes
- Streamlit app prefers `demo_data/` for lower memory usage in cloud.
- Flask/Render files are kept in repo but optional.

## Copyright
Copyright (c) Mutian He 2026
