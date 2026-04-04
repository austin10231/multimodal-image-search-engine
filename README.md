# Multimodal Image Search (CLIP + FAISS)

A lightweight multimodal image search project:
- Encode images with CLIP
- Encode text queries with CLIP
- Retrieve top-k similar images with FAISS
- View results in a Streamlit web UI

## Project Structure
- `src/load_data.py`: load image-caption data
- `src/data_processing.py`: clean data (one caption per image)
- `src/image_embedding.py`: generate image embeddings
- `src/text_embedding.py`: generate query embeddings
- `src/search_basic.py`: baseline similarity search
- `src/search_faiss.py`: FAISS search
- `streamlit_app.py`: Streamlit app (deploy entry)
- `demo_data/`: deployable demo embeddings + images

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Streamlit:
```bash
streamlit run streamlit_app.py
```

3. Open:
`http://localhost:8501`

## Streamlit Cloud Deploy
1. Push this repo to GitHub
2. In Streamlit Community Cloud, click **New app**
3. Select:
   - Repository: `austin10231/multimodal-image-search-engine`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Click **Deploy**

## Notes
- The app uses `demo_data/` by default for cloud-friendly deployment.
- First run may take longer because CLIP model needs to download.

## Copyright
Copyright (c) Mutian He 2026
