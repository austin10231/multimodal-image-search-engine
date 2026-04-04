# Multimodal Image Search (CLIP + FAISS)

A lightweight multimodal image search project:
- Encode images with CLIP
- Encode text queries with CLIP
- Retrieve top-k similar images with FAISS
- View results in a simple web UI

## Project Structure
- `src/load_data.py`: load image-caption data
- `src/data_processing.py`: clean data (one caption per image)
- `src/image_embedding.py`: generate image embeddings
- `src/text_embedding.py`: generate query embeddings
- `src/search_basic.py`: baseline similarity search
- `src/search_faiss.py`: FAISS search
- `web_app.py`: Flask web app

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web app:
```bash
python web_app.py
```

3. Open:
`http://127.0.0.1:8000`

## Notes
- If Python crashes on macOS (OpenMP conflict), keep thread limits enabled in code.
- Default data files are under `data/`.

## Copyright
Copyright (c) Mutian He 2026
