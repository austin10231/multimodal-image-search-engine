# Multimodal Image Search (CLIP + FAISS)

A lightweight multimodal image search project with Flask web UI.

## Quick Start (Flask)
```bash
pip install -r requirements.txt
python web_app.py
```
Open: `http://127.0.0.1:8000`

## Render Deploy (Recommended for Flask UI)
This repo includes `render.yaml`.

1. Push latest code to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select this repository.
4. Render auto-detects `render.yaml`; click **Apply**.
5. After deploy, open the generated Render URL.

## Data Mode
`web_app.py` auto-selects data:
- `full`: uses `data/image_embeddings.jsonl` + `data/Images/`
- `demo`: fallback to `demo_data/image_embeddings_demo.jsonl` + `demo_data/images/`

For cloud deploy, `demo` mode is used by default.

## Copyright
Copyright (c) Mutian He 2026
