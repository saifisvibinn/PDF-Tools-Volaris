# PDF Layout Extraction Companion

A streamlined workflow for extracting figures, tables, annotated layouts, and markdown text from scientific PDFs using [DocLayout-YOLO](https://github.com/juliozhao/DocLayout-YOLO), PyMuPDF, and Flask. The project exposes a command-line pipeline (`main.py`) and a modern Flask web UI (`app.py`).

---

## Features
- **Layout-aware extraction** of figures and tables with YOLO-based detection
- **Cross-page stitching** for multi-page tables, captions, titles, and body text
- **Annotated PDF output** with bounding boxes for detected regions
- **Markdown export** powered by `pymupdf4llm` / `pymupdf-layout`
- **Flask Web UI** with modern design, dark/light theme, GPU/CPU status, and individual PDF viewing
- Unified `output/<PDF stem>/` directory structure for CLI + UI runs

---

## Requirements
- Python 3.12+
- [uv](https://docs.astral.sh/uv/latest/) (recommended) or `pip`
- GPU optional (DocLayout-YOLO runs on CPU as well)

Install dependencies:
```bash
uv pip install
```

> If you prefer a virtualenv, create/activate it first, then run `uv pip install` inside.

---

## Quick Start

### Command Line Pipeline
Process all PDFs in `./pdfs` and write outputs to `./output/<PDF stem>/`:
```bash
uv run python main.py
```

Each subdirectory contains:
- `* _content_list.json` – metadata for extracted figures/tables
- `*_layout.pdf` – annotated PDF with layout boxes
- `*.md` – markdown export (if `pymupdf4llm` is installed)
- `figures/` & `tables/` – cropped PNGs with stitched captions/titles

### Flask Web App (Recommended)
Launch the modern Flask web interface locally:
```bash
python run_flask_gpu.py
```
Then open your browser to `http://localhost:5000`

**Features:**
- Clean, modern UI with dark/light theme support
- Multiple PDF upload and processing
- Individual PDF output viewing with sidebar navigation
- Real-time GPU/CPU status display
- Image gallery for figures and tables
- Markdown preview and download
- Responsive design for mobile and desktop

All Flask app runs also write into `./output/<PDF stem>/` using the same structure as the CLI.

### Deploy to Modal.com (Cloud with GPU)
Deploy your Flask app online with GPU support using Modal:
```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal token new

# Deploy to Modal
modal deploy modal_app.py
```

See [MODAL_DEPLOYMENT.md](MODAL_DEPLOYMENT.md) for detailed instructions.

**Benefits:**
- GPU support (T4, A10G, or A100)
- Pay-per-use pricing
- Automatic HTTPS
- Auto-scaling
- Global deployment

---

## Configuration Highlights
- **Detection model:** DocLayout-YOLO (`doclayout_yolo_docstructbench_imgsz1024.pt`)
- **Detection thresholds:** configurable in `main.py`
- **Layout stitching:** tables, captions, titles, body text
- **Markdown extraction:** defaults to enabled (`pymupdf4llm.to_markdown`); falls back gracefully if the package is missing
- **Output directory:** `./output` (configurable near the bottom of `main.py`)

---

## File Overview
| Path | Description |
|------|-------------|
| `main.py` | CLI pipeline for batch PDF processing |
| `app.py` | Flask web application (recommended UI) |
| `run_flask_gpu.py` | Local Flask runner with GPU support |
| `modal_app.py` | Modal.com deployment configuration (cloud GPU) |
| `MODAL_DEPLOYMENT.md` | Modal.com deployment guide |
| `templates/` | Flask HTML templates |
| `static/` | Flask static files (CSS, JS) |
| `pdfs/` | Source PDFs (gitignored) |
| `output/` | Generated outputs per PDF |
| `pyproject.toml` | Project metadata & dependency list |
| `uv.lock` | Locked dependency versions (auto-maintained by `uv`) |

---

## Troubleshooting
- **`ModuleNotFoundError: pymupdf4llm`** – install it via `uv pip install pymupdf4llm` (already listed in `pyproject.toml`).
- **Slow performance** – ensure GPU CUDA drivers are available or reduce concurrency by toggling `USE_MULTIPROCESSING` in `main.py`.
- **Large outputs** – clean the `output/` directory before reruns to avoid confusing duplicates.

For additional logging, set `LOG_LEVEL` or edit the `logger` configuration in `main.py`.

---

## Acknowledgements
- [DocLayout-YOLO](https://github.com/juliozhao/DocLayout-YOLO)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [PyMuPDF4LLM](https://github.com/pymupdf/RAG/blob/main/pymupdf4llm.md)
- [Flask](https://flask.palletsprojects.com/)

Happy extracting! 🎉
