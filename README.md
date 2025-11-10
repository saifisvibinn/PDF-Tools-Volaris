# PDF Layout Extraction Companion

A streamlined workflow for extracting figures, tables, annotated layouts, and markdown text from scientific PDFs using [DocLayout-YOLO](https://github.com/juliozhao/DocLayout-YOLO), PyMuPDF, and Streamlit. The project exposes both a command-line pipeline (`main.py`) and a full-featured web UI (`streamlit_app.py`).

---

## Features
- **Layout-aware extraction** of figures and tables with YOLO-based detection
- **Cross-page stitching** for multi-page tables, captions, titles, and body text
- **Annotated PDF output** with bounding boxes for detected regions
- **Markdown export** powered by `pymupdf4llm` / `pymupdf-layout`
- **Streamlit UI** with extraction modes: _Images only_, _Markdown only_, or _Images & Markdown_
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

### Streamlit UI
Launch the interactive app:
```bash
uv run streamlit run streamlit_app.py
```
- Choose extraction mode (images / markdown / both)
- Upload one or multiple PDFs
- Results appear immediately with download buttons and previews

All UI runs also write into `./output/<PDF stem>/` using the same structure as the CLI.

---

## Configuration Highlights
- **Detection model:** DocLayout-YOLO (`doclayout_yolo_docstructbench_imgsz1024.pt`)
- **Detection thresholds:** configurable in `main.py`
- **Layout stitching:** tables, captions, titles, body text
- **Markdown extraction:** defaults to enabled (`pymupdf4llm.to_markdown`); falls back gracefully if the package is missing
- **Output directory:** `./output` (configurable near the bottom of `main.py` and top of `streamlit_app.py`)

---

## File Overview
| Path | Description |
|------|-------------|
| `main.py` | CLI pipeline for batch PDF processing |
| `streamlit_app.py` | Streamlit UI for uploads & previews |
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
- [Streamlit](https://streamlit.io/)

Happy extracting! 🎉
