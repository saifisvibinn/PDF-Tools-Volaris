import os
import json
import signal
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial

import fitz  # PyMuPDF (Still needed for drawing output PDF)
import pypdfium2 as pdfium
import torch
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from loguru import logger
from PIL import Image
import numpy as np

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model options
MODEL_SIZE = 1024
REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
WEIGHTS_FILE = f"doclayout_yolo_docstructbench_imgsz{MODEL_SIZE}.pt"

# Detection settings
CONF_THRESHOLD = 0.25

# Multiprocessing settings
NUM_WORKERS = None  # None = auto (cpu_count - 1), or set to specific number like 4
USE_MULTIPROCESSING = True  # Set to False to disable parallel processing entirely

# ----------------------------------------------------------------------
# Color map for the layout classes
# ----------------------------------------------------------------------
CLASS_COLORS = {
    "text": (0, 128, 0),          # Dark Green
    "title": (192, 0, 0),        # Dark Red
    "figure": (0, 0, 192),       # Dark Blue
    "table": (218, 165, 32),     # Goldenrod (Dark Yellow)
    "list": (128, 0, 128),       # Purple
    "header": (0, 128, 128),     # Teal
    "footer": (100, 100, 100),   # Dark Gray
    "figure_caption": (0, 0, 128), # Navy
    "table_caption": (139, 69, 19),  # Saddle Brown
    "table_footnote": (128, 0, 128), # Purple
}

# Global model instance (will be None in worker processes until loaded)
_model = None
_shutdown_requested = False

# ----------------------------------------------------------------------
# Signal handler for graceful shutdown
# ----------------------------------------------------------------------
def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _shutdown_requested
    if not _shutdown_requested:
        _shutdown_requested = True
        logger.warning("\n⚠️  Interrupt received! Finishing current page and shutting down gracefully...")
        logger.warning("Press Ctrl+C again to force quit (may leave incomplete files)")
    else:
        logger.error("\n❌ Force quit requested. Exiting immediately.")
        sys.exit(1)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ----------------------------------------------------------------------
# Model loader function
# ----------------------------------------------------------------------
def get_model():
    """Lazy load the model (only once per process)."""
    global _model
    if _model is None:
        weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_FILE)
        _model = YOLOv10(weights_path)
        logger.info(f"✓ Model loaded in worker process (PID: {os.getpid()})")
    return _model

# ----------------------------------------------------------------------
# Worker initialization function
# ----------------------------------------------------------------------
def init_worker():
    """Initialize worker process - loads model once at startup."""
    try:
        get_model()
        logger.success(f"Worker {os.getpid()} ready")
    except Exception as e:
        logger.error(f"Failed to initialize worker {os.getpid()}: {e}")
        raise

# ----------------------------------------------------------------------
# Run layout detection on a single page image (YOLO)
# ----------------------------------------------------------------------
def detect_page(pil_img: Image.Image) -> List[dict]:
    """Detect layout elements using YOLO model."""
    model = get_model()  # Will return already-loaded model in worker
    img_cv = np.array(pil_img)
    results = model.predict(
        img_cv,
        imgsz=MODEL_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False
    )
    dets = []
    for i, box in enumerate(results[0].boxes):
        cls_id = int(box.cls.item())
        name = results[0].names[cls_id]
        conf = float(box.conf.item())
        x0, y0, x1, y1 = box.xyxy[0].cpu().numpy().tolist()
        dets.append({
            "name": name,
            "bbox": [x0, y0, x1, y1],
            "conf": conf,
            "source": "yolo",
            "index": i
        })
    return dets

# ----------------------------------------------------------------------
# Crop & save figure/table regions (with captions)
# ----------------------------------------------------------------------
def get_union_box(box1: List[float], box2: List[float]) -> List[float]:
    """Get the bounding box enclosing two boxes."""
    x0 = min(box1[0], box2[0])
    y0 = min(box1[1], box2[1])
    x1 = max(box1[2], box2[2])
    y1 = max(box1[3], box2[3])
    return [x0, y0, x1, y1]

def find_closest_element_below(element: Dict, all_dets: List[Dict],
                               target_name: str) -> Optional[Dict]:
    """Find the closest matching element directly below the given one."""
    element_bottom = element["bbox"][3]
    element_center_x = (element["bbox"][0] + element["bbox"][2]) / 2
    
    candidates = []
    for d in all_dets:
        if d["name"] == target_name:
            candidate_top = d["bbox"][1]
            if candidate_top > element_bottom:
                candidate_center_x = (d["bbox"][0] + d["bbox"][2]) / 2
                horizontal_dist = abs(element_center_x - candidate_center_x)
                
                element_width = element["bbox"][2] - element["bbox"][0]
                if horizontal_dist < element_width:
                    vertical_dist = candidate_top - element_bottom
                    candidates.append((vertical_dist, d))

    if not candidates:
        return None
    
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def save_layout_elements(pil_img: Image.Image, page_num: int, 
                         dets: List[dict], out_dir: Path) -> List[dict]:
    """Save figure and table crops, merging captions."""
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    infos = []
    fig_count = 0
    tab_count = 0
    
    processed_indices = set()

    for i, d in enumerate(dets):
        if d["index"] in processed_indices:
            continue
        
        name = d["name"].lower()
        final_box = d["bbox"]
        caption_info = None
        
        if name == "figure":
            elem_type = "figure"
            path_template = fig_dir / f"page_{page_num + 1}_fig_{fig_count}.png"
            fig_count += 1
            caption = find_closest_element_below(d, dets, "figure_caption")
            if caption:
                final_box = get_union_box(d["bbox"], caption["bbox"])
                processed_indices.add(caption["index"])
                caption_info = caption
        
        elif name == "table":
            elem_type = "table"
            path_template = tab_dir / f"page_{page_num + 1}_tab_{tab_count}.png"
            tab_count += 1
            caption = find_closest_element_below(d, dets, "table_caption")
            if caption:
                final_box = get_union_box(d["bbox"], caption["bbox"])
                processed_indices.add(caption["index"])
                caption_info = caption
        else:
            continue
            
        x0, y0, x1, y1 = map(int, final_box)
        crop = pil_img.crop((x0, y0, x1, y1))
        
        if crop.mode == "CMYK":
            crop = crop.convert("RGB")
            
        crop.save(path_template)
        
        info_data = {
            "type": elem_type,
            "page": page_num + 1,
            "bbox_pixels": final_box,
            "conf": d["conf"],
            "source": d.get("source", "yolo"),
            "image_path": str(path_template.relative_to(out_dir)),
            "width": int(x1 - x0),
            "height": int(y1 - y0)
        }
        if caption_info:
            info_data["caption_bbox"] = caption_info["bbox"]
        
        infos.append(info_data)
    
    return infos

# ----------------------------------------------------------------------
# Draw layout boxes on the original PDF
# ----------------------------------------------------------------------
def draw_layout_pdf(pdf_bytes: bytes, all_dets: List[List[dict]],
                    scale: float, out_path: Path):
    """Annotate PDF with semi-transparent bounding boxes and labels."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_no, dets in enumerate(all_dets):
        page = doc[page_no]

        for d in dets:
            rgb = CLASS_COLORS.get(d["name"], (0, 0, 0))
            rect = fitz.Rect([c / scale for c in d["bbox"]])

            border_color = [c / 255 for c in rgb]
            fill_color = [c / 255 for c in rgb]
            fill_opacity = 0.15
            border_width = 1.5

            page.draw_rect(
                rect,
                color=border_color,
                fill=fill_color,
                width=border_width,
                overlay=True,
                fill_opacity=fill_opacity
            )

            label = f"{d['name']} {d['conf']:.2f}"
            if d.get("source"):
                label += f" [{d['source'][0].upper()}]"

            text_bg = fitz.Rect(rect.x0, rect.y0 - 10, rect.x0 + 60, rect.y0)
            page.draw_rect(text_bg, color=None, fill=(1, 1, 1, 0.6), overlay=True)

            page.insert_text(
                (rect.x0 + 2, rect.y0 - 8),
                label,
                fontsize=6.5,
                color=border_color,
                overlay=True
            )

    doc.save(str(out_path))
    doc.close()

# ----------------------------------------------------------------------
# Process a single PDF Page (for parallel execution)
# ----------------------------------------------------------------------
def process_page(task_data: Tuple[int, bytes, float, Path, str]) -> Optional[Tuple[int, List[dict], List[dict]]]:
    """
    Process a single page of a PDF in a worker process.
    Returns: (page_number, detections, elements) or None on failure
    """
    pno, pdf_bytes, scale, out_dir, pdf_name = task_data
    
    if _shutdown_requested:
        return None
    
    pdf_pdfium = None
    try:
        pdf_pdfium = pdfium.PdfDocument(pdf_bytes)
        
        page = pdf_pdfium[pno]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()

        dets = detect_page(pil)
        elements = save_layout_elements(pil, pno, dets, out_dir)
        
        page_figures = len([d for d in dets if d['name'] == 'figure'])
        page_tables = len([d for d in dets if d['name'] == 'table'])
        logger.info(f"  [{pdf_name}] Page {pno + 1}: {page_figures} figs, {page_tables} tables")

        page.close()
        pdf_pdfium.close()
        
        return (pno, dets, elements)

    except Exception as e:
        logger.error(f"Failed to process page {pno + 1} of {pdf_name}: {e}")
        if pdf_pdfium:
            pdf_pdfium.close()
        return None

# ----------------------------------------------------------------------
# Process a full PDF using the persistent worker pool
# ----------------------------------------------------------------------
def process_pdf_with_pool(pdf_path: Path, out_dir: Path, pool: Optional[Pool] = None):
    """
    Main processing pipeline for a PDF file.
    If pool is provided, uses it. Otherwise processes serially.
    """
    
    if _shutdown_requested:
        logger.warning(f"Skipping {pdf_path.name} due to shutdown request")
        return
    
    stem = pdf_path.stem
    logger.info(f"Processing {pdf_path.name}")

    pdf_bytes = pdf_path.read_bytes()
    
    try:
        with pdfium.PdfDocument(pdf_bytes) as doc:
            page_count = len(doc)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path.name}: {e}. Skipping.")
        return

    scale = 2.0
    all_dets = [None] * page_count  # Pre-allocate to maintain page order
    all_elements = []
    
    # Use pool if provided and multiprocessing is enabled
    if pool is not None and USE_MULTIPROCESSING:
        logger.info(f"  Using worker pool for {page_count} pages...")
        
        # Create tasks for all pages
        tasks = [
            (pno, pdf_bytes, scale, out_dir, pdf_path.name)
            for pno in range(page_count)
        ]
        
        try:
            # Process all pages using the persistent pool
            results = pool.map(process_page, tasks)
            
            # Collect results in correct order
            for res in results:
                if res:
                    pno, dets, elements = res
                    all_dets[pno] = dets
                    all_elements.extend(elements)
                    
        except KeyboardInterrupt:
            logger.warning("Processing interrupted during parallel execution")
            raise
    
    # Serial execution fallback
    else:
        logger.info("Using serial processing...")
        
        try:
            pdf_pdfium = pdfium.PdfDocument(pdf_bytes)
            
            for pno in range(page_count):
                if _shutdown_requested:
                    logger.warning(f"Stopping at page {pno + 1}/{page_count} due to shutdown request")
                    break
                
                try:
                    logger.info(f"  Processing page {pno + 1}/{page_count}")
                    
                    page = pdf_pdfium[pno]
                    bitmap = page.render(scale=scale)
                    pil = bitmap.to_pil()

                    dets = detect_page(pil)
                    all_dets[pno] = dets
                    
                    elements = save_layout_elements(pil, pno, dets, out_dir)
                    all_elements.extend(elements)
                    
                    page_figures = len([d for d in dets if d['name'] == 'figure'])
                    page_tables = len([d for d in dets if d['name'] == 'table'])
                    logger.info(f"    Found {page_figures} figures and {page_tables} tables")
                    
                    page.close()
                
                except Exception as e:
                    logger.error(f"Failed to process page {pno + 1}: {e}. Skipping page.")
            
            pdf_pdfium.close()
            
        except Exception as e:
            logger.error(f"Fatal error processing {pdf_path.name}: {e}")
            if 'pdf_pdfium' in locals() and pdf_pdfium:
                pdf_pdfium.close()
            return

    # Filter out None entries (failed pages)
    all_dets = [d for d in all_dets if d is not None]

    # Save final outputs (even if interrupted, save what we have)
    if all_elements:
        content_list_path = out_dir / f"{stem}_content_list.json"
        with open(content_list_path, 'w', encoding='utf-8') as f:
            json.dump(all_elements, f, ensure_ascii=False, indent=4)
        logger.info(f"  Saved {len(all_elements)} elements to JSON")

    if all_dets:
        draw_layout_pdf(pdf_bytes, all_dets, scale,
                        out_dir / f"{stem}_layout.pdf")
        logger.info("  Generated annotated PDF")
    else:
        logger.warning(f"No detections found for {stem}. Skipping layout PDF.")

    if _shutdown_requested:
        logger.warning(f"⚠️  Partial results saved for {stem} → {out_dir}")
    else:
        logger.success(f"✓ {stem} → {out_dir} ({len(all_elements)} elements extracted)")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Important for multiprocessing on Windows/macOS
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    INPUT_DIR = Path("./pdfs")
    OUTPUT_DIR = Path("./output4")

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in ./pdfs")
        logger.info("Please add PDF files to the ./pdfs directory")
        raise SystemExit(1)

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    logger.info(f"Settings: MODEL_SIZE={MODEL_SIZE}, CONF={CONF_THRESHOLD}")
    
    # Determine worker count
    total_cpus = cpu_count()
    if NUM_WORKERS is None:
        num_workers = max(1, total_cpus - 1)
    else:
        num_workers = max(1, min(NUM_WORKERS, total_cpus))
    
    # Decide whether to use multiprocessing
    use_pool = USE_MULTIPROCESSING and DEVICE == "cpu" and total_cpus >= 4
    
    if use_pool:
        logger.info(f"🚀 Creating persistent worker pool with {num_workers} workers...")
    else:
        if not USE_MULTIPROCESSING:
            logger.info("Multiprocessing disabled by configuration")
        elif DEVICE != "cpu":
            logger.info(f"Using serial GPU processing (device: {DEVICE})")
        else:
            logger.info(f"Using serial CPU processing (CPU count {total_cpus} too low)")

    pool = None
    try:
        # Create persistent pool ONCE for all PDFs
        if use_pool:
            pool = Pool(processes=num_workers, initializer=init_worker)
            logger.success(f"✓ Worker pool ready with {num_workers} workers\n")
        else:
            # Load model in main process for serial execution
            logger.info("Initializing model in main process...")
            get_model()
            logger.success(f"✓ Model loaded (device: {DEVICE})\n")

        # Process all PDFs using the same pool
        for i, pdf_path in enumerate(pdf_files, 1):
            if _shutdown_requested:
                logger.warning(f"\nShutdown requested. Processed {i-1}/{len(pdf_files)} files.")
                break
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📄 File {i}/{len(pdf_files)}: {pdf_path.name}")
            logger.info(f"{'='*60}")
            
            sub_out = OUTPUT_DIR / pdf_path.stem
            os.makedirs(sub_out, exist_ok=True)
            
            try:
                process_pdf_with_pool(pdf_path, sub_out, pool)
            except KeyboardInterrupt:
                logger.warning(f"\nInterrupted while processing {pdf_path.name}")
                break
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                if _shutdown_requested:
                    break
                logger.info("Continuing with next file...")
                continue

        if _shutdown_requested:
            logger.warning(f"\n⚠️  Processing interrupted. Partial results saved in {OUTPUT_DIR}")
        else:
            logger.success(f"\n✨ All done! Results are in {OUTPUT_DIR}")
            
    except KeyboardInterrupt:
        logger.error("\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        # Clean up pool if it exists
        if pool is not None:
            logger.info("\n🧹 Shutting down worker pool...")
            pool.close()
            pool.join()
            logger.success("✓ Worker pool closed cleanly")