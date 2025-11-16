import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import torch

import main as extractor
from loguru import logger

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './output'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global model instance
_model = None


def get_device_info() -> Dict[str, any]:
    """Get information about GPU/CPU availability."""
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    
    info = {
        "device": device,
        "cuda_available": cuda_available,
        "device_name": None,
        "device_count": 0,
    }
    
    if cuda_available:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    
    return info


def load_model_once():
    """Load the model once and cache it."""
    global _model
    if _model is None:
        logger.info("Loading DocLayout-YOLO model...")
        _model = extractor.get_model()
        logger.info("Model loaded successfully")
    return _model


@app.route('/')
def index():
    """Main page."""
    device_info = get_device_info()
    return render_template('index.html', device_info=device_info)


@app.route('/api/device-info')
def device_info():
    """API endpoint to get device information."""
    return jsonify(get_device_info())


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle multiple PDF file uploads."""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    extraction_mode = request.form.get('extraction_mode', 'images')
    include_images = extraction_mode != 'markdown'
    include_markdown = extraction_mode != 'images'
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    
    for file in files:
        if file and file.filename.endswith('.pdf'):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                stem = Path(filename).stem
                upload_path = Path(app.config['UPLOAD_FOLDER']) / filename
                file.save(str(upload_path))
                
                # Prepare output directory
                output_dir = Path(app.config['OUTPUT_FOLDER']) / stem
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy PDF to output directory
                pdf_path = output_dir / filename
                upload_path.rename(pdf_path)
                
                # Process PDF
                extractor.USE_MULTIPROCESSING = False
                logger.info(f"Processing {filename} (images={include_images}, markdown={include_markdown})")
                
                if include_images:
                    load_model_once()
                
                extractor.process_pdf_with_pool(
                    pdf_path,
                    output_dir,
                    pool=None,
                    extract_images=include_images,
                    extract_markdown=include_markdown,
                )
                
                # Collect results
                json_path = output_dir / f"{stem}_content_list.json"
                elements = []
                if include_images and json_path.exists():
                    elements = json.loads(json_path.read_text(encoding='utf-8'))
                
                annotated_pdf = None
                if include_images:
                    candidate_pdf = output_dir / f"{stem}_layout.pdf"
                    if candidate_pdf.exists():
                        annotated_pdf = str(candidate_pdf.relative_to(app.config['OUTPUT_FOLDER']))
                
                markdown_path = None
                if include_markdown:
                    candidate_md = output_dir / f"{stem}.md"
                    if candidate_md.exists():
                        markdown_path = str(candidate_md.relative_to(app.config['OUTPUT_FOLDER']))
                
                # Get figure and table counts
                figures = [e for e in elements if e.get('type') == 'figure']
                tables = [e for e in elements if e.get('type') == 'table']
                
                results.append({
                    'filename': filename,
                    'stem': stem,
                    'output_dir': str(output_dir.relative_to(app.config['OUTPUT_FOLDER'])),
                    'figures_count': len(figures),
                    'tables_count': len(tables),
                    'elements_count': len(elements),
                    'annotated_pdf': annotated_pdf,
                    'markdown_path': markdown_path,
                    'include_images': include_images,
                    'include_markdown': include_markdown,
                })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
    
    return jsonify({'results': results})


@app.route('/api/pdf-list')
def pdf_list():
    """Get list of processed PDFs."""
    output_dir = Path(app.config['OUTPUT_FOLDER'])
    pdfs = []
    
    for item in output_dir.iterdir():
        if item.is_dir():
            # Check if this directory has processed content
            json_files = list(item.glob('*_content_list.json'))
            md_files = list(item.glob('*.md'))
            pdf_files = list(item.glob('*.pdf'))
            
            if json_files or md_files or pdf_files:
                stem = item.name
                pdfs.append({
                    'stem': stem,
                    'output_dir': str(item.relative_to(app.config['OUTPUT_FOLDER'])),
                })
    
    return jsonify({'pdfs': pdfs})


@app.route('/api/pdf-details/<path:pdf_stem>')
def pdf_details(pdf_stem):
    """Get detailed information about a processed PDF."""
    output_dir = Path(app.config['OUTPUT_FOLDER']) / pdf_stem
    
    if not output_dir.exists():
        return jsonify({'error': 'PDF not found'}), 404
    
    # Load content list
    json_files = list(output_dir.glob('*_content_list.json'))
    elements = []
    if json_files:
        elements = json.loads(json_files[0].read_text(encoding='utf-8'))
    
    # Get figures and tables
    figures = [e for e in elements if e.get('type') == 'figure']
    tables = [e for e in elements if e.get('type') == 'table']
    
    # Get file paths
    annotated_pdf = None
    pdf_files = list(output_dir.glob('*_layout.pdf'))
    if pdf_files:
        annotated_pdf = str(pdf_files[0].relative_to(app.config['OUTPUT_FOLDER']))
    
    markdown_path = None
    md_files = list(output_dir.glob('*.md'))
    if md_files:
        markdown_path = str(md_files[0].relative_to(app.config['OUTPUT_FOLDER']))
    
    # Get figure and table images
    figure_dir = output_dir / 'figures'
    table_dir = output_dir / 'tables'
    
    figure_images = []
    if figure_dir.exists():
        figure_images = [str(f.relative_to(app.config['OUTPUT_FOLDER'])) 
                        for f in sorted(figure_dir.glob('*.png'))]
    
    table_images = []
    if table_dir.exists():
        table_images = [str(t.relative_to(app.config['OUTPUT_FOLDER'])) 
                       for t in sorted(table_dir.glob('*.png'))]
    
    return jsonify({
        'stem': pdf_stem,
        'figures': figures,
        'tables': tables,
        'figures_count': len(figures),
        'tables_count': len(tables),
        'elements_count': len(elements),
        'annotated_pdf': annotated_pdf,
        'markdown_path': markdown_path,
        'figure_images': figure_images,
        'table_images': table_images,
    })


@app.route('/output/<path:filename>')
def output_file(filename):
    """Serve output files (PDFs, images, markdown)."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


def _delete_by_stem(stem_raw: str):
    stem = (stem_raw or "").strip()
    if not stem:
        return jsonify({'error': 'Missing stem'}), 400

    # Resolve output directory safely
    output_root = Path(app.config['OUTPUT_FOLDER']).resolve()
    target_dir = (output_root / stem).resolve()

    # Prevent path traversal - ensure target is within output_root
    if output_root not in target_dir.parents and target_dir != output_root:
        return jsonify({'error': 'Invalid stem path'}), 400

    if not target_dir.exists() or not target_dir.is_dir():
        return jsonify({'error': 'Not found'}), 404

    # Delete the directory
    shutil.rmtree(target_dir, ignore_errors=False)
    logger.info(f"Deleted processed output: {target_dir}")

    return jsonify({'ok': True, 'deleted': stem})


@app.route('/api/delete', methods=['POST'])
def delete_pdf():
    """Delete a processed PDF directory by stem (JSON or form body)."""
    try:
        data = request.get_json(silent=True) or {}
        stem = (data.get('stem') or request.form.get('stem') or '').strip()
        return _delete_by_stem(stem)
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete/<path:stem>', methods=['POST', 'GET'])
def delete_pdf_by_path(stem: str):
    """Alternate endpoint to delete using URL path, for clients avoiding bodies."""
    try:
        return _delete_by_stem(stem)
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


