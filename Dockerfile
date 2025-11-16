# Hugging Face Spaces Dockerfile for PDF Toolkit Volaris (CPU-only)
# Runtime: Docker (select in Space settings)

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf \
    PIP_NO_CACHE_DIR=1

# System dependencies for Tesseract, OpenCV, PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency spec first (better Docker layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application source
COPY . .

# Ensure directories exist that app expects
RUN mkdir -p uploads output pdfs static templates

# Expose default Streamlit / Gradio / FastAPI port
EXPOSE 7860

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -f http://localhost:7860/ || exit 1

# Default command: run Streamlit UI (modify if switching to FastAPI)
CMD ["streamlit", "run", "pdf_extractor_gui.py", "--server.address=0.0.0.0", "--server.port=7860"]
