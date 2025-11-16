"""
Modal deployment configuration for PDF Layout Extractor Flask app.
Deploy with: modal deploy modal_app.py
"""
import modal

# Create a Modal image with GPU support and all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        "gcc",
        "g++",
        "libgl1",
        "libglib2.0-0",  # Required for cv2 (provides libgthread-2.0.so.0)
        "libsm6",        # Required for cv2
        "libxext6",      # Required for cv2
        "libxrender-dev",  # Required for cv2
        "libgomp1",      # Required for cv2
        "libffi-dev",
        "libjpeg62-turbo-dev",
        "zlib1g-dev",
        "netcat-openbsd",
    )
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "doclayout-yolo>=0.0.4",
        "huggingface-hub>=1.1.2",
        "loguru>=0.7.3",
        "pillow>=12.0.0",
        "pymupdf>=1.26.6",
        "pymupdf-layout>=0.0.15",
        "pypdfium2>=5.0.0",
        "pymupdf4llm>=0.1.9",
        "flask>=3.0.0",
        "fastapi>=0.109.0",  # Required for Modal web endpoints
        "werkzeug>=3.0.0",
        "gunicorn>=21.2.0",
        "asgiref>=3.7.0",  # For WSGI-to-ASGI conversion
    )
    .run_commands(
        "mkdir -p /app/uploads /app/output /app/static /app/templates"
    )
    # Copy application files directly into the image
    .add_local_dir("static", remote_path="/app/static")
    .add_local_dir("templates", remote_path="/app/templates")
    .add_local_file("app.py", remote_path="/app/app.py")
    .add_local_file("main.py", remote_path="/app/main.py")
)

# Create the Modal app
app = modal.App("pdf-layout-extractor", image=image)

# GPU configuration - using T4 for cheapest option (~$0.50/hour while active)
# For no GPU (CPU only), set gpu=None (much cheaper but slower)
# Valid options: "T4", "A10G", "A100", or None


@app.function(
    image=image,
    gpu="T4",  # Cheapest GPU option (~$0.50/hour while active)
    secrets=[
        # Add any secrets here if needed (e.g., HUGGINGFACE_TOKEN)
        # modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=3600,  # 1 hour timeout for long PDF processing
    max_containers=10,  # Handle up to 10 concurrent requests
)
@modal.asgi_app()
def flask_app():
    """
    Expose the Flask app as an ASGI application for Modal.
    Flask is WSGI, so we convert it to ASGI using a wrapper.
    """
    import sys
    import os
    from pathlib import Path
    
    # Set working directory
    os.chdir("/app")
    sys.path.insert(0, "/app")
    
    # Import Flask app
    from app import app as flask_app_instance
    
    # Convert Flask WSGI app to ASGI for Modal
    # Using asgiref's WSGI-to-ASGI adapter
    from asgiref.wsgi import WsgiToAsgi
    
    asgi_app = WsgiToAsgi(flask_app_instance)
    return asgi_app


# Alternative: Deploy as a web endpoint with automatic HTTPS
@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    max_containers=10,
)
@modal.fastapi_endpoint(method="GET", label="pdf-extractor")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "pdf-layout-extractor"}


if __name__ == "__main__":
    # For local testing with Modal dev server:
    # Run: modal serve modal_app.py
    pass

