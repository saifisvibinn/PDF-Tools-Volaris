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
        "werkzeug>=3.0.0",
        "gunicorn>=21.2.0",
        "asgiref>=3.7.0",  # For WSGI-to-ASGI conversion
    )
    .run_commands(
        "mkdir -p /app/uploads /app/output /app/static /app/templates"
    )
)

# Create mounts for local files and directories
# Modal uses instance methods: create Mount() then add files/dirs
mount = modal.mount.Mount()
mount.add_local_dir("static", remote_path="/app/static")
mount.add_local_dir("templates", remote_path="/app/templates")
mount.add_local_file("app.py", remote_path="/app/app.py")
mount.add_local_file("main.py", remote_path="/app/main.py")

# Create the Modal app
app = modal.App("pdf-layout-extractor", image=image)

# GPU configuration - using T4 for cost efficiency, can upgrade to A10G or A100
GPU_CONFIG = modal.gpu.T4(count=1)  # Change to A10G or A100 for better performance


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    mounts=[mount],  # Mount local files and directories
    secrets=[
        # Add any secrets here if needed (e.g., HUGGINGFACE_TOKEN)
        # modal.Secret.from_name("huggingface-secret"),
    ],
    allow_concurrent_inputs=10,  # Handle up to 10 concurrent requests
    timeout=3600,  # 1 hour timeout for long PDF processing
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
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=10,
    timeout=3600,
)
@modal.web_endpoint(method="GET", label="pdf-extractor")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "pdf-layout-extractor"}


if __name__ == "__main__":
    # For local testing
    app.serve()

