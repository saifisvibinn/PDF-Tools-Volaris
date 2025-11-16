# Modal.com Deployment Guide

This guide explains how to deploy the PDF Layout Extractor Flask app to Modal.com with GPU support.

## Prerequisites

1. **Sign up for Modal**: Go to [modal.com](https://modal.com) and create an account
2. **Install Modal CLI**:
   ```bash
   pip install modal
   ```
3. **Authenticate with Modal**:
   ```bash
   modal token new
   ```
   This will open your browser to authenticate. Follow the instructions.

## Quick Start

1. **Deploy the app**:
   ```bash
   modal deploy modal_app.py
   ```

2. **Modal will provide you with a URL** like:
   ```
   https://your-username--pdf-layout-extractor-flask-app.modal.run
   ```

3. **Access your API**:
   - Open the URL in your browser to see the Flask UI
   - API endpoints are available at:
     - `GET /` - Main UI
     - `GET /api/device-info` - GPU/CPU status
     - `POST /api/upload` - Upload PDFs
     - `GET /api/pdf-list` - List processed PDFs
     - `GET /api/pdf-details/<stem>` - Get PDF details
     - `POST /api/delete/<stem>` - Delete a processed PDF

## Configuration

### GPU Options

The default GPU is **T4** (cheapest GPU option at ~$0.50/hour). You can change it in `modal_app.py`:

```python
# For T4 (CHEAPEST, default: ~$0.50/hour while active)
GPU_CONFIG = modal.gpu.T4(count=1)

# For A10G (faster, more expensive: ~$1.50/hour while active)
GPU_CONFIG = modal.gpu.A10G(count=1)

# For A100 (fastest, most expensive: ~$3.50/hour while active)
GPU_CONFIG = modal.gpu.A100(count=1)

# For CPU only (cheapest, no GPU: ~$0.10/hour but much slower)
GPU_CONFIG = None
```

### Timeout Settings

The current timeout is 3600 seconds (1 hour) for PDF processing. Adjust if needed:

```python
timeout=3600,  # Change to your desired timeout
```

### Concurrent Requests

The app is configured to handle 10 concurrent requests. Adjust if needed:

```python
allow_concurrent_inputs=10,  # Change as needed
```

## Cost

**GPU Options (cheapest to most expensive):**
- **CPU only**: ~$0.10/hour while active (slowest, no GPU)
- **T4 GPU**: ~$0.50/hour while active (cheapest GPU, current setting) ✅
- **A10G GPU**: ~$1.50/hour while active (faster)
- **A100 GPU**: ~$3.50/hour while active (fastest)

**Additional costs:**
- **Idle time**: $0 (pays only when processing requests)
- **Storage**: Free for reasonable usage
- **Bandwidth**: Free for reasonable usage

## Local Testing

Test locally before deploying:

```bash
modal run modal_app.py
```

This will run the app locally using Modal's infrastructure.

## Monitoring

View logs and monitor usage:

```bash
# View logs
modal app logs pdf-layout-extractor

# View app status
modal app list
```

## Troubleshooting

### Import Errors

If you get import errors, make sure all dependencies are in the image definition in `modal_app.py`.

### GPU Not Detected

If GPU is not detected:
1. Verify GPU configuration in `modal_app.py`
2. Check Modal dashboard for GPU availability
3. Try a different GPU type (T4, A10G, A100)

### Timeout Errors

If processing times out:
1. Increase the `timeout` value in `modal_app.py`
2. Consider using a faster GPU (A10G or A100)
3. Split large PDFs into smaller batches

### Storage Issues

Modal provides ephemeral storage. Files are lost when the container stops. For persistence:
1. Use Modal Volumes (see Modal docs)
2. Store results in external storage (S3, etc.)
3. Download results via the API

## API Usage Examples

### Upload PDF

```bash
curl -X POST https://your-url.modal.run/api/upload \
  -F "files[]=@document.pdf" \
  -F "extraction_mode=both"
```

### Get PDF List

```bash
curl https://your-url.modal.run/api/pdf-list
```

### Get PDF Details

```bash
curl https://your-url.modal.run/api/pdf-details/document_name
```

### Delete PDF

```bash
curl -X POST https://your-url.modal.run/api/delete/document_name
```

## Updates

To update your deployed app:

```bash
modal deploy modal_app.py
```

Modal will automatically update the deployment.

## Need Help?

- [Modal Documentation](https://modal.com/docs)
- [Modal Discord Community](https://modal.com/discord)
- Check logs: `modal app logs pdf-layout-extractor`

