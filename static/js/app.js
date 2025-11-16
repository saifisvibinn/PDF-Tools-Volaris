// Application State
const AppState = {
    currentPdf: null,
    pdfs: [],
    deviceInfo: null,
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeTheme();
    loadDeviceInfo();
    initializeEventListeners();
    loadPdfList();
});

// Theme Toggle
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
    
    document.getElementById('themeToggle').addEventListener('click', function() {
        const currentTheme = document.body.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });
}

function updateThemeIcon(theme) {
    const icon = document.getElementById('themeIcon');
    if (icon) {
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
}

// Load Device Info
async function loadDeviceInfo() {
    try {
        const response = await fetch('/api/device-info');
        const data = await response.json();
        AppState.deviceInfo = data;
        updateDeviceStatus(data);
    } catch (error) {
        console.error('Error loading device info:', error);
        updateDeviceStatus({ device: 'unknown', cuda_available: false });
    }
}

function updateDeviceStatus(info) {
    const badge = document.getElementById('deviceBadge');
    const deviceName = document.getElementById('deviceName');
    
    if (info.cuda_available) {
        badge.textContent = 'GPU';
        badge.className = 'badge bg-success';
        deviceName.textContent = info.device_name || 'CUDA Device';
    } else {
        badge.textContent = 'CPU';
        badge.className = 'badge bg-secondary';
        deviceName.textContent = 'CPU Processing';
    }
}

// Event Listeners
function initializeEventListeners() {
    const uploadForm = document.getElementById('uploadForm');
    uploadForm.addEventListener('submit', handleUpload);
}

// Handle File Upload
async function handleUpload(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;
    
    if (files.length === 0) {
        alert('Please select at least one PDF file');
        return;
    }
    
    const extractionMode = document.querySelector('input[name="extractionMode"]:checked').value;
    
    // Show processing section
    document.getElementById('processingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('emptyState').style.display = 'none';
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files[]', files[i]);
    }
    formData.append('extraction_mode', extractionMode);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Hide processing section
        document.getElementById('processingSection').style.display = 'none';
        
        // Reload PDF list and show results
        await loadPdfList();
        
        // Show first PDF details if available
        if (data.results && data.results.length > 0) {
            const firstPdf = data.results[0];
            if (!firstPdf.error) {
                showPdfDetails(firstPdf.stem);
            }
        }
        
        // Reset form
        fileInput.value = '';
        
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error processing files: ' + error.message);
        document.getElementById('processingSection').style.display = 'none';
    }
}

// Load PDF List
async function loadPdfList() {
    try {
        const response = await fetch('/api/pdf-list');
        const data = await response.json();
        AppState.pdfs = data.pdfs || [];
        renderPdfList();
        
        if (AppState.pdfs.length > 0) {
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('emptyState').style.display = 'none';
        } else {
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('emptyState').style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading PDF list:', error);
    }
}

// Render PDF List
function renderPdfList() {
    const pdfList = document.getElementById('pdfList');
    pdfList.innerHTML = '';
    
    if (AppState.pdfs.length === 0) {
        pdfList.innerHTML = '<div class="text-center text-muted p-3">No PDFs processed yet</div>';
        return;
    }
    
    AppState.pdfs.forEach((pdf, index) => {
        const item = document.createElement('div');
        item.className = `list-group-item d-flex align-items-center justify-content-between ${index === 0 && !AppState.currentPdf ? 'active' : ''} ${AppState.currentPdf === pdf.stem ? 'active' : ''}`;
        
        const left = document.createElement('a');
        left.href = '#';
        left.className = 'flex-grow-1 text-decoration-none text-reset';
        left.innerHTML = `
            <div class="d-flex w-100 justify-content-between">
                <h6 class="mb-0">
                    <i class="fas fa-file-pdf me-2"></i>
                    ${pdf.stem}
                </h6>
            </div>
        `;
        left.addEventListener('click', function(e) {
            e.preventDefault();
            // Update active state
            document.querySelectorAll('#pdfList .list-group-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            showPdfDetails(pdf.stem);
        });
        
        const delBtn = document.createElement('button');
        delBtn.className = 'btn btn-sm btn-outline-danger ms-3';
        delBtn.innerHTML = '<i class="fas fa-trash-alt"></i>';
        delBtn.title = `Delete "${pdf.stem}"`;
        delBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            const confirmed = confirm(`Delete processed outputs for "${pdf.stem}"? This cannot be undone.`);
            if (!confirmed) return;
            try {
                // Use form-encoded POST to the body endpoint for widest compatibility
                const resp = await fetch('/api/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
                    body: new URLSearchParams({ stem: pdf.stem }).toString()
                });
                const raw = await resp.text();
                let res;
                try { res = JSON.parse(raw); } catch (_) { res = null; }
                if (!resp.ok || (res && res?.error)) {
                    throw new Error((res && res?.error) || raw || 'Delete failed');
                }
                // Refresh list and clear details if we deleted the active item
                if (AppState.currentPdf === pdf.stem) {
                    AppState.currentPdf = null;
                    const details = document.getElementById('pdfDetails');
                    if (details) {
                        details.innerHTML = `
                            <div class="alert alert-success">
                                <i class="fas fa-check-circle me-2"></i>
                                Deleted "${pdf.stem}" successfully.
                            </div>
                        `;
                    }
                }
                await loadPdfList();
            } catch (err) {
                console.error('Delete error:', err);
                alert('Failed to delete: ' + (err?.message || err));
            }
        });
        
        item.appendChild(left);
        item.appendChild(delBtn);
        pdfList.appendChild(item);
    });
}

// Show PDF Details
async function showPdfDetails(pdfStem) {
    AppState.currentPdf = pdfStem;
    
    // Update active state in list
    document.querySelectorAll('#pdfList .list-group-item').forEach((item, index) => {
        item.classList.remove('active');
        const pdfStemFromItem = AppState.pdfs[index]?.stem;
        if (pdfStemFromItem === pdfStem) {
            item.classList.add('active');
        }
    });
    
    try {
        const response = await fetch(`/api/pdf-details/${encodeURIComponent(pdfStem)}`);
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        renderPdfDetails(data);
    } catch (error) {
        console.error('Error loading PDF details:', error);
        document.getElementById('pdfDetails').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                Error loading PDF details: ${error.message}
            </div>
        `;
    }
}

// Render PDF Details
function renderPdfDetails(data) {
    const container = document.getElementById('pdfDetails');
    
    let html = `
        <div class="card shadow-sm mb-4">
            <div class="card-header bg-primary-custom text-white">
                <h5 class="mb-0">
                    <i class="fas fa-file-pdf me-2"></i>
                    ${data.stem}
                </h5>
                <button class="btn btn-sm btn-danger float-end" id="deletePdfBtn" title="Delete this processed PDF">
                    <i class="fas fa-trash-alt me-1"></i> Delete
                </button>
            </div>
            <div class="card-body">
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="stat-card">
                            <i class="fas fa-images fa-2x text-primary mb-2"></i>
                            <div class="stat-value">${data.figures_count || 0}</div>
                            <div class="stat-label">Figures</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <i class="fas fa-table fa-2x text-primary mb-2"></i>
                            <div class="stat-value">${data.tables_count || 0}</div>
                            <div class="stat-label">Tables</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <i class="fas fa-list fa-2x text-primary mb-2"></i>
                            <div class="stat-value">${data.elements_count || 0}</div>
                            <div class="stat-label">Total Elements</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <i class="fas fa-microchip fa-2x text-primary mb-2"></i>
                            <div class="stat-value">${AppState.deviceInfo?.device === 'cuda' ? 'GPU' : 'CPU'}</div>
                            <div class="stat-label">Device</div>
                        </div>
                    </div>
                </div>
                
                <div class="download-btn-group">
    `;
    
    if (data.annotated_pdf) {
        html += `
            <a href="/output/${data.annotated_pdf}" class="btn btn-primary" download>
                <i class="fas fa-download me-2"></i>
                Download Annotated PDF
            </a>
        `;
    }
    
    if (data.markdown_path) {
        html += `
            <a href="/output/${data.markdown_path}" class="btn btn-outline-primary" download>
                <i class="fas fa-download me-2"></i>
                Download Markdown
            </a>
        `;
    }
    
    html += `
                </div>
            </div>
        </div>
    `;
    
    // Figures Section
    if (data.figure_images && data.figure_images.length > 0) {
        html += `
            <div class="card shadow-sm mb-4">
                <div class="card-header">
                    <h5 class="mb-0">
                        <i class="fas fa-images me-2"></i>
                        Figures (${data.figure_images.length})
                    </h5>
                </div>
                <div class="card-body">
                    <div class="image-gallery">
        `;
        
        data.figure_images.forEach((imgPath, index) => {
            const figure = data.figures[index] || {};
            html += `
                <div class="image-item">
                    <img src="/output/${imgPath}" alt="Figure ${index + 1}" loading="lazy">
                    <div class="image-caption">
                        <strong>Figure ${index + 1}</strong>
                        ${figure.page ? `<br><small class="text-muted">Page ${figure.page}</small>` : ''}
                    </div>
                </div>
            `;
        });
        
        html += `
                    </div>
                </div>
            </div>
        `;
    }
    
    // Tables Section
    if (data.table_images && data.table_images.length > 0) {
        html += `
            <div class="card shadow-sm mb-4">
                <div class="card-header">
                    <h5 class="mb-0">
                        <i class="fas fa-table me-2"></i>
                        Tables (${data.table_images.length})
                    </h5>
                </div>
                <div class="card-body">
                    <div class="image-gallery">
        `;
        
        data.table_images.forEach((imgPath, index) => {
            const table = data.tables[index] || {};
            html += `
                <div class="image-item">
                    <img src="/output/${imgPath}" alt="Table ${index + 1}" loading="lazy">
                    <div class="image-caption">
                        <strong>Table ${index + 1}</strong>
                        ${table.page ? `<br><small class="text-muted">Page ${table.page}</small>` : ''}
                    </div>
                </div>
            `;
        });
        
        html += `
                    </div>
                </div>
            </div>
        `;
    }
    
    // Markdown Preview
    if (data.markdown_path) {
        html += `
            <div class="card shadow-sm">
                <div class="card-header">
                    <h5 class="mb-0">
                        <i class="fas fa-file-code me-2"></i>
                        Markdown Preview
                    </h5>
                </div>
                <div class="card-body">
                    <div class="markdown-preview" id="markdownPreview">
                        Loading markdown...
                    </div>
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
    
    // Load markdown preview if available
    if (data.markdown_path) {
        loadMarkdownPreview(data.markdown_path);
    }

    // Wire delete button
    const deleteBtn = document.getElementById('deletePdfBtn');
    if (deleteBtn) {
        deleteBtn.addEventListener('click', async () => {
            const confirmed = confirm(`Delete processed outputs for "${data.stem}"? This cannot be undone.`);
            if (!confirmed) return;
            try {
                // Use form-encoded POST to the body endpoint for widest compatibility
                const resp = await fetch('/api/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
                    body: new URLSearchParams({ stem: data.stem }).toString()
                });
                const raw = await resp.text();
                let res;
                try { res = JSON.parse(raw); } catch (_) { res = null; }
                if (!resp.ok || (res && res.error)) {
                    throw new Error((res && res.error) || raw || 'Delete failed');
                }
                // Refresh list and clear details
                await loadPdfList();
                document.getElementById('pdfDetails').innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i>
                        Deleted "${data.stem}" successfully.
                    </div>
                `;
                AppState.currentPdf = null;
            } catch (err) {
                console.error('Delete error:', err);
                alert('Failed to delete: ' + (err?.message || err));
            }
        });
    }
}

// Load Markdown Preview
async function loadMarkdownPreview(markdownPath) {
    try {
        const response = await fetch(`/output/${markdownPath}`);
        const text = await response.text();
        document.getElementById('markdownPreview').textContent = text;
    } catch (error) {
        console.error('Error loading markdown:', error);
        document.getElementById('markdownPreview').textContent = 'Error loading markdown content';
    }
}

