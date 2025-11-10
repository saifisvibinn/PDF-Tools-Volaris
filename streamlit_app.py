import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from loguru import logger

import main as extractor


OUTPUT_BASE = Path("./output")


@dataclass
class ExtractionResult:
    output_dir: Path
    pdf_path: Path
    stem: str
    elements: List[Dict]
    annotated_pdf: Optional[Path]
    markdown_path: Optional[Path]
    include_images: bool
    include_markdown: bool
    markdown_path: Optional[Path]
    include_images: bool
    include_markdown: bool


@st.cache_resource(show_spinner="Loading DocLayout-YOLO weights...")
def load_model():
    """
    Load the DocLayout-YOLO model once per Streamlit session.
    """
    return extractor.get_model()


def _prepare_output_dir(stem: str) -> Path:
    """
    Create (or increment) an output subdirectory for the given PDF stem.
    """
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    candidate = OUTPUT_BASE / stem
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    counter = 1
    while True:
        alternative = OUTPUT_BASE / f"{stem}_{counter}"
        if not alternative.exists():
            alternative.mkdir(parents=True, exist_ok=True)
            return alternative
        counter += 1


def run_extraction(
    uploaded_file,
    *,
    include_images: bool,
    include_markdown: bool,
) -> ExtractionResult:
    """
    Persist the uploaded PDF to disk, run the extraction pipeline, and
    collect the generated artifacts for downstream visualization.
    """
    original_name = Path(uploaded_file.name).stem or "uploaded_pdf"
    output_dir = _prepare_output_dir(original_name)

    pdf_path = output_dir / f"{original_name}.pdf"
    pdf_path.write_bytes(uploaded_file.getvalue())

    extractor.USE_MULTIPROCESSING = False
    logger.info(
        "Streamlit UI: running extraction in serial mode "
        f"(images={include_images}, markdown={include_markdown})"
    )

    extractor.process_pdf_with_pool(
        pdf_path,
        output_dir,
        pool=None,
        extract_images=include_images,
        extract_markdown=include_markdown,
    )

    json_path = output_dir / f"{pdf_path.stem}_content_list.json"
    elements: List[Dict] = []
    if include_images and json_path.exists():
        elements = json.loads(json_path.read_text(encoding="utf-8"))

    annotated_pdf: Optional[Path] = None
    if include_images:
        candidate_pdf = output_dir / f"{pdf_path.stem}_layout.pdf"
        if candidate_pdf.exists():
            annotated_pdf = candidate_pdf

    markdown_path: Optional[Path] = None
    if include_markdown:
        candidate_md = output_dir / f"{pdf_path.stem}.md"
        if candidate_md.exists():
            markdown_path = candidate_md

    return ExtractionResult(
        output_dir=output_dir,
        pdf_path=pdf_path,
        stem=pdf_path.stem,
        elements=elements,
        annotated_pdf=annotated_pdf,
        markdown_path=markdown_path,
        include_images=include_images,
        include_markdown=include_markdown,
    )


def render_summary(result: ExtractionResult):
    """
    Display a high-level summary of the extracted figures and tables.
    """
    figures = [elem for elem in result.elements if elem.get("type") == "figure"]
    tables = [elem for elem in result.elements if elem.get("type") == "table"]

    st.subheader("Summary")
    st.caption(f"Output directory: `{result.output_dir}`")

    if result.include_images:
        st.metric("Figures", len(figures))
        st.metric("Tables", len(tables))

        if result.annotated_pdf:
            with open(result.annotated_pdf, "rb") as pdf_file:
                st.download_button(
                    "Download Annotated PDF",
                    data=pdf_file.read(),
                    file_name=result.annotated_pdf.name,
                    mime="application/pdf",
                )
        else:
            st.info("Annotated PDF was not generated for this document.")
    else:
        st.info("Figure/table extraction was skipped for this run.")

    if result.include_markdown:
        md_source = result.markdown_path if result.markdown_path and result.markdown_path.exists() else None

        if md_source:
            markdown_bytes = md_source.read_bytes()
            st.download_button(
                "Download Markdown",
                data=markdown_bytes,
                file_name=md_source.name,
                mime="text/markdown",
            )
            if st.checkbox(
                "Preview Markdown",
                key=f"preview_{result.output_dir.name}",
            ):
                st.markdown(
                    markdown_bytes.decode("utf-8"),
                    unsafe_allow_html=False,
                )
        else:
            st.warning("Markdown output was requested but not generated.")

    if result.include_images and result.elements:
        st.write("Detected elements (first 200 shown):")
        st.dataframe(result.elements[:200])


def render_media(result: ExtractionResult):
    """
    Show extracted figures and tables as images within Streamlit.
    """
    if not result.include_images:
        st.info("Image extraction was skipped, so no media preview is available.")
        return

    if not result.elements:
        st.warning("No figures or tables detected.")
        return

    figures = [elem for elem in result.elements if elem.get("type") == "figure"]
    tables = [elem for elem in result.elements if elem.get("type") == "table"]

    if figures:
        st.subheader("Figures")
        for item in figures:
            image_path = result.output_dir / item["image_path"]
            if image_path.exists():
                st.image(str(image_path), caption=f"Page {item['page']}")
            else:
                st.warning(f"Missing figure image: {image_path.name}")

    if tables:
        st.subheader("Tables")
        for item in tables:
            image_path = result.output_dir / item["image_path"]
            if image_path.exists():
                st.image(str(image_path), caption=f"Page {item['page']}")
            else:
                st.warning(f"Missing table image: {image_path.name}")


def main():
    st.set_page_config(page_title="PDF Layout Extraction", layout="wide")
    st.title("PDF Layout Extraction Companion")
    st.markdown(
        "Upload one or more PDFs to extract figures, tables, and an annotated layout using the DocLayout-YOLO model."
    )

    uploaded_files = st.file_uploader(
        "Select PDF files", type=["pdf"], accept_multiple_files=True
    )

    extraction_choice = st.radio(
        "Select extraction mode",
        ("Images only", "Markdown only", "Images and Markdown"),
        index=0,
        horizontal=True,
    )
    include_images = extraction_choice != "Markdown only"
    include_markdown = extraction_choice != "Images only"

    if not uploaded_files:
        st.info("Awaiting PDF uploads.")
    else:
        st.success(f"{len(uploaded_files)} PDF(s) ready for processing.")
        with st.expander("Selected files", expanded=False):
            for file in uploaded_files:
                st.write(file.name)

    with st.sidebar:
        st.header("Settings")
        st.caption("Multiprocessing is disabled in Streamlit to keep the UI responsive.")
        st.toggle("Use multiprocessing", value=False, disabled=True)

    process_button = st.button("Run extraction", disabled=not uploaded_files)

    if process_button and uploaded_files:
        if include_images:
            with st.spinner("Initializing model..."):
                load_model()

        results: Dict[str, Dict] = {}
        errors: Dict[str, str] = {}
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        total = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files, 1):
            status_text.info(f"Processing {uploaded_file.name} ({idx}/{total})")
            try:
                result = run_extraction(
                    uploaded_file,
                    include_images=include_images,
                    include_markdown=include_markdown,
                )
            except Exception as exc:  # pragma: no cover
                logger.exception("Extraction failed")
                errors[uploaded_file.name] = str(exc)
            else:
                label = uploaded_file.name
                if label in results:
                    suffix = 2
                    while f"{label} ({suffix})" in results:
                        suffix += 1
                    label = f"{label} ({suffix})"
                results[label] = {
                    "output_dir": str(result.output_dir),
                    "pdf_path": str(result.pdf_path),
                    "stem": result.stem,
                    "elements": result.elements,
                    "annotated_pdf": str(result.annotated_pdf)
                    if result.annotated_pdf
                    else None,
                    "markdown_path": str(result.markdown_path)
                    if result.markdown_path
                    else None,
                    "include_images": result.include_images,
                    "include_markdown": result.include_markdown,
                }
            progress_bar.progress(idx / total)

        status_text.empty()
        progress_bar.progress(1.0)
        st.session_state["results"] = results
        st.session_state["errors"] = errors

        if results:
            st.success(f"Processed {len(results)} PDF(s).")
            first_key = next(iter(results.keys()))
            st.session_state["pdf_choice"] = first_key
        else:
            st.warning("No PDFs were processed successfully.")
            st.session_state.pop("pdf_choice", None)

    errors = st.session_state.get("errors") or {}
    if errors:
        st.error(
            "\n".join(f"{name}: {message}" for name, message in errors.items())
        )

    stored_results = st.session_state.get("results") or {}
    if stored_results:
        options = list(stored_results.keys())
        default_idx = 0
        if "pdf_choice" in st.session_state and st.session_state["pdf_choice"] in options:
            default_idx = options.index(st.session_state["pdf_choice"])
        selected_label = st.selectbox(
            "Select a processed PDF to preview",
            options,
            index=default_idx,
            key="pdf_choice",
        )

        stored = stored_results[selected_label]
        result = ExtractionResult(
            output_dir=Path(stored["output_dir"]),
            pdf_path=Path(stored["pdf_path"]),
            stem=stored["stem"],
            elements=stored["elements"],
            annotated_pdf=Path(stored["annotated_pdf"])
            if stored["annotated_pdf"]
            else None,
            markdown_path=Path(stored["markdown_path"])
            if stored.get("markdown_path")
            else None,
            include_images=stored.get("include_images", True),
            include_markdown=stored.get("include_markdown", False),
        )
        st.markdown(f"### Results for `{selected_label}`")
        render_summary(result)
        render_media(result)
    else:
        st.info("Run extraction to preview results here.")


if __name__ == "__main__":
    main()

