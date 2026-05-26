import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import os
import pytesseract

from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    layout="wide",
    page_title="Scanned Table Extractor",
    page_icon="📊"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: #0f1117; }
    h1 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600; color: #e8f4f8 !important;
        letter-spacing: -1px;
        border-bottom: 2px solid #00d4aa;
        padding-bottom: 12px; margin-bottom: 8px !important;
    }
    .subtitle {
        font-family: 'IBM Plex Mono', monospace;
        color: #5a7a8a; font-size: 13px;
        margin-bottom: 32px; letter-spacing: 1px;
    }
    .step-badge {
        display: inline-block;
        background: #00d4aa22; border: 1px solid #00d4aa55;
        color: #00d4aa; font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; padding: 3px 10px; border-radius: 2px;
        margin-bottom: 8px; letter-spacing: 2px;
    }
    .info-box {
        background: #161b22; border-left: 3px solid #00d4aa;
        padding: 14px 18px; border-radius: 0 4px 4px 0;
        margin: 16px 0; font-size: 13px; color: #8baab8;
    }
    .stButton > button {
        background: #00d4aa !important; color: #0f1117 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important; border: none !important;
        border-radius: 2px !important; padding: 10px 28px !important;
        letter-spacing: 1px !important; font-size: 13px !important;
        width: 100%;
    }
    .stDownloadButton > button {
        background: transparent !important; color: #00d4aa !important;
        border: 1px solid #00d4aa55 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important; letter-spacing: 1px !important;
        border-radius: 2px !important;
    }
    .metric-card {
        background: #161b22; border: 1px solid #1e2d38;
        padding: 16px 20px; border-radius: 4px; text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px; font-weight: 600; color: #00d4aa;
    }
    .metric-label {
        font-size: 11px; color: #5a7a8a;
        letter-spacing: 1px; text-transform: uppercase; margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Scanned Table Extractor")
st.markdown('<div class="subtitle">// SCANNED IMAGE → STRUCTURED CSV</div>', unsafe_allow_html=True)


# =========================================================
# STEP 1 — PREPROCESS FOR SCANNED DOCUMENTS
# Scans need: upscale → deskew → denoise → sharpen → binarize
# =========================================================
def preprocess_scan(pil_image: Image.Image) -> np.ndarray:
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # --- upscale to at least 300 DPI equivalent ---
    h, w = img.shape[:2]
    if max(h, w) < 2000:
        scale = 2.0
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- deskew ---
    gray = _deskew(gray)

    # --- remove scan noise (salt & pepper, dust) ---
    gray = cv2.medianBlur(gray, 3)

    # --- sharpen to crisp up text and lines ---
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1,  0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # --- binarize: Otsu works well on clean scans ---
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- if image is dark-background, invert ---
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    return binary


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct up to ±10° rotation common in hand-placed scans."""
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                 threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return gray
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if not angles:
            return gray
        median_angle = np.median(angles)
        if abs(median_angle) > 10:
            return gray
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


# =========================================================
# STEP 2 — DETECT TABLE GRID (morphological line detection)
# This is the core of why scanned table extraction works well:
# we find the actual drawn lines, not guessing from text position.
# =========================================================
def detect_grid_cells(binary: np.ndarray):
    """
    Find all table cells by detecting horizontal + vertical lines
    via morphological operations, then finding enclosed rectangles.
    Returns sorted list of (x, y, w, h) cell bounding boxes.
    """
    h, w = binary.shape

    # invert so lines are white on black (easier for morphology)
    inv = cv2.bitwise_not(binary)

    # ---- horizontal lines ----
    h_len = max(w // 20, 40)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=2)
    h_lines = cv2.dilate(h_lines, cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 3)), iterations=1)

    # ---- vertical lines ----
    v_len = max(h // 20, 40)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_lines = cv2.dilate(v_lines, cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 1)), iterations=1)

    # ---- combine grid ----
    grid = cv2.add(h_lines, v_lines)

    # ---- find enclosed cell contours ----
    contours, _ = cv2.findContours(grid, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) * 0.0003
    cells = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        # filter: not too small, not the whole image
        if (cw * ch > min_area
                and cw > 20 and ch > 10
                and cw < w * 0.98 and ch < h * 0.98):
            cells.append((cx, cy, cw, ch))

    return cells, grid


# =========================================================
# STEP 3 — CLUSTER CELLS INTO ROWS & COLUMNS
# =========================================================
def cells_to_grid(cells, binary):
    """
    Group detected cell boxes into a 2D grid (rows × cols),
    OCR each cell individually, return a DataFrame.
    """
    if not cells:
        return pd.DataFrame()

    # cluster by Y to find rows
    cells_sorted = sorted(cells, key=lambda c: c[1])
    row_threshold = 12

    rows_map = {}
    for (cx, cy, cw, ch) in cells_sorted:
        matched = None
        for ry in rows_map:
            if abs(cy - ry) <= row_threshold:
                matched = ry
                break
        if matched is None:
            rows_map[cy] = []
            matched = cy
        rows_map[matched].append((cx, cy, cw, ch))

    # sort rows top→bottom, cells left→right
    sorted_rows = []
    for ry in sorted(rows_map.keys()):
        row = sorted(rows_map[ry], key=lambda c: c[0])
        sorted_rows.append(row)

    # OCR each cell
    table_data = []
    for row in sorted_rows:
        row_texts = []
        for (cx, cy, cw, ch) in row:
            pad = 4
            cell_img = binary[
                max(0, cy + pad): cy + ch - pad,
                max(0, cx + pad): cx + cw - pad
            ]
            if cell_img.size == 0:
                row_texts.append("")
                continue
            # scale up tiny cells for better OCR
            ch2, cw2 = cell_img.shape
            if cw2 < 60 or ch2 < 20:
                cell_img = cv2.resize(cell_img, None, fx=3, fy=3,
                                       interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(
                cell_img,
                config="--psm 7 --oem 3 -c tessedit_char_blacklist=|"
            ).strip()
            row_texts.append(text)
        table_data.append(row_texts)

    # normalize column count
    max_cols = max(len(r) for r in table_data) if table_data else 0
    normalized = [r + [""] * (max_cols - len(r)) for r in table_data]

    return pd.DataFrame(normalized)


# =========================================================
# STEP 4 — FALLBACK: no grid lines detected
# Use img2table if available, else word-grouping
# =========================================================
def fallback_extract(binary: np.ndarray, pil_image: Image.Image,
                     has_header: bool):
    """Try img2table first (handles borderless tables), then word-grouping."""

    # -- try img2table --
    try:
        from img2table.document import Image as I2TImage
        from img2table.ocr import TesseractOCR

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(binary).save(tmp.name)
            tmp_path = tmp.name

        ocr = TesseractOCR(n_threads=1, lang="eng")
        doc = I2TImage(src=tmp_path, detect_rotation=False)
        result = doc.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            implicit_columns=True,
            borderless_tables=True,
            min_confidence=30,
        )
        os.unlink(tmp_path)

        tables = []
        for t in result:
            df = t.df
            if has_header and len(df) > 1:
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
            tables.append(df)
        if tables:
            return tables, "img2table (borderless)"
    except Exception:
        pass

    # -- word-grouping last resort --
    df = _word_group_extract(binary)
    return [df], "word-grouping (no grid detected)"


def _word_group_extract(binary: np.ndarray) -> pd.DataFrame:
    data = pytesseract.image_to_data(
        binary,
        output_type=pytesseract.Output.DATAFRAME,
        config="--psm 6 --oem 3"
    )
    data = data.dropna(subset=["text"])
    data = data[data["text"].astype(str).str.strip() != ""]
    data = data[data["conf"] > 25]
    if data.empty:
        return pd.DataFrame()

    data = data.sort_values(["top", "left"])
    row_thr = 12
    rows, cur, last_y = [], [], None
    for _, tok in data.iterrows():
        y = tok["top"]
        if last_y is None or abs(y - last_y) <= row_thr:
            cur.append(tok)
            last_y = y if last_y is None else (last_y + y) / 2
        else:
            rows.append(sorted(cur, key=lambda t: t["left"]))
            cur, last_y = [tok], y
    if cur:
        rows.append(sorted(cur, key=lambda t: t["left"]))

    # X-cluster for columns
    all_x = [t["left"] for row in rows for t in row]
    cols = _cluster_x(all_x, gap=30)

    table_rows = []
    for row in rows:
        rd = [""] * len(cols)
        for tok in row:
            ci = min(range(len(cols)), key=lambda i: abs(cols[i] - tok["left"]))
            rd[ci] = (rd[ci] + " " + str(tok["text"]).strip()).strip()
        table_rows.append(rd)

    df = pd.DataFrame(table_rows)
    if len(df) > 1:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
    return df


def _cluster_x(lefts, gap=30):
    if not lefts:
        return [0]
    sl = sorted(set(lefts))
    clusters = [[sl[0]]]
    for x in sl[1:]:
        if x - clusters[-1][-1] <= gap:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [int(np.mean(c)) for c in clusters]


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    has_header = st.toggle("First row is header", value=True)
    show_preprocessed = st.toggle("Show preprocessed image", value=False)
    show_grid = st.toggle("Show detected grid overlay", value=False)

    st.markdown("---")
    st.markdown('<div class="info-box">Best results with:<br>• Scans ≥ 200 DPI<br>• Tables with visible border lines<br>• Black text on white background<br>• Minimal skew / shadows</div>',
                unsafe_allow_html=True)

    st.markdown("### 📦 Install")
    st.code("pip install streamlit opencv-python\npip install pytesseract Pillow\npip install pandas img2table", language="bash")


# =========================================================
# MAIN
# =========================================================
st.markdown('<div class="step-badge">STEP 01 — UPLOAD</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your scanned table image",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
)

if uploaded_file:
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("**Original**")
        st.image(pil_image, use_container_width=True)

    # preprocess immediately so we can show it
    binary = preprocess_scan(pil_image)

    if show_preprocessed:
        with col2:
            st.markdown("**Preprocessed (binarized)**")
            st.image(binary, use_container_width=True, clamp=True)

    st.markdown("---")
    st.markdown('<div class="step-badge">STEP 02 — EXTRACT</div>', unsafe_allow_html=True)

    if st.button("🔍  EXTRACT TABLE"):

        with st.spinner("Detecting grid lines and reading cells..."):

            cells, grid_img = detect_grid_cells(binary)

            if show_grid:
                # overlay detected grid on original for debugging
                overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                overlay[grid_img > 0] = [0, 212, 170]
                st.image(overlay, caption="Detected Grid (teal = found lines)",
                         use_container_width=True)

            method = "morphological grid + cell OCR"

            if len(cells) >= 4:
                # --- happy path: grid found ---
                df_raw = cells_to_grid(cells, binary)

                if df_raw.empty:
                    tables, method = fallback_extract(binary, pil_image, has_header)
                else:
                    if has_header and len(df_raw) > 1:
                        df_raw.columns = df_raw.iloc[0]
                        df_raw = df_raw[1:].reset_index(drop=True)
                    tables = [df_raw]
            else:
                # --- no grid lines: use fallback ---
                tables, method = fallback_extract(binary, pil_image, has_header)

        # ---- display results ----
        if not tables or all(df.empty for df in tables):
            st.warning("⚠️ No table data detected. Tips: ensure the image has visible table borders, or try a higher resolution scan.")
        else:
            good_tables = [df for df in tables if not df.empty]
            st.success(f"✅ Extracted **{len(good_tables)}** table(s)  •  Method: `{method}`")

            for i, df in enumerate(good_tables):
                label = f"Table {i + 1}" if len(good_tables) > 1 else "Extracted Table"
                st.markdown(f"### {label}")

                c1, c2, c3 = st.columns(3)
                filled = df.replace("", pd.NA).notna().sum().sum()
                total = df.size
                pct = int(filled / total * 100) if total > 0 else 0

                c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="metric-value">{len(df.columns)}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="metric-value">{pct}%</div><div class="metric-label">Fill Rate</div></div>', unsafe_allow_html=True)

                st.dataframe(df, use_container_width=True)

                fname = f"table_{i+1}.csv" if len(good_tables) > 1 else "table.csv"
                st.download_button(
                    label=f"⬇  Download {label} as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=fname,
                    mime="text/csv",
                    key=f"dl_{i}"
                )
