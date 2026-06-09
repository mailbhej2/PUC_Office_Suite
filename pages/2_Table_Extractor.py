import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract
import re

st.set_page_config(layout="wide", page_title="Table Extractor", page_icon="📊")
st.title("📊 Table Extractor")


# =========================================================
# PREPROCESS
# =========================================================
def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if max(h, w) < 2000:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary


# =========================================================
# DETECT CELLS
# =========================================================
def detect_cells(binary):
    h, w = binary.shape
    inv = cv2.bitwise_not(binary)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 50, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 50))
    h_lines  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_lines  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=2)
    h_lines  = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    v_lines  = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    grid = cv2.add(h_lines, v_lines)

    contours, hierarchy = cv2.findContours(grid, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], grid

    min_area = w * h * 0.0003
    max_area = w * h * 0.6
    raw = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            continue
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if min_area < cw * ch < max_area and cw > 15 and ch > 8:
            raw.append((cx, cy, cw, ch))

    raw = sorted(raw, key=lambda c: c[2] * c[3], reverse=True)
    kept = []
    for box in raw:
        if not any(_iou(box, k) > 0.4 for k in kept):
            kept.append(box)
    return kept, grid


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union else 0


# =========================================================
# OCR A SINGLE CELL — upscale heavily, clean noise
# =========================================================
def ocr_cell(cell_img):
    # Upscale to at least 200px tall for reliable OCR
    ch, cw = cell_img.shape
    scale = max(3.0, 150 / ch) if ch > 0 else 3.0
    cell_img = cv2.resize(cell_img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    # Add white border so characters aren't clipped
    cell_img = cv2.copyMakeBorder(cell_img, 10, 10, 10, 10,
                                   cv2.BORDER_CONSTANT, value=255)

    # Re-threshold after upscaling
    _, cell_img = cv2.threshold(cell_img, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # PSM 6 = uniform block of text (best for multi-line cells)
    text = pytesseract.image_to_string(
        cell_img,
        config="--psm 6 --oem 3 -c preserve_interword_spaces=1"
    ).strip()

    # collapse internal newlines → single space
    text = " ".join(text.split())

    # strip common OCR junk chars that aren't real content
    text = re.sub(r'[|\\}{~`]', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # remove lone punctuation left after stripping
    if re.fullmatch(r'[\W_]+', text):
        text = ""

    return text


# =========================================================
# BUILD DATAFRAME
# =========================================================
def cells_to_df(cells, binary, has_header):
    cells_sorted = sorted(cells, key=lambda c: c[1])
    thr = max(8, int(np.median([c[3] for c in cells_sorted])) // 3)

    rows_map = {}
    for box in cells_sorted:
        cy = box[1]
        match = next((ry for ry in rows_map if abs(cy - ry) <= thr), None)
        if match is None:
            rows_map[cy] = []
            match = cy
        rows_map[match].append(box)

    table = []
    for ry in sorted(rows_map):
        row_cells = sorted(rows_map[ry], key=lambda c: c[0])
        row_texts = []
        for (cx, cy, cw, ch) in row_cells:
            pad = 6
            cell = binary[max(0, cy+pad):cy+ch-pad,
                          max(0, cx+pad):cx+cw-pad]
            text = ocr_cell(cell) if cell.size > 0 else ""
            row_texts.append(text)
        table.append(row_texts)

    if not table:
        return pd.DataFrame()

    max_cols = max(len(r) for r in table)
    df = pd.DataFrame([r + [""] * (max_cols - len(r)) for r in table])

    if has_header and len(df) > 1:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

    return _sanitize(df)


def _sanitize(df):
    df = df.astype(str).replace("nan", "")
    seen, cols = {}, []
    for col in df.columns:
        col = str(col).strip() or f"Col_{len(cols)}"
        if col in seen:
            seen[col] += 1; col = f"{col}_{seen[col]}"
        else:
            seen[col] = 0
        cols.append(col)
    df.columns = cols
    return df.reset_index(drop=True)


# =========================================================
# UI
# =========================================================
uploaded = st.file_uploader("Upload table image",
                             type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded:
    pil = Image.open(uploaded)
    st.image(pil, use_container_width=True)
    has_header = st.checkbox("First row is header", value=True)
    show_debug = st.checkbox("Show debug overlay")

    if st.button("Extract Table", type="primary"):
        with st.spinner("Processing..."):
            binary = preprocess(pil)
            cells, grid = detect_cells(binary)

        if show_debug:
            d1, d2 = st.columns(2)
            d1.caption("Preprocessed")
            d1.image(binary, use_container_width=True, clamp=True)
            overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            overlay[grid > 0] = [0, 200, 100]
            for (cx, cy, cw, ch) in cells:
                cv2.rectangle(overlay, (cx, cy), (cx+cw, cy+ch), (0, 100, 255), 2)
            d2.caption(f"Detected: {len(cells)} cells")
            d2.image(overlay, use_container_width=True, clamp=True)

        if len(cells) >= 4:
            with st.spinner("Reading cell contents..."):
                df = cells_to_df(cells, binary, has_header)
            method = f"grid ({len(cells)} cells)"
        else:
            st.warning(f"Only {len(cells)} cell(s) detected. "
                        "Enable debug overlay to inspect.")
            df = pd.DataFrame()
            method = "failed"

        if df is None or df.empty:
            st.warning("No data extracted.")
        else:
            st.success(f"✅  Method: `{method}`")
            m1, m2 = st.columns(2)
            m1.metric("Data Rows", len(df))
            m2.metric("Columns", len(df.columns))
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇ Download CSV",
                                df.to_csv(index=False).encode(),
                                "table.csv", "text/csv")
