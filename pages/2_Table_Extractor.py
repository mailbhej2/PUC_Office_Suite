import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract

st.set_page_config(layout="wide", page_title="Table Extractor", page_icon="📊")
st.title("📊 Table Extractor")

# =========================================================
# PREPROCESS — scan-optimised
# =========================================================
def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    if max(h, w) < 2000:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.filter2D(gray, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary


# =========================================================
# DETECT CELLS via morphological line detection
# =========================================================
def detect_cells(binary):
    h, w = binary.shape
    inv = cv2.bitwise_not(binary)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 20, 40), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 20, 40)))

    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=2)

    grid = cv2.morphologyEx(cv2.add(h_lines, v_lines),
                             cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, hierarchy = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    min_area = w * h * 0.0003
    raw = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] != -1:   # skip child contours
            continue
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if cw * ch > min_area and cw > 20 and ch > 10 and cw < w * 0.98 and ch < h * 0.98:
            raw.append((cx, cy, cw, ch))

    # deduplicate overlapping boxes
    raw = sorted(raw, key=lambda c: c[2] * c[3], reverse=True)
    kept = []
    for box in raw:
        ax, ay, aw, ah = box
        if not any(_iou(box, k) > 0.5 for k in kept):
            kept.append(box)
    return kept


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union else 0


# =========================================================
# BUILD DATAFRAME — cluster cells → rows → OCR each cell
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
        row_texts = []
        for (cx, cy, cw, ch) in sorted(rows_map[ry], key=lambda c: c[0]):
            pad = 4
            cell = binary[max(0, cy+pad):cy+ch-pad, max(0, cx+pad):cx+cw-pad]
            if cell.size == 0:
                row_texts.append("")
                continue
            if min(cell.shape) < 20:
                cell = cv2.resize(cell, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(cell, config="--psm 7 --oem 3").strip()
            row_texts.append(text)
        table.append(row_texts)

    max_cols = max(len(r) for r in table)
    df = pd.DataFrame([r + [""] * (max_cols - len(r)) for r in table])

    if has_header and len(df) > 1:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

    return _sanitize(df)


def _sanitize(df):
    df = df.astype(str)
    seen, cols = {}, []
    for col in df.columns:
        col = str(col).strip() or f"Col_{len(cols)}"
        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        else:
            seen[col] = 0
        cols.append(col)
    df.columns = cols
    return df.reset_index(drop=True)


# =========================================================
# UI
# =========================================================
uploaded = st.file_uploader("Upload scanned table image",
                             type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded:
    pil = Image.open(uploaded)
    st.image(pil, use_container_width=True)

    has_header = st.checkbox("First row is header", value=True)

    if st.button("Extract Table", type="primary"):
        with st.spinner("Detecting grid and reading cells..."):
            binary = preprocess(pil)
            cells  = detect_cells(binary)

        if len(cells) < 4:
            st.warning("No table grid detected. Ensure the image has visible border lines.")
        else:
            df = cells_to_df(cells, binary, has_header)

            r, c = len(df), len(df.columns)
            m1, m2 = st.columns(2)
            m1.metric("Data Rows", r)
            m2.metric("Columns", c)

            st.dataframe(df, use_container_width=True)
            st.download_button("⬇ Download CSV",
                                df.to_csv(index=False).encode(),
                                "table.csv", "text/csv")
