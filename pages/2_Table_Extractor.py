import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract

st.set_page_config(layout="wide", page_title="Table Extractor", page_icon="📊")
st.title("📊 Table Extractor")


# =========================================================
# PREPROCESS — deskew + binarize
# =========================================================
def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    if max(h, w) < 2000:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # auto-deskew via Hough lines
    gray = _deskew(gray)

    gray = cv2.medianBlur(gray, 3)
    gray = cv2.filter2D(gray, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

    # CLAHE to even out lighting / shadows
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _deskew(gray):
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                 threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return gray
        angles = [np.degrees(np.arctan2(y2-y1, x2-x1))
                  for x1,y1,x2,y2 in lines[:,0] if x2 != x1]
        angle = np.median(angles)
        if abs(angle) > 10:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


# =========================================================
# DETECT CELLS — lenient parameters for photos/angled scans
# =========================================================
def detect_cells(binary, h_scale, v_scale, min_area_pct):
    h, w = binary.shape
    inv = cv2.bitwise_not(binary)

    # lenient: shorter minimum line lengths catch partial/skewed lines
    h_len = max(w // h_scale, 10)
    v_len = max(h // v_scale, 10)

    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)), iterations=2)
    h_lines = cv2.dilate(h_lines,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4)), iterations=1)

    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)), iterations=2)
    v_lines = cv2.dilate(v_lines,
                cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1)), iterations=1)

    grid = cv2.morphologyEx(cv2.add(h_lines, v_lines),
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    contours, hierarchy = cv2.findContours(grid, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], grid

    min_area = w * h * (min_area_pct / 100.0)
    raw = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            continue
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if (cw * ch > min_area and cw > 15 and ch > 8
                and cw < w * 0.98 and ch < h * 0.98):
            raw.append((cx, cy, cw, ch))

    # deduplicate
    raw = sorted(raw, key=lambda c: c[2]*c[3], reverse=True)
    kept = []
    for box in raw:
        if not any(_iou(box, k) > 0.5 for k in kept):
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
# FALLBACK — word-group OCR when grid detection fails
# This works well even for angled/phone photos
# =========================================================
def ocr_fallback(binary, has_header):
    """
    Use Tesseract's own layout analysis (--psm 6) to extract text,
    then group tokens into rows by Y-position and columns by X-clustering.
    Handles multi-line header cells by merging lines with same column.
    """
    data = pytesseract.image_to_data(
        binary,
        output_type=pytesseract.Output.DATAFRAME,
        config="--psm 6 --oem 3"
    )
    data = data.dropna(subset=["text"])
    data = data[data["text"].astype(str).str.strip() != ""]
    data = data[data["conf"] > 20].copy()

    if data.empty:
        return pd.DataFrame(), "ocr-fallback (no text found)"

    # cluster X positions into columns
    all_lefts = sorted(data["left"].tolist())
    col_centers = _cluster_x(all_lefts, gap=max(20, int(data["width"].median() * 0.6)))

    # assign each token to a column
    data["col_idx"] = data["left"].apply(
        lambda x: min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - x))
    )

    # group tokens into lines by Y proximity
    data = data.sort_values("top")
    line_thr = max(8, int(data["height"].median() * 0.6))
    lines = []
    cur_line = []
    last_y = None
    for _, tok in data.iterrows():
        y = tok["top"]
        if last_y is None or abs(y - last_y) <= line_thr:
            cur_line.append(tok)
            last_y = y if last_y is None else (last_y + y) / 2
        else:
            lines.append(cur_line)
            cur_line = [tok]
            last_y = y
    if cur_line:
        lines.append(cur_line)

    # build row text: each line → one row dict {col_idx: text}
    raw_rows = []
    for line in lines:
        row = {}
        for tok in line:
            ci = int(tok["col_idx"])
            row[ci] = (row.get(ci, "") + " " + str(tok["text"]).strip()).strip()
        raw_rows.append(row)

    # merge consecutive lines that share the same column pattern
    # (handles multi-line header cells like "Existing\nRemune\nration as on")
    merged = [raw_rows[0]]
    for row in raw_rows[1:]:
        prev = merged[-1]
        # if the two lines share ≥50% of the same columns, merge
        shared = len(set(row.keys()) & set(prev.keys()))
        total  = len(set(row.keys()) | set(prev.keys()))
        if total > 0 and shared / total >= 0.5:
            for ci, txt in row.items():
                prev[ci] = (prev.get(ci, "") + " " + txt).strip()
        else:
            merged.append(row)

    n_cols = len(col_centers)
    table = []
    for row in merged:
        table.append([row.get(i, "") for i in range(n_cols)])

    df = pd.DataFrame(table)
    if has_header and len(df) > 1:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

    return _sanitize(df), f"ocr-fallback ({len(table)} raw rows, {n_cols} cols)"


def _cluster_x(lefts, gap=30):
    if not lefts:
        return [0]
    clusters = [[lefts[0]]]
    for x in lefts[1:]:
        if x - clusters[-1][-1] <= gap:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [int(np.mean(c)) for c in clusters]


# =========================================================
# BUILD DATAFRAME FROM DETECTED CELLS
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
            cell = binary[max(0,cy+pad):cy+ch-pad, max(0,cx+pad):cx+cw-pad]
            if cell.size == 0:
                row_texts.append("")
                continue
            if min(cell.shape) < 20:
                cell = cv2.resize(cell, None, fx=3, fy=3,
                                   interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(
                cell, config="--psm 6 --oem 3").strip()
            # collapse internal newlines (multi-line cell content)
            text = " ".join(text.split())
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
uploaded = st.file_uploader("Upload table image (scan or photo)",
                             type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded:
    pil = Image.open(uploaded)
    st.image(pil, use_container_width=True)

    has_header = st.checkbox("First row is header", value=True)

    with st.expander("⚙️ Detection tuning"):
        st.caption("If grid detection fails, the app auto-falls back to OCR layout mode.")
        col1, col2, col3 = st.columns(3)
        h_scale  = col1.slider("H line sensitivity", 5, 60, 30)
        v_scale  = col2.slider("V line sensitivity", 5, 60, 30)
        min_area = col3.slider("Min cell area %", 0.01, 1.0, 0.02, step=0.01)
        show_debug = st.checkbox("Show debug overlay")

    if st.button("Extract Table", type="primary"):
        with st.spinner("Processing..."):
            binary = preprocess(pil)
            cells, grid = detect_cells(binary, h_scale, v_scale, min_area)

        if show_debug:
            d1, d2 = st.columns(2)
            d1.caption("Preprocessed")
            d1.image(binary, use_container_width=True, clamp=True)
            overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            overlay[grid > 0] = [0, 200, 100]
            for (cx, cy, cw, ch) in cells:
                cv2.rectangle(overlay, (cx,cy), (cx+cw,cy+ch), (0,100,255), 2)
            d2.caption(f"Grid — {len(cells)} cells detected")
            d2.image(overlay, use_container_width=True, clamp=True)

        if len(cells) >= 4:
            df = cells_to_df(cells, binary, has_header)
            method = f"grid detection ({len(cells)} cells)"
        else:
            st.info(f"Grid detection found only {len(cells)} cells — using OCR layout fallback.")
            df, method = ocr_fallback(binary, has_header)

        if df.empty:
            st.warning("No table data extracted. Try enabling debug overlay to inspect the image.")
        else:
            st.success(f"Method: `{method}`")
            m1, m2 = st.columns(2)
            m1.metric("Data Rows", len(df))
            m2.metric("Columns", len(df.columns))
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇ Download CSV",
                                df.to_csv(index=False).encode(),
                                "table.csv", "text/csv")
