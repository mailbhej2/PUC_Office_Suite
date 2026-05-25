import re
import cv2
import tempfile
import pandas as pd
import streamlit as st
import pytesseract

from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    layout="wide",
    page_title="Table Extractor"
)

st.title("Table Extractor")


# =========================================================
# SMART TABLE PARSER
# =========================================================
def parse_table(text):

    # -----------------------------------------------------
    # CLEAN TEXT
    # -----------------------------------------------------
    text = re.sub(
        r"[|_\[\]]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    # -----------------------------------------------------
    # SPLIT ROWS USING SERIAL NUMBERS
    # -----------------------------------------------------
    rows = re.split(
        r"(?=\s\d{1,2}\.?\s)",
        text
    )

    final_rows = []


    # -----------------------------------------------------
    # PROCESS EACH ROW
    # -----------------------------------------------------
    for row in rows:

        row = row.strip()

        if not row:
            continue


        # Extract serial no
        sr_match = re.match(
            r"(\d{1,2})\.?\s+(.*)",
            row
        )

        if not sr_match:
            continue

        sr_no = sr_match.group(1)

        remaining = sr_match.group(2)


        # Detect status
        status_match = re.search(
            r"\b(Serving|Retired)\b",
            remaining,
            re.IGNORECASE
        )

        if not status_match:
            continue

        status = status_match.group(1)

        split_index = status_match.start()

        name = remaining[:split_index].strip()

        address = remaining[
            status_match.end():
        ].strip()


        # Remove garbage OCR chars
        name = re.sub(
            r"[^A-Za-z0-9.&()\\-\\s]",
            "",
            name
        )

        address = re.sub(
            r"\\s+",
            " ",
            address
        ).strip()


        final_rows.append([

            sr_no,

            name,

            status,

            address
        ])


    return pd.DataFrame(

        final_rows,

        columns=[
            "Sr No",
            "Name",
            "Status",
            "Address"
        ]
    )


# =========================================================
# UPLOAD IMAGE
# =========================================================
uploaded_file = st.file_uploader(
    "Upload Table Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PROCESS
# =========================================================
if uploaded_file:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        use_container_width=True
    )


    # -----------------------------------------------------
    # SAVE TEMP FILE
    # -----------------------------------------------------
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as tmp:

        image.save(tmp.name)

        temp_path = tmp.name


    # -----------------------------------------------------
    # PREPROCESS IMAGE
    # -----------------------------------------------------
    img = cv2.imread(temp_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # upscale
    gray = cv2.resize(

        gray,

        None,

        fx=2,

        fy=2,

        interpolation=cv2.INTER_CUBIC
    )

    # denoise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # threshold
    thresh = cv2.adaptiveThreshold(

        gray,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        31,

        11
    )


    # =====================================================
    # EXTRACT
    # =====================================================
    if st.button("Extract Table"):

        with st.spinner(
            "Extracting..."
        ):

            try:

                text = pytesseract.image_to_string(

                    thresh,

                    config="--psm 6"
                )


                # -----------------------------------------
                # RAW OCR
                # -----------------------------------------
                with st.expander(
                    "Raw OCR Text",
                    expanded=False
                ):

                    st.text(text)


                # -----------------------------------------
                # PARSE TABLE
                # -----------------------------------------
                df = parse_table(text)


                if df.empty:

                    st.warning(
                        "Could not detect table properly."
                    )

                else:

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    st.download_button(

                        "Download CSV",

                        df.to_csv(
                            index=False
                        ).encode("utf-8"),

                        "table.csv",

                        "text/csv"
                    )

            except Exception as e:

                st.error(str(e))