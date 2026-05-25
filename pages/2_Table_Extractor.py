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
# UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload Table Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# OCR LINE PARSER
# =========================================================
def parse_table(lines):

    rows = []

    for line in lines:

        # split on multiple spaces / tabs
        parts = [

            x.strip()

            for x in line
            .replace("\t", "  ")
            .split("  ")

            if x.strip()
        ]

        if len(parts) > 1:
            rows.append(parts)

    if not rows:
        return pd.DataFrame()

    max_cols = max(len(r) for r in rows)

    rows = [

        r + [""] * (max_cols - len(r))

        for r in rows
    ]

    headers = rows[0]

    data = rows[1:]

    return pd.DataFrame(
        data,
        columns=headers
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
    # PREPROCESS
    # -----------------------------------------------------
    img = cv2.imread(temp_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # upscale for better OCR
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


    # -----------------------------------------------------
    # EXTRACT
    # -----------------------------------------------------
    if st.button("Extract Table"):

        with st.spinner(
            "Extracting..."
        ):

            try:

                text = pytesseract.image_to_string(

                    thresh,

                    config="--psm 6"
                )

                lines = [

                    line.strip()

                    for line in text.split("\n")

                    if line.strip()
                ]


                # -----------------------------------------
                # RAW OCR TEXT
                # -----------------------------------------
                with st.expander(
                    "Raw OCR Text",
                    expanded=False
                ):

                    st.text(text)


                # -----------------------------------------
                # TABLE DATAFRAME
                # -----------------------------------------
                df = parse_table(lines)


                if df.empty:

                    st.warning(
                        "Could not structure table properly."
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