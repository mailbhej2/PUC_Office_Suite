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
# ROBUST TABLE PARSER
# =========================================================
def parse_table(text):

    # -----------------------------------------------------
    # CLEAN LINES
    # -----------------------------------------------------
    lines = [

        re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        for line in text.split("\n")

        if line.strip()
    ]


    # -----------------------------------------------------
    # BUILD LOGICAL ROWS
    # -----------------------------------------------------
    rows = []

    current_row = ""


    for line in lines:

        # remove OCR garbage chars
        line = re.sub(
            r"[|_\[\]]",
            " ",
            line
        )

        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()


        # detect new row using serial no
        is_new_row = re.match(
            r"^\d{1,2}[\.\)]?\s",
            line
        )


        if is_new_row:

            if current_row:

                rows.append(
                    current_row.strip()
                )

            current_row = line

        else:

            current_row += " " + line


    if current_row:

        rows.append(
            current_row.strip()
        )


    # -----------------------------------------------------
    # PARSE ROWS
    # -----------------------------------------------------
    final_rows = []


    for row in rows:

        # extract serial number
        sr_match = re.match(
            r"^(\d{1,2})[\.\)]?\s+(.*)",
            row
        )

        if not sr_match:
            continue


        sr_no = sr_match.group(1)

        remaining = sr_match.group(2)


        # detect status
        status_match = re.search(
            r"\b(Serving|Retired)\b",
            remaining,
            re.IGNORECASE
        )

        if not status_match:
            continue


        status = status_match.group(1)


        # split name and address
        name = remaining[
            :status_match.start()
        ].strip()

        address = remaining[
            status_match.end():
        ].strip()


        # cleanup
        name = re.sub(
            r"[^A-Za-z0-9.,&()\-\/\s]",
            "",
            name
        )

        address = re.sub(
            r"\s+",
            " ",
            address
        ).strip()


        # skip false rows
        if len(name) < 3:
            continue


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
# FILE UPLOAD
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
    # SAVE TEMP IMAGE
    # -----------------------------------------------------
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as tmp:

        image.save(tmp.name)

        temp_path = tmp.name


    # -----------------------------------------------------
    # IMAGE PREPROCESSING
    # -----------------------------------------------------
    img = cv2.imread(temp_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    # upscale image
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


    # adaptive threshold
    thresh = cv2.adaptiveThreshold(

        gray,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        31,

        11
    )


    # =====================================================
    # EXTRACT TABLE
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
                # RAW OCR TEXT
                # -----------------------------------------
                with st.expander(
                    "Raw OCR Text",
                    expanded=False
                ):

                    st.text(text)


                # -----------------------------------------
                # STRUCTURED TABLE
                # -----------------------------------------
                df = parse_table(text)


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