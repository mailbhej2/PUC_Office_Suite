import cv2
import tempfile
import pandas as pd
import pytesseract
import streamlit as st

from PIL import Image


st.set_page_config(
    layout="wide",
    page_title="Table Extractor"
)

st.title("Table Extractor")


uploaded_file = st.file_uploader(
    "Upload Table Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        use_container_width=True
    )


    # =====================================================
    # SAVE TEMP IMAGE
    # =====================================================
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as tmp:

        image.save(tmp.name)

        temp_path = tmp.name


    # =====================================================
    # PREPROCESS
    # =====================================================
    img = cv2.imread(temp_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    thresh = cv2.adaptiveThreshold(

        gray,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        11,

        2
    )

    cv2.imwrite(
        temp_path,
        thresh
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
                    thresh
                )

                lines = [

                    line.strip()

                    for line in text.split("\n")

                    if line.strip()
                ]

                df = pd.DataFrame(
                    {"Extracted Text": lines}
                )

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