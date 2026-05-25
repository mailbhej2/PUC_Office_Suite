import cv2
import tempfile
import pandas as pd
import streamlit as st

from PIL import Image as PILImage
from img2table.document import Image
from img2table.ocr import TesseractOCR


st.set_page_config(
    layout="wide",
    page_title="Table Extractor"
)

st.title("Table Extractor")


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = PILImage.open(
        uploaded_file
    )

    st.image(
        image,
        use_container_width=True
    )


    # =====================================================
    # SAVE TEMP FILE
    # =====================================================
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as tmp:

        image.save(tmp.name)

        temp_path = tmp.name


    # =====================================================
    # PREPROCESS IMAGE
    # =====================================================
    img = cv2.imread(temp_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )[1]

    cv2.imwrite(
        temp_path,
        thresh
    )


    # =====================================================
    # EXTRACT TABLE
    # =====================================================
    if st.button("Extract Table"):

        with st.spinner(
            "Extracting..."
        ):

            try:

                ocr = TesseractOCR(
                    n_threads=1
                )

                doc = Image(temp_path)

                tables = doc.extract_tables(
                    ocr=ocr,
                    borderless_tables=True
                )


                if not tables:

                    st.warning(
                        "No table detected."
                    )

                else:

                    df = pd.concat(

                        [
                            table.df
                            for table in tables
                        ],

                        ignore_index=True
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