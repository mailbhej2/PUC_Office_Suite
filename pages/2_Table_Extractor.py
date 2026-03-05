import streamlit as st
import pandas as pd
import tempfile
from PIL import Image as PILImage
from img2table.document import Image
from img2table.ocr import TesseractOCR

st.set_page_config(layout="wide", page_title="Table Extractor")
st.title("Table Extractor from Image")

uploaded_file = st.file_uploader(
    "Upload table image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = PILImage.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    if st.button("Extract Table"):

        with st.spinner("Extracting table..."):

            ocr = TesseractOCR(n_threads=1)
            doc = Image(temp_path)

            tables = doc.extract_tables(ocr=ocr)

            if tables:

                df = tables[0].df
                st.subheader("Extracted Data")
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="extracted_table.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No table detected in the image.")