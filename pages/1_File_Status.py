import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

st.set_page_config(layout="wide")

st.set_page_config(page_title="File Status")
st.title("File Status")
load_dotenv()

# Read Google Sheet
df = pd.read_csv(os.getenv("GOOGLE_SHEET_URL"))
df.index += 2
# 🔹 Specify which columns you want to show
# Just change this list anytime
columns_to_show = [
    "File No.",
    "Subject",
    "Last Dealt On",
    "Reminder Date",
    "Remarks",
    "CFMS No.",
    "Current Status",
    "Current Status Date"
]

# Filter only available columns (avoids errors)
available_cols = [col for col in columns_to_show if col in df.columns]

st.title("File Status")
st.dataframe(df[available_cols])