import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

st.set_page_config(layout="wide", page_title="File Status")
st.title("File Status")

load_dotenv()

# Read Google Sheet
df = pd.read_csv(os.getenv("GOOGLE_SHEET_URL"))
df.index += 1

# Convert to datetime for proper sorting
df["Last Dealt On"] = pd.to_datetime(df["Last Dealt On"], dayfirst=True, errors='coerce')
df["Reminder Date"] = pd.to_datetime(df["Reminder Date"], dayfirst=True, errors='coerce')

# Sort while still datetime
df = df.sort_values("Last Dealt On",ascending=False)

# After sorting -> convert for display only
df["Last Dealt On"] = df["Last Dealt On"].dt.strftime("%d-%m-%y")
df["Reminder Date"] = df["Reminder Date"].dt.strftime("%d-%m-%y")

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

available_cols = [col for col in columns_to_show if col in df.columns]
df = df[df["File No."].notna() & (df["File No."].astype(str).str.strip() != "")]
st.dataframe(df[available_cols], use_container_width=True)
