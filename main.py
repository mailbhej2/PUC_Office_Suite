import streamlit as st
import pandas as pd
import google.generativeai as genai
from docx import Document
from datetime import date
from io import BytesIO
import os
from dotenv import load_dotenv
from style import local_css

# Load CSS
st.markdown(f"<style>{local_css()}</style>", unsafe_allow_html=True)

# Configure API and paths
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
OUTPUT_FOLDER = os.path.join(downloads_path, "output_folder")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Reminder Table
df = pd.read_csv(os.getenv('GOOGLE_SHEET_URL'))
df["Reminder Date"] = pd.to_datetime(df["Reminder Date"], errors="coerce")
start_date = pd.Timestamp("today") - pd.Timedelta(days=3)
end_date = pd.Timestamp("today") + pd.Timedelta(days=7)
df_filtered = df[df["Reminder Date"].between(start_date, end_date)][["File No.", "Subject", "Reminder Date"]].copy()
df_filtered["Reminder Date"] = df_filtered["Reminder Date"].dt.strftime("%d-%m-%y")
df_filtered.index += 2
st.set_page_config(layout="wide", page_title="Deepak | PUC Office Suite")
st.title("Reminder")
st.dataframe(df_filtered.sort_values(by="Reminder Date"))

# Form UI
st.title("Deepak | PUC Office Suite")
with st.form("puc_form"):
    file_no_col,branch_cfms_no_col, branch_cfms_date_col,draft_type_col= st.columns(4)
    file_number = file_no_col.text_input("File Number")
    branch_cfms_no = branch_cfms_no_col.text_input("Branch CFMS No.")
    branch_cfms_date = branch_cfms_date_col.date_input("Branch CFMS Date", value=date.today())
    draft_type = draft_type_col.selectbox("Draft Type", ["Single_Draft", "Multi_Draft", "Hindi_Draft"])
    col1, col2, col3 = st.columns(3)
    puc_no = col1.text_input("PUC No.")
    puc_date = col2.date_input("PUC Date", value=date.today())
    puc_sender = col3.text_input("PUC Sender")
    puc_subject = st.text_input("PUC Subject")
    puc_body = st.text_area("PUC Body")
    col4, col5, col6 = st.columns(3)
    same_puc_body = col4.selectbox("Same PUC Body?", ["No", "Yes"])
    forward = col5.selectbox("Forward?", ["No", "Yes"])
    forwarding_dept = col6.text_input("Forwarding Deptt.")
    submitted = st.form_submit_button("Generate Files")

# Generate mid-body para
def get_mid_body(puc_body,same_puc_body):
    if same_puc_body== "Yes":
        response = f"{puc_body}"
        return response
    else:
        directions = """
        1. Start the first para with 'Vide PUC they have informed...'
        2. Language should be easy to under stand and formal.
        """
        puc_body_with_direction = f"""
        Please make brief paras on {puc_body}. You must strictly compliance with below directions:
        {directions}
        """
        response = model.generate_content(puc_body_with_direction)
        return response.text


# Replace placeholders in doc
def replace_placeholder(doc, placeholder, value):
    for p in doc.paragraphs:
        if placeholder in p.text:
            full_text = "".join(run.text for run in p.runs)
            new_text = full_text.replace(placeholder, str(value))
            for run in p.runs[1:]:
                p._element.remove(run._element)
            p.runs[0].text = new_text

# Create download button
def download_btn(buffer, label, file_no, subject):
    st.download_button(
        label=f"Download {label}",
        data=buffer.getvalue(),
        file_name=f"{file_no.replace('/', '-')}_{subject}-{label}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"{label}"
    )

# Generate docs
if submitted:
    doc_noting = Document("files/Noting.docx")
    doc_draft = Document(f"files/{draft_type}.docx")

    first_para = f"PUC no. {puc_no} dated {puc_date.strftime('%d.%m.%Y')} sent by {puc_sender} and received through CFMS no. {branch_cfms_no} dated {branch_cfms_date.strftime('%d.%m.%Y')}, may kindly be perused."
    mid_para = get_mid_body(puc_body, same_puc_body)
    last_para = "In view of the above..." if forward == "No" else f"In view of the above, a copy of the PUC may be sent to {forwarding_dept} for further necessary action, as per the draft outlined below."

    # For Noting placeholder
    replace_placeholder(doc_noting, "{{PUC_SUBJECT}}", puc_subject)
    replace_placeholder(doc_noting, "{{FIRST_PARA}}", first_para)
    replace_placeholder(doc_noting, "{{MID_PARA}}", mid_para)
    replace_placeholder(doc_noting, "{{LAST_PARA}}", last_para)
    replace_placeholder(doc_noting, "{{FILE_NUMBER}}", file_number)
    # For Draft placeholder
    replace_placeholder(doc_draft, "{{FILE_NUMBER}}", file_number)
    replace_placeholder(doc_draft, "{{BRANCH_CFMS_DATE}}", branch_cfms_date.strftime("%d.%m.%Y"))
    replace_placeholder(doc_draft, "{{PUC_SUBJECT}}", puc_subject)
    replace_placeholder(doc_draft, "{{PUC_NUMBER}}", puc_no)
    replace_placeholder(doc_draft, "{{PUC_DATE}}", puc_date)
    replace_placeholder(doc_draft, "{{PUC_SENDER}}", puc_sender)

    noting_buffer = BytesIO(); doc_noting.save(noting_buffer); noting_buffer.seek(0)
    draft_buffer = BytesIO(); doc_draft.save(draft_buffer); draft_buffer.seek(0)

    st.session_state.noting_buffer = noting_buffer
    st.session_state.draft_buffer = draft_buffer
    st.session_state.puc_subject = puc_subject
    st.session_state.file_number = file_number

# Show download buttons...
if "noting_buffer" in st.session_state:
    download_btn(st.session_state.noting_buffer, "Noting", st.session_state.file_number, st.session_state.puc_subject)
if "draft_buffer" in st.session_state:
    download_btn(st.session_state.draft_buffer, "Draft", st.session_state.file_number, st.session_state.puc_subject)
