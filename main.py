import streamlit as st
import pandas as pd
import google.generativeai as genai
from docx import Document
from datetime import date
from io import BytesIO
import os
from dotenv import load_dotenv
from style import local_css


# ---------------------------------------------------------
# 🔹 1. PAGE CONFIG (must be first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(page_title="Deepak | PUC Office Suite")


# ---------------------------------------------------------
# 🔹 2. TOP-RIGHT NAV BUTTON
# ---------------------------------------------------------
col_nav1, col_nav2 = st.columns([9, 1])
with col_nav2:
    if st.button("File Status"):
        st.switch_page("pages/file_status.py")


# ---------------------------------------------------------
# 🔹 3. LOAD CSS
# ---------------------------------------------------------
st.markdown(f"<style>{local_css()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 🔹 4. LOAD ENV + AI MODEL CONFIG
# ---------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
OUTPUT_FOLDER = os.path.join(downloads_path, "output_folder")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# 🔹 5. REMINDER TABLE
# ---------------------------------------------------------
df = pd.read_csv(os.getenv('GOOGLE_SHEET_URL'))
df["Reminder Date"] = pd.to_datetime(df["Reminder Date"], errors="coerce", dayfirst=True)

start_date = pd.Timestamp("today") - pd.Timedelta(days=5)
end_date = pd.Timestamp("today") + pd.Timedelta(days=15)

df_filtered = df[df["Reminder Date"].between(start_date, end_date)][["File No.", "Subject", "Reminder Date"]]
df_filtered = df_filtered.sort_values("Reminder Date")
df_filtered["Reminder Date"] = df_filtered["Reminder Date"].dt.strftime("%d-%m-%y")
df_filtered.index += 2

st.title("Reminder")
st.dataframe(df_filtered[['File No.', 'Reminder Date', 'Subject']])


# ---------------------------------------------------------
# 🔹 6. FORM UI
# ---------------------------------------------------------
st.title("Deepak | PUC Office Suite")

with st.form("puc_form"):

    file_no_col, branch_cfms_no_col, branch_cfms_date_col, draft_type_col = st.columns(4)
    file_number = file_no_col.text_input("File Number")
    branch_cfms_no = branch_cfms_no_col.text_input("Branch CFMS No.")
    branch_cfms_date = branch_cfms_date_col.date_input("Branch CFMS Date", value=date.today())
    draft_type = draft_type_col.selectbox(
        "Draft Type",
        ["Single_Draft", "Multi_Draft", "Hindi_Draft", "UO_Draft"]
    )

    col1, col2, col3 = st.columns(3)
    puc_no = col1.text_input("PUC No.")
    puc_date = col2.date_input("PUC Date", value=date.today())
    puc_sender = col3.text_input("PUC Sender")

    puc_subject = st.text_input("PUC Subject")
    puc_body = st.text_area("PUC Body")

    col4, col5, col6 = st.columns(3)
    same_puc_body = col4.selectbox("Same PUC Body?", ["Yes", "No"])
    forward = col5.selectbox("Forward?", ["No", "Yes"])
    forwarding_dept = col6.text_input("Forwarding Deptt.")

    submitted = st.form_submit_button("Generate Files")


# ---------------------------------------------------------
# 🔹 7. GENERATE MID-PARA USING AI
# ---------------------------------------------------------
def get_mid_body(puc_body, same_puc_body):
    if same_puc_body == "Yes":
        return puc_body
    else:
        directions = """
        1. Start the first para with 'Vide PUC they have informed...'
        2. Language should be easy to understand and formal.
        """
        prompt = f"""
        Please make brief paras on {puc_body}. You must strictly comply with below directions:
        {directions}
        """
        response = model.generate_content(prompt)
        return response.text


# ---------------------------------------------------------
# 🔹 8. REPLACE PLACEHOLDERS
# ---------------------------------------------------------
def replace_placeholder(doc, placeholder, value):
    for p in doc.paragraphs:
        if placeholder in p.text:
            full_text = "".join(run.text for run in p.runs)
            new_text = full_text.replace(placeholder, str(value))

            for run in p.runs[1:]:
                p._element.remove(run._element)
            p.runs[0].text = new_text


# ---------------------------------------------------------
# 🔹 9. DOWNLOAD BUTTON HELPER
# ---------------------------------------------------------
def download_btn(buffer, label, file_no, subject):
    st.download_button(
        label=f"Download {label}",
        data=buffer.getvalue(),
        file_name=f"{file_no.replace('/', '-')} {subject}-{label}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"{label}"
    )


# ---------------------------------------------------------
# 🔹 10. GENERATE DOC FILES
# ---------------------------------------------------------
if submitted:

    doc_noting = Document("files/Noting.docx")
    doc_draft = Document(f"files/{draft_type}.docx")

    first_para = (
        f"PUC no. {puc_no} dated {puc_date.strftime('%d.%m.%Y')} sent by {puc_sender} "
        f"and received through CFMS no. {branch_cfms_no} dated {branch_cfms_date.strftime('%d.%m.%Y')}, "
        f"may kindly be perused."
    )

    mid_para = get_mid_body(puc_body, same_puc_body)

    last_para = (
        "In view of the above..."
        if forward == "No"
        else f"In view of the above, a copy of the PUC may be sent to {forwarding_dept} "
             "for further necessary action, as per the draft outlined below."
    )

    # Noting replacements
    replace_placeholder(doc_noting, "{{PUC_SUBJECT}}", puc_subject)
    replace_placeholder(doc_noting, "{{FIRST_PARA}}", first_para)
    replace_placeholder(doc_noting, "{{MID_PARA}}", mid_para)
    replace_placeholder(doc_noting, "{{LAST_PARA}}", last_para)
    replace_placeholder(doc_noting, "{{FILE_NUMBER}}", file_number)

    # Draft replacements
    replace_placeholder(doc_draft, "{{FILE_NUMBER}}", file_number)
    replace_placeholder(doc_draft, "{{BRANCH_CFMS_DATE}}", branch_cfms_date.strftime("%d.%m.%Y"))
    replace_placeholder(doc_draft, "{{PUC_SUBJECT}}", puc_subject)
    replace_placeholder(doc_draft, "{{PUC_NUMBER}}", puc_no)
    replace_placeholder(doc_draft, "{{PUC_DATE}}", puc_date.strftime("%d.%m.%Y"))
    replace_placeholder(doc_draft, "{{PUC_SENDER}}", puc_sender)

    noting_buffer = BytesIO()
    doc_noting.save(noting_buffer)
    noting_buffer.seek(0)

    draft_buffer = BytesIO()
    doc_draft.save(draft_buffer)
    draft_buffer.seek(0)

    st.session_state.noting_buffer = noting_buffer
    st.session_state.draft_buffer = draft_buffer
    st.session_state.puc_subject = puc_subject
    st.session_state.file_number = file_number


# ---------------------------------------------------------
# 🔹 11. SHOW DOWNLOAD BUTTONS
# ---------------------------------------------------------
if "noting_buffer" in st.session_state:
    download_btn(
        st.session_state.noting_buffer,
        "Noting",
        st.session_state.file_number,
        st.session_state.puc_subject
    )

if "draft_buffer" in st.session_state:
    download_btn(
        st.session_state.draft_buffer,
        "Draft",
        st.session_state.file_number,
        st.session_state.puc_subject
    )
