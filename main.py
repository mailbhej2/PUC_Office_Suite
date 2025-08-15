#pip install -r requirements.txt
import streamlit as st
from style import local_css
import pandas as pd
import google.generativeai as genai
from docx import Document
from datetime import date
import os
from dotenv import load_dotenv
#CSS Style
st.markdown(f"<style>{local_css()}</style>", unsafe_allow_html=True)
# AI
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
OUTPUT_FOLDER = fr"{downloads_path}\output_folder"
# Make sure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Downloads folder path:", downloads_path)
def get_mid_body(puc_body,same_puc_body):
    if same_puc_body== "Yes":
        response = f"{puc_body}"
        return response
    else:
        directions = """
        1. Start the first para with 'Vide PUC they have informed...'
        2. Must use keywords when starting a para when necessary 'It is pertinent to mention here that', 'It is also pertinent to mention here that', 'In this regard' etc.
        3. Language should be easy to under stand and formal.
        """
        puc_body_with_direction = f"""
        Please make brief paras on {puc_body}. You must strictly compliance with below directions:
        {directions}
        """
        response = model.generate_content(puc_body_with_direction)
        return response.text
# Reminder table
xl_file_path = r"C:\Users\deepa\OneDrive\Desktop\Dropbox\Deepak Soni\Office Work\Industries Branch (3IB-II)\All Files\A_File_Status.xlsx"
df = pd.read_excel(xl_file_path)
df["Reminder Date"] = pd.to_datetime(df["Reminder Date"], errors="coerce")
st.set_page_config(layout="wide")
st.title("Reminder")
df_filtered = df.loc[
    df["Reminder Date"].between(pd.Timestamp("today"), pd.Timestamp("today") + pd.Timedelta(days=7)),
    ["File No.", "Subject", "Reminder Date"]
].copy()
# Format the Reminder Date as mm-dd-yy string
df_filtered["Reminder Date"] = df_filtered["Reminder Date"].dt.strftime("%d-%m-%y")
df_filtered.index = df_filtered.index+2
df_filtered= df_filtered.sort_values(by="Reminder Date", ascending=True)
st.dataframe(df_filtered)

# Streamlit form
st.set_page_config(page_title="Deepak | PUC Office Suite")
st.title("Deepak | PUC Office Suite")
with st.form("puc_form"):
    col9, col10 = st.columns(2)
    with col9:
        file_number = st.text_input("File Number")
    with col10:
        draft_type = st.selectbox("Draft Type", ["Single_Draft", "Multi_Draft","Hindi_Draft"])
    col1, col2,col3= st.columns(3)
    with col1:
        puc_no = st.text_input("PUC No.")
    with col2:
        puc_date = st.date_input("PUC Date", value=date.today())
    with col3:
        puc_sender = st.text_input("PUC Sender")
    col4, col5 = st.columns(2)
    with col4:
        branch_cfms_no = st.text_input("Branch CFMS No.")
    with col5:
        branch_cfms_date = st.date_input("Branch CFMS Date", value=date.today())
    puc_subject = st.text_input("PUC Subject")
    puc_body = st.text_area("PUC Body")
    col6, col7,col8 = st.columns(3)
    with col6:
        same_puc_body = st.selectbox("Same PUC Body ?", ["No", "Yes"])
    with col7:
        forward = st.selectbox("Forward ?", ["No", "Yes"])
    with col8:
        forwarding_dept = st.text_input("Forwarding Deptt.")

    submitted = st.form_submit_button("Generate Noting")

if submitted:
    # Load template
    TEMPLATE_NOTING_PATH = "files/Noting.docx"  # replace with your file path
    TEMPLATE_DRAFT_PATH = f"files/{draft_type}.docx"
    doc_noting = Document(TEMPLATE_NOTING_PATH)
    doc_draft =  Document(TEMPLATE_DRAFT_PATH)
    first_para = f'PUC no. {puc_no} dated {puc_date.strftime("%d.%m.%Y")} sent by {puc_sender} and received in this branch through CFMS no. {branch_cfms_no} dated {branch_cfms_date.strftime("%d.%m.%Y")}, may kindly be perused.'
    mid_para = get_mid_body(puc_body,same_puc_body)
    last_para = "In view of the above..." if forward =="No" else f"In view of the above, if officers agree a copy of the PUC along with enclosures may be sent to {forwarding_dept} for further necessary action, as per the draft outlined below."
    # Replace placeholders in the document
    def replace_placeholder(doc, placeholder, value):
        for p in doc.paragraphs:
            if placeholder in p.text:
                # Combine all runs text
                full_text = "".join(run.text for run in p.runs)
                # Replace placeholder
                new_text = full_text.replace(placeholder, str(value))

                # Clear all runs except the first
                for run in p.runs[1:]:
                    p._element.remove(run._element)

                # Replace text in the first run
                p.runs[0].text = new_text

    # Write to Noting
    replace_placeholder(doc_noting, "{{PUC_SUBJECT}}", puc_subject)
    replace_placeholder(doc_noting, "{{FIRST_PARA}}", first_para)
    replace_placeholder(doc_noting, "{{MID_PARA}}", mid_para)
    replace_placeholder(doc_noting, "{{LAST_PARA}}", last_para)
    # Save noting file
    output_path = os.path.join(OUTPUT_FOLDER, f"{file_number.replace('/','-')} {puc_subject}- Noting.docx")
    doc_noting.save(output_path)
    # Write to Draft
    replace_placeholder(doc_draft, "{{FILE_NUMBER}}", file_number)
    replace_placeholder(doc_draft, "{{BRANCH_CFMS_DATE}}", branch_cfms_date.strftime("%d.%m.%Y"))
    replace_placeholder(doc_draft, "{{PUC_SUBJECT}}", puc_subject)
    # Save draft file
    output_path = os.path.join(OUTPUT_FOLDER, f"{file_number.replace('/','-')} {puc_subject}- Draft.docx")
    doc_draft.save(output_path)

    st.success(f"Document saved:\n {output_path}")