import streamlit as st
from docx import Document
from datetime import date
from io import BytesIO
from style import local_css
from utils import get_reminders, replace_placeholder
from database import add_task, update_status, delete_task, get_tasks

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Deepak | PUC Office Suite"
)

st.markdown(
    f"<style>{local_css()}</style>",
    unsafe_allow_html=True
)

st.title("Deepak | PUC Office Suite")


# =========================================================
# TOP NAVIGATION
# =========================================================
nav1, nav2, nav3 = st.columns([6, 2, 2])

with nav2:
    if st.button(
        "📂 File Status",
        use_container_width=True
    ):
        st.switch_page(
            "pages/1_File_Status.py"
        )

with nav3:
    if st.button(
        "📊 Table Extractor",
        use_container_width=True
    ):
        st.switch_page(
            "pages/2_Table_Extractor.py"
        )


# =======================================================
# Task planner
# ===================================================
with st.expander(
    "Task Planner",
    expanded=True
):

    c1, c2 = st.columns(2)

    task_title = c1.text_input(
        "Task Title"
    )

    scheduled_date = c2.date_input(
        "Scheduled Date"
    )

    if st.button("Save Task"):

        if task_title.strip():

            add_task(
                task_title,
                scheduled_date
            )

            st.rerun()


    st.divider()

    st.subheader("Upcoming Tasks")

    tasks = get_tasks()


    if not tasks:

        st.info(
            "No pending tasks available."
        )

    else:

        for task in tasks:

            c1, c2, c3, c4 = st.columns(
                [0.5, 5, 2, 1]
            )

            checked = c1.checkbox(
                "",
                value=task["status"] == "Done",
                key=f"done_{task['id']}"
            )

            new_status = (
                "Done"
                if checked
                else "Pending"
            )

            if new_status != task["status"]:

                update_status(
                    task["id"],
                    new_status
                )

                st.rerun()

            c2.write(
                task["task_title"]
            )

            c3.write(
                task["scheduled_date"]
            )

            if c4.button(
                "Delete",
                key=f"del_{task['id']}"
            ):

                delete_task(
                    task["id"]
                )

                st.rerun()
# =========================================================
# REMINDERS
# =========================================================
with st.expander(
    "🔔 Upcoming Reminders",
    expanded=True
):

    get_reminders()


# =========================================================
# TYPE SELECTOR
# =========================================================
draft_mode = st.radio(
    "File Maker",
    ["General", "Medical"],
    horizontal=True
)


# =========================================================
# FORM
# =========================================================
with st.form("main_form"):

    # -----------------------------------------------------
    # COMMON FIELDS
    # -----------------------------------------------------
    c1, c2, c3 = st.columns(3)

    file_number = c1.text_input(
        "File Number"
    )

    branch_cfms_number = c2.text_input(
        "Branch CFMS No."
    )

    branch_cfms_date = c3.date_input(
        "Branch CFMS Date",
        value=date.today()
    )

    c4, c5, c6 = st.columns(3)

    puc_number = c4.text_input(
        "PUC No."
    )

    puc_date = c5.date_input(
        "PUC Date",
        value=date.today()
    )

    puc_sender = c6.text_input(
        "PUC Sender"
    )

    puc_subject = st.text_input(
        "PUC Subject"
    )


    # -----------------------------------------------------
    # GENERAL
    # -----------------------------------------------------
    if draft_mode == "General":

        draft_type = st.selectbox(
            "Draft Type",
            [
                "Single_Draft",
                "Multi_Draft",
                "Hindi_Draft",
                "UO_Draft"
            ],
            index=0
        )


    # -----------------------------------------------------
    # MEDICAL
    # -----------------------------------------------------
    else:

        c7, c8 = st.columns(2)

        claimant_name = c7.text_input(
            "Claimant Name"
        )

        claimant_office = c8.text_input(
            "Claimant Office"
        )

        c9, c10 = st.columns(2)

        patient_name = c9.text_input(
            "Patient Name"
        )

        relation_with_claimant = c10.text_input(
            "Relation With Claimant"
        )

        c11, c12 = st.columns(2)

        hospital_name = c11.text_input(
            "Hospital Name"
        )

        civil_surgeon = c12.text_input(
            "Civil Surgeon"
        )

        c13, c14 = st.columns(2)

        treatment_from = c13.date_input(
            "Treatment From",
            value=date.today()
        )

        treatment_to = c14.date_input(
            "Treatment To",
            value=date.today()
        )

        c15, c16 = st.columns(2)

        claim_amount = c15.text_input(
            "Claim Amount"
        )

        head = c16.text_input(
            "Head"
        )


    submitted = st.form_submit_button(
        "Generate Files"
    )


# =========================================================
# GENERATE FILES
# =========================================================
if submitted:

    # -----------------------------------------------------
    # TEMPLATE PATHS
    # -----------------------------------------------------
    if draft_mode == "General":

        noting_path = (
            "files/Noting.docx"
        )

        draft_path = (
            f"files/{draft_type}.docx"
        )

    else:

        noting_path = (
            "files/Medical_Noting.docx"
        )

        draft_path = (
            "files/Medical_Draft.docx"
        )


    # -----------------------------------------------------
    # LOAD DOCUMENTS
    # -----------------------------------------------------
    doc_noting = Document(
        noting_path
    )

    doc_draft = Document(
        draft_path
    )
    # -----------------------------------------------------
    # Common DATA
    # -----------------------------------------------------
    data = {

        "{{FILE_NUMBER}}":
            file_number,

        "{{BRANCH_CFMS_NUMBER}}":
            branch_cfms_number,

        "{{BRANCH_CFMS_DATE}}":
            branch_cfms_date.strftime(
                "%d.%m.%Y"
            ),

        "{{PUC_NUMBER}}":
            puc_number,

        "{{PUC_DATE}}":
            puc_date.strftime(
                "%d.%m.%Y"
            ),

        "{{PUC_SENDER}}":
            puc_sender,

        "{{PUC_SUBJECT}}":
            puc_subject
    }

    # -----------------------------------------------------
    # MEDICAL DATA
    # -----------------------------------------------------
    if draft_mode == "Medical":

        data.update({

            "{{CLAIMANT_NAME}}":
                claimant_name,

            "{{CLAIMANT_OFFICE}}":
                claimant_office,

            "{{PATIENT_NAME}}":
                patient_name,

            "{{RELATION_WITH_CLAIMANT}}":
                relation_with_claimant,

            "{{HOSPITAL_NAME}}":
                hospital_name,

            "{{CIVIL_SURGEON}}":
                civil_surgeon,

            "{{CLAIM_AMOUNT}}":
                claim_amount,

            "{{HEAD}}":
                head,

            "{{TREATMENT_FROM}}":
                treatment_from.strftime(
                    "%d.%m.%Y"
                ),

            "{{TREATMENT_TO}}":
                treatment_to.strftime(
                    "%d.%m.%Y"
                )
        })


    # -----------------------------------------------------
    # REPLACE PLACEHOLDERS
    # -----------------------------------------------------
    replace_placeholder(
        doc_noting,
        data
    )

    replace_placeholder(
        doc_draft,
        data
    )


    # -----------------------------------------------------
    # SAVE BUFFERS IN SESSION STATE
    # -----------------------------------------------------
    noting_buffer = BytesIO()

    doc_noting.save(
        noting_buffer
    )

    noting_buffer.seek(0)


    draft_buffer = BytesIO()

    doc_draft.save(
        draft_buffer
    )

    draft_buffer.seek(0)


    st.session_state.noting_buffer = (
        noting_buffer
    )

    st.session_state.draft_buffer = (
        draft_buffer
    )

    st.session_state.file_number = (
        file_number
    )

    st.session_state.puc_subject = (
        puc_subject
    )

    st.success(
        "Files Generated Successfully"
    )


# =========================================================
# DOWNLOAD BUTTONS
# =========================================================
if "noting_buffer" in st.session_state:

    st.download_button(
        label="Download Noting",
        data=st.session_state.noting_buffer.getvalue(),
        file_name=(
            f"{st.session_state.file_number.replace('/', '-')}"
            f" {st.session_state.puc_subject}-Noting.docx"
        ),
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
    )


if "draft_buffer" in st.session_state:

    st.download_button(
        label="Download Draft",
        data=st.session_state.draft_buffer.getvalue(),
        file_name=(
            f"{st.session_state.file_number.replace('/', '-')}"
            f" {st.session_state.puc_subject}-Draft.docx"
        ),
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
    )