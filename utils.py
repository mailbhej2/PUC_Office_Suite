import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from database import get_tasks, add_task, update_status, delete_task

def task_planner():

        st.markdown(
            "##### Add New Task"
        )

        c1, c2, c3 = st.columns([5, 2, 1])

        task_title = c1.text_input(
            "Task Title",
            label_visibility="collapsed",
            placeholder="Enter task..."
        )

        scheduled_date = c2.date_input(
            "Scheduled Date",
            label_visibility="collapsed"
        )

        save_clicked = c3.button(
            "Save",
            use_container_width=True
        )

        if save_clicked:

            if task_title.strip():
                add_task(
                    task_title,
                    scheduled_date
                )

                st.rerun()

        st.markdown("---")

        st.markdown(
            "##### Pending Tasks"
        )

        tasks = get_tasks()

        if not tasks:

            st.caption(
                "No pending tasks available."
            )

        else:

            for task in tasks:

                with st.container(border=True):

                    c1, c2, c3, c4 = st.columns(
                        [0.5, 7, 2, 0.6]
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

                    c2.markdown(
                        f"""
                        <div style="
                            font-size:14px;
                            padding-top:2px;
                        ">
                            {task['task_title']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    c3.caption(
                        task["scheduled_date"]
                    )

                    if c4.button(
                            "🗑",
                            key=f"del_{task['id']}",
                            use_container_width=True
                    ):
                        delete_task(
                            task["id"]
                        )

                        st.rerun()



def get_reminders():
    # =========================================================
    # REMINDER TABLE
    # =========================================================
    try:

        load_dotenv()

        google_sheet_url = os.getenv(
            "GOOGLE_SHEET_URL"
        )

        if google_sheet_url:
            df = pd.read_csv(
                google_sheet_url
            )

            df["Reminder Date"] = pd.to_datetime(
                df["Reminder Date"],
                errors="coerce",
                dayfirst=True
            )

            start_date = (
                    pd.Timestamp("today")
                    - pd.Timedelta(days=5)
            )

            end_date = (
                    pd.Timestamp("today")
                    + pd.Timedelta(days=15)
            )

            df_filtered = df[
                df["Reminder Date"].between(
                    start_date,
                    end_date
                )
            ][
                [
                    "File No.",
                    "Subject",
                    "Reminder Date",
                    "Last Dealt On"
                ]
            ]

            df_filtered = df_filtered.sort_values(
                "Reminder Date"
            )

            df_filtered["Reminder Date"] = (
                df_filtered["Reminder Date"]
                .dt.strftime("%d-%m-%y")
            )

            df_filtered.index += 2

            df_filtered.index.name = "#"
            st.markdown(
                f"<h4 style='font-size:20px;'>Reminders ({len(df_filtered)})</h4>",
                unsafe_allow_html=True
            )

            st.dataframe(
                df_filtered[
                    [
                        "File No.",
                        "Last Dealt On",
                        "Reminder Date",
                        "Subject"
                    ]
                ],
                use_container_width=True
            )

    except:
        pass

# ----------------------- Place holders ----------------------------------

def replace_placeholder(doc, data):

    for p in doc.paragraphs:

        full_text = "".join(
            run.text for run in p.runs
        )

        for key, value in data.items():

            full_text = full_text.replace(
                key,
                str(value)
            )

        if p.runs:

            p.runs[0].text = full_text

            for run in p.runs[1:]:
                p._element.remove(
                    run._element
                )


