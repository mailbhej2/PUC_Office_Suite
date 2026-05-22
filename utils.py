import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

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
                    "Reminder Date"
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

            st.subheader(f"Reminders {len(df)}")

            st.dataframe(
                df_filtered[
                    [
                        "File No.",
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



