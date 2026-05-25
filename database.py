from supabase import create_client
from dotenv import load_dotenv
import os


load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


# =====================================================
# GET TASKS
# =====================================================
def get_tasks():

    response = (
        supabase.table("tasks")
        .select("*")
        .order("scheduled_date")
        .execute()
    )

    return response.data


# =====================================================
# ADD TASK
# =====================================================
def add_task(
    task_title,
    scheduled_date
):

    supabase.table("tasks").insert({

        "task_title": task_title,

        "scheduled_date":
            str(scheduled_date),

        "status": "Pending"

    }).execute()


# =====================================================
# UPDATE STATUS
# =====================================================
def update_status(
    task_id,
    status
):

    supabase.table("tasks").update({

        "status": status

    }).eq(
        "id",
        task_id
    ).execute()


# =====================================================
# DELETE TASK
# =====================================================
def delete_task(task_id):

    supabase.table("tasks").delete().eq(
        "id",
        task_id
    ).execute()