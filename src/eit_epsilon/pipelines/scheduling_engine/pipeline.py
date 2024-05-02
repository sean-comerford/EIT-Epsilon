from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_jobs, schedule_earliest_due_date
# from .nodes import get_starting_jobs, fill_schedule

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_jobs,
                inputs=["croom_open_orders", "params:scheduling_options"],
                outputs="job_list",
                name="load_jobs",
            ),
            node(
                func=schedule_earliest_due_date,
                inputs=["job_list", "params:scheduling_options"],
                outputs="final_schedule",
                name="schedule_earliest_due_date",
            ),
        ]
    )
