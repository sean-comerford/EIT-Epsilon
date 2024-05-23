from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_jobs, schedule_earliest_due_date, create_chart, save_chart_to_html

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=load_jobs,
            #     inputs=["croom_open_orders", "params:scheduling_options"],
            #     outputs="job_list",
            #     name="load_jobs",
            # ),
            # node(
            #     func=schedule_earliest_due_date,
            #     inputs=["job_list", "params:scheduling_options"],
            #     outputs="final_schedule",
            #     name="schedule_earliest_due_date",
            # ),
            node(
                func=create_chart,
                inputs=["final_schedule", "params:visualization_options"],
                outputs="gantt_chart_json",
                name="schedule_chart",
            ),
            node(
                func=save_chart_to_html,
                inputs="gantt_chart_json",
                outputs=None,
                name="save_chart_to_html",
            ),
        ]
    )
