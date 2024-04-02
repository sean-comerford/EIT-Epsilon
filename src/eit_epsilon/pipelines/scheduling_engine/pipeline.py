from kedro.pipeline import Pipeline, node, pipeline
import datetime

from .nodes import order_generator,get_starting_jobs, fill_schedule


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_starting_jobs,
                inputs=["orders", "params:scheduling_options"],
                outputs=["starting_schedule", "remaining_orders"],
                name="initialize_schedule",
            ),
            node(
                func=fill_schedule,
                inputs=["remaining_orders", "starting_schedule"],
                outputs="final_schedule",
                name="finalize_schedule",
            ),
        ]
    )