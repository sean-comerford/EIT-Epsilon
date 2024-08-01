from kedro.pipeline import Pipeline, node

from .nodes import (
    JobShop,
    create_chart,
    save_chart_to_html,
    reformat_output,
    GeneticAlgorithmScheduler,
    create_start_end_time,
)


def create_pipeline(**kwargs) -> Pipeline:
    # Instantiate a jobshop object
    jobshop = JobShop()
    genetic_algorithm = GeneticAlgorithmScheduler()

    # Create a pipeline with the jobshop object

    pipeline = Pipeline(
        [
            node(
                func=jobshop.preprocess_orders,
                inputs="croom_open_orders",
                outputs="croom_processed_orders",
                name="preprocess_orders",
            ),
            node(
                func=jobshop.preprocess_cycle_times,
                inputs=["monza_cycle_times_op1", "monza_cycle_times_op2"],
                outputs=["ps_cycle_times", "cr_cycle_times", "op2_cycle_times"],
                name="preprocess_cycle_times",
            ),
            node(
                func=jobshop.build_ga_representation,
                inputs=[
                    "croom_processed_orders",
                    "cr_cycle_times",
                    "ps_cycle_times",
                    "op2_cycle_times",
                    "params:task_to_machines",
                    "params:scheduling_options",
                ],
                outputs="input_repr_dict",
                name="build_ga_representation",
            ),
            node(
                func=jobshop.build_changeover_compatibility,
                inputs=["croom_processed_orders", "params:size_categories_op2"],
                outputs="compatibility_dict",
                name="build_changeover_compatibility",
            ),
            node(
                func=genetic_algorithm.run,
                inputs=[
                    "input_repr_dict",
                    "params:scheduling_options",
                    "compatibility_dict",
                ],
                outputs=["best_schedule", "best_scores"],
                name="mock_genetic_algorithm",
            ),
            node(
                func=reformat_output,
                inputs=[
                    "croom_processed_orders",
                    "best_schedule",
                    "params:column_mapping_reformat",
                    "params:machine_dict",
                ],
                outputs="croom_reformatted_orders",
                name="reformat_output",
            ),
            node(
                func=create_start_end_time,
                inputs=["croom_reformatted_orders", "params:scheduling_options"],
                outputs="final_schedule",
                name="create_start_end_time",
            ),
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

    return pipeline
