from kedro.pipeline import Pipeline, node

from .nodes import (
    JobShop,
    create_chart,
    create_mix_charts,
    save_charts_to_html,
    reformat_output,
    identify_changeovers,
    GeneticAlgorithmScheduler,
    create_start_end_time,
    calculate_kpi,
    order_to_id,
    split_and_save_schedule,
    output_schedule_per_machine,
    reorder_jobs_by_starting_time,
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
                inputs=[
                    "croom_open_orders",
                ],
                outputs="croom_processed_orders",
                name="preprocess_orders",
            ),
            node(
                func=jobshop.build_ga_representation,
                inputs=[
                    "croom_processed_orders",
                    "timecards",
                    "croom_task_durations",
                    "params:task_to_machines",
                    "params:task_to_names",
                    "params:scheduling_options",
                    "params:machine_dict",
                    "params:timecard_ctd_mapping",
                    "params:timecard_op1_mapping",
                    "params:timecard_op2_mapping",
                    "params:manual_HAAS_starting_part_ids",
                ],
                outputs=["input_repr_dict", "HAAS_starting_part_ids"],
                name="build_ga_representation",
            ),
            node(
                func=jobshop.build_changeover_compatibility,
                inputs=[
                    "croom_processed_orders",
                    "params:size_categories_op2_cr",
                    "params:size_categories_op2_ps",
                ],
                outputs="compatibility_dict",
                name="build_changeover_compatibility",
            ),
            node(
                func=jobshop.generate_arbor_mapping,
                inputs=[
                    "input_repr_dict",
                    "params:cemented_arbors",
                    "params:cementless_arbors",
                    "HAAS_starting_part_ids",
                ],
                outputs="arbor_dict",
                name="generate_arbor_mapping",
            ),
            node(
                func=genetic_algorithm.run,
                inputs=[
                    "input_repr_dict",
                    "params:scheduling_options",
                    "compatibility_dict",
                    "arbor_dict",
                    "params:ghost_machine_dict",
                    "params:cemented_arbors",
                    "params:arbor_quantities",
                    "HAAS_starting_part_ids",
                ],
                outputs=["best_schedule", "best_scores"],
                name="genetic_algorithm",
            ),
            node(
                func=reformat_output,
                inputs=[
                    "croom_processed_orders",
                    "best_schedule",
                    "params:column_mapping_reformat",
                    "params:machine_dict",
                    "params:task_to_names",
                ],
                outputs="croom_reformatted_orders",
                name="reformat_output",
            ),
            node(
                func=identify_changeovers,
                inputs=[
                    "croom_reformatted_orders",
                    "params:scheduling_options",
                ],
                outputs="changeovers",
                name="identify_changeovers",
            ),
            node(
                func=create_start_end_time,
                inputs=["croom_reformatted_orders", "changeovers", "params:scheduling_options"],
                outputs=["final_schedule", "final_changeovers"],
                name="create_start_end_time",
            ),
            node(
                func=calculate_kpi,
                inputs="final_schedule",
                outputs="kpi_results",
                name="calculate_kpi",
            ),
            node(
                func=order_to_id,
                inputs=["mapping_dict_read", "final_schedule"],
                outputs=["mapping_dict_write", "final_schedule_with_id"],
                name="order_to_id",
            ),
            node(
                func=reorder_jobs_by_starting_time,
                inputs=["final_schedule_with_id", "params:scheduling_options", "params:machine_dict"],
                outputs="final_schedule_reordered",
                name="reorder_jobs_by_starting_time",
            ),
            node(
                func=create_chart,
                inputs=[
                    "final_schedule_reordered",
                    "params:visualization_options",
                    "params:scheduling_options",
                ],
                outputs="gantt_chart_json",
                name="schedule_chart",
            ),
            node(
                func=create_mix_charts,
                inputs="final_schedule_reordered",
                outputs=[
                    "op_mix_by_date_excel",
                    "op_mix_by_date_chart_json",
                    "op_mix_by_week_excel",
                    "op_mix_by_week_chart_json",
                    "part_mix_by_week_excel",
                    "part_mix_by_week_chart_json",
                ],
                name="schedule_op_chart",
            ),
            node(
                func=save_charts_to_html,
                inputs=[
                    "gantt_chart_json",
                    "op_mix_by_date_chart_json",
                    "op_mix_by_week_chart_json",
                    "part_mix_by_week_chart_json",
                ],
                outputs=None,
                name="save_chart_to_html",
            ),
            node(
                func=split_and_save_schedule,
                inputs="final_schedule_reordered",
                outputs=[
                    "ctd_mapping",
                    "op1_mapping",
                    "op2_mapping",
                ],
                name="split_and_save_schedule",
            ),
            node(
                func=output_schedule_per_machine,
                inputs=["final_schedule_reordered", "params:task_to_names"],
                outputs="machine_schedules",
                name="output_schedule_per_machine",
            ),
        ]
    )

    return pipeline
