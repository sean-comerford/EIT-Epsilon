from kedro.pipeline import Pipeline, node

from .nodes import (
    JobShop,
    create_chart,
    create_op_mix,
    save_charts_to_html,
    reformat_output,
    identify_changeovers,
    GeneticAlgorithmScheduler,
    create_start_end_time,
    calculate_kpi,
    order_to_id,
    split_and_save_schedule,
    output_schedule_per_machine,
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
                func=jobshop.build_ga_representation,
                inputs=[
                    "croom_processed_orders",
                    "croom_task_durations",
                    "params:task_to_machines",
                    "params:scheduling_options",
                    "params:machine_dict",
                ],
                outputs="input_repr_dict",
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
                    "params:HAAS_starting_part_ids",
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
                    "params:HAAS_starting_part_ids",
                    "params:custom_tasks_dict",
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
                func=create_chart,
                inputs=["final_schedule", "params:visualization_options"],
                outputs="gantt_chart_json",
                name="schedule_chart",
            ),
            node(
                func=create_op_mix,
                inputs="final_schedule",
                outputs=["op_mix_excel", "op_mix_chart_json"],
                name="schedule_op_chart",
            ),
            node(
                func=save_charts_to_html,
                inputs=["gantt_chart_json", "op_mix_chart_json"],
                outputs=None,
                name="save_chart_to_html",
            ),
            node(
                func=order_to_id,
                inputs=["mapping_dict_read", "final_schedule", "croom_processed_orders"],
                outputs=["mapping_dict_write", "final_schedule_with_id"],
                name="order_to_id",
            ),
            node(
                func=split_and_save_schedule,
                inputs="final_schedule_with_id",
                outputs=[
                    "ctd_mapping",
                    "op1_mapping",
                    "op2_mapping",
                ],
                name="split_and_save_schedule",
            ),
            node(
                func=output_schedule_per_machine,
                inputs=["final_schedule_with_id", "params:task_to_names"],
                outputs="machine_schedules",
                name="output_schedule_per_machine",
            ),
        ]
    )

    return pipeline
