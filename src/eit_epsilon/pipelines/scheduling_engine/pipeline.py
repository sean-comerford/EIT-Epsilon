from kedro.pipeline import Pipeline, node, pipeline

from .nodes import JobShop, create_chart, save_chart_to_html, reformat_output, mock_genetic_algorithm


def create_pipeline(**kwargs) -> Pipeline:

    # Instantiate a jobshop object
    jobshop = JobShop()

    # Create a pipeline with the jobshop object
    pipeline = Pipeline([
        node(
            func=jobshop.preprocess_orders,
            inputs='croom_open_orders',
            outputs='croom_processed_orders',
            name='preprocess_orders'
        ),
        node(
            func=jobshop.preprocess_cycle_times,
            inputs='monza_cycle_times',
            outputs=['ps_cycle_times', 'cr_cycle_times'],
            name='preprocess_cycle_times'
        ),
        node(
            func=jobshop.build_ga_representation,
            inputs=['croom_processed_orders', 'cr_cycle_times', 'ps_cycle_times',
                    'params:machine_qty_dict', 'params:task_to_machines'],
            outputs='input_repr_dict',
            name='build_ga_representation'
        ),
        node(
            func=mock_genetic_algorithm,
            inputs='input_repr_dict',
            outputs='best_schedule',
            name='mock_genetic_algorithm',
        ),
        node(
            func=reformat_output,
            inputs=['croom_processed_orders', 'best_schedule', 'params:scheduling_options'],
            outputs='final_schedule_alt',
            name='reformat_output',
        ),
        node(
            func=create_chart,
            inputs=['final_schedule_alt', 'params:visualization_options'],
            outputs='gantt_chart_json',
            name='schedule_chart',
        ),
        node(
            func=save_chart_to_html,
            inputs='gantt_chart_json',
            outputs=None,
            name='save_chart_to_html',
        ),
    ])

    return pipeline
