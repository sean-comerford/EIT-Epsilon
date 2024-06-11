from kedro.pipeline import Pipeline, node, pipeline

from .nodes import JobShop


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
    ])

    return pipeline
