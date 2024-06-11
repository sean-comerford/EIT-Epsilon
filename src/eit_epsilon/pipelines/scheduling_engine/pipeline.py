from kedro.pipeline import Pipeline, node, pipeline

from nodes import JobShop


def create_pipeline(**kwargs) -> Pipeline:

    jobshop = JobShop()

    pipeline = Pipeline([
        node(
            func=jobshop.load_jobs,
            inputs="croom_open_orders",
            outputs="job_list",
            name="load_jobs",
        ),
        node(
            func=jobshop.schedule_earliest_due_date,
            inputs='jobs_df',
            outputs='schedule',
            name='schedule_earliest_due_date_node'
        ),
        node(
            func=jobshop.create_chart,
            inputs='schedule',
            outputs='chart',
            name='create_chart_node'
        ),
        node(
            func=jobshop.save_chart_to_html,
            inputs='chart',
            outputs=None,
            name='save_chart_to_html_node'
        ),
    ])

    return pipeline
