from kedro.pipeline import Pipeline, node

from .preprocessing_forecast_orders import (
    apply_add_random_op,
    apply_split_quantity,
    explode_quantities,
    rename_and_select_columns,
    process_product_description,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=apply_add_random_op,
                inputs="order_forecast",
                outputs="raw_orders",
                name="apply_add_random_op",
            ),
            node(
                func=apply_split_quantity,
                inputs="raw_orders",
                outputs="orders_with_quantities",
                name="apply_split_quantity_node",
            ),
            node(
                func=explode_quantities,
                inputs="orders_with_quantities",
                outputs="exploded_orders",
                name="explode_quantities_node",
            ),
            node(
                func=rename_and_select_columns,
                inputs="exploded_orders",
                outputs="renamed_orders",
                name="rename_and_select_columns_node",
            ),
            node(
                func=process_product_description,
                inputs=["renamed_orders", "params:preprocess_options"],
                outputs="croom_processed_forecast_orders",
                name="process_product_description_node",
            ),
        ]
    )
