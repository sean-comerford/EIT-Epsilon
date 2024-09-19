"""Project pipelines."""

from pathlib import Path
from typing import Dict

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.pipeline import Pipeline, pipeline

import eit_epsilon.pipelines.scheduling_engine as scheduling_engine
import eit_epsilon.pipelines.scheduling_engine.forecast_orders as forecast_orders

conf_path = str(Path.cwd() / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
parameters = conf_loader["parameters"]


def register_pipelines_scheduling_engine() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    scheduling_engine_pipeline = scheduling_engine.create_pipeline()
    forecast_orders_pipeline = forecast_orders.create_pipeline()

    pipeline_scheduling_engine = pipeline(pipe=scheduling_engine_pipeline)
    pipeline_forecast_orders = pipeline(pipe=forecast_orders_pipeline)

    pipelines = {
        "__default__": pipeline_scheduling_engine,
        "preprocess_forecast": pipeline_forecast_orders,
    }
    return pipelines
