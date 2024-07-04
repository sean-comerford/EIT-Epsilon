"""Project pipelines."""

from typing import Dict
from kedro.pipeline import Pipeline

try:
    from eit_epsilon.pipelines_scheduling_engine import (
        register_pipelines_scheduling_engine,
    )

    scheduling_engine_available = True
except ImportError:
    scheduling_engine_available = False

# Try to import the energy forecasting pipelines, if available
try:
    from eit_epsilon.pipelines_energy_forecasting import (
        register_pipelines_energy_forecasting,
    )

    energy_forecasting_available = True
except ImportError:
    energy_forecasting_available = False


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    This function registers pipelines from the energy forecasting code or the scheduling engine code or both,
    depending on what is available.

    The function first checks if the energy forecasting and scheduling engine pipelines are available. If they are,
    it imports and registers the respective pipelines. The registered pipelines are then returned as a mapping from
    pipeline names to Pipeline objects.

    Args:
        None

    Returns:
        A mapping from pipeline names to Pipeline objects.

    Raises:
        ImportError: If either the energy forecasting or scheduling engine pipelines are not available.
    """
    pipelines = {}

    if energy_forecasting_available:
        energy_forecasting_pipeline_dict = register_pipelines_energy_forecasting()
        pipelines.update(energy_forecasting_pipeline_dict)

    if scheduling_engine_available:
        scheduling_engine_pipeline_dict = register_pipelines_scheduling_engine()
        pipelines.update(scheduling_engine_pipeline_dict)

    return pipelines
