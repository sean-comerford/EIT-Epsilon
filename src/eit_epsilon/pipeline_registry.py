"""Project pipelines."""
from pathlib import Path
from typing import Dict

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.pipeline import Pipeline, pipeline

import eit_epsilon.pipelines.energy_forecasting.data_extraction.energy_production as extract_energy_production
import eit_epsilon.pipelines.energy_forecasting.data_extraction.weather as extract_weather
import eit_epsilon.pipelines.energy_forecasting.data_funnel as data_funnel
import eit_epsilon.pipelines.energy_forecasting.exploration as exploration
import eit_epsilon.pipelines.energy_forecasting.modeling as modeling
import eit_epsilon.pipelines.energy_forecasting.preprocessing as preprocessing
import eit_epsilon.pipelines.scheduling_engine as scheduling_engine

conf_path = str(Path.cwd() / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
parameters = conf_loader["parameters"]

data_to_use = parameters["energy_production_data_to_use"]


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()

    energy_production_pipeline = (
        extract_energy_production.create_pipeline()
        + preprocessing.create_pipeline_preprocess_energy_production()
    )

    weather_stations_pipeline = (
        extract_weather.create_pipeline_weather_stations()
        + data_funnel.create_pipeline_linking_weather_stations()
    )

    weather_measurements_pipeline = (
        extract_weather.create_pipeline_measurements()
        + preprocessing.create_pipeline_preprocess_weather_data()
    )

    weather_measurements_cz_pipeline = extract_weather.create_pipeline_measurements_czechia()

    modeling_training_pipeline = modeling.create_pipeline_training_model()

    weather_forecasts_pipeline = (
        extract_weather.create_pipeline_forecasting()
        + preprocessing.create_pipeline_preprocess_weather_forecasts()
    )

    modeling_testing_pipeline = modeling.create_pipeline_testing_model()

    scheduling_engine_pipeline = (scheduling_engine.create_pipeline())

    weather_forecasts_testing_pipeline = pipeline(
        pipe=weather_forecasts_pipeline,
        parameters={
            "params:start_date": "params:testing.start_date",
            "params:stop_date": "params:testing.stop_date",
            "params:locations": "params:testing.locations",
            "params:pilot_locations_coordinates": "params:pilot_locations_coordinates",
        },
        outputs={
            "weather_forecasts": "hourly_weather_forecasts_for_testing",
            "closest_coordinates": "closest_coordinates_testing",
            "preprocessed_weather_forecasts": "preprocessed_weather_forecasts_for_testing",
        },
        namespace="weather_forecasts_testing",
    )

    modeling_forecasting_pipeline = modeling.create_pipeline_forecasting_model()
    exploration_pipeline = exploration.create_pipeline()
    exploration_pipeline_czechia = exploration.create_pipeline_predictions()
    data_funnel_forecasting_pipeline = data_funnel.create_pipeline_weather_forecasts()

    # Day ahead forecasting pipelines
    # Model training day ahead
    data_funnel_training_and_testing_pipeline = (
        data_funnel.create_pipeline_merging_production_weather_data()
    )

    day_ahead_forecasting_pipeline = pipeline(
        pipe=weather_forecasts_pipeline
        + data_funnel_forecasting_pipeline
        + modeling_forecasting_pipeline,
        parameters={
            "params:pilot_locations_coordinates": "params:pilot_locations_coordinates",
            "params:meta_data": "params:meta_data",
            "params:start_date": "params:day_ahead_forecasting_pipeline.forecasting.start_date",
            "params:stop_date": "params:day_ahead_forecasting_pipeline.forecasting.stop_date",
            "params:locations": "params:day_ahead_forecasting_pipeline.forecasting.locations",
            "params:feature_engineering_selection": "params:day_ahead_model_training.feature_engineering_selection",
            "params:data_interpolation": "params:day_ahead_model_training.data_interpolation",
        },
        inputs={"trained_model": "trained_model_day_ahead"},
        outputs={
            "forecasting_data": "forecasting_day_ahead_data",
            "forecasting_predictions": "forecasting_day_ahead_predictions",
            "closest_coordinates": "closest_coordinates_day_ahead_forecasting",
            "selected_models": "selected_day_ahead_models",
        },
        namespace="day_ahead_forecasting_pipeline",
    )


    czechia_prediction_pipeline = pipeline(
        pipe=weather_measurements_cz_pipeline,
        parameters={"params:ecmwf_grib_file", "params:pilot_locations_coordinates",
                    "params:czechia_prediction_pipeline"},
        outputs={
            "weather_measurements": "weather_measurements_czechia",
            "preprocessed_weather_forecasts": "preprocessed_weather_czechia",
        },
        namespace="czechia_prediction_pipeline",
    ) + pipeline(
        pipe=data_funnel_forecasting_pipeline + modeling_forecasting_pipeline + exploration_pipeline_czechia,
        parameters={
            "params:pilot_locations_coordinates": "params:pilot_locations_coordinates",
            "params:meta_data": "params:meta_data",
            "params:feature_engineering_selection": "params:day_ahead_model_training.feature_engineering_selection",
        },
        inputs={
            "trained_model": "trained_model_day_ahead",
            "preprocessed_weather_forecasts": "preprocessed_weather_czechia",
            "energy_consumption": "voestalpine_consumption"
        },
        outputs={
            "forecasting_data": "forecasting_day_ahead_data",
            "forecasting_predictions": "forecasting_day_ahead_predictions",
            "selected_models": "selected_day_ahead_models",
        },
        namespace="czechia_prediction_pipeline",
    )


    # Model forecasting day ahead
    day_ahead_model_training_pipeline = pipeline(
        pipe=energy_production_pipeline
        + weather_stations_pipeline
        + weather_measurements_pipeline
        + data_funnel_training_and_testing_pipeline
        + weather_forecasts_testing_pipeline
        + modeling_training_pipeline
        + modeling_testing_pipeline
        + exploration_pipeline,
        parameters={
            "params:knmi.weather_stations.start": "params:knmi.weather_stations.start",
            "params:knmi.weather_stations.end": "params:knmi.weather_stations.end",
            "params:meta_data": "params:meta_data",
            "params:pilot_locations_coordinates": "params:pilot_locations_coordinates",
            "params:knmi.hourly_measurements.variables": "params:knmi.hourly_measurements.variables",
            "params:weather_forecasts_testing.data_interpolation": "params:day_ahead_model_training.data_interpolation",
            "params:energy_production_data_to_use": "params:energy_production_data_to_use",
        },
        inputs={
            "data_to_use": data_to_use,
        },
        outputs={
            "trained_model": "trained_model_day_ahead",
            "training_data": "training_day_ahead_data",
            "testing_data": "testing_day_ahead_data",
            "permuted_feature_importance_training": "permuted_feature_importance_training_day_ahead",
            "preprocessed_energy_production_data": "preprocessed_energy_production_data",
            "solar_energy_production_per_location": "solar_energy_production_per_location",
            "performance_testing_using_measurements": "performance_day_ahead_testing_using_measurements",
            "performance_testing_using_forecasts": "performance_day_ahead_testing_using_forecasts",
            "weather_stations": "weather_stations",
            "nearest_weather_stations": "nearest_weather_stations",
            "unique_weather_stations": "unique_weather_stations",
            "weather_measurements": "weather_measurements",
            "production_and_weather_data": "production_and_weather_data_nl"
        },
        namespace="day_ahead_model_training",
    )
    pipeline_scheduling_engine = pipeline(
        pipe=scheduling_engine_pipeline
    )

    pipelines = {
        # TODO: remove/replace old pipelines when czechia stable
        "__default__": day_ahead_model_training_pipeline + day_ahead_forecasting_pipeline,
        # "__default__": day_ahead_model_training_pipeline,
        "modeling_training_day_ahead": day_ahead_model_training_pipeline,
        # "forecasting_day_ahead": day_ahead_forecasting_pipeline,
        # "day_ahead": day_ahead_model_training_pipeline + day_ahead_forecasting_pipeline,
        "czechia": czechia_prediction_pipeline,
        'scheduling_engine': pipeline_scheduling_engine
    }

    # pipelines["__default__"] = sum(pipelines.values())
    return pipelines
