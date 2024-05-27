import os
from pathlib import Path
import datetime

import pandas as pd
from pandas.api.types import is_string_dtype
import plotly

from eit_epsilon.scheduling_engine.heuristics import EarliestDueDate
from eit_epsilon.scheduling_engine.data_loading import JobShop


def load_jobs(order_book: pd.DataFrame, parameters: dict):
    js = JobShop(order_book)
    js.load_jobs_from_orderbook(parameters["column_mapping"], parameters["change_over_time"])
    js.filter_jobs()
    js.get_rand_processing_times(
        parameters["min_processing_time_per_item"], parameters["max_processing_time_per_item"]
    )
    jobs_df = js.return_jobs_as_df()
    return jobs_df


def schedule_earliest_due_date(jobs_df: pd.DataFrame, parameters: dict):
    initial_schedule, remaining_orders = EarliestDueDate.get_starting_jobs(jobs_df, parameters)
    job_schedule = EarliestDueDate.fill_schedule(remaining_orders, initial_schedule)
    return job_schedule


def create_chart(schedule: pd.DataFrame, parameters: dict):
    if not is_string_dtype(schedule[[parameters["column_mapping"]["Resource"]]]):
        schedule[parameters["column_mapping"]["Resource"]] = schedule[parameters["column_mapping"]["Resource"]].apply(str)
    schedule = schedule.rename(columns=
        parameters["column_mapping"]
    )
    schedule['Late'] = schedule['Due_date'] < schedule['End_time']
    return schedule


#TODO: Create custom data catalog item to store html file using kedro data catalog
def save_chart_to_html(gantt_chart):
    filepath = Path(os.getcwd()) / 'data/08_reporting/gantt_chart.html'
    plotly.offline.plot(gantt_chart, filename=str(filepath))
