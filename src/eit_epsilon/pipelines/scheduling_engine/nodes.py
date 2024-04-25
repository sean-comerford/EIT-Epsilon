import pandas as pd
import datetime
from eit_epsilon.scheduling_engine.heuristics import EarliestDueDate
from eit_epsilon.scheduling_engine.data_loading import JobShop


def load_jobs(order_book: pd.DataFrame, parameters: dict):
    js = JobShop(order_book)
    js.load_jobs_from_orderbook(parameters['column_mapping'], parameters['change_over_time'])
    js.filter_jobs()
    js.get_rand_processing_times(parameters['min_processing_time_per_item'], parameters['max_processing_time_per_item'])
    jobs_df = js.return_jobs_as_df()
    return jobs_df


def schedule_earliest_due_date(jobs_df: pd.DataFrame, parameters: dict):
    initial_schedule, remaining_orders = EarliestDueDate.get_starting_jobs(jobs_df, parameters)
    job_schedule = EarliestDueDate.fill_schedule(remaining_orders, initial_schedule)
    return job_schedule




# def order_generator(df: pd.DataFrame):
#     """Generate a new oder from the dataframe each time the generator is called"""
#     df.sort_values('Due_date')
#     for index,row in df.sort_values('Due_date').iterrows():
#         yield(index)
#
# def get_starting_jobs(df: pd.DataFrame, parameters: dict):
#     """Get a starting schedule by putting the orders with the earliest due date on the machines"""
#     starting_time = datetime.datetime.strptime(parameters["start_date"], "%Y-%m-%d")
#     og = order_generator(df)
#     job_schedule = pd.DataFrame(columns=df.columns)
#     for machine in range(0, parameters["machines"]):
#         idx = next(og)
#         job_schedule = pd.concat([job_schedule, pd.DataFrame(df.loc[[idx]])])
#         job_schedule.at[idx,"Machine"] = machine+1
#         job_schedule.at[idx,"Order_in_sequence"] = 1
#         df = df.loc[~df.index.isin([idx])]
#
#     job_schedule["Start_time"] = starting_time
#     job_schedule["End_time"] = job_schedule.apply(lambda row: row["Start_time"] + datetime.timedelta(days=row["Time"]), axis = 1)
#     job_schedule["Lead_time"] = job_schedule.apply(lambda row: (row["End_time"] - row["Order_date"]).days, axis = 1)
#     return job_schedule, df
#
#
# def fill_schedule(df: pd.DataFrame, job_schedule: pd.DataFrame):
#     """Fill a schedule by continuously selecting the order with the earliest due date"""
#     remaining_orders = df.copy()
#     og = order_generator(remaining_orders)
#     while len(remaining_orders) > 0:
#         finished_job = job_schedule.loc[[
#             job_schedule.loc[list(job_schedule.groupby('Machine')['Order_in_sequence'].idxmax())]["End_time"].idxmin()]]
#         idx = next(og)
#
#         job_schedule = pd.concat([job_schedule, pd.DataFrame(remaining_orders.loc[[idx]])], ignore_index=True)
#         job_in_schedule_index = job_schedule.iloc[[-1]].index.values[0]
#         job_schedule.at[job_in_schedule_index, "Machine"] = finished_job["Machine"].iloc[0]
#         job_schedule.at[job_in_schedule_index, "Order_in_sequence"] = finished_job["Order_in_sequence"].iloc[0] + 1
#         if finished_job["Product"].iloc[0] == job_schedule.iloc[-1]["Product"]:
#             change_over_time = datetime.timedelta(days=0)
#         else:
#             change_over_time = datetime.timedelta(days=int(job_schedule.loc[job_in_schedule_index]["Change-over time"]))
#         job_schedule.at[job_in_schedule_index, "Start_time"] = finished_job["End_time"].iloc[0] + change_over_time
#         job_schedule.at[job_in_schedule_index, "End_time"] = job_schedule.loc[job_in_schedule_index]["Start_time"] + datetime.timedelta(
#             days=int(job_schedule.loc[job_in_schedule_index]["Time"]))
#         job_schedule.at[job_in_schedule_index, "Lead_time"] = (
#                     job_schedule.loc[job_in_schedule_index]["End_time"] - job_schedule.loc[job_in_schedule_index]["Order_date"]).days
#         remaining_orders.drop(idx, inplace=True)
#         print(f"Schedule with {len(remaining_orders)} orders remaining")
#         print(job_schedule)
#     return job_schedule
