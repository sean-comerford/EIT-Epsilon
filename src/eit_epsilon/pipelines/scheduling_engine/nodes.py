import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import random
import logging
from pandas.api.types import is_string_dtype
import plotly
from typing import List, Dict, Tuple, Union

# Instantiate logger
logger = logging.getLogger(__name__)


class Job:
    """
    The Job class contains methods for preprocessing and extracting information from open orders that need
    to be processed in a manufacturing workshop.
    """
    @staticmethod
    def filter_in_scope_op1(data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the data to include only in-scope operations for OP 1.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The filtered data.
        """
        # Debug statement
        logger.info(f"Total order data: {data.shape}")

        # Apply the filter
        inscope_data = data[(data['Part Description'].str.contains('OP 1') |
                            data['Part Description'].str.contains('ATT ')) &
                            (~data['On Hold?']) &
                            (~data['Part Description'].str.contains('OP 2'))]

        # Debug statement
        logger.info(f"In-scope data for OP 1: {inscope_data.shape}")

        return inscope_data

    @staticmethod
    def extract_info(data: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts type, size, and orientation from the part description.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The data with extracted information.
        """
        data = data.assign(
            Type=lambda x: x['Part Description'].apply(
                lambda y: 'CR' if 'CR' in y else ('PS' if 'PS' in y else '')),
            Size=lambda x: x['Part Description'].apply(
                lambda y: re.search(r'Sz (\d+N?)', y).group(1) if re.search(r'Sz (\d+N?)', y) else ''),
            Orientation=lambda x: x['Part Description'].apply(
                lambda y: 'LEFT' if 'LEFT' in y.upper() else ('RIGHT' if 'RIGHT' in y.upper() else '')),
            Cementless=lambda x: x['Part Description'].apply(
                lambda y: 'CLS' if 'CLS' in y.upper() else 'C')
        )

        # Debug statement
        if data[['Type', 'Size', 'Orientation']].isna().sum().sum() > 0:
            logger.warning(f"Data with extracted information: {data[['Type', 'Size', 'Orientation']].isna().sum()}")
        else:
            logger.info(f"No missing values in Type, Size, and Orientation columns")

        return data

    @staticmethod
    def check_part_id_consistency(data: pd.DataFrame) -> None:
        """
        Checks the consistency of Part IDs.

        Args:
            data (pd.DataFrame): The input data.

        Raises:
            LoggerError: If Part ID is not unique for every combination of Type, Size, and Orientation.
        """
        grouped = data.groupby('Part ID')[['Type', 'Size', 'Orientation']].nunique()

        if (grouped > 1).any().any():
            logger.error("[bold red blink]Part ID not unique for every combination of Type, Size, and Orientation[/]",
                         extra={"markup": True})
        else:
            logger.info(f"Part ID consistency check passed")

    @staticmethod
    def create_jobs_op1(data: pd.DataFrame) -> List[List[int]]:
        """
        Creates jobs representation for the GA, defined as J.
        Cemented products do not have to go through manual prep in OP 1.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            List[List[int]]: The list of jobs.
        """
        # Find proportion of cementless products
        cementless_count = data[data['Cementless'] == 'CLS'].shape[0]
        total_products = data.shape[0]

        cementless_percentage = (cementless_count / total_products) * 100
        logger.info(f"Proportion of cementless products: {cementless_percentage:.1f}%")

        # Populate J
        J = [[1, 2, 3, 4, 5, 6, 7] if cementless == 'CLS' else [1, 2, 3, 6, 7] for cementless in data['Cementless']]

        if not len(J) == len(data):
            logger.error("[bold red blink]J is not of the same length as processed orders![/]",
                         extra={"markup": True})

        # Debug statement
        logger.info(f"Snippet of Jobs for OP 1: {J[:2]}")

        return J


class Shop:
    """
    The Shop class contains methods for creating machine lists, compatibility matrices,
    duration matrices and due dates as input for a genetic algorithm.
    It creates a digital representation of the processes in and the setup of a manufacturing workshop.
    """
    @staticmethod
    def create_machines(machine_qty_dict: Dict[str, int]) -> List[int]:
        """
        Creates a list of machines based on the quantity dictionary.

        Args:
            machine_qty_dict (Dict[str, int]): The dictionary of machine quantities.

        Returns:
            List[int]: The list of machines.
        """
        total_machines = sum(machine_qty_dict.values())
        M = list(range(1, total_machines + 1))

        # Debug statement
        logger.info(f"Machine list (M): {M}")

        return M

    @staticmethod
    def get_compatibility(J: List[List[int]], task_to_machines: Dict[int, List[int]]) -> List[List[List[int]]]:
        """
        Gets the compatibility of tasks to machines.

        Example:
            compat = [[[1], [2, 3], [4, 5]],  -- Job 1, task 1 is only compatible with machine 1
                    [[1, 2],[2, 3],[4, 5]]]  -- Job 2, task 1 is compatible with machine 1 and 2

        Args:
            J (List[List[int]]): The list of jobs.
            task_to_machines (Dict[int, List[int]]): The dictionary mapping tasks to machines.

        Returns:
            List[List[List[int]]]: The compatibility list.
        """
        compat = []
        for job_tasks in J:
            job_compat = []
            for task in job_tasks:
                if task in task_to_machines:
                    machines = task_to_machines[task]
                else:
                    raise ValueError("Invalid task number!")
                job_compat.append(machines)
            compat.append(job_compat)

        # Debug statement
        logger.info(f"Snippet of compatability matrix: {compat[0]}")

        return compat

    @staticmethod
    def preprocess_cycle_times(cycle_times: pd.DataFrame, last_task_minutes: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the cycle times:
            1.) Remove empty rows and columns
            2.) Create new index starting from 1
            3.) Reduce column names to only the size
            4.) Fill the missing values in final inspection with 4 minutes
            5.) Split data for cruciate retaining and posterior stabilizing products

        Args:
            cycle_times (pd.DataFrame): The cycle times data.
            last_task_minutes (int, optional): The duration of the last task. Defaults to 4 minutes.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The preprocessed PS and CR cycle times.
        """
        cycle_times.columns = cycle_times.iloc[1]
        cycle_times = cycle_times.iloc[2:, 1:]
        cycle_times.index = range(1, len(cycle_times) + 1)

        def extract_number(s: str) -> str:
            match = re.search(r'\d+N?', s)
            return match.group(0) if match else s

        cycle_times.columns = [extract_number(col) for col in cycle_times.columns]
        cycle_times.loc[7] = cycle_times.loc[7].fillna(last_task_minutes)
        ps_times, cr_times = cycle_times.iloc[:, :cycle_times.shape[1]//2], cycle_times.iloc[:, cycle_times.shape[1]//2 + 1:]

        # Debug statement
        logger.info(f"PS times dim.: {ps_times.shape}, CR times dim.: {cr_times.shape}")

        return ps_times, cr_times

    @staticmethod
    def get_duration_matrix(J: List[List[int]], inscope_orders: pd.DataFrame, cr_times: pd.DataFrame, ps_times: pd.DataFrame) -> List[List[float]]:
        """
        Gets the duration matrix for the jobs.

        Example:
             dur = [[3, 2, 2],  -- Job 1; task 1, 2, 3 will take 3, 2, 2 minutes respectively
                    [3, 3, 3]]  -- Job 2; task 1, 2, 3 will take 3, 3, 3 minutes respectively

        Args:
            J (List[List[int]]): The list of jobs.
            inscope_orders (pd.DataFrame): The in-scope orders.
            cr_times (pd.DataFrame): The CR cycle times.
            ps_times (pd.DataFrame): The PS cycle times.

        Returns:
            List[List[float]]: The duration matrix.
        """
        dur = []
        for i, job in enumerate(J):
            job_dur = []
            for task in job:
                times = cr_times if inscope_orders.iloc[i]['Type'] == 'CR' else ps_times
                duration = round(times.loc[task, inscope_orders.iloc[i]['Size']] * inscope_orders.iloc[i]['Order Qty'], 1)
                job_dur.append(duration)
            dur.append(job_dur)

        # Debug statement
        logger.info(f"Snippet of duration: {dur[0]}")

        return dur

    @staticmethod
    def get_due_date(inscope_orders: pd.DataFrame, date: str = '2024-03-18') -> List[int]:
        """
        Gets the due dates for the in-scope orders.

        Args:
            inscope_orders (pd.DataFrame): The in-scope orders.
            date (str): The reference date.

        Returns:
            List[int]: The list of due dates in working minutes.
        """
        due = []
        for due_date in inscope_orders['Due Date ']:
            if pd.Timestamp(date) > due_date:
                working_days = -len(pd.bdate_range(due_date, date)) * 480
            else:
                working_days = len(pd.bdate_range(date, due_date)) * 480
            due.append(working_days)

        # Debug statement
        logger.info(f"Snippet of due: {due[:5]}")

        return due


class JobShop(Job, Shop):

    def preprocess_orders(self, croom_open_orders: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the open orders.

        Args:
            croom_open_orders (pd.DataFrame): The open orders.

        Returns:
            pd.DataFrame: The preprocessed orders.
        """
        inscope_data = (
            self.filter_in_scope_op1(croom_open_orders)
                .pipe(self.extract_info)
        )
        self.check_part_id_consistency(inscope_data)

        return inscope_data

    def build_ga_representation(self, croom_processed_orders: pd.DataFrame, cr_cycle_times: pd.DataFrame, ps_cycle_times: pd.DataFrame,
                                machine_qty_dict: Dict[str, int], task_to_machines: Dict[int, List[int]]) -> Dict[str, any]:
        """
        Builds the GA representation:
            J: List[List[int]]: The list of jobs, each job is a list of tasks.
            M: List[int]: The list of machines.
            compat: List[List[List[int]]]: The compatibility list.
            dur: List[List[float]]: The duration matrix.
            due: List[int]: The list of due dates in working minutes.

        Additionally, checks if all output variables are valid; they must be (nested) lists of integers/floats.

        Args:
            croom_processed_orders (pd.DataFrame): The processed orders.
            cr_cycle_times (pd.DataFrame): The CR cycle times.
            ps_cycle_times (pd.DataFrame): The PS cycle times.
            machine_qty_dict (Dict[str, int]): The machine quantity dictionary.
            task_to_machines (Dict[int, List[int]]): The task to machines dictionary.

        Returns:
            Dict[str, any]: The GA representation.
        """
        J = self.create_jobs_op1(croom_processed_orders)
        M = self.create_machines(machine_qty_dict)
        compat = self.get_compatibility(J, task_to_machines)
        dur = self.get_duration_matrix(J, croom_processed_orders, cr_cycle_times, ps_cycle_times)
        due = self.get_due_date(croom_processed_orders)

        def is_nested_list_of_numbers(lst):
            if isinstance(lst, list):
                return all(is_nested_list_of_numbers(item) if isinstance(item, list) else isinstance(item, (int, float)) for item in lst)
            return False

        input_repr_dict = {
            'J': J,
            'M': M,
            'compat': compat,
            'dur': dur,
            'due': due,
        }

        # Check if J, M, compat, dur, and due are (nested) lists of integers or floats
        for var_name, var in input_repr_dict.items():
            if not is_nested_list_of_numbers(var):
                logger.error(
                    f"[bold red blink]{var_name} is not a nested list of integers/floats: {var[0]}[/]",
                    extra={"markup": True})

        return input_repr_dict


def mock_genetic_algorithm(input_repr_dict: Dict[str, any], n: int = 20000) -> Dict[str, any]:
    """
    This function is a mock implementation of a genetic algorithm (GA) for scheduling tasks on machines.
    It initializes a population of schedules, evaluates them, and selects the best schedule without performing
    crossover/offspring or mutation operations and without running multiple generations.

    Args:
        input_repr_dict (Dict[str, any]): A dictionary containing the necessary input variables for the GA.
        n (int, optional): The population size. Defaults to 200.

    Returns:
        best_schedule: Dictionary containing schedule with the best score.
    """

    # Unpack input into local variables
    for key, value in input_repr_dict.items():
        globals()[key] = value

    # INIT
    P = []
    i = 0
    percentages = np.arange(10, 101, 10)

    while i < n:
        i += 1
        avail_m = {m: 0 for m in M}
        P_j = []

        J_temp = list(range(len(J)))
        while J_temp:
            job_idx = random.choice(J_temp)
            J_temp.remove(job_idx)
            job = J[job_idx]
            for task in range(len(job)):
                m = random.choice(compat[job_idx][task])
                if task == 0:
                    start = avail_m[m]
                else:
                    start = max(avail_m[m], P_j[-1][3] + dur[job_idx][task - 1])
                P_j.append((job_idx, job[task], m, start, dur[job_idx][task]))
                avail_m[m] = start + dur[job_idx][task]

        if P_j not in P:
            P.append(P_j)
        else:
            i -= 1

        # Print a message when a certain percentage of schedules have been created
        if i * 100 / n in percentages:
            logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {i * 100 / n}% of schedules have been created.")

    # EVAL
    scores = []
    for schedule in P:
        score = 0
        for (job_idx, task, machine, start_time, job_task_dur) in schedule:
            job = J[job_idx]
            if task+1 == len(job):  # Applies to last task of each job
                completion_time = start_time + dur[job_idx][task]   # Assuming tasks are 1 indexed
                if completion_time <= due[job_idx]:
                    score += (10000 + due[job_idx] - completion_time)
                else:
                    score -= (completion_time - due[job_idx])  # Penalty for late completion

        scores.append(round(score))

    # Score metrics
    logger.info(f"Best score: {max(scores)}, Median score: {np.median(scores)}, Worst score: {min(scores)}")

    # Zip schedules and scores together
    schedules_and_scores = list(zip(P, scores))

    # Sort schedules by score and extract best schedule
    schedules_and_scores.sort(key=lambda x: x[1], reverse=True)
    best_schedule = schedules_and_scores[0][0]

    # Debug statement
    logger.info(f"Snippet of best schedule (job, task, machine, start_time, duration): {best_schedule[:4]}")

    return best_schedule


def reformat_output(croom_processed_orders: pd.DataFrame, best_schedule: Dict[str, any], column_mapping: dict) -> pd.DataFrame:

    # Convert best schedule into a dataframe
    schedule_df = pd.DataFrame(best_schedule, columns=['job', 'task', 'machine', 'starting_time', 'duration'])

    # Round the starting time
    schedule_df['starting_time'] = schedule_df['starting_time'].round(1)

    # Rename columns of processed orders
    mapping = {
        'Job ID': 'Order',
        'Created Date': 'Order_date',
        'Part Description': 'Product',
        'Due Date ': 'Due_date',
        'Order Qty': 'Order Qty'
    }
    croom_processed_orders = croom_processed_orders.rename(columns=mapping)

    # Reset and drop index
    croom_processed_orders.reset_index(inplace=True, drop=True)

    # Join best schedule to processed orders
    croom_processed_orders = croom_processed_orders.merge(schedule_df,
                                                          left_index=True,
                                                          right_on='job',
                                                          how='left')

    # Define end time
    croom_processed_orders['end_time'] = croom_processed_orders['starting_time'] + croom_processed_orders['duration']

    # Rename round two
    croom_processed_orders = croom_processed_orders.rename(columns={'job': 'Job',
                                                                    'starting_time': 'Start_time',
                                                                    'end_time': 'End_time',
                                                                    'machine': 'Machine'
                                                                    })

    def add_duration_to_start_time(df: pd.DataFrame, base_date: str = '2024-03-18T09:00') -> pd.DataFrame:
        """
        Adds the duration to the start time.

        Args:
            df (pd.DataFrame): The input dataframe with 'start_time' and 'duration' columns.
            base_date (str): The base date in the format 'YYYY-MM-DDTHH:MM'.

        Returns:
            pd.DataFrame: The dataframe with 'start_time' and 'end_time' columns.
        """
        base_datetime = datetime.strptime(base_date, '%Y-%m-%dT%H:%M')
        df['Start_time'] = pd.to_timedelta(df['Start_time'], unit='m') + base_datetime
        df['End_time'] = df['Start_time'] + pd.to_timedelta(df['duration'], unit='m')
        return df

    # Apply duration function
    croom_adjusted = add_duration_to_start_time(croom_processed_orders)

    # Define mapping for machine names
    machine_dict = {
        1: 'HAAS-1',
        2: 'HAAS-2',
        3: 'HAAS-3',
        4: 'HAAS-4',
        5: 'HAAS-5',
        6: 'HAAS-6',
        7: 'Inspection-1',
        8: 'Inspection-2',
        9: 'Inspection-3',
        10: 'Inspection-4',
        11: 'Wash-1',
        12: 'Manual Prep-1',
        13: 'Manual Prep-2',
        14: 'Manual Prep-3',
        15: 'Final Wash-1',
        16: 'Final Inspect-1',
        17: 'Final Inspect-2'
    }

    # Apply machine name mapping
    croom_adjusted['Machine'] = croom_adjusted['Machine'].map(machine_dict)

    return croom_adjusted


def create_chart(schedule: pd.DataFrame, parameters: Dict[str, Union[str, Dict[str, str]]]) -> pd.DataFrame:
    """
    Creates a Gantt chart based on the schedule and parameters.

    Args:
        schedule (pd.DataFrame): The schedule data.
        parameters (Dict[str, Union[str, Dict[str, str]]]): The parameters for creating the chart.

    Returns:
        pd.DataFrame: The updated schedule data with additional columns for the chart.
    """
    if not is_string_dtype(schedule[[parameters["column_mapping"]["Resource"]]]):
        schedule[parameters["column_mapping"]["Resource"]] = schedule[parameters["column_mapping"]["Resource"]].apply(str)
    schedule = schedule.rename(columns=parameters["column_mapping"])

    return schedule


def save_chart_to_html(gantt_chart: plotly.graph_objs.Figure) -> None:
    """
    Saves the Gantt chart to an HTML file.

    Args:
        gantt_chart (plotly.graph_objs.Figure): The Gantt chart to be saved.
    """
    filepath = Path(os.getcwd()) / 'data/08_reporting/gantt_chart.html'
    plotly.offline.plot(gantt_chart, filename=str(filepath))
