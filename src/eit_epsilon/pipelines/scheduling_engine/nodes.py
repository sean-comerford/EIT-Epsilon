import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import math
import re
import copy
import random
import logging
from pandas.api.types import is_string_dtype
import plotly
from typing import List, Dict, Tuple, Union
from collections import defaultdict

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
        inscope_data = data[
            (
                data["Part Description"].str.contains("OP 1")
                | data["Part Description"].str.contains("ATT ")
            )
            & (~data["On Hold?"])
            & (~data["Part Description"].str.contains("OP 2"))
        ]

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
            Type=lambda x: x["Part Description"].apply(
                lambda y: "CR" if "CR" in y else ("PS" if "PS" in y else "")
            ),
            Size=lambda x: x["Part Description"].apply(
                lambda y: (re.search(r"Sz (\d+N?)", y).group(1) if re.search(r"Sz (\d+N?)", y) else "")
            ),
            Orientation=lambda x: x["Part Description"].apply(
                lambda y: ("LEFT" if "LEFT" in y.upper() else ("RIGHT" if "RIGHT" in y.upper() else ""))
            ),
            Cementless=lambda x: x["Part Description"].apply(
                lambda y: "CLS" if "CLS" in y.upper() else "C"
            ),
        )

        # Debug statement
        if data[["Type", "Size", "Orientation"]].isna().sum().sum() > 0:
            logger.warning(
                f"Data with extracted information: {data[['Type', 'Size', 'Orientation']].isna().sum()}"
            )
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
        grouped = data.groupby("Part ID")[["Type", "Size", "Orientation"]].nunique()

        if (grouped > 1).any().any():
            logger.error(
                "[bold red blink]Part ID not unique for every combination of Type, Size, and Orientation[/]",
                extra={"markup": True},
            )
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
        cementless_count = data[data["Cementless"] == "CLS"].shape[0]
        total_products = data.shape[0]

        cementless_percentage = (cementless_count / total_products) * 100
        logger.info(f"Proportion of cementless products: {cementless_percentage:.1f}%")

        # Populate J
        J = [
            [1, 2, 3, 4, 5, 6, 7] if cementless == "CLS" else [1, 2, 3, 6, 7]
            for cementless in data["Cementless"]
        ]

        if not len(J) == len(data):
            logger.error(
                "[bold red blink]J is not of the same length as processed orders![/]",
                extra={"markup": True},
            )

        # Debug statement
        logger.info(f"Snippet of Jobs for OP 1: {J[:2]}")

        return J

    @staticmethod
    def get_part_id(data: pd.DataFrame) -> List[int]:
        part_id = data["Part ID"].to_list()

        # Show snippet of Part IDs
        logger.info(f"Snippet of Part IDs: {part_id[:5]}")

        return part_id


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
    def get_compatibility(
        J: List[List[int]], task_to_machines: Dict[int, List[int]]
    ) -> List[List[List[int]]]:
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
    def preprocess_cycle_times(
        cycle_times: pd.DataFrame, last_task_minutes: int = 4
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            match = re.search(r"\d+N?", s)
            return match.group(0) if match else s

        cycle_times.columns = [extract_number(col) for col in cycle_times.columns]
        cycle_times.loc[7] = cycle_times.loc[7].fillna(last_task_minutes)
        ps_times, cr_times = (
            cycle_times.iloc[:8, 2 : math.ceil(cycle_times.shape[1] / 2) + 1],
            cycle_times.iloc[:8, math.ceil(cycle_times.shape[1] / 2) + 1 :],
        )

        # Debug statement
        logger.info(f"PS times dim.: {ps_times.shape}, CR times dim.: {cr_times.shape}")

        # Convert everything to float
        ps_times = ps_times.astype(float)
        cr_times = cr_times.astype(float)

        # Check if all cycle times are numeric values
        if not all(isinstance(i, (int, float)) for i in ps_times.values.flatten()) or not all(
            isinstance(i, (int, float)) for i in cr_times.values.flatten()
        ):
            logger.warning(
                "[bold red blink]All cycle times should be numeric values. Please check the input data.[/]",
                extra={"markup": True},
            )

        # Define required sizes
        required_sizes = {"1", "2", "3", "3N", "4", "4N", "5", "5N", "6", "6N", "7", "8", "9", "10"}

        # Check if all required sizes are in the columns of both dataframes
        if not required_sizes.issubset(set(cr_times.columns)) or not required_sizes.issubset(
            set(ps_times.columns)
        ):
            logger.warning(
                "[bold red blink]Either cr_times or ps_times is missing some of the sizes in the columns.[/]",
                extra={"markup": True},
            )

        return ps_times, cr_times

    @staticmethod
    def get_duration_matrix(
        J: List[List[int]],
        inscope_orders: pd.DataFrame,
        cr_times: pd.DataFrame,
        ps_times: pd.DataFrame,
    ) -> List[List[float]]:
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
                times = cr_times if inscope_orders.iloc[i]["Type"] == "CR" else ps_times
                duration = round(
                    times.loc[task, inscope_orders.iloc[i]["Size"]]
                    * inscope_orders.iloc[i]["Order Qty"],
                    1,
                )
                job_dur.append(duration)
            dur.append(job_dur)

        # Debug statement
        logger.info(f"Snippet of duration: {dur[0]}")

        return dur

    @staticmethod
    def get_due_date(
        inscope_orders: pd.DataFrame,
        date: str = "2024-03-18",
        working_minutes: int = 480,
    ) -> List[int]:
        """
        Gets the due dates for the in-scope orders.

        Args:
            inscope_orders (pd.DataFrame): The in-scope orders.
            date (str): The reference date in 'YYYY-MM-DD' format. Defaults to '2024-03-18'.
            working_minutes (int, optional): The number of working minutes per day. Defaults to 480.

        Returns:
            List[int]: The list of due dates in working minutes.
        """
        due = []
        for due_date in inscope_orders["Due Date "]:
            if pd.Timestamp(date) > due_date:
                working_days = -len(pd.bdate_range(due_date, date)) * working_minutes
            else:
                working_days = len(pd.bdate_range(date, due_date)) * working_minutes
            due.append(working_days)

        # Debug statement
        logger.info(f"Snippet of due: {due[:5]}")

        return due


class JobShop(Job, Shop):
    """
    The JobShop class combines the functionality of the Job and Shop classes.
    The Job class is used to preprocess orders into the correct representation for a genetic algorithm.
    The Shop class is used to create a representation of the factory floor: the machines and the constraints.

    By using the JobShop class all the functionality of the Job and Shop classes can be accessed.
    """

    def preprocess_orders(self, croom_open_orders: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the open orders.

        Args:
            croom_open_orders (pd.DataFrame): The open orders.

        Returns:
            pd.DataFrame: The preprocessed orders.
        """
        inscope_data = self.filter_in_scope_op1(croom_open_orders).pipe(self.extract_info)
        self.check_part_id_consistency(inscope_data)

        return inscope_data

    def build_ga_representation(
        self,
        croom_processed_orders: pd.DataFrame,
        cr_cycle_times: pd.DataFrame,
        ps_cycle_times: pd.DataFrame,
        machine_qty_dict: Dict[str, int],
        task_to_machines: Dict[int, List[int]],
    ) -> Dict[str, any]:
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
        part_id = self.get_part_id(croom_processed_orders)

        def is_nested_list_of_numbers(lst):
            if isinstance(lst, list):
                return all(
                    (
                        is_nested_list_of_numbers(item)
                        if isinstance(item, list)
                        else isinstance(item, (int, float))
                    )
                    for item in lst
                )
            return False

        input_repr_dict = {
            "J": J,
            "M": M,
            "compat": compat,
            "dur": dur,
            "due": due,
            "part_id": part_id,
        }

        # Check if J, M, compat, dur, and due are (nested) lists of integers or floats
        # Part ID is a list of strings, so should be excluded
        for var_name, var in input_repr_dict.items():
            if var_name != "part_id":
                if not is_nested_list_of_numbers(var):
                    logger.error(
                        f"[bold red blink]{var_name} is not a nested list of integers/floats: {var[0]}[/]",
                        extra={"markup": True},
                    )

        return input_repr_dict


class GeneticAlgorithmScheduler:
    """
    Contains all functions to run a genetic algorithm for a flexible job shop scheduling problem (FJSSP):

    - Initialize population
    - Evaluate fitness
    - Crossover/Offspring
    - Mutation (Currently not implemented)
    - Run
    """

    def __init__(self):
        self.J = None
        self.M = None
        self.compat = None
        self.dur = None
        self.due = None
        self.part_id = None
        self.n = None
        self.n_e = None
        self.n_c = None
        self.P = None
        self.scores = None
        self.day_range = None
        self.best_schedule = None
        self.minutes_per_day = None
        self.change_over_time = None
        self.urgent_orders = None
        self.urgent_multiplier = None
        self.max_iterations = None

    def find_avail_m(self, start: int, job_idx: int, task_num: int) -> int:
        """
        Finds the next available time for a machine to start a task, considering the working day duration.

        Args:
            start (int): The starting time in the schedule in minutes.
            job_idx (int): The index of the job in the job list.
            task_num (int): The task number within the job.

        Returns:
            int: The next available time for the machine to start the task.
        """
        for day in self.day_range:
            if start < day <= start + self.dur[job_idx][task_num]:
                return day
        return start + self.dur[job_idx][task_num]

    def init_population(self, num_inds: int = None, fill_inds: bool = False):
        """
        Initializes the population of schedules. Each schedule is a list of tasks assigned to machines with start times.
        """
        if num_inds is None:
            num_inds = self.n

        P = []
        percentages = np.arange(10, 101, 10)

        for i in range(num_inds):
            avail_m = {m: 0 for m in self.M}
            product_m = {m: 0 for m in self.M}
            changeover_finish_time = 0
            P_j = []

            J_temp = list(range(len(self.J)))
            random_roll = random.random()

            if random_roll < 0.4:
                random.shuffle(J_temp)
            elif random_roll < 0.6:
                J_temp.sort(key=lambda x: self.part_id[x])
            elif random_roll < 0.7:
                J_temp.sort(key=lambda x: self.due[x], reverse=True)
            elif random_roll < 0.8:
                J_temp.sort(key=lambda x: (self.part_id[x], self.due[x]), reverse=True)
            else:
                # Reorder J_temp according to the urgent order list
                for job in self.urgent_orders:
                    J_temp.remove(job)  # Remove the job from its current position
                    J_temp.append(job)  # Append the job to the end of the list

            while J_temp:
                # Take the index at the end of the list and remove it from the list
                job_idx = J_temp.pop()
                job = self.J[job_idx]
                for task in range(len(job)):
                    random_roll = random.random()

                    if task == 0:
                        compat_task_0 = self.compat[job_idx][task]
                        preferred_machines = [
                            key
                            for key in compat_task_0
                            if product_m.get(key) == self.part_id[job_idx] or product_m.get(key) == 0
                        ]

                        if not preferred_machines:
                            m = (
                                min(compat_task_0, key=lambda x: avail_m.get(x))
                                if random_roll < 0.7
                                else random.choice(compat_task_0)
                            )
                        else:
                            m = (
                                min(preferred_machines, key=lambda x: avail_m.get(x))
                                if random_roll < 0.7
                                else random.choice(preferred_machines)
                            )

                        start = (
                            avail_m[m]
                            if product_m[m] == 0 or self.part_id[job_idx] == product_m[m]
                            else avail_m[m]
                            + self.change_over_time
                            + max((changeover_finish_time - avail_m[m]), 0)
                        )

                        if product_m[m] != 0 and self.part_id[job_idx] != product_m[m]:
                            changeover_finish_time = start
                    else:
                        compat_with_task = self.compat[job_idx][task]
                        m = (
                            min(compat_with_task, key=lambda x: avail_m.get(x))
                            if random_roll < 0.85
                            else random.choice(compat_with_task)
                        )
                        start = max(avail_m[m], P_j[-1][3] + self.dur[job_idx][task - 1])

                    P_j.append(
                        (
                            job_idx,
                            job[task],
                            m,
                            start,
                            self.dur[job_idx][task],
                            task,
                            self.part_id[job_idx],
                        )
                    )
                    avail_m[m] = self.find_avail_m(start, job_idx, task)
                    product_m[m] = self.part_id[job_idx]

            if P_j not in P:
                P.append(P_j)
            else:
                i -= 1

            if not fill_inds and i * 100 / num_inds in percentages:
                logger.info(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {i * 100 / num_inds}% of schedules have been created."
                )

        if not fill_inds:
            self.P = P
        else:
            return P

    def evaluate_population(
        self, best_scores: list = None, display_scores: bool = True, on_time_bonus: int = 5000
    ):
        """
        Evaluates the population of schedules by calculating a score for each schedule based on the completion times
        of jobs vs. the required due date.
        """
        # Calculate scores for each schedule
        self.scores = [
            round(
                sum(
                    (
                        # Difference between due date and completion time, multiplied by urgent_multiplier if urgent
                        self.due[job_idx]
                        - (start_time + job_task_dur)
                        * (self.urgent_multiplier if job_idx in self.urgent_orders else 1)
                        + (
                            # Bonus for completing tasks on time
                            on_time_bonus
                            if (self.due[job_idx] - (start_time + job_task_dur)) > 0
                            else 0
                        )
                    )
                    for (
                        job_idx,
                        task,
                        machine,
                        start_time,
                        job_task_dur,
                        _,
                        _,
                    ) in schedule
                    # Only consider the completion time of the final task
                    if task + 1 == max(self.J[job_idx])
                )
            )
            # Evaluate each schedule in the population
            for schedule in self.P
        ]

        if display_scores:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Best score: {max(self.scores)}, "
                f"Median score: {np.median(self.scores)}, Worst score: {min(self.scores)}"
            )
            best_scores.append(max(self.scores))

    def resolve_conflict(
        self, P_prime: List[Tuple[int, int, int, int, int, int, str]]
    ) -> (List)[Tuple[int, int, int, int, int, int, str]]:
        """
        This function resolves conflicts in a given schedule. If tasks are planned on the same machine at the same time,
        it finds the first available time for each task to start on the machine.

        Parameters:
        P_prime (List[Tuple[int, int, int, int, int, int, str]]): A list of tuples where each tuple represents a task.
        Each task is represented as (job index, task, machine, start time, duration, task number).

        Returns:
        P_prime_sorted (List[Tuple[int, int, int, int, int, int, str]]): A sorted list of tuples where each tuple
        represents a task. Each task is represented as (job index, job, machine, start time, duration, task number).
        """
        P_prime_sorted = []
        avail_m = {m: 0 for m in self.M}
        product_m = {m: 0 for m in self.M}
        changeover_finish_time = 0
        start_times = {j: 0 for j in range(len(self.J))}

        for job_idx in range(len(self.J)):
            job = self.J[job_idx]
            job_tasks = sorted([entry for entry in P_prime if entry[0] == job_idx], key=lambda x: x[1])

            for task_entry in job_tasks:
                _, task, m, _, _, task_num, _ = task_entry

                if task_num == 0:
                    start = (
                        avail_m[m]
                        if product_m[m] == 0 or self.part_id[job_idx] == product_m[m]
                        else avail_m[m]
                        + self.change_over_time
                        + max((changeover_finish_time - avail_m[m]), 0)
                    )
                    if product_m[m] != 0 and self.part_id[job_idx] != product_m[m]:
                        changeover_finish_time = start
                else:
                    start = max(
                        avail_m[m],
                        start_times[job_idx] + self.dur[job_idx][task_num - 1],
                    )
                start_times[job_idx] = start

                avail_m[m] = self.find_avail_m(start, job_idx, task_num)
                product_m[m] = self.part_id[job_idx]

                P_prime_sorted.append(
                    (
                        job_idx,
                        job[task_num],
                        m,
                        start,
                        self.dur[job_idx][task_num],
                        task_num,
                        self.part_id[job_idx],
                    )
                )

        return P_prime_sorted

    def find_best_schedules(self) -> List:
        """
        This method evaluates the population, sorts them based on their scores in descending order,
        and retains the top schedules based on a specified retention count. The retention count is
        the maximum of 3 or the product of the length of the population and a specified ratio.

        Returns:
            P_0 (List): The list of top schedules based on their scores.
        """

        self.evaluate_population(display_scores=False)
        scored_population = sorted(zip(self.scores, self.P), key=lambda x: x[0], reverse=True)
        retain_count = max(3, int(len(self.P) * self.n_e))
        P_0 = [schedule for score, schedule in scored_population[:retain_count]]

        return P_0

    def offspring(self) -> None:
        """
        This function generates offspring for the next generation of the population. It identifies the best schedules
        in the population, after which it randomly selects each job from the first or the second schedule.
        Conflicts are resolved by another function.

        Returns:
        None. The function updates the population in-place.
        """
        P_0 = self.find_best_schedules()

        iter_count = len(self.P) * (self.n_c + self.n_e)
        while len(P_0) < iter_count:
            P1, P2 = random.sample(P_0, 2)
            P_prime = [
                entry
                for job_idx in range(len(self.J))
                for entry in random.choice([P1, P2])
                if entry[0] == job_idx
            ]
            P_prime = self.resolve_conflict(P_prime)
            if P_prime not in P_0:
                P_0.append(P_prime)

        self.P = P_0

    def mutate(self):
        """
        Perform mutation on the current population of schedules to create a new set of schedules.

        This function identifies pairs of jobs within each schedule that have the same number
        of tasks and the same durations. It then swaps the start times and machines of tasks
        between the selected pairs of jobs, creating new schedules. If the new schedule does
        not already exist in the population, it is added to the population.

        Steps:
        1. Find the best schedules based on evaluation score.
        2. Make a deep copy of these schedules.
        3. For each schedule:
            a. Group tasks by their job index.
            b. Identify pairs of jobs with the same number of tasks and identical durations.
            c. Randomly select a pair of jobs.
            d. Swap the start times and machines of tasks between the selected jobs.
            e. If the new schedule is unique, add it to the list of new schedules.
        4. Add all new schedules to the population.

        This method helps in exploring new potential solutions by making modifications to
        existing ones, promoting diversity in the population.

        Returns:
            None
        """

        P_0 = self.find_best_schedules()

        # Make a deep copy of P_0
        P_1 = copy.deepcopy(P_0)

        # Initialize a list of new schedules
        new_schedules = []

        for schedule in P_1:
            # Group tasks by job_idx
            jobs = defaultdict(list)
            for task in schedule:
                jobs[task[0]].append(task)

            # Find pairs of jobs with same number of tasks and same durations
            job_pairs = []
            job_list = list(jobs.items())

            for i in range(len(job_list)):
                for j in range(i + 1, len(job_list)):
                    job1, task_details_1 = job_list[i]
                    job2, task_details_2 = job_list[j]

                    # Append if the part ID matches for a pair
                    if task_details_1[-1] == task_details_2[-1]:
                        # task_details format: (job_index, task, machine, start_time, duration, task_num, part_id)
                        # hence the last field is the part_id
                        job_pairs.append((job1, job2))

            # If no pairs found, continue to the next schedule
            if not job_pairs:
                continue

            # Randomly select a pair of jobs
            job1, job2 = random.choice(job_pairs)

            # Swap the start times of the tasks in the selected jobs
            tasks1 = jobs[job1]
            tasks2 = jobs[job2]

            for i in range(len(tasks1)):
                task1 = tasks1[i]
                task2 = tasks2[i]

                # Create new tasks with swapped start times and machines
                # task_details follow this format: (job_index, task_num, machine, start_time, duration, task_index)
                new_task1 = (task1[0], task1[1], task2[2], task2[3], task1[4], task1[5], task1[6])
                new_task2 = (task2[0], task2[1], task1[2], task1[3], task2[4], task2[5], task2[6])

                # Update the schedule
                schedule[schedule.index(task1)] = new_task1
                schedule[schedule.index(task2)] = new_task2

            # If the new schedule does not exist yet in the population, add it to P_0
            if schedule not in P_0 and schedule not in new_schedules:
                new_schedules.append(schedule)

        # Add all schedules from P_0 to the population
        self.P = P_0 + new_schedules

    def run(
        self,
        input_repr_dict: Dict[str, any],
        scheduling_options: dict,
    ):
        """
        Runs the genetic algorithm by initializing the population, evaluating it, and selecting the best schedule.

        Args:
            input_repr_dict (Dict[str, any]): A dictionary containing the necessary input variables for the GA.
            scheduling_options (dict): Dictionary containing hyperparameters for running the algorithm.

        Returns:
            Tuple[List[Tuple[int, int, int, int, float]], List[int]]: The best schedule with the highest score and the list of best scores per generation.
        """
        self.J = input_repr_dict["J"]
        self.M = input_repr_dict["M"]
        self.compat = input_repr_dict["compat"]
        self.dur = input_repr_dict["dur"]
        self.due = input_repr_dict["due"]
        self.part_id = input_repr_dict["part_id"]
        self.n = scheduling_options["n"]
        self.n_e = scheduling_options["n_e"]
        self.n_c = scheduling_options["n_c"]
        self.minutes_per_day = scheduling_options["minutes_per_day"]
        self.change_over_time = scheduling_options["change_over_time"]
        self.max_iterations = scheduling_options["max_iterations"]
        self.urgent_multiplier = scheduling_options["urgent_multiplier"]
        self.urgent_orders = [job_idx - 1 for job_idx in scheduling_options["urgent_orders"]]
        self.day_range = np.arange(
            self.minutes_per_day,
            len(self.J) // 5 * self.minutes_per_day,
            self.minutes_per_day,
        )

        self.init_population()
        best_scores = []

        for iteration in range(self.max_iterations):
            logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Iteration {iteration + 1}")
            self.offspring()
            self.mutate()
            if len(self.P) < self.n:
                self.P += self.init_population(num_inds=self.n - len(self.P), fill_inds=True)
            self.evaluate_population(best_scores=best_scores)

        schedules_and_scores = sorted(zip(self.P, self.scores), key=lambda x: x[1], reverse=True)
        self.best_schedule = schedules_and_scores[0][0]
        logger.info(
            f"Snippet of best schedule (job, task, machine, start_time, duration, task_num, part_id): "
            f"{self.best_schedule[:2]}"
        )

        return self.best_schedule, best_scores


def reformat_output(
    croom_processed_orders: pd.DataFrame,
    best_schedule: Dict[str, any],
    column_mapping_reformat: dict,
    machine_dict: dict,
) -> pd.DataFrame:
    """
    Reformats the output of the genetic algorithm by converting the best schedule into a dataframe,
    rounding the starting time, resetting and dropping the index, joining the best schedule to processed orders,
    defining the end time, renaming columns, applying machine name mapping, and creating start and end datetime
    based on a hypothetical start date.

    Args:
        croom_processed_orders (pd.DataFrame): The processed orders.
        best_schedule (Dict[str, any]): The best schedule generated by the genetic algorithm.
        column_mapping_reformat (dict): The mapping for renaming columns in the output dataframe.
        machine_dict (dict): The dictionary mapping machine numbers to machine names.

    Returns:
        pd.DataFrame: The reformatted output dataframe.
    """
    # Convert best schedule into a dataframe
    schedule_df = pd.DataFrame(
        best_schedule,
        columns=["job", "task", "machine", "starting_time", "duration", "task_num", "part_id"],
    )

    # Round the starting time and duration
    schedule_df["starting_time"] = schedule_df["starting_time"].round(5)
    schedule_df["duration"] = schedule_df["duration"].round(5)

    # Reset and drop index
    croom_processed_orders.reset_index(inplace=True, drop=True)

    # Join best schedule to processed orders
    croom_processed_orders = croom_processed_orders.merge(
        schedule_df, left_index=True, right_on="job", how="left"
    )

    # Define end time
    croom_processed_orders["end_time"] = (
        croom_processed_orders["starting_time"] + croom_processed_orders["duration"]
    )

    # Rename columns
    croom_processed_orders = croom_processed_orders.rename(columns=column_mapping_reformat)

    # Apply machine name mapping
    croom_processed_orders["Machine"] = croom_processed_orders["Machine"].map(machine_dict)

    return croom_processed_orders


def create_start_end_time(
    croom_reformatted_orders: pd.DataFrame, scheduling_options: dict
) -> pd.DataFrame:
    """
    Adjusts the start and end times of tasks in the given DataFrame to fit within working hours (09:00 to 17:00)
    and ensures that tasks are scheduled sequentially within each job and machine.

    The function first converts the 'Start_time' from minutes to a datetime based on a hypothetical start date.
    It then adjusts the start and end times of tasks to fit within working hours, pushing tasks to the next day
    if they start after 17:00 or before 09:00. The function also checks for consistency in task scheduling within
    each job and machine.

    Args:
        croom_reformatted_orders (pd.DataFrame): The DataFrame containing reformatted orders with 'Start_time' in minutes.
        scheduling_options (dict): A dictionary containing scheduling options, including the 'start_date'.

    Returns:
        pd.DataFrame: The adjusted DataFrame with updated start and end times.
    """

    # Parse the date string into a datetime object
    base_date = datetime.strptime(scheduling_options["start_date"], "%Y-%m-%dT%H:%M")

    # Sort by start time ascending
    croom_reformatted_orders.sort_values("Start_time", inplace=True)

    # Initialize empty 'Start_time_date' column
    croom_reformatted_orders["Start_time_date"] = None

    def working_hours_shift(row):
        days = [d * scheduling_options["minutes_per_day"] for d in range(1, 25)]

        for k, day in enumerate(days):
            if row["Start_time"] < day:
                row["Start_time_date"] = (
                    base_date
                    + pd.to_timedelta(row["Start_time"], unit="m")
                    + pd.Timedelta(days=k)
                    - pd.Timedelta(minutes=scheduling_options["minutes_per_day"] * k)
                )
                break
        return row

    # Apply function
    croom_reformatted_orders = croom_reformatted_orders.apply(working_hours_shift, axis=1)

    # Overwrite the integer start time with the calculated datetimes
    croom_reformatted_orders["Start_time"] = croom_reformatted_orders["Start_time_date"]
    croom_reformatted_orders["End_time"] = croom_reformatted_orders["Start_time"] + pd.to_timedelta(
        croom_reformatted_orders["duration"], unit="m"
    )

    # Reorder by earliest start time
    croom_reformatted_orders.sort_values(by="Start_time", inplace=True)

    # Check if the start time for each task within each job is later than the completion time of the previous task
    for job_id in croom_reformatted_orders["Job"].unique():
        job_schedule = croom_reformatted_orders[croom_reformatted_orders["Job"] == job_id]
        for i in range(1, len(job_schedule)):
            if not job_schedule.iloc[i]["Start_time"] >= job_schedule.iloc[i - 1]["End_time"]:
                logger.warning(
                    f"The start time for task {job_schedule.iloc[i]['task']} in job {job_id} "
                    f"is earlier than the completion time of the previous task!"
                )

    # Check if the start time for each task within each job is later than the completion time of the previous task
    for machine in croom_reformatted_orders["Machine"].unique():
        machine_schedule = croom_reformatted_orders[
            croom_reformatted_orders["Machine"] == machine
        ].sort_values("Start_time")
        for i in range(1, len(machine_schedule)):
            if not machine_schedule.iloc[i]["Start_time"] >= machine_schedule.iloc[i - 1]["End_time"]:
                logger.warning(
                    f"The start time for job {machine_schedule.iloc[i]['Job']}, task {machine_schedule.iloc[i]['task']}"
                    f" in machine {machine} is earlier than the completion time of the previous task!"
                )

    # Sort again by job and task before plotting
    croom_reformatted_orders = croom_reformatted_orders.sort_values(["Job", "task"])

    return croom_reformatted_orders


def create_chart(
    schedule: pd.DataFrame, parameters: Dict[str, Union[str, Dict[str, str]]]
) -> pd.DataFrame:
    """
    Creates a Gantt chart based on the schedule and parameters.

    Args:
        schedule (pd.DataFrame): The schedule data.
        parameters (Dict[str, Union[str, Dict[str, str]]]): The parameters for creating the chart.

    Returns:
        pd.DataFrame: The updated schedule data with additional columns for the chart.
    """
    if not is_string_dtype(schedule[[parameters["column_mapping"]["Resource"]]]):
        schedule[parameters["column_mapping"]["Resource"]] = schedule[
            parameters["column_mapping"]["Resource"]
        ].apply(str)
    schedule = schedule.rename(columns=parameters["column_mapping"])

    return schedule


def save_chart_to_html(gantt_chart: plotly.graph_objs.Figure) -> None:
    """
    Saves the Gantt chart to an HTML file.

    Args:
        gantt_chart (plotly.graph_objs.Figure): The Gantt chart to be saved.
    """
    filepath = Path(os.getcwd()) / "data/08_reporting/gantt_chart.html"
    plotly.offline.plot(gantt_chart, filename=str(filepath))
