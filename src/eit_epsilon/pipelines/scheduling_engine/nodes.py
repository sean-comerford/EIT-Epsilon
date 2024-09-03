import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import time
import re
import copy
import random
import logging
from pandas.api.types import is_string_dtype
import plotly
from typing import List, Dict, Tuple, Deque, Union, Optional, Any
import itertools
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from collections import deque

# Instantiate logger
logger = logging.getLogger(__name__)


class Job:
    """
    The Job class contains methods for preprocessing and extracting information from open orders that need
    to be processed in a manufacturing workshop.
    """

    @staticmethod
    def filter_in_scope(data: pd.DataFrame, operation: str = "OP 1") -> pd.DataFrame:
        """
        Filters the data to include only in-scope operations for OP 1.

        Args:
            data (pd.DataFrame): The input data.
            operation (str, optional): The operation for which to filter data. Defaults to 'OP 1'.

        Returns:
            pd.DataFrame: The filtered data.
        """
        # Debug statement
        logger.info(f"Total order data: {data.shape}")

        # Apply the filter
        if operation == "OP 1":
            in_scope_data = data[
                (
                    data["Part Description"].str.contains("OP 1")
                    | data["Part Description"].str.contains("ATT ")
                )
                & (~data["On Hold?"])
                & (~data["Part Description"].str.contains("OP 2"))
            ]

        elif operation == "OP 2":
            in_scope_data = data[
                (data["Part Description"].str.contains("OP 2"))
                & (~data["On Hold?"])
                & (~data["Part Description"].str.contains("OP 1"))
            ]

        else:
            logger.error(f"Invalid operation: {operation} - Only 'OP 1' and 'OP 2' are supported")
            raise ValueError("Invalid operation")

        # Debug statement
        logger.info(f"In-scope data for {operation}: {in_scope_data.shape}")

        return in_scope_data

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
            # CR: Cruciate retaining, PS: Posterior stabilizing
            Type=lambda x: x["Part Description"].apply(
                lambda y: "CR" if "CR" in y else ("PS" if "PS" in y else "")
            ),
            # Range 1-10 with optional 'N' for some sizes; e.g. '5N' (Not sure what this stands for)
            Size=lambda x: x["Part Description"].apply(
                lambda y: (re.search(r"Sz (\d+N?)", y).group(1) if re.search(r"Sz (\d+N?)", y) else "")
            ),
            # LEFT or RIGHT orientation
            Orientation=lambda x: x["Part Description"].apply(
                lambda y: ("LEFT" if "LEFT" in y.upper() else ("RIGHT" if "RIGHT" in y.upper() else ""))
            ),
            # CLS: Cementless, CTD: Cemented
            Cementless=lambda x: x["Part Description"].apply(
                lambda y: "CLS" if "CLS" in y.upper() else "CTD"
            ),
        )

        # Create custom Part ID
        data["Custom Part ID"] = (
            data["Orientation"] + "-" + data["Type"] + "-" + data["Size"] + "-" + data["Cementless"]
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
        grouped = data.groupby("Part ID")[["Type", "Size", "Orientation", "Custom Part ID"]].nunique()

        if (grouped > 1).any().any():
            logger.error(
                "[bold red blink]Part ID not unique for every combination of Type, Size, and Orientation[/]",
                extra={"markup": True},
            )
        else:
            logger.info(f"Part ID consistency check passed")

    @staticmethod
    def create_jobs(
        data: pd.DataFrame, scheduling_options: dict, operation: str = "OP 1"
    ) -> Dict[int, Tuple[str, int]]:
        """Extract the Job ID and corresponding Part ID from the data, calculate the due date for each job
            and store the result in a dict object

        Args:
            data (pd.DataFrame): The input data i.e. the list of jobs
            scheduling_options (dict): A dictionary containing scheduling options, including start date, working minutes
            operation (str, optional): The operation for which to create jobs. Defaults to 'OP 1'.

        Returns:
            Dict[int, Tuple[str, int]]: A dict, each entry of which contains a job ID, part ID and due time
            e.g. {
                4421322: ('MP0389', 2400)
                4421321: ('MP0389', 2400)
                4420709: ('MP0442', 1440)
            }
        """

        # Find proportion of cementless products
        cementless_count = data[data["Cementless"] == "CLS"].shape[0]
        total_products = data.shape[0]

        cementless_percentage = (cementless_count / total_products) * 100
        logger.info(f"Proportion of cementless products: {cementless_percentage:.1f}%")

        if operation == "OP 1":
            data = data[~data["Part Description"].str.contains("OP 2")]
        elif operation == "OP 2":
            data = data[data["Part Description"].str.contains("OP 2")]
        else:
            logger.error(f"Invalid operation: {operation} - Only 'OP 1' and 'OP 2' are supported")
            raise ValueError("Invalid operation")

        J = dict(
            zip(data["Job ID"], zip(data["Custom Part ID"], Shop.get_due_date(data, scheduling_options)))
        )

        # Debug statement
        sample_of_keys = random.sample(list(J), 2)
        sample = [(k, v) for k, v in J.items() if k in sample_of_keys]
        logger.info(f"Snippet of Jobs for {operation}: {sample}")

        return J

    @staticmethod
    def create_part_id_to_task_seq(data: pd.DataFrame) -> Dict[str, List[int]]:
        """Create a dictionary which maps from a part ID to the list of tasks for that part

        Example:

        Part_to_task_sequence =    {'MP0389': [1, 2, 3, 4, 5, 6, 7],
                                    'MP0523': [1, 2, 3, 6, 7],
                                    }

        Args:
            data (pd.DataFrame): The input data i.e. the list of jobs

        Returns:
            Dict[str, List[int]]: A dictionary that maps from part_id to the sequence of tasks for that part
        """

        unique_task_types = data[["Custom Part ID", "Part Description", "Cementless"]].drop_duplicates()
        unique_task_types = unique_task_types.reset_index()

        unique_custom_part_ids = data[["Custom Part ID"]].drop_duplicates()

        if len(unique_task_types) != len(unique_custom_part_ids):
            logger.info("Combination of part/description/cementless is not unique")

        result = {}
        for _, row in unique_task_types.iterrows():
            if "OP 1" in row["Part Description"]:
                result[row["Custom Part ID"]] = list(range(1, 8))
            elif "OP 2" in row["Part Description"]:
                result[row["Custom Part ID"]] = list(range(10, 21))
            else:
                result[row["Custom Part ID"]] = list(range(30, 45))

        return result


class Shop:
    """
    The Shop class contains methods for creating machine lists, compatibility matrices,
    duration matrices and due dates as input for a genetic algorithm.
    It creates a digital representation of the processes in and the setup of a manufacturing workshop.
    """

    @staticmethod
    def create_machines(task_to_machines: Dict[int, List[int]]) -> List[int]:
        """
        Creates a list of machines based on all the unique machines in the task_to_machines dictionary.

        Args:
            task_to_machines (Dict[int, List[int]]): The dictionary of machine quantities.

        Returns:
            List[int]: The list of machines.
        """
        M = list(set([machine for machines in task_to_machines.values() for machine in machines]))

        return M

    @staticmethod
    def preprocess_cycle_times(
        monza_cycle_times_op1: pd.DataFrame,
        monza_cycle_times_op2: pd.DataFrame,
        last_task_minutes: int = 4,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the cycle times:
            1.) Remove empty rows and columns
            2.) Create new index starting from 1
            3.) Reduce column names to only the size
            4.) Fill the missing values in final inspection with 4 minutes
            5.) Split data for cruciate retaining and posterior stabilizing products

        Args:
            monza_cycle_times_op1 (pd.DataFrame): The OP 1 cycle times data.
            monza_cycle_times_op2 (pd.DataFrame): The OP 2 cycle times data.
            last_task_minutes (int, optional): The duration of the last task. Defaults to 4 minutes.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The preprocessed PS-, CR-, and OP 2 cycle times.
        """
        monza_cycle_times_op1.columns = monza_cycle_times_op1.iloc[1]
        monza_cycle_times_op1 = monza_cycle_times_op1.iloc[2:, 1:]
        monza_cycle_times_op1.index = range(1, len(monza_cycle_times_op1) + 1)

        def extract_number(s: str) -> str:
            match = re.search(r"\d+N?", s)
            return match.group(0) if match else s

        monza_cycle_times_op1.columns = [extract_number(col) for col in monza_cycle_times_op1.columns]
        monza_cycle_times_op1.loc[7] = monza_cycle_times_op1.loc[7].fillna(last_task_minutes)
        ps_times, cr_times = (
            monza_cycle_times_op1.iloc[:8, 2 : math.ceil(monza_cycle_times_op1.shape[1] / 2) + 1],
            monza_cycle_times_op1.iloc[:8, math.ceil(monza_cycle_times_op1.shape[1] / 2) + 1 :],
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
        required_sizes = {"1", "2", "3N", "3", "4N", "4", "5N", "5", "6N", "6", "7", "8", "9", "10"}

        # Check if all required sizes are in the columns of both dataframes
        if not required_sizes.issubset(set(cr_times.columns)) or not required_sizes.issubset(
            set(ps_times.columns)
        ):
            logger.warning(
                "[bold red blink]Either cr_times or ps_times is missing some of the sizes in the columns.[/]",
                extra={"markup": True},
            )

        # Operation 2 - set headers
        monza_cycle_times_op2.columns = monza_cycle_times_op2.iloc[1]
        monza_cycle_times_op2 = monza_cycle_times_op2.iloc[2:, 1:6]

        # As per email from Bryan 26/7/24: FPI and RA testing belong to another product group, so they do not need
        # to be considered for our schedule
        monza_cycle_times_op2 = monza_cycle_times_op2[
            ~monza_cycle_times_op2["Operation type"].isin(["FPI", "RA testing "])
        ]

        # Update the index according to the tasks of OP2
        monza_cycle_times_op2.index = range(10, 10 + len(monza_cycle_times_op2))

        return ps_times, cr_times, monza_cycle_times_op2

    @staticmethod
    def get_duration_matrix(
        J: Dict[int, Tuple[str, int]],
        part_id_to_task_seq: Dict[str, List[int]],
        in_scope_orders: pd.DataFrame,
        croom_task_durations: pd.DataFrame,
    ) -> Dict[Tuple[int, int], Any]:
        """
        Gets the duration matrix for the jobs.

        Example:
              dur =  {
                    # (Part, Task): Duration
                    ('MP0389', 1): 162,
                    ('MP0389', 2): 19.8
                }

        Args:
            J (List[List[int]]): The list of jobs.
            part_id_to_task_seq (Dict[str, List[int]]): The dictionary mapping part IDs to their tasks.
            in_scope_orders (pd.DataFrame): The in-scope orders.
            croom_task_durations (pd.DataFrame): The task durations.

        Returns:
            Dict[Tuple[int, int], Any]: The duration matrix.
        """

        # Set the task number as the index
        croom_task_durations.set_index("Task", inplace=True)

        # Remove whitespace from column names
        croom_task_durations.columns = croom_task_durations.columns.str.strip()

        dur = {}
        for job_id, (part_id, due_time) in J.items():
            # Find the corresponding row for the given job_id
            rows = in_scope_orders.loc[in_scope_orders["Job ID"] == job_id]

            if len(rows) > 1:
                logger.warning(f"Multiple rows found for JobID {job_id}. Using the first row.")

            # Extract the first row if needed
            row = rows.iloc[0]

            for task in part_id_to_task_seq[part_id]:
                # Construct the type_size string
                type_size = f"{row['Type']}-{row['Size']}"

                # Determine which DataFrame to use based on the task number
                duration = croom_task_durations.loc[task, type_size] * 12

                # Store the duration in the dictionary with key (part_id, task)
                dur[(job_id, task)] = duration

        return dur

    @staticmethod
    def get_due_date(
        in_scope_orders: pd.DataFrame,
        scheduling_options: dict,
    ) -> deque:
        """
        Gets the due dates for the in-scope orders.

        Args:
            in_scope_orders (pd.DataFrame): The in-scope orders.
            scheduling_options (dict): The scheduling options dictionary containing start date and daily working minutes

        Returns:
            List[int]: The list of due dates in working minutes.
        """

        # Extract base date
        base_date = scheduling_options["start_date"]

        # Extract working minutes per day
        total_minutes = scheduling_options["total_minutes_per_day"]

        due = deque()
        for due_date in in_scope_orders["Due Date "]:
            if pd.Timestamp(base_date) > due_date:
                working_days = -len(pd.bdate_range(due_date, base_date)) * total_minutes
            else:
                working_days = len(pd.bdate_range(base_date, due_date)) * total_minutes
            due.append(working_days)

        # Debug statement
        logger.info(f"Snippet of due: {list(due)[:2]}")

        return due


class JobShop(Job, Shop):
    """
    The JobShop class combines the functionality of the Job and Shop classes.
    The Job class is used to preprocess orders into the correct representation for a genetic algorithm.
    The Shop class is used to create a representation of the factory floor: the machines and the constraints.

    By using the JobShop class all the functionality of the Job and Shop classes can be accessed.
    """

    def __init__(self):
        self.input_repr_dict = None

    def preprocess_orders(self, croom_open_orders: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the open orders by filtering, extracting information, and performing various checks.

        This function performs the following steps:
        1. Filters the open orders to include only in-scope operations for OP 1 and OP 2.
        2. Extracts type, size, orientation, and cementing information from the part description.
        3. Adds an 'operation' column to distinguish between OP 1 and OP 2.
        4. Combines the data for OP 1 and OP 2.
        5. Adds the operation to the custom part ID.
        6. Checks the consistency of part IDs across different operations.
        7. Resets the index of the combined data.
        8. Checks for batch size limits, valid product values, and ensures no products are on hold.

        Args:
            croom_open_orders (pd.DataFrame): The open orders.

        Returns:
            pd.DataFrame: The preprocessed orders.
        """
        in_scope_data_op1 = self.filter_in_scope(croom_open_orders).pipe(self.extract_info)
        in_scope_data_op2 = self.filter_in_scope(croom_open_orders, operation="OP 2").pipe(
            self.extract_info
        )

        # Add an extra column
        in_scope_data_op1["operation"] = "OP1"
        in_scope_data_op2["operation"] = "OP2"

        # Combine both Operation 1 and Operation 2 data
        in_scope_data = pd.concat([in_scope_data_op1, in_scope_data_op2], axis=0)

        # Add the operation to the custom part id
        in_scope_data["Custom Part ID"] = (
            in_scope_data["Custom Part ID"] + "-" + in_scope_data["operation"]
        )

        if not len(in_scope_data) == (len(in_scope_data_op1) + len(in_scope_data_op2)):
            logging.warning(
                f"Length of concatenated data: {len(in_scope_data)}, "
                f"while length of OP 1 data: {len(in_scope_data_op1)}, "
                f"and length of OP 2 data: {len(in_scope_data_op2)}"
            )

        # Check if the custom part ID is in the correct format
        pattern = re.compile(r"^(LEFT|RIGHT)-(PS|CR)-([1-9]|10)N?-(CLS|CTD)-(OP1|OP2)$")
        for index, row in in_scope_data.iterrows():
            custom_part_id = row["Custom Part ID"]
            assert isinstance(
                custom_part_id, str
            ), f"Custom Part ID is not a string for job {row['Job ID']}: {custom_part_id}."
            assert pattern.match(
                custom_part_id
            ), f"Invalid Custom Part ID format for job {row['Job ID']}: {custom_part_id}."

        # Check if all part IDs are consistent across different operations
        self.check_part_id_consistency(in_scope_data)

        # Reset index
        in_scope_data.reset_index(inplace=True, drop=True)

        # Check batch size limit
        for index, row in in_scope_data.iterrows():
            assert (
                row["Production Qty"] <= 12
            ), f"Production Qty exceeds limit for job {row['Job ID']}: {row['Production Qty']}."
            assert (
                row["Production Qty"] > 0
            ), f"Production Qty is nonsensical for job {row['Job ID']}: {row['Production Qty']}."

        # Check in-scope orders
        valid_substrings = ["OP 1", "OP 2", "ATT Primary"]
        for index, row in in_scope_data.iterrows():
            assert any(
                substring in row["Part Description"] for substring in valid_substrings
            ), f"Invalid product value found for job {row['Job ID']}: {row['Part Description']}."

        # Check no products on hold
        for index, row in in_scope_data.iterrows():
            assert not row[
                "On Hold?"
            ], f"On Hold? is not False for job {row['Job ID']}: {row['On Hold?']}."

        return in_scope_data

    @staticmethod
    def build_changeover_compatibility(
        croom_processed_orders: pd.DataFrame,
        size_categories_cr: Dict[str, List[str]],
        size_categories_ps: Dict[str, List[str]],
    ) -> Dict[str, deque]:
        """
        Build a compatibility dictionary for changeovers between different operations (OP1 and OP2).

        The compatibility rules are:
        - For OP1: Parts are compatible if they have the exact same size and cementing status.
        - For OP2: Parts are compatible if they belong to the same type (CR or PS) and size category.
                   For PS type, they must also have the same cementing status.

        Args:
            croom_processed_orders (pd.DataFrame): DataFrame containing the processed orders with columns
                                                   'Size', 'Orientation', 'Type', 'Cementless', and 'operation'.
            size_categories_cr (Dict[str, List[str]]): Size categories for CR type.
            size_categories_ps (Dict[str, List[str]]): Size categories for PS type.

        Returns:
            Dict[str, List[str]]: A dictionary where each key is a part ID and the value is a list of compatible part IDs.
        """

        # Extract unique values for attributes from the DataFrame
        sizes = croom_processed_orders["Size"].unique()
        orientations = croom_processed_orders["Orientation"].unique()
        types = croom_processed_orders["Type"].unique()
        cementing_methods = croom_processed_orders["Cementless"].unique()
        operations = croom_processed_orders["operation"].unique()

        # Helper function to determine the size category based on type
        def get_size_category(size: str, prod_type: str, cementing: str) -> Union[str, None]:
            if prod_type == "PS" and cementing == "CLS":
                size_categories = size_categories_ps
            else:
                size_categories = size_categories_cr

            for category, cat_sizes in size_categories.items():
                if size in cat_sizes:
                    return category
            return None

        # Generate all possible part IDs
        part_ids = [
            f"{orientation}-{prod_type}-{size}-{cementing}-{op}"
            for orientation, prod_type, size, cementing, op in itertools.product(
                orientations, types, sizes, cementing_methods, operations
            )
        ]

        # Create the combined compatibility dictionary
        combined_compatibility_dict = {}

        for part_id in part_ids:
            # Split the part ID into its components
            orientation, prod_type, size, cementing, op = part_id.split("-")
            size_category = get_size_category(size, prod_type, cementing)

            # Initialize a list to hold compatible parts for the current part ID
            compatible_parts = deque()

            for other_part_id in part_ids:
                # Skip if comparing the part ID with itself
                if other_part_id == part_id:
                    continue

                # Split the other part ID into its components
                (
                    other_orientation,
                    other_type,
                    other_size,
                    other_cementing,
                    other_op,
                ) = other_part_id.split("-")
                other_size_category = get_size_category(other_size, other_type, other_cementing)

                # Compatibility rules for OP1
                if op == "OP1" and other_op == "OP1":
                    if size == other_size and cementing == other_cementing:
                        compatible_parts.append(other_part_id)

                # Compatibility rules for OP2
                elif op == "OP2" and other_op == "OP2":
                    if type == other_type and size_category == other_size_category:
                        if type == "CR" or (type == "PS" and cementing == other_cementing):
                            compatible_parts.append(other_part_id)

            # Assign the list of compatible parts to the current part ID in the dictionary
            combined_compatibility_dict[part_id] = compatible_parts

        return combined_compatibility_dict

    def build_ga_representation(
        self,
        croom_processed_orders: pd.DataFrame,
        croom_task_durations: pd.DataFrame,
        task_to_machines: Dict[int, List[int]],
        scheduling_options: dict,
    ) -> Dict[str, any]:
        """
        Builds the GA input data.

        Args:
            croom_processed_orders (pd.DataFrame): The processed orders.
            croom_task_durations (pd.DataFrame): The task durations.
            task_to_machines (Dict[int, List[int]]): The task to machines dictionary.
            scheduling_options (dict): The scheduling options.

        Returns:
            Dict[str, any]: The GA representation.
        """
        J = self.create_jobs(croom_processed_orders, scheduling_options)
        J_op_2 = self.create_jobs(croom_processed_orders, scheduling_options, operation="OP 2")

        # Combine jobs from both operations (Operation 1 and Operation 2) into one list of jobs (J)
        J.update(J_op_2)
        part_id_to_task_seq = self.create_part_id_to_task_seq(croom_processed_orders)
        M = self.create_machines(task_to_machines)
        dur = self.get_duration_matrix(
            J, part_id_to_task_seq, croom_processed_orders, croom_task_durations
        )

        input_repr_dict = {
            "J": J,
            "part_to_tasks": part_id_to_task_seq,
            "M": M,
            "dur": dur,
            "task_to_machines": task_to_machines,
        }

        return input_repr_dict

    @staticmethod
    def generate_arbor_mapping(input_repr_dict, cemented_arbors, cementless_arbors):
        # Initialize the dictionary to store the results
        arbor_mapping = {}

        # Extract part_ids from class attribute
        part_ids = [part_id for part_id, _ in input_repr_dict["J"].values()]

        # Filter and process part IDs
        for part_id in part_ids:
            if part_id.endswith("OP1"):
                # Extract components
                orientation, type_id, size, cement_type, operation = part_id.split("-")

                # Determine the correct arbor number based on cement type and size
                if cement_type == "CTD":
                    arbor_number = cemented_arbors.get(size)
                elif cement_type == "CLS":
                    arbor_number = cementless_arbors.get(size)
                else:
                    arbor_number = None

                # Add to the dictionary
                if arbor_number:
                    arbor_mapping[part_id] = arbor_number

        return arbor_mapping


class GeneticAlgorithmScheduler:
    """
    Contains all functions to run a genetic algorithm for a flexible job shop scheduling problem (FJSSP):
    - Initialize population
    - Evaluate fitness
    - Crossover/Offspring
    - Mutation (Currently not implemented)
    - Run

    Additional utility and helper functions are available.
    """

    def __init__(self):
        self.J = None
        self.M = None
        self.dur = None
        self.task_to_machines = None
        self.n = None
        self.n_e = None
        self.n_c = None
        self.P = None
        self.start_date = None
        self.scores = None
        self.day_range = None
        self.part_to_tasks = None
        self.best_schedule = None
        self.working_minutes_per_day = None
        self.total_minutes_per_day = None
        self.drag_machine_setup_time = None
        self.change_over_time_op1 = None
        self.change_over_time_op2 = None
        self.change_over_machines_op1 = None
        self.change_over_machines_op2 = None
        self.cemented_only_haas_machines = None
        self.compatibility_dict = None
        self.arbor_dict = None
        self.cemented_arbors = None
        self.urgent_orders = None
        self.urgent_multiplier = None
        self.max_iterations = None
        self.task_time_buffer = None

    def adjust_start_time(self, start: float, task: int = None) -> float:
        """
        Adjusts the start time to ensure it falls within working hours. If the start time is outside the
        working hours, it is pushed to the start of the next working day. Additionally, if the start time
        falls on a weekend, it is pushed to the following Monday.

        Args:
            start (float): The initial start time in minutes from the reference start date.
            task (int): The task number as in parameters task_to_machines dictionary.

        Returns:
            float: The adjusted start time in minutes from the reference start date.
        """
        # Convert start_date to datetime object
        starting_date: datetime = datetime.fromisoformat(self.start_date)

        # Determine the current day cycle start time in minutes
        current_day_start: float = (start // self.total_minutes_per_day) * self.total_minutes_per_day

        # TODO: Fix this
        # Adjust if start time is outside of working hours
        if start >= current_day_start + self.working_minutes_per_day:
            start = current_day_start + self.total_minutes_per_day

        # Determine the adjusted start date and time
        actual_start_datetime: datetime = starting_date + timedelta(minutes=start)

        # Adjust for weekends
        while True:
            weekday = actual_start_datetime.weekday()

            if weekday == 6:  # Sunday
                # Push to Monday morning
                start += self.total_minutes_per_day
            elif weekday == 5:  # Saturday
                if task not in [1, 0]:  # If not task type 1, push to Monday
                    start += self.total_minutes_per_day * 2
                else:
                    break  # Task type 1 can stay on Saturday
            else:
                break  # It's a weekday, no adjustment needed

            actual_start_datetime = starting_date + timedelta(minutes=start)

        return start

    def find_avail_m(self, start: int, job_id: int, task_id: int, after_hours_starts: int = 0) -> int:
        """
        Finds the next available time for a machine to start a task, considering the working day duration.
        Add 'task_time_buffer' between each task on a machine as switching time.

        Args:
            start (int): The starting time in the schedule in minutes.
            job_id (int): The index of the job in the job list.
            task_id (int): The task number within the job.
            after_hours_starts (int): The number of hours after which jobs can run overnight.

        Returns:
            int: The next available time for the machine to start the task.
        """

        # Extract part_id
        part_id, _ = self.J[job_id]

        # Duration of the task
        duration = self.dur[(job_id, task_id)]
        # Calculate next available time after the task and buffer
        next_avail_time = start + duration + self.task_time_buffer

        # Start date
        starting_date = datetime.fromisoformat(self.start_date)

        if task_id in [1, 30]:  # Task 1, 30 corresponds to HAAS machines
            if (
                after_hours_starts < 3
            ):  # Message from Bryan: 3 batches (36 parts total) can be preloaded in HAAS
                actual_start_datetime = starting_date + timedelta(minutes=next_avail_time)
                weekday = actual_start_datetime.weekday()
                time_in_day = actual_start_datetime.time()

                # If the start time is Sunday or Saturday after 7PM, push to next working day
                if weekday == 6 or (
                    weekday == 5 and time_in_day >= datetime.strptime("19:00", "%H:%M").time()
                ):
                    return int(self.adjust_start_time(next_avail_time))
                else:
                    return next_avail_time
            else:
                if next_avail_time >= self.adjust_start_time(next_avail_time, task_id):
                    return next_avail_time
                else:
                    return (
                        self.adjust_start_time(next_avail_time, task_id) // self.total_minutes_per_day
                    ) * self.total_minutes_per_day
        else:
            # For other tasks, ensure they are scheduled during working hours
            next_avail_time = self.adjust_start_time(next_avail_time, task_id)

            # Determine if next_avail_time needs to be adjusted further for working hours
            day_offset = (next_avail_time // self.total_minutes_per_day) * self.total_minutes_per_day
            time_in_day = next_avail_time % self.total_minutes_per_day

            if time_in_day >= self.working_minutes_per_day:
                overflow_time = time_in_day - self.working_minutes_per_day
                next_avail_time = day_offset + self.total_minutes_per_day + overflow_time
                logger.info(
                    f"next_avail_time: {next_avail_time}, day_offset: {day_offset}, self.working_minutes_per_day: {self.working_minutes_per_day}, time_in_day: {time_in_day}, overflow_time: {overflow_time}"
                )
            elif time_in_day < 0:
                next_avail_time = day_offset

            return next_avail_time

    def adjust_changeover_finish_time(self, start_time: float) -> float:
        """
        Adjusts the changeover start time to ensure it completes before the working day ends.

        If the start time is after the last valid time in the working day, it will be pushed to the next working day.

        Args:
        start_time (float): The original changeover start time in minutes.

        Returns:
        float: The adjusted changeover start time.
        """
        # Calculate the valid start time window
        changeover_duration = self.change_over_time_op1
        valid_start_window_end = self.working_minutes_per_day - changeover_duration

        # Check if the start time is within the valid start window
        if (
            start_time
            < (start_time // self.total_minutes_per_day) * self.total_minutes_per_day
            + valid_start_window_end
        ):
            return start_time
        else:
            # If not, move to the start of the next working day
            next_working_day_start = (
                start_time // self.total_minutes_per_day
            ) * self.total_minutes_per_day + self.total_minutes_per_day

            # Adjust for weekends
            starting_date = datetime.fromisoformat(self.start_date)
            next_start_datetime = starting_date + timedelta(minutes=next_working_day_start)
            while next_start_datetime.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
                next_working_day_start += self.total_minutes_per_day
                next_start_datetime = starting_date + timedelta(minutes=next_working_day_start)

            return next_working_day_start

    def count_arbor_frequencies(self) -> Counter:
        """
        Counts the frequency of each arbor used for parts that require the 'OP1' operation.

        This method:
        1. Extracts the arbor values associated with each part ID that ends with 'OP1'.
        2. Uses the Counter class to count how often each arbor appears in the list.

        Returns:
            Counter: A Counter object where keys are arbor numbers and values are their corresponding frequencies.
        """

        # Step 1: Extract the arbor values for each part_id that ends with 'OP1'
        # We use a nested list comprehension to filter part_ids that end with 'OP1' after first extracting part_ids
        # from self.J, and then use the arbor_dict to get the corresponding arbor value.
        jobs_per_arbor = [
            self.arbor_dict.get(part_id)
            for part_id in [part_id for part_id, _ in self.J.values()]
            if part_id.endswith("OP1")
        ]

        # Step 2: Use Counter to count the frequencies of each arbor value
        # Counter will create a dictionary-like object where the keys are arbor numbers
        # and the values are the counts of how often they appear in the jobs_per_arbor list.
        arbor_frequency = Counter(jobs_per_arbor)

        return arbor_frequency

    def assign_arbors_to_machines(self, arbor_frequency: Counter) -> Dict[str, List[int]]:
        """
        Assigns arbors to machines based on their frequency and type (cemented or cementless).

        This method:
        1. Initializes an empty assignment dictionary where each arbor is associated with a list of machines.
        2. Iterates over the arbors, checking if they are cemented or cementless.
        3. Assigns each arbor to a set number of machines based on its frequency.
        4. Cementless arbors can be assigned to machines [1, 2, 3, 4, 5, 6] while cemented arbors are limited to machines [1, 2, 3].
        5. The method uses a random choice mechanism to vary the machines to which arbors are assigned.

        Args:
            arbor_frequency (Counter): A Counter object with arbors as keys and their frequency of use as values.

        Returns:
            Dict[str, List[int]]: A dictionary where the keys are arbor numbers and the values are lists of machine numbers to which they are assigned.
        """

        # Initialize the assignment dictionary
        arbor_to_machines = {arbor: [] for arbor in arbor_frequency}

        # Initialize indices for cementless and cemented machines
        machine_index_cementless = 0
        machine_index_cemented = 0

        # Initialize machine lists
        machines_cementless = list(reversed(self.change_over_machines_op1))
        machines_cemented = self.cemented_only_haas_machines

        # Randomly select machines
        # For cementless: [6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2] or [6, 5, 4, 3]
        # The latter options have fewer machines but also less overlap with cemented machines
        selected_machines_cls = random.choice(
            [machines_cementless, machines_cementless[:-1], machines_cementless[:-2]]
        )
        selected_machines_ctd = random.choice([machines_cemented, machines_cemented[:-1]])

        for arbor, frequency in arbor_frequency.items():
            # Create a boolean for the cemented status
            cemented = arbor in self.cemented_arbors.values()

            # Determine if the arbor is cementless or cemented based on its number
            if not cemented:
                # Cementless arbors
                # Randomly select all: [6, 5, 4, 3, 2, 1], first five: [6, 5, 4, 3, 2], or first four: [6, 5, 4, 3]
                # Selecting less will lead to less overlap with cemented arbors
                machines = selected_machines_cls
                machine_index = machine_index_cementless
            else:
                # Cemented arbors
                # Randomly select from cemented machines: [1, 2, 3] or [1, 2]
                machines = selected_machines_ctd
                machine_index = machine_index_cemented

            # Determine the number of machines to assign based on frequency
            num_machines_to_assign = random.choice([1, 2])

            # Assign the arbor to the appropriate number of machines
            for _ in range(num_machines_to_assign):
                arbor_to_machines[arbor].append(machines[machine_index])
                machine_index = (machine_index + 1) % len(machines)

            # Update the machine index for the next iteration
            if not cemented:
                machine_index_cementless = machine_index
            else:
                machine_index_cemented = machine_index

        return arbor_to_machines

    def pick_early_machine(
        self,
        task_id: int,
        avail_m: Dict[int, int],
        random_roll: float,
        prob: float = 0.75,
    ) -> int:
        """
        Selects a machine for the given task based on availability and compatibility.
        There is a chance of 'prob' to select the machine that comes available earliest,
        otherwise a random machine is picked.

        Parameters:
        [- job_id (int): ID of the job.] No longer needed
        - task (int): Index of the task within the job.
        - avail_m (Dict[int, int]): A dictionary with machine IDs as keys and their available times as values.
        - random_roll (float): A random number to decide the selection strategy.
        - prob (float): Probability to pick the earliest available compatible machine.

        Returns:
        - int: The selected machine ID.
        """
        compat_with_task = self.task_to_machines[
            task_id
        ]  # A list of machines that are compatible with this task

        if random_roll < prob:
            m = min(compat_with_task, key=lambda x: avail_m.get(x))
        else:
            m = random.choice(compat_with_task)

        return m

    def slack_window_check(self, slack: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Check and adjust a given slack time window to ensure it falls within valid working hours.

        Args:
            slack (Tuple[float, float]): A tuple containing the start and end times of the slack window.

        Returns:
            Optional[Tuple[float, float]]: The adjusted slack window if valid, or None if invalid.

        The function operates as follows:
        1. Extracts the start and end times from the slack tuple.
        2. Validates that both times are floats.
        3. Calculates the current working day's start and end times based on the total minutes per day and working minutes per day.
        4. Checks if the start time falls within the valid working window.
        5. Adjusts the end time if it exceeds the valid working window.
        6. Checks if the adjusted end time falls within the valid working window.
        7. Returns the adjusted slack window if valid, otherwise returns None.
        """

        start_time, end_time = slack

        # Check if the times are valid floats
        assert isinstance(start_time, (int, float)) and isinstance(
            end_time, (int, float)
        ), f"Expected numeric type for start- and end time, received: {start_time}, {end_time}"

        # Determine the window in which the start_time falls
        window_start = (start_time // self.total_minutes_per_day) * self.total_minutes_per_day
        window_end = window_start + self.working_minutes_per_day

        # Check if the start_time is within the valid window
        if start_time < window_start or start_time >= window_end:
            return None

        # Adjust the end_time if it exceeds the valid window
        if end_time > window_end:
            end_time = window_end

        # Check if the end_time falls within the valid window
        if end_time <= window_start or end_time > window_end:
            return None

        return start_time, end_time

    def count_after_hours_start(self, P_j: deque, m: int, start: int) -> int:
        """
        Counts the number of tasks in the schedule `P_j` where the machine `m` is used
        and the start time is within dynamically calculated threshold limits based on `start`.

        Parameters:
        P_j (deque of tuples): The schedule list where each tuple has the following format:
                               (job_idx, task_num, m, start, duration, task_idx, part_id)
        m (int): The machine identifier to filter by.
        start (int): The start time to determine the threshold range.

        Returns:
        int: The count of tasks where `m == m` and the start time falls within the calculated threshold range.
        """

        # Calculate the threshold limits only once
        multiplier = (start // self.total_minutes_per_day) + 1
        threshold_lower = (multiplier - 1) * self.total_minutes_per_day + self.working_minutes_per_day
        threshold_upper = multiplier * self.total_minutes_per_day

        # Use sum with a generator expression to count the filtered tasks without creating a list
        return sum(1 for task in P_j if task[2] == m and threshold_lower < task[3] < threshold_upper)

    def get_preferred_machines(
        self,
        compat_task_0: List[int],
        product_m: Dict[int, int],
        job_id: int,
        fixture_to_machine_assignment: Dict[str, List[int]],
    ) -> List[int]:
        """
        Get the preferred machines for a given task based on the compatibility, product ID, job index,
        and fixture to machine assignment.

        Args:
            compat_task_0 (List[int]): A list of machines that can process the task.
            product_m (Dict[int, int]): A dictionary mapping machines to product IDs.
            job_id (int): The ID-number of the job.
            fixture_to_machine_assignment (Dict[str, List[int]]): A dictionary mapping fixtures to machines.

        Returns:
            List[int]: A list of preferred machines for the given task.
        """

        # Extract the job id
        part_id, _ = self.J[job_id]

        # Find preferred machines
        # 1.) Machines that processed the exact part_id
        # 2.) Machines that processed a compatible part_id
        # Machines that have not processed anything yet
        preferred_machines = [
            machine
            for machine in compat_task_0
            if product_m[machine] == 0
            or product_m[machine] == part_id
            or product_m[machine] in self.compatibility_dict[part_id]
        ]

        # Extract the appropriate arbor from custom part ID
        arbor = self.arbor_dict[part_id]

        # Extract the valid machines from the fixture_to_machine_assignment
        valid_machines = fixture_to_machine_assignment[arbor]

        # Filter the preferred machines to only include those that are valid for the current arbor
        preferred_machines = [machine for machine in preferred_machines if machine in valid_machines]

        # If no preferred machines are found, use the valid machines for the current arbor
        if not preferred_machines:
            preferred_machines = valid_machines

        # Return the preferred machines
        return preferred_machines

    def slack_logic(
        self,
        m: int,
        avail_m: Dict[int, int],
        slack_m: Dict[Any, deque],
        slack_time_used: bool,
        previous_task_start: float,
        previous_task_dur: float,
        current_task_dur: float,
        changeover_duration: int = 0,
    ):
        """
        Determine the start time for a task on a machine, considering machine availability and existing slack time.

        Args:
            m (int): The machine identifier.
            avail_m (Dict[int, int]): Dictionary mapping machine IDs to their available times.
            slack_m (Dict[int, List]): Dictionary mapping machine IDs to their slack windows (tuples of start and end times).
            slack_time_used (bool): Flag indicating whether slack time has been used.
            previous_task_start (float): The start time of the previous task.
            previous_task_dur (float): The duration of the previous task.
            current_task_dur (float): The duration of the current task.
            changeover_duration (int): Changeover duration in whole minutes.

        Returns:
            Tuple[float, bool]: A tuple containing the determined start time for the current task and a boolean indicating
            whether slack time was used.

        The function operates as follows:
        1. Initializes the `start` variable to `None`.
        2. Checks if the previous task ends after the machine becomes available. If so:
            - Sets `start` to the completion time of the previous task.
            - Adds a slack window representing the time between the machine becoming available and the new task's start time.
        3. If the previous task does not overlap the machine's availability:
            - Iterates over existing slack windows for the machine.
            - Checks if the task can fit within any slack window:
                - Sets `start` to the later of the slack window's start or the previous task's end.
                - Removes the used slack window.
                - Adds new slack windows for any remaining time before or after the task within the original slack window.
                - Sets `slack_time_used` to `True` if a slack window is used.
        4. If no slack time is used, sets `start` to the machine's available time.
        5. Logs a warning if no valid start time is determined.
        6. Returns the `start` time and the `slack_time_used` flag.
        """

        # Initialize start variable
        start = None

        # Define previous task finish
        previous_task_finish = previous_task_start + previous_task_dur

        # If the previous task is completed later than the new machine comes available
        if previous_task_finish > avail_m[m]:
            # Start time is the completion of the previous task of the job in question
            start = self.adjust_start_time(
                previous_task_finish + changeover_duration + self.task_time_buffer
            )

            # Difference between the moment the machine becomes available and the new tasks starts is slack
            # e.g.: machine comes available at 100, new task can only start at 150, slack = (100, 150)
            # We subtract changeover_duration, because even though the task actually starts later,
            # the changeover_duration cannot be used for a different task
            slack_window_upd = self.slack_window_check(
                (avail_m[m], start - changeover_duration - self.task_time_buffer)
            )

            if slack_window_upd:
                slack_m[m].append(slack_window_upd)

        else:
            # Loop over slack in this machine
            for unused_time in slack_m[m]:
                # If the unused time + duration of task is less than the end of the slack window
                if (
                    max(unused_time[0], previous_task_finish)
                    + changeover_duration
                    + current_task_dur
                    + self.task_time_buffer
                ) < unused_time[1]:
                    # New starting time is the largest of the beginning of the slack time or the time when the
                    # previous task of the job is completed
                    # Task can only start once changeover is completed
                    start = (
                        max(unused_time[0], previous_task_finish)
                        + changeover_duration
                        + self.task_time_buffer
                    )

                    # Remove the slack period if it has been used
                    slack_m[m].remove(unused_time)

                    # We add the remaining time between when the task finishes and the end of the slack window
                    # as a new slack window
                    # e.g.: original slack = (100, 150), task planned now takes (110, 130), new slack = (130, 150)
                    # changeover_duration must be added because it delays the task
                    slack_window_upd = self.slack_window_check(
                        (
                            (
                                max(unused_time[0], previous_task_finish)
                                + changeover_duration
                                + current_task_dur
                                + self.task_time_buffer
                            ),
                            unused_time[1],
                        )
                    )

                    if slack_window_upd:
                        slack_m[m].append(slack_window_upd)

                    # Append another slack window if previous start was not at the beginning of the slack window,
                    # in this case there is still some time between when the machine comes available and when the
                    # task starts
                    # e.g. original slack = (100, 150), task planned now takes (110, 130), new slack = (100, 110)
                    # We subtract changeover_duration, because even though the task actually starts later,
                    # the changeover_duration cannot be used for a different task
                    if start == (previous_task_finish + changeover_duration + self.task_time_buffer):
                        slack_window_upd = self.slack_window_check(
                            (unused_time[0], previous_task_finish)
                        )

                        if slack_window_upd:
                            slack_m[m].append(slack_window_upd)

                    slack_time_used = True
                    # break the loop if a suitable start time has been found in the slack
                    break

            # If slack time is not used, start when the machine becomes available
            if not slack_time_used:
                start = self.adjust_start_time(avail_m[m] + changeover_duration)

        if start is None:
            logger.warning("No real start time was defined!")

        return start, slack_time_used

    def generate_individual(
        self, arbor_frequencies: Counter
    ) -> Deque[Tuple[int, int, int, float, float, int, str]]:
        """
        Generates an individual schedule (list of tasks assigned to machines with start times).

        Args:
            arbor_frequencies (Dict[int, int]): Frequencies of arbors to guide machine assignments.

        Returns:
            Deque[Tuple[int, int, int, float, float, int, str]]:
                - A deque representing a proposed schedule, where each element is a tuple containing:
                  (job_id, task_id, machine_id, start_time, duration, placeholder, part_id).

        The function operates as follows:
        1. Initializes availability and product tracking dictionaries for machines.
        2. Shuffles or sorts the job list based on a random roll.
        3. Assigns tasks to machines based on compatibility, availability, and operation type.
        4. Updates machine availability and product tracking after each task assignment.
        5. Returns the generated individual schedule.
        """
        avail_m = {m: 0 for m in self.M}
        slack_m = {m: deque() for m in self.M}
        product_m = {m: 0 for m in self.M}
        changeover_finish_time = deque([0])
        P_j = deque()

        # Create a temporary copy of the Job IDs
        J_temp = list(self.J.keys())

        # Generate a random float [0, 1]
        random_roll = random.random()

        # Create machine assignment based on fixtures
        fixture_to_machine_assignment = self.assign_arbors_to_machines(arbor_frequencies)

        # Based on the random number we either randomly shuffle or apply some sorting logic
        if random_roll < 0.6:
            random.shuffle(J_temp)
        elif random_roll < 0.7:
            J_temp.sort(key=lambda x: self.J[x][0])  # Sort on the part ID
        elif random_roll < 0.8:
            J_temp.sort(key=lambda x: self.J[x][1], reverse=True)  # Sort on the due time
        elif random_roll < 0.9:
            J_temp.sort(key=lambda x: (self.J[x][0], self.J[x][1]), reverse=True)
        else:
            for job in self.urgent_orders:
                J_temp.remove(job)
                J_temp.append(job)

        # While our list of jobs is not empty
        while J_temp:
            # Take the job id at the end of the list and remove it from the list
            job_id = J_temp.pop()
            part_id, _ = self.J[job_id]

            # Loop over the tasks in the job (i.e. get the tasks for the part ID for this job)
            for task_id in self.part_to_tasks[part_id]:
                # Generate random float [0, 1]
                random_roll = random.random()

                # New boolean to track if the task is planned during slack time,
                # if so we do not need to update avail_m
                slack_time_used = False

                # Conditional logic; separate flow for first task of OP1 (HAAS)
                if task_id in [1, 30]:
                    compat_task_0 = self.task_to_machines[task_id]
                    preferred_machines = self.get_preferred_machines(
                        compat_task_0, product_m, job_id, fixture_to_machine_assignment
                    )
                    m = (
                        min(preferred_machines, key=lambda x: avail_m.get(x))
                        if random_roll < 0.5
                        else random.choice(preferred_machines)
                    )
                    if (
                        product_m.get(m) == 0
                        or product_m.get(m) == part_id
                        or product_m.get(m) in self.compatibility_dict[part_id]
                    ):
                        start = avail_m[m]
                    else:
                        start = (
                            self.adjust_changeover_finish_time(
                                avail_m[m] + max((changeover_finish_time[-1] - avail_m[m]), 0)
                            )
                            + self.change_over_time_op1
                        )
                        changeover_finish_time.append(start)
                else:
                    m = self.pick_early_machine(task_id, avail_m, random_roll)
                    changeover_duration = 0

                    # Determine the changeover duration. self.task_time_buffer will also be added
                    # in all cases, either in slack_logic() or find_avail_m()
                    if m in self.change_over_machines_op2:
                        if (
                            product_m.get(m) == 0
                            or product_m.get(m) == part_id
                            or product_m.get(m) in self.compatibility_dict[part_id]
                        ):  # Previous part was the same or compatible, or there wasn't a previous part
                            changeover_duration = self.drag_machine_setup_time
                        else:
                            changeover_duration = self.change_over_time_op2

                    if task_id in [1, 10, 30]:
                        start = self.adjust_start_time(avail_m[m] + changeover_duration, task_id)

                    else:
                        start, slack_time_used = self.slack_logic(
                            m,
                            avail_m,
                            slack_m,
                            slack_time_used,
                            P_j[-1][3],  # Previous task start
                            P_j[-1][4],  # Previous task duration
                            self.dur[(job_id, task_id)],
                            changeover_duration,
                        )

                P_j.append(
                    (
                        job_id,
                        task_id,
                        m,
                        start,
                        self.dur[(job_id, task_id)],
                        0,
                        part_id,
                    )
                )

                after_hours_starts = 0
                if m in self.change_over_machines_op1:
                    after_hours_starts = self.count_after_hours_start(P_j, m, start)

                if not slack_time_used:
                    avail_m[m] = self.find_avail_m(start, job_id, task_id, after_hours_starts)

                product_m[m] = part_id

        return P_j

    def parallel_init_population(
        self, num_inds: int = None, arbor_frequencies: Counter = None, fill_inds: bool = False
    ) -> Deque[Deque[Tuple[int, int, int, float, float, int, str]]]:
        """
        Initializes the population of schedules in parallel using multiprocessing.

        Args:
            num_inds (int, optional): Number of individuals (schedules) to generate. If None, uses `self.n`. Defaults to None.
            arbor_frequencies (Counter, optional): Frequencies of arbors to guide machine assignments. Defaults to None.
            fill_inds (bool, optional): Flag indicating whether to fill individuals in the population or return them.
                                        Defaults to False.

        Returns:
            Union[None, Deque[Deque[Tuple[int, int, int, float, float, int, str]]]]:
                - If `fill_inds` is False, updates `self.P` with the generated population.
                - If `fill_inds` is True, returns the generated population.

        The function operates as follows:
        1. Sets `num_inds` to `self.n` if it is not provided.
        2. Counts arbor frequencies to guide machine assignments.
        3. Initializes a deque `P` to store the population and sets up progress logging.
        4. Uses `ProcessPoolExecutor` to generate individuals in parallel.
        5. Adds each generated individual to the population if it is unique.
        6. Returns `P`
        """
        if num_inds is None:
            num_inds = self.n

        if not fill_inds:
            logger.info(f"Arbor frequencies: {arbor_frequencies}")

        P = deque()
        percentages = np.arange(10, 101, 10)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.generate_individual, arbor_frequencies) for _ in range(num_inds)
            ]
            for i, future in enumerate(futures):
                P_j = future.result()

                # We risk creating duplicates because once we have a large population P, it becomes very time-consuming
                # to check for every new schedule if it is/a duplicate of any of the existing schedules
                P.append(P_j)

                if not fill_inds and i * 100 / num_inds in percentages:
                    logger.info(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {i * 100 / num_inds}% of schedules have been created."
                    )

        return P

    @staticmethod
    def negative_exponentiation(value, exponent):
        """
        Performs a negative exponentiation operation. This is used to penalize jobs that are completed extremely late.
        If this is not done, the model will consider one extremely late order and one extremely early order equivalent
        to one slightly late and one slightly early order. We effectively want to penalize extremely long lead times.

        Note: This function is used to handle negative values, as the original model does not support negative values.

        The function operates as follows:
        1. Checks if the value is less than 0. If so, it returns -(absolute value of the value raised to the exponent).
        2. If the value is greater than or equal to 0, it returns the value itself raised to the exponent.
        Args:
            value: The number to exponentiate
            exponent: The order to exponentiate by

        Returns:
            value: Exponentiated input number
        """
        if value < 0:
            return -(abs(value) ** exponent)
        if value >= 0:
            return value

    def evaluate_population(
        self,
        best_scores: deque = None,
        display_scores: bool = True,
        on_time_bonus: int = 5000,
    ):
        """
        Evaluates the population of schedules by calculating a score for each schedule based on the completion times
        of jobs versus their due dates.

        The function iterates over each schedule in the population, calculates the score for each schedule, and updates
        the scores list. The score for each schedule is determined by the difference between the due date and the
        completion time of the final task, with penalties for lateness and bonuses for on-time completion.

        Args:
            best_scores (deque, optional): A deque to store the best scores of the population. Defaults to None.
            display_scores (bool, optional): If True, logs the best, median, and worst scores. Defaults to True.
            on_time_bonus (int, optional): A fixed bonus added to the score for jobs completed on time. Defaults to 10000.

        Returns:
            None

        Notes:
            - The function uses the `negative_exponentiation` method to penalize jobs that are completed extremely late.
            - The final task for OP1 is task 7 and for OP2 is task 19.
            - The HAAS tasks are defined by task IDs 1 and 0.
            - The `urgent_multiplier` is applied to urgent jobs to increase their penalty for lateness.
        """
        # Calculate scores for each schedule
        # Note: self.J[job_id] gives the tuple (Due time, Part ID) for a given job ID
        self.scores = [
            round(
                sum(
                    (
                        # Difference between due date and completion time, multiplied by urgent_multiplier if urgent.
                        self.negative_exponentiation(
                            (self.J[job_id][1] - (start_time + job_task_dur)),
                            1.02,
                        )
                        * (2 if task_id in [1, 30] else 1)
                        * (self.urgent_multiplier if job_id in self.urgent_orders else 1)
                        + (
                            # Fixed size bonus for completing the job on time (only applies if the final task is
                            # completed on time)
                            on_time_bonus
                            if (self.J[job_id][1] - (start_time + job_task_dur)) > 0
                            and task_id in [7, 20, 44]
                            else 0
                        )
                    )
                    for (
                        job_id,
                        task_id,
                        machine,
                        start_time,
                        job_task_dur,
                        _,
                        _,
                    ) in schedule
                    # Only consider the completion time of the final task
                    if task_id in [7, 20, 44] or task_id in [1, 30]
                )
            )
            # Evaluate each schedule in the population
            for schedule in self.P
        ]

        # if display_scores:
        #     logger.info(
        #         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Best score: {max(self.scores)}, "
        #         f"Median score: {np.median(self.scores)}, Worst score: {min(self.scores)}"
        #     )
        #     best_scores.append(max(self.scores))

        if display_scores:
            best_score = round(max(self.scores))
            top_1_percent = round(np.percentile(self.scores, 99))
            top_5_percent = round(np.percentile(self.scores, 95))
            top_10_percent = round(np.percentile(self.scores, 90))
            median_score = round(np.median(self.scores))
            worst_score = round(min(self.scores))

            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Best score: {best_score}, "
                f"Top 1% score: {top_1_percent}, Top 5% score: {top_5_percent}, Top 10% score: {top_10_percent}, "
                f"Median score: {median_score}, Worst score: {worst_score}"
            )
            best_scores.append(best_score)

    def resolve_conflict(self, P_prime: list) -> deque:
        """
        This function resolves conflicts in a given schedule. If tasks are planned on the same machine at the same time,
        it finds the first available time for each task to start on the machine.

        Parameters:
        P_prime (deque): A list of tuples where each tuple represents a task.
        Each task is represented as (job id, task id, machine, start time, duration, task index, part id).

        Returns:
        P_prime_sorted (List[Tuple[int, int, int, int, int, int, str]]): A sorted list of tuples where each tuple
        represents a task. Each task is represented as (job id, task id, machine, start time, duration, task index, part id).

        """
        # Initialize an empty list to hold tasks for this proposed schedule
        P_prime_sorted = deque()
        avail_m = {m: 0 for m in self.M}
        slack_m = {m: deque() for m in self.M}
        product_m = {m: 0 for m in self.M}
        changeover_finish_time = deque([0])

        # Count arbor frequencies
        arbor_frequencies = self.count_arbor_frequencies()

        # Create machine assignment based on fixtures
        fixture_to_machine_assignment = self.assign_arbors_to_machines(arbor_frequencies)

        # Loop over the jobs in the job list (J)
        for job_id in list(self.J.keys()):
            part_id, _ = self.J[job_id]

            # We have a list of tuples, where each tuple stands for a task in a proposed schedule
            # We filter all the tuples for ones belonging to a specific job_idx (first field of the tuple)
            job_tasks = sorted([entry for entry in P_prime if entry[0] == job_id], key=lambda x: x[1])

            # Loop over the tasks one by one
            for task_entry in job_tasks:
                _, task_id, m, _, _, task_idx, _ = task_entry
                slack_time_used = False

                if task_id in [1, 30]:
                    # Start time is the time that the machine comes available if no changeover is required
                    # else, the changeover time is added, and an optional waiting time if we need to wait
                    # for another changeover to finish first (only one changeover can happen concurrently)

                    # Extract compatible HAAS machines for first task
                    compat_task_0 = self.task_to_machines[task_id]

                    # New preferred machines logic
                    preferred_machines = self.get_preferred_machines(
                        compat_task_0, product_m, job_id, fixture_to_machine_assignment
                    )

                    # If the selected machine is not in preferred machines yet, select from preferred machines
                    if m not in preferred_machines:
                        m = random.choice(preferred_machines)

                    if (
                        product_m.get(m) == 0
                        or product_m.get(m) == part_id
                        or product_m.get(m) in self.compatibility_dict[part_id]
                    ):
                        start = avail_m[m]
                    else:
                        # If the changeover would not be finished before the end of day,
                        # it is pushed to the next morning
                        start = (
                            self.adjust_changeover_finish_time(
                                avail_m[m] + max((changeover_finish_time[-1] - avail_m[m]), 0)
                            )
                            + self.change_over_time_op1
                        )

                        # Update time that a mechanic becomes available for a new changeover
                        changeover_finish_time.append(start)

                else:
                    # Initialize changeover time to 0
                    changeover_duration = 0

                    # Determine the changeover duration. self.task_time_buffer will also be added
                    # in all cases, either in slack_logic() or find_avail_m()
                    if m in self.change_over_machines_op2:
                        if (
                            product_m.get(m) == 0
                            or product_m.get(m) == part_id
                            or product_m.get(m) in self.compatibility_dict[part_id]
                        ):  # Previous part was the same or compatible, or there wasn't a previous part
                            changeover_duration = self.drag_machine_setup_time
                        else:
                            changeover_duration = self.change_over_time_op2

                    # First tasks of OP1 and OP2 do not need to consider slack
                    if task_id in [10]:
                        start = self.adjust_start_time(avail_m[m] + changeover_duration, task_id)
                    else:
                        start, slack_time_used = self.slack_logic(
                            m,
                            avail_m,
                            slack_m,
                            slack_time_used,
                            P_prime_sorted[-1][3],
                            P_prime_sorted[-1][4],
                            self.dur[(job_id, task_id)],
                            changeover_duration,
                        )

                # If slack time is used no need to update latest machine availability
                if not slack_time_used:
                    avail_m[m] = self.find_avail_m(start, job_id, task_id)

                # Record part ID of the latest product to be processed on a machine for changeovers
                product_m[m] = part_id

                # Issue warning if 'start' is still not defined after loop
                if start is None:
                    logger.warning(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - No start time found for job ID {job_id}, "
                        f"task ID {task_id}, machine {m}"
                    )

                # Add the task to the sorted list of tasks in this proposed schedule
                P_prime_sorted.append(
                    (
                        job_id,
                        task_id,
                        m,
                        start,
                        self.dur[(job_id, task_id)],
                        task_idx,
                        part_id,
                    )
                )

                # Count the number of after hours HAAS starts
                after_hours_starts = self.count_after_hours_start(P_prime_sorted, m, start)

                # If slack time is used no need to update latest machine availability
                if not slack_time_used:
                    avail_m[m] = self.find_avail_m(start, job_id, task_id, after_hours_starts)

                # Record part ID of the latest product to be processed on a machine for changeovers
                product_m[m] = part_id

        return P_prime_sorted

    def find_best_schedules(self) -> deque:
        """
        This method evaluates the population, sorts them based on their scores in descending order,
        and retains the top schedules based on a specified retention count. The retention count is
        the maximum of 3 or the product of the length of the population and a specified ratio.

        Returns:
            P_0 (deque): The list of top schedules based on their scores.
        """
        self.evaluate_population(display_scores=False)
        scored_population = sorted(zip(self.scores, self.P), key=lambda x: x[0], reverse=True)
        retain_count = max(5, int(len(self.P) * self.n_e))
        P_0 = deque()
        for score, schedule in scored_population[:retain_count]:
            P_0.append(schedule)

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
            sch1, sch2 = random.sample(P_0, 2)
            P_prime = [
                entry
                for job_id in list(self.J.keys())
                for entry in random.choice([sch1, sch2])
                if entry[0] == job_id
            ]
            P_prime = self.resolve_conflict(P_prime)
            if P_prime not in P_0:
                P_0.append(P_prime)

        self.P = P_0

    def mutate(self) -> None:
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
            b. Identify pairs of jobs with the same number of tasks, identical durations and compatible part IDs.
            c. Randomly select a pair of jobs up to four times.
            d. Swap the start times and machines of tasks between the selected jobs.
            e. If the new schedule is unique, add it to the list of new schedules.
        4. Add all new schedules to the population.

        This method helps in exploring new potential solutions by making modifications to
        existing ones, promoting diversity in the population.

        Returns:
            None
        """

        # After the offspring function the population has already shrunk significantly,
        # so we can use the whole population
        P_0 = self.find_best_schedules()

        # Make a deep copy of P_0
        P_1 = copy.deepcopy(P_0)

        # Initialize a list of new schedules
        new_schedules = deque()

        for schedule in P_1:
            # Group tasks by job_id
            jobs = defaultdict(list)
            for task in schedule:
                jobs[task[0]].append(task)

            # Find pairs of jobs with same number of tasks and same durations
            job_pairs = deque()
            job_list = list(jobs.items())

            for i in range(len(job_list)):
                for j in range(i + 1, len(job_list)):
                    job1, task_details_1 = job_list[i]
                    job2, task_details_2 = job_list[j]

                    # Append if the part ID matches for a pair
                    # Each job consists of multiple tuples for tasks, we will check if the number of tasks is the same
                    # and then if the duration of all tasks matches
                    if len(task_details_1) == len(task_details_2):
                        # The third last field in the task tuple is the duration
                        all_durations_match = all(
                            task1[-3] == task2[-3]
                            for task1, task2 in zip(task_details_1, task_details_2)
                        )

                        if all_durations_match:
                            # Extract custom part ID's
                            # Should we remove the part ID from the schedule?
                            custom_part_id_1 = task_details_1[0][-1]
                            custom_part_id_2 = task_details_2[0][-1]

                            # Check compatibility in both dictionaries safely
                            # This is required to make sure no extra changeover will be needed due to mutation
                            compat_op = custom_part_id_1 in self.compatibility_dict.get(
                                custom_part_id_2, []
                            )

                            # If the part ID between the jobs is the same, or they are compatible append to job pairs
                            if (custom_part_id_1 == custom_part_id_2) or compat_op:
                                job_pairs.append((job1, job2))

            # If no pairs found, continue to the next schedule
            if not job_pairs:
                continue

            # Randomly select a pair of jobs (at least once, but up to three times)
            for _ in range(random.randint(1, 4)):
                job1, job2 = random.choice(job_pairs)

                # Remove all entries in job_pairs where either job1 or job2 appears
                # This ensures that we do not mutate the same job multiple times
                job_pairs = [
                    (j1, j2) for j1, j2 in job_pairs if j1 not in (job1, job2) and j2 not in (job1, job2)
                ]

                # Swap the start times of the tasks in the selected jobs
                tasks1 = jobs[job1]
                tasks2 = jobs[job2]

                for i in range(len(tasks1)):
                    task1 = tasks1[i]
                    task2 = tasks2[i]

                    # Create new tasks with swapped start times and machines
                    # task_details follow this format:
                    # (job_id, task_id, machine, start_time, duration, task_index, part_id)
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

    def perform_iteration(self, iteration: int, best_scores: Deque, gene_pool: Deque) -> int:
        """
        Perform a single iteration of the genetic algorithm.

        This method logs the current iteration, evaluates the population, mutates the population,
        checks and adjusts the population size, and increments the iteration counter.

        Args:
            iteration (int): The current iteration number.
            best_scores (Deque): A deque to store the best scores.
            gene_pool (Deque): The gene pool from which new individuals can be sampled.

        Returns:
            int: The incremented iteration number.
        """
        logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Iteration {iteration + 1}")
        self.evaluate_population(best_scores=best_scores)
        self.mutate()
        if len(self.P) < self.n:
            self.P += random.sample(gene_pool, self.n - len(self.P))
        iteration += 1
        return iteration

    def run(
        self,
        input_repr_dict: Dict[str, Any],
        scheduling_options: Dict[str, Any],
        compatibility_dict: Dict[str, Any],
        arbor_dict: Dict[str, Any],
        cemented_arbors: Dict[str, str],
    ) -> Tuple[Any, deque]:
        """
        Runs the genetic algorithm by initializing the population, evaluating it, and selecting the best schedule.

        Args:
            input_repr_dict (Dict[str, Any]): A dictionary containing the necessary input variables for the GA.
            scheduling_options (Dict[str, Any]): Dictionary containing hyperparameters for running the algorithm.
            compatibility_dict (Dict[str, Any]): Dictionary containing the compatibility information for changeovers.
            arbor_dict (Dict[str, Any]): Dictionary containing the arbor information for changeovers [custom_part_id: arbor_num].
            cemented_arbors (Dict[str, str]): Dictionary containing the cemented arbor information.

        Returns:
            Tuple[Any, deque]: The best schedule with the highest score and the
            deque of best scores per generation.
        """
        self.J = input_repr_dict["J"]
        self.M = input_repr_dict["M"]
        self.dur = input_repr_dict["dur"]
        self.task_to_machines = input_repr_dict["task_to_machines"]
        self.part_to_tasks = input_repr_dict["part_to_tasks"]
        self.n = scheduling_options["n"]
        self.n_e = scheduling_options["n_e"]
        self.n_c = scheduling_options["n_c"]
        self.start_date = scheduling_options["start_date"]
        self.working_minutes_per_day = scheduling_options["working_minutes_per_day"]
        self.total_minutes_per_day = scheduling_options["total_minutes_per_day"]
        self.drag_machine_setup_time = scheduling_options["drag_machine_setup_time"]
        self.change_over_time_op1 = scheduling_options["change_over_time_op1"]
        self.change_over_time_op2 = scheduling_options["change_over_time_op2"]
        self.change_over_machines_op1 = scheduling_options["change_over_machines_op1"]
        self.change_over_machines_op2 = scheduling_options["change_over_machines_op2"]
        self.cemented_only_haas_machines = scheduling_options["cemented_only_haas_machines"]
        self.compatibility_dict = compatibility_dict
        self.arbor_dict = arbor_dict
        self.cemented_arbors = cemented_arbors
        self.max_iterations = scheduling_options["max_iterations"]
        self.urgent_multiplier = scheduling_options["urgent_multiplier"]
        self.task_time_buffer = scheduling_options["task_time_buffer"]
        self.urgent_orders = [job_idx - 1 for job_idx in scheduling_options["urgent_orders"]]
        self.day_range = np.arange(
            self.working_minutes_per_day,
            len(self.J) // 5 * self.working_minutes_per_day,
            self.working_minutes_per_day,
        )

        # Initialize start time
        start_time = time.time()

        # Count arbor frequencies
        arbor_frequencies = self.count_arbor_frequencies()

        # Create gene pool
        gene_pool = self.parallel_init_population(
            num_inds=self.n * 10, arbor_frequencies=arbor_frequencies
        )

        # Randomly sample the initial population from the improved gene pool
        self.P = random.sample(gene_pool, self.n)

        # Create double ended queue to append the results to
        best_scores = deque()

        # Initialize iteration
        iteration = 0

        # Extract time budget
        time_budget = scheduling_options["time_budget"]

        # max_iterations is used if time_budget is set to zero
        if time_budget != 0:
            while True:
                # Check if time_budget has expired
                elapsed_time = round(time.time() - start_time)
                if elapsed_time > time_budget:
                    logger.info(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Time budget has expired"
                    )
                    break
                iteration = self.perform_iteration(iteration, best_scores, gene_pool)
        else:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Time budget not set, using max_iterations"
            )

            for _ in range(scheduling_options["max_iterations"]):
                iteration = self.perform_iteration(iteration, best_scores, gene_pool)

        schedules_and_scores = sorted(zip(self.P, self.scores), key=lambda x: x[1], reverse=True)
        self.best_schedule = schedules_and_scores[0][0]
        logger.info(
            f"Snippet of best schedule (job, task, machine, start_time, duration, task_idx, part_id): "
            f"{list(self.best_schedule)[-2:]}"
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
        columns=["job_id", "task", "machine", "starting_time", "duration", "task_idx", "part_id"],
    )

    # Round the starting time and duration
    schedule_df["starting_time"] = schedule_df["starting_time"].round(5)
    schedule_df["duration"] = schedule_df["duration"].round(5)

    # Reset and drop index
    croom_processed_orders.reset_index(inplace=True, drop=True)

    # Join best schedule to processed orders
    croom_processed_orders = croom_processed_orders.merge(
        # Schedule contains the job ID instead of the job index now, so join on this instead
        # schedule_df, left_index=True, right_on="job", how="left"
        schedule_df,
        left_on="Job ID",
        right_on="job_id",
        how="left",
    )

    # Define end time
    croom_processed_orders["end_time"] = (
        croom_processed_orders["starting_time"] + croom_processed_orders["duration"]
    )

    # Rename columns
    croom_processed_orders = croom_processed_orders.rename(columns=column_mapping_reformat)

    # Use the index as the job index
    croom_processed_orders["Job"] = croom_processed_orders.index

    # Apply machine name mapping
    croom_processed_orders["Machine"] = croom_processed_orders["Machine"].map(machine_dict)

    return croom_processed_orders


def identify_changeovers(df: pd.DataFrame, scheduling_options: Dict[str, Any]) -> pd.DataFrame:
    """
    Identify and return a DataFrame of changeovers for specified machines. Changeovers occur when there is more than
    a specified threshold of minutes between the end time of one task and the start time of the next task.

    Parameters:
    df (pd.DataFrame): DataFrame containing scheduling data with columns ['Machine', 'Start_time', 'End_time'].
    scheduling_options (Dict[str, List[str]]): Dictionary containing machine names with key 'changeover_machines_op1_full_name'.

    Returns:
    pd.DataFrame: DataFrame containing changeover periods with columns ['Machine', 'Start_time', 'End_time'].
    """

    # Initialize a list to store all changeover periods
    all_changeovers = deque()

    # Extract the list of machines from the scheduling options
    machines = scheduling_options["changeover_machines_op1_full_name"]
    threshold = scheduling_options["change_over_time_op1"]

    # Loop through each machine in the list
    for machine in machines:
        # Filter the DataFrame for the current machine and sort by 'Start_time'
        machine_tasks = (
            df[df["Machine"] == machine][["Machine", "Start_time", "End_time"]]
            .sort_values(["Start_time"])
            .reset_index(drop=True)
        )

        # Iterate through the tasks to find gaps
        for i in range(len(machine_tasks) - 1):
            current_end_time: float = machine_tasks.at[i, "End_time"]
            next_start_time: float = machine_tasks.at[i + 1, "Start_time"]

            # Calculate the gap between the current task's end time and the next task's start time
            gap: float = next_start_time - current_end_time

            # If the gap is greater than the threshold, identify the changeover period
            if gap > threshold:
                changeover_end_time = next_start_time
                changeover_start_time = next_start_time - threshold
                # Append the changeover period to the list
                all_changeovers.append(
                    {
                        "Machine": machine,
                        "Start_time": changeover_start_time,
                        "End_time": changeover_end_time,
                    }
                )

    # Create a DataFrame from the list of changeovers
    changeovers_df: pd.DataFrame = pd.DataFrame(all_changeovers)

    return changeovers_df


def create_start_end_time(
    croom_reformatted_orders: pd.DataFrame, changeovers: pd.DataFrame, scheduling_options: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adjusts the start and end times of tasks in the given DataFrame to fit within working hours (06:30 to 14:30)
    and ensures that tasks are scheduled sequentially within each job and machine.

    The function first converts the 'Start_time' from minutes to a datetime based on a specified start date.
    It then adjusts the start and end times of tasks to fit within working hours, pushing tasks to the next day
    if they start after 14:30 or before 06:30. The function also checks for consistency in task scheduling within
    each job and machine.

    Args:
        croom_reformatted_orders (pd.DataFrame): The DataFrame containing reformatted orders with 'Start_time' in minutes.
        changeovers (pd.DataFrame): The DataFrame containing changeover periods with 'Start_time' in minutes.
        scheduling_options (Dict[str, Any]): A dictionary containing scheduling options, including the 'start_date' and
        'change_over_time_op1'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames - the adjusted reformatted orders and changeovers with
        updated start and end times.
    """

    # Parse the date string into a datetime object
    base_date = datetime.strptime(scheduling_options["start_date"], "%Y-%m-%dT%H:%M")

    # Sort by start time ascending
    croom_reformatted_orders.sort_values("Start_time", inplace=True)
    changeovers.sort_values("Start_time", inplace=True)

    # Initialize empty 'Start_time_date' column
    croom_reformatted_orders["Start_time_date"] = None

    def working_hours_shift(row):
        """
        Maps minutes since the start of scheduling to a real job starting datetime,
        assuming there are 1440 minutes in a day.
        """

        row["Start_time_date"] = base_date + pd.to_timedelta(row["Start_time"], unit="m")

        return row

    # Apply function
    croom_reformatted_orders = croom_reformatted_orders.apply(working_hours_shift, axis=1)
    changeovers = changeovers.apply(working_hours_shift, axis=1)

    # Overwrite the integer start time with the calculated datetimes
    croom_reformatted_orders["Start_time"] = croom_reformatted_orders["Start_time_date"]
    croom_reformatted_orders["End_time"] = croom_reformatted_orders["Start_time"] + pd.to_timedelta(
        croom_reformatted_orders["duration"], unit="m"
    )

    # Overwrite the integer start time with the calculated datetimes for changeovers
    changeovers["Start_time"] = changeovers["Start_time_date"]
    changeovers["End_time"] = changeovers["Start_time"] + pd.Timedelta(
        minutes=scheduling_options["change_over_time_op1"]
    )

    # Define the time range for valid changeovers
    # Define start_time_min based on the time part of base_date
    start_time_min = base_date.time()

    # Convert total_minutes_per_day and change_over_time_op1 to timedelta
    working_minutes = timedelta(minutes=scheduling_options["working_minutes_per_day"])
    change_over_time = timedelta(minutes=scheduling_options["change_over_time_op1"])

    # Calculate the end time (start_time_max) by adding total_minutes and subtracting change_over_time
    end_time = base_date + working_minutes - change_over_time
    start_time_max = end_time.time()

    # Filter out changeovers outside the valid time range
    changeovers = changeovers[
        (changeovers["Start_time"].dt.time >= start_time_min)
        & (changeovers["Start_time"].dt.time <= start_time_max)
    ]

    # Reorder by earliest start time
    croom_reformatted_orders.sort_values(by="Start_time", inplace=True)

    # Check if the start time for each task within each job is later than the completion time of the previous task
    # If this error is raised the schedule is invalid
    for job_id in croom_reformatted_orders["Order"].unique():
        job_schedule = croom_reformatted_orders[croom_reformatted_orders["Order"] == job_id]
        for i in range(1, len(job_schedule)):
            if not job_schedule.iloc[i]["Start_time"] >= job_schedule.iloc[i - 1]["End_time"]:
                logger.warning(
                    f"The start time for task {job_schedule.iloc[i]['task']} in job {job_id} "
                    f"is earlier than the completion time of the previous task!"
                )

    # Check if the start time for each task on each machine is later than the completion time of the previous task
    # on that machine
    # If this error is raised the schedule is invalid
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

    # Check if the time between the end of one task and the start of the next is at least 15 minutes
    # for machines with 'Drag' in the name
    for machine in croom_reformatted_orders["Machine"].unique():
        if "Drag" in machine:
            machine_schedule = croom_reformatted_orders[
                croom_reformatted_orders["Machine"] == machine
            ].sort_values("Start_time")
            for i in range(1, len(machine_schedule)):
                previous_end_time = machine_schedule.iloc[i - 1]["End_time"]
                current_start_time = machine_schedule.iloc[i]["Start_time"]
                time_diff = current_start_time - previous_end_time
                if time_diff < timedelta(
                    minutes=15
                ):  # TODO: Load from params after merging with test branch
                    logger.warning(
                        f"The time between tasks on {machine} is less than 15 minutes: {time_diff}"
                    )

    # Sort again by job and task before plotting
    croom_reformatted_orders = croom_reformatted_orders.sort_values(["Job", "task"])
    changeovers = changeovers.sort_values(["Machine", "Start_time"])

    return croom_reformatted_orders, changeovers


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
