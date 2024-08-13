import logging
import pandas as pd

from typing import List, Dict, Tuple, Union

logger = logging.getLogger(__name__)


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
        part_ID_to_task_seq: Dict[str, List[int]],
        in_scope_orders: pd.DataFrame,
        cr_times: pd.DataFrame,
        ps_times: pd.DataFrame,
        op2_times: pd.DataFrame,
    ) -> List[List[float]]:   
    
        dur = {}
        for job_id, (part_id, due_time) in J.items():
            # Find the corresponding row for the given job_id
            rows = in_scope_orders.loc[in_scope_orders['Job ID'] == job_id]
            
            if len(rows) > 1:
                print(f"Error: Multiple rows found for JobID {job_id}. Using the first row.")

            row = rows.iloc[0]
           
            for task in part_ID_to_task_seq[part_id]:
                if task < 10:  # Operation 1 tasks
                    times = cr_times if row["Type"] == "CR" else ps_times
                    duration = round(times.loc[task, row["Size"]] * row["Order Qty"], 1)
                    
                # Tasks of type 99 have the same duration as tasks of type 1
                elif task == 99:
                    times = cr_times if row["Type"] == "CR" else ps_times
                    # Use task 1 here, instead of 99
                    duration = round(times.loc[1, row["Size"]] * row["Order Qty"], 1)                     
                    
                else:  # Operation 2 tasks
                    #print(f'Job {job_id}\t Task: {task}\tTime: {op2_times.loc[task, "Actual "]} \tQty: {row["Order Qty"]}')
                    
                    # TODO: Should we read the quantity or use a batch size of 12?
                    duration = round(op2_times.loc[task, "Actual "] * 12, 1)
                    #duration = round(op2_times.loc[task, "Actual "] * row["Order Qty"], 1)
                    
                # Store the duration in the dictionary with key (job_id, task)
                dur[(part_id, task)] = duration

        return dur

    @staticmethod
    def get_due_date(
        in_scope_orders: pd.DataFrame,
        date: str = "2024-03-18",
        working_minutes: int = 480,
    ) -> List[int]:
        """
        Gets the due dates for the in-scope orders.

        Args:
            in_scope_orders (pd.DataFrame): The in-scope orders.
            date (str): The reference date in 'YYYY-MM-DD' format. Defaults to '2024-03-18'.
            working_minutes (int, optional): The number of working minutes per day. Defaults to 480.

        Returns:
            List[int]: The list of due dates in working minutes.
        """
        due = []
        for due_date in in_scope_orders["Due Date "]:
            if pd.Timestamp(date) > due_date:
                working_days = -len(pd.bdate_range(due_date, date)) * working_minutes
            else:
                working_days = len(pd.bdate_range(date, due_date)) * working_minutes
            due.append(working_days)

        # Debug statement
        logger.info(f"Snippet of due: {due[:5]}")

        return due