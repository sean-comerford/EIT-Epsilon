import pandas as pd
import logging
from typing import List, Dict, Tuple, Union

from .Job import Job
from .Shop import Shop

logger = logging.getLogger(__name__)

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

        # Check if all part IDs are consistent across different operations
        self.check_part_id_consistency(in_scope_data)

        # Reset index
        in_scope_data.reset_index(inplace=True, drop=True)

        return in_scope_data
    
    
    
    def build_ga_representation(
        self,
        croom_processed_orders: pd.DataFrame,
        cr_cycle_times: pd.DataFrame,
        ps_cycle_times: pd.DataFrame,
        op2_cycle_times: pd.DataFrame,
        task_to_machines: Dict[int, List[int]],
        scheduling_options: Dict[str, any],
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
            op2_cycle_times (pd.DataFrame): The Operation 2 cycle times.
            task_to_machines (Dict[int, List[int]]): The task to machines dictionary.
            scheduling_options (Dict[str, any]): The scheduling options dictionary.

        Returns:
            Dict[str, any]: The GA representation.
        """
        J = self.create_jobs(croom_processed_orders)
        J_op_2 = self.create_jobs(croom_processed_orders, operation="OP 2")

        # Combine jobs from both operations (Operation 1 and Operation 2) into one list of jobs (J)
        J.update(J_op_2)
        
        partToTasks = self.create_partID_to_task_seq(croom_processed_orders)
        
        input_repr_dict = {
            "J": J,
            "part_to_tasks": partToTasks,
            # "M": M,
            # "compat": compat,
            # "dur": dur,
            # "due": due,
            # "part_id": part_id,
        }
        
        return input_repr_dict
        
        