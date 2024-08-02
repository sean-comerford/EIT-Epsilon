from typing import List, Dict, Tuple, Union
import random
import numpy as np

def blah():
    return 7

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
        self.task_to_machines = None
        self.n = None
        self.n_e = None
        self.n_c = None
        self.P = None
        self.scores = None
        self.day_range = None
        self.best_schedule = None
        self.minutes_per_day = None
        self.change_over_time_op1 = None
        self.change_over_time_op2 = None
        self.change_over_machines_op2 = None
        self.compatibility_dict = None
        self.urgent_orders = None
        self.urgent_multiplier = None
        self.max_iterations = None

    def find_avail_m(self, start: int, job_idx: int, task_idx: int) -> int:
        """
        Finds the next available time for a machine to start a task, considering the working day duration.

        Args:
            start (int): The starting time in the schedule in minutes.
            job_idx (int): The index of the job in the job list.
            task_idx (int): The task number within the job.

        Returns:
            int: The next available time for the machine to start the task.
        """
        for day in self.day_range:
            if start < day <= start + self.dur[job_idx][task_idx]:
                return day
        return start + self.dur[job_idx][task_idx]
    
    def pick_early_machine(
        self,
        task_idx: int,
        avail_m: Dict[int, int],
        random_roll: float,
        prob: float = 0.75,
    ) -> int:
        """
        Selects a machine for the given task based on availability and compatibility.
        There is a chance of 'prob' to select the machine that comes available earliest,
        otherwise a random machine is picked.

        Parameters:
        - job_id (int): ID of the job.
        - task (int): Index of the task within the job.
        - avail_m (Dict[int, int]): A dictionary with machine IDs as keys and their available times as values.
        - random_roll (float): A random number to decide the selection strategy.
        - prob (float): Probability to pick the earliest available compatible machine.

        Returns:
        - int: The selected machine ID.
        """
        compat_with_task = self.task_to_machines[task_idx] # A list of machines that are compatible with this task
        if random_roll < prob:
            m = min(compat_with_task, key=lambda x: avail_m.get(x))
        else:
            m = random.choice(compat_with_task)

        return m
    
    def run(self, input_repr_dict: Dict[str, any], scheduling_options: dict, compatibility_dict: dict):
        """
        Runs the genetic algorithm by initializing the population, evaluating it, and selecting the best schedule.

        Args:
            input_repr_dict (Dict[str, any]): A dictionary containing the necessary input variables for the GA.
            scheduling_options (dict): Dictionary containing hyperparameters for running the algorithm.
            compatibility_dict (dict): Dictionary containing the compatibility information for changeovers.

        Returns:
            Tuple[List[Tuple[int, int, int, int, float]], List[int]]: The best schedule with the highest score and the list of best scores per generation.
        """
        self.J = input_repr_dict["J"]
        self.M = input_repr_dict["M"]
        self.compat = input_repr_dict["compat"]
        self.dur = input_repr_dict["dur"]
        self.due = input_repr_dict["due"]
        self.part_id = input_repr_dict["part_id"]
        self.task_to_machines = input_repr_dict["task_to_machines"]
        self.n = scheduling_options["n"]
        self.n_e = scheduling_options["n_e"]
        self.n_c = scheduling_options["n_c"]
        self.minutes_per_day = scheduling_options["minutes_per_day"]
        self.change_over_time_op1 = scheduling_options["change_over_time_op1"]
        self.change_over_time_op2 = scheduling_options["change_over_time_op2"]
        self.change_over_machines_op2 = scheduling_options["change_over_machines_op2"]
        self.compatibility_dict = compatibility_dict
        self.max_iterations = scheduling_options["max_iterations"]
        self.urgent_multiplier = scheduling_options["urgent_multiplier"]
        self.urgent_orders = [job_idx - 1 for job_idx in scheduling_options["urgent_orders"]]
        self.day_range = np.arange(
            self.minutes_per_day,
            len(self.J) // 5 * self.minutes_per_day,
            self.minutes_per_day,
        )

        self.init_population()