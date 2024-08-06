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
    
    def init_population(
        self, num_inds: int = None, fill_inds: bool = False
    ) -> Union[None, List[List[Tuple[int, int, int, float, float, int, str]]]]:
        """
        Initializes the population of schedules. Each schedule is a list of tasks assigned to machines with start times.

        Args:
            num_inds (int, optional): Number of individuals (schedules) to generate. If None, uses self.n. Defaults to None.
            fill_inds (bool, optional): Flag indicating whether to fill individuals in the population or return them.
                                        Defaults to False.

        Returns:
            Union[None, List[List[Tuple[int, int, int, float, float, int, str]]]]:
                - If fill_inds is False, updates self.P with the generated population.
                - If fill_inds is True, returns the generated population.

        The function operates as follows:
        1. Sets `num_inds` to `self.n` if it is not provided.
        2. Extracts part sizes and operations from `self.part_id`.
        3. Initializes the population list `P` and a range of percentages for logging progress.
        4. For each individual:
            - Initializes availability and product tracking dictionaries for machines.
            - Creates a temporary job list and shuffles or sorts it based on a random roll.
            - Processes each job, assigning tasks to machines based on compatibility, availability, and operation type.
            - Updates machine availability and product tracking after each task assignment.
            - Adds the proposed schedule to the population if it is unique.
            - Logs progress at certain percentages if `fill_inds` is False.
        5. Sets `self.P` to the generated population or returns it based on `fill_inds`.

        Note:
            - OP1 and OP2 represent different operation types with specific logic for task assignment.
            - Compatibility and availability are considered for machine assignment, with preference given to certain conditions.
        """
        # If the number of individuals to create is not strictly defined we create the same amount as the
        # whole population (this happens in generation 0)
        if num_inds is None:
            num_inds = self.n

        # Extract the operation from custom part id
        # def getOP(id):
        #     return id.split("-")[-1]
        # jobIDToOperation = {jobID: getOP(partID) for jobID, (partID, _) in self.J}
        
        # partIDs = [partID for _, (partID, _) in self.J]
        # operation = [id.split("-")[-1] for id in partIDs]

        P = []
        percentages = np.arange(10, 101, 10)
            
        for i in range(num_inds):
            avail_m = {m: 0 for m in self.M}
            slack_m = {m: [] for m in self.M}
            product_m = {m: 0 for m in self.M}
            changeover_finish_time = 0
            P_j = []            
            
            # Create a temporary copy of the Job IDs
            J_temp = list(self.J.keys())
            
            for jobID in J_temp:                
                print(f"J temp before:  {jobID} {self.J[jobID][0]} {self.J[jobID][1]}")

            # Generate a random float [0, 1]
            random_roll = random.random()
            random_roll = 0.9
            
            # Based on the random number we either randomly shuffle or apply some sorting logic
            if random_roll < 0.4:
                random.shuffle(J_temp)
            elif random_roll < 0.6:
                J_temp.sort(key=lambda x: self.J[x][0]) # Sort on the part ID
            elif random_roll < 0.7:
                J_temp.sort(key=lambda x: self.J[x][1], reverse=True) # Sort on the due time
            elif random_roll < 0.8:
                J_temp.sort(key=lambda x: (self.J[x][0], self.J[x][1]), reverse=True)
            else:
                # Reorder J_temp according to the urgent order list. Urgent orders should now contain the job IDs (instead of the index)
                for job in self.urgent_orders:
                    J_temp.remove(job)  # Remove the job from its current position
                    J_temp.append(job)  # Append the job to the end of the list (this means it will be picked first)
                    
            for jobID in J_temp:                
                print(f"J temp after:  {jobID} {self.J[jobID][0]} {self.J[jobID][1]}")
                    
                    
            # While our list of jobs is not empty
            while J_temp:
                # Take the job id at the end of the list and remove it from the list
                job_id = J_temp.pop()

                # Loop over the tasks in the job (i.e. get the tasks for the part ID for this job)
                for task_id in self.part_to_tasks[self.J[job_id][0]]:
                    # Generate random float [0, 1]
                    random_roll = random.random()

                    # New boolean to track if the task is planned during slack time,
                    # if so we do not need to update avail_m
                    slack_time_used = False
                
                
                
                

            
            
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
        #self.compat = input_repr_dict["compat"]  No longer required (?)
        self.dur = input_repr_dict["dur"]
        #self.due = input_repr_dict["due"] # No longer required (part of the jobs dictionary)
        #self.part_id = input_repr_dict["part_id"]
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
        self.urgent_orders = scheduling_options["urgent_orders"]
        self.day_range = np.arange(
            self.minutes_per_day,
            len(self.J) // 5 * self.minutes_per_day,
            self.minutes_per_day,
        )

        self.init_population()