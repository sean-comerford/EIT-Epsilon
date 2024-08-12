from typing import List, Dict, Tuple, Union
import random
import numpy as np
import logging
from datetime import datetime
import copy 
from collections import defaultdict

logger = logging.getLogger(__name__)

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

    def find_avail_m(self, start: int, part_id: int, task_id: int) -> int:
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
            if start < day <= start + self.dur[(part_id, task_id)]:
                return day
        return start + self.dur[(part_id, task_id)]
    
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
        compat_with_task = self.task_to_machines[task_id] # A list of machines that are compatible with this task
        if random_roll < prob:
            m = min(compat_with_task, key=lambda x: avail_m.get(x))
        else:
            m = random.choice(compat_with_task)

        return m
    
    @staticmethod
    def slack_logic(
        m: int,
        avail_m: Dict[int, int],
        slack_m: Dict[int, List],
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

        if m in [""]:
            start = avail_m[m] + changeover_duration

        # If the previous task is completed later than the new machine comes available
        if previous_task_finish > avail_m[m]:
            # Start time is the completion of the previous task of the job in question
            start = previous_task_finish + changeover_duration

            # Difference between the moment the machine becomes available and the new tasks starts is slack
            # e.g.: machine comes available at 100, new task can only start at 150, slack = (100, 150)
            # We subtract changeover_duration, because even though the task actually starts later,
            # the changeover_duration cannot be used for a different task
            slack_m[m].append((avail_m[m], start - changeover_duration))

        else:
            # Loop over slack in this machine
            for unused_time in slack_m[m]:
                # If the unused time + duration of task is less than the end of the slack window
                if (
                    max(unused_time[0], previous_task_finish) + changeover_duration + current_task_dur
                ) < unused_time[1]:
                    # New starting time is the largest of the beginning of the slack time or the time when the
                    # previous task of the job is completed
                    # Task can only start once changeover is completed
                    start = max(unused_time[0], previous_task_finish) + changeover_duration

                    # Remove the slack period if it has been used
                    slack_m[m].remove(unused_time)

                    # We add the remaining time between when the task finishes and the end of the slack window
                    # as a new slack window
                    # e.g.: original slack = (100, 150), task planned now takes (110, 130), new slack = (130, 150)
                    # changeover_duration must be added because it delays the task
                    slack_m[m].append(
                        (
                            (
                                max(unused_time[0], previous_task_finish)
                                + changeover_duration
                                + current_task_dur
                            ),
                            unused_time[1],
                        )
                    )

                    # Append another slack window if previous start was not at the beginning of the slack window,
                    # in this case there is still some time between when the machine comes available and when the
                    # task starts
                    # e.g. original slack = (100, 150), task planned now takes (110, 130), new slack = (100, 110)
                    # We subtract changeover_duration, because even though the task actually starts later,
                    # the changeover_duration cannot be used for a different task
                    if start == previous_task_finish:
                        slack_m[m].append((unused_time[0], start - changeover_duration))

                    slack_time_used = True
                    # break the loop if a suitable start time has been found in the slack
                    break

            # If slack time is not used, start when the machine becomes available
            if not slack_time_used:
                start = avail_m[m] + changeover_duration

        if start is None:
            logger.warning("No real start time was defined!")

        return start, slack_time_used
    
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
    
        #random.seed(191919)
            
        for i in range(num_inds):
            avail_m = {m: 0 for m in self.M}
            slack_m = {m: [] for m in self.M}
            product_m = {m: 0 for m in self.M}
            changeover_finish_time = 0
            P_j = []            
            
            # Create a temporary copy of the Job IDs
            J_temp = list(self.J.keys())
            
            # for jobID in J_temp:                
            #     print(f"J temp before:  {jobID} {self.J[jobID][0]} {self.J[jobID][1]}")

            # Generate a random float [0, 1]
            random_roll = random.random()
            #random_roll = 0.5
            
            
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
                    J_temp.remove(job)  # Remove the job from its current position (TODO: Should we print an error if the job isn't in J_temp?)
                    J_temp.append(job)  # Append the job to the end of the list (this means it will be picked first)
                    
            # for jobID in J_temp:                
            #     print(f"J temp after:  {jobID} {self.J[jobID][0]} {self.J[jobID][1]}")                    
                    
            # While our list of jobs is not empty
            while J_temp:
                # Take the job id at the end of the list and remove it from the list
                job_id = J_temp.pop()
                part_id = self.J[job_id][0]

                # Loop over the tasks in the job (i.e. get the tasks for the part ID for this job)
                for task_index, task_id in enumerate(self.part_to_tasks[part_id]):
                    # Generate random float [0, 1]
                    random_roll = random.random()

                    # New boolean to track if the task is planned during slack time,
                    # if so we do not need to update avail_m
                    slack_time_used = False
                
                    # Conditional logic; separate flow for first task of OP1 (HAAS)
                    
                    
                    if part_id.split("-")[-1] == "OP1" and (task_id == 1 or task_id == 99):  # TODO: Modify this?
                        # Logic for first task in OP1 (HAAS machines)
                        compat_task_0 = self.task_to_machines[task_id] # A list of machines that are compatible with this task
                        # Preferred machines are those that have not been used yet or processed the same
                        # size previously (in this case no changeover is required)
                        # Note: no changeover needed for the first task on a HAAS machine is an assumption
                        preferred_machines = [
                            key for key in compat_task_0
                            if (
                                product_m.get(key) == 0
                                or product_m.get(key) == part_id
                                or product_m.get(key) in self.compatibility_dict[part_id]
                            )
                        ]
                        
                        # If no preferred machines can be found, pick one that comes available earliest
                        # with a higher probability
                        if not preferred_machines:
                            m = self.pick_early_machine(
                                task_id, avail_m, random_roll, prob=0.7
                            )
                        # If there are preferred machines, pick the preferred machine that comes available
                        # earliest with a higher probability
                        else:
                            m = (
                                min(preferred_machines, key=lambda x: avail_m.get(x))
                                if random_roll < 0.7
                                else random.choice(preferred_machines)
                            )
                            
                        # Start time is the time that the machine comes available if no changeover is required
                        # else, the changeover time is added, and an optional waiting time if we need to wait
                        # for another changeover to finish first (only one changeover can happen concurrently)
                        start = (
                            avail_m[m]
                            if ( 
                                # No changeover required (there was no previous part on the machine, or the previous part was the same or compatible)
                                product_m.get(m) == 0
                                or product_m.get(m) == part_id
                                or product_m.get(m) in self.compatibility_dict[part_id]
                            )
                            else avail_m[m]
                            + self.change_over_time_op1
                            + max((changeover_finish_time - avail_m[m]), 0)
                        )

                        # If a changeover happened, we update the time someone comes available to do another
                        # changeover
                        if product_m[m] != 0 and part_id != product_m[m]:
                            changeover_finish_time = start
                            
                    else:
                        # Pick a machine with preference for one available earliest
                        m = self.pick_early_machine(task_id, avail_m, random_roll)

                        # Initialize changeover time to 0
                        changeover_time = 0

                        # If m is in changeover machines and the last size was not the same
                        if m in self.change_over_machines_op2:
                            changeover_time = (
                                0
                                if (
                                    product_m.get(m) == 0
                                    or product_m.get(m) == part_id
                                    or product_m.get(m) in self.compatibility_dict[part_id]
                                )
                                else self.change_over_time_op2
                            )

                        
                        if task_index == 0: # TODO: Double check this. We are looking to see if its the first task in the sequence?
                            start = avail_m[m] + changeover_time
                        else:
                            start, slack_time_used = self.slack_logic(
                                m,
                                avail_m,
                                slack_m,
                                slack_time_used,
                                P_j[-1][3],
                                P_j[-1][4],
                                self.dur[(part_id, task_id)],
                                changeover_time,
                            )
                            
                    # Append the task to our proposed schedule
                    P_j.append(
                        (
                            job_id,
                            task_id,  # Actual task number (as in task_to_machines_dict)
                            m,
                            start,
                            self.dur[(part_id, task_id)],
                            task_index,  # Is this needed?
                            part_id,
                        )
                    )

                    if not slack_time_used:
                        avail_m[m] = self.find_avail_m(start, part_id, task_id)

                    product_m[m] = part_id

            # Add proposed schedule to population (list of proposed schedules) if it is not in there already
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
                
    def evalPopTest(self, best_scores: list = None, display_scores: bool = True, on_time_bonus: int = 5000):
        for i, schedule in enumerate(self.P):
            #if i == 5: return
            score = round(
                        sum(
                            (
                                # Difference between due date and completion time, multiplied by urgent_multiplier if urgent
                                (self.J[job_id][1] - (start_time + job_task_dur))
                                * (self.urgent_multiplier if job_id in self.urgent_orders else 1)
                                + (
                                    # Fixed size bonus for completing the job on time
                                    on_time_bonus
                                    if (self.J[job_id][1] - (start_time + job_task_dur)) > 0
                                    else 0
                                )
                            )
                            for (job_id, task_id, machine, start_time, job_task_dur, _, _,) in schedule
                            # Only consider the completion time of the final task
                            # if task + 1 == max(self.J[job_idx])
                            if task_id == self.part_to_tasks[self.J[job_id][0]][-1]
                        )
                    )
            print(f"Schedule {i} score: {score}")
  
    def evaluate_population(
        self, best_scores: list = None, display_scores: bool = True, on_time_bonus: int = 5000
    ):
        """
        Evaluates the population of schedules by calculating a score for each schedule based on the completion times
        of jobs vs. the required due date.
        """
        # Calculate scores for each schedule
        # Note: self.J[job_id] gives the tuple (Due time, Part ID) for a given job ID
        self.scores = [
            round(
                sum(
                    (
                        # Difference between due date and completion time, multiplied by urgent_multiplier if urgent
                        (self.J[job_id][1] - (start_time + job_task_dur))
                        * (self.urgent_multiplier if job_id in self.urgent_orders else 1)
                        + (
                            # Fixed size bonus for completing the job on time
                            on_time_bonus
                            if (self.J[job_id][1] - (start_time + job_task_dur)) > 0
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
                    if task_id == self.part_to_tasks[self.J[job_id][0]][-1]
                    #if task + 1 == max(self.J[job_idx])
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
        Each task is represented as (job id, task id, machine, start time, duration, task index, part id).

        Returns:
        P_prime_sorted (List[Tuple[int, int, int, int, int, int, str]]): A sorted list of tuples where each tuple
        represents a task. Each task is represented as (job id, task id, machine, start time, duration, task index, part id).
        """
        # Initialize an empty list to hold tasks for this proposed schedule
        P_prime_sorted = []
        avail_m = {m: 0 for m in self.M}
        slack_m = {m: [] for m in self.M}
        product_m = {m: 0 for m in self.M}
        changeover_finish_time = 0

        # Loop over the jobs in the job list (J)
        for job_id in list(self.J.keys()):
        #for job_idx in range(len(self.J)):
            
            part_id = self.J[job_id][0]

            # We have a list of tuples, where each tuple stands for a task in a proposed schedule
            # We filter all the tuples for ones belonging to a specific job_idx (first field of the tuple) 
            # and sort by the task index 
            # TODO: Double check with with Sean and Jean Luc - we can't sort by task_id as 99 might be the first one
            job_tasks = sorted([entry for entry in P_prime if entry[0] == job_id], key=lambda x: x[5])

            # Loop over the tasks one by one
            for task_entry in job_tasks:
                _, task_id, m, _, _, task_idx, _ = task_entry
                slack_time_used = False

                if part_id.split("-")[-1] == "OP1" and task_idx == 0: 
                    # Start whenever the first machine becomes available + optional changeover time
                    # + optional time we have to wait for changeover mechanic to complete previous changeover
                    start = (
                        avail_m[m]
                        if (
                            product_m.get(m) == 0
                            or product_m.get(m) == part_id
                            or product_m.get(m) in self.compatibility_dict[part_id]
                        )
                        else avail_m[m]
                        + self.change_over_time_op1
                        + max((changeover_finish_time - avail_m[m]), 0)
                    )
                    # Update changeover mechanic availability
                    if (
                        product_m.get(m) != 0
                        and product_m.get(m) != part_id
                        and product_m.get(m) not in self.compatibility_dict[part_id]
                    ):
                        changeover_finish_time = start

                else:
                    # Initialize changeover time to 0
                    changeover_time = 0

                    # If m is in changeover machines and the last size was not the same
                    if m in self.change_over_machines_op2:
                        changeover_time = (
                            0
                            if (
                                product_m.get(m) == 0
                                or product_m.get(m) == part_id
                                or product_m.get(m) in self.compatibility_dict[part_id]
                            )
                            else self.change_over_time_op2
                        )
                    if task_idx == 0:
                        start = avail_m[m] + changeover_time
                    else:
                        start, slack_time_used = self.slack_logic(
                            m,
                            avail_m,
                            slack_m,
                            slack_time_used,
                            P_prime_sorted[-1][3],
                            P_prime_sorted[-1][4],
                            self.dur[(part_id, task_id)],
                            changeover_time,
                        )

                # If slack time is used no need to update latest machine availability
                if not slack_time_used:
                    avail_m[m] = self.find_avail_m(start, part_id, task_id)

                # Record part ID of the latest product to be processed on a machine for changeovers
                product_m[m] = part_id

                # Issue warning if 'start' is still not defined after loop
                if start is None:
                    logger.warning(
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - No start time found for job ID {job_id}, "
                        f"task ID {task_id}, machine {m}")

                # Add the task to the sorted list of tasks in this proposed schedule
                P_prime_sorted.append(
                    (
                        job_id,
                        task_id,
                        m,
                        start,
                        self.dur[(part_id, task_id)],
                        task_idx,
                        part_id,
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

        P_0 = self.find_best_schedules()

        # Make a deep copy of P_0
        P_1 = copy.deepcopy(P_0)

        # Initialize a list of new schedules
        new_schedules = []

        for schedule in P_1:
            # Group tasks by job_id
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
                            # TODO: We can easily look up the part ID from the job ID now
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

            # Randomly select a pair of jobs (at least once, but up to four times)
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
        self.part_to_tasks = input_repr_dict["part_to_tasks"]
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
        
        # best_scores = []        
        # self.evaluate_population(best_scores=best_scores)
        
        return self.P