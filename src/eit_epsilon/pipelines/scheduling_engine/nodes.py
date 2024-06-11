import os
from pathlib import Path
import datetime

import pandas as pd
import re
from pandas.api.types import is_string_dtype
import plotly


class Job:
    def filter_in_scope_op1(self, data):
        # Add your preprocessing logic here
        inscope_data = data[(data['Part Description'].str.contains('OP 1') |
                            data['Part Description'].str.contains('ATT ')) &
                            (~data['On Hold?']) &
                            (~data['Part Description'].str.contains('OP 2'))]

        return inscope_data

    def extract_info(self, data):
        # Extract type, size, and orientation using.assign
        data = data.assign(
            Type=lambda x: x['Part Description'].apply(
                lambda y: 'CR' if 'CR' in y else ('PS' if 'PS' in y else '')),
            Size=lambda x: x['Part Description'].apply(
                lambda y: re.search(r'Sz (\d+N?)', y).group(1) if re.search(r'Sz (\d+N?)', y) else ''),
            Orientation=lambda x: x['Part Description'].apply(
                lambda y: 'LEFT' if 'LEFT' in y.upper() else ('RIGHT' if 'RIGHT' in y.upper() else ''))
        )

        return data

    def check_part_id_consistency(self, data):
        grouped = data.groupby('Part ID')[['Type', 'Size', 'Orientation']].nunique()
        assert not (grouped > 1).any().any(), "Part ID should be unique for every combination of Type, Size, and Orientation"

    def create_jobs_op1(self, data):
        # Seven tasks per job in OP 1
        J = [[1, 2, 3, 4, 5, 6, 7] for _ in range(len(data))]

        return J


class Shop:

    def create_machines(self, machine_qty_dict):
        total_machines = sum(machine_qty_dict.values())

        # Create representation of M
        M = list(range(1, total_machines + 1))

        return M

    def get_compatibility(self, J, task_to_machines):
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
        return compat

    def preprocess_cycle_times(self, cycle_times):

        # Set the first row as the headers
        cycle_times.columns = cycle_times.iloc[1]

        # Drop the first two rows and first column
        cycle_times = cycle_times.iloc[2:, 1:]

        # Create new index starting from 1
        cycle_times.index = range(1, len(cycle_times) + 1)

        def extract_number(s):
            """ Extracts valid size from column name"""
            match = re.search(r'\d+N?', s)
            return match.group(0) if match else s

        # Apply size extraction
        cycle_times.columns = [extract_number(col) for col in cycle_times.columns]

        # Fill times for final inspection
        cycle_times.loc[7].fillna(1, inplace=True)

        # Split cycle times for PS and CR into separate dataframes
        ps_times, cr_times = cycle_times.iloc[:, :19], cycle_times.iloc[:, 19:]

        return ps_times, cr_times

    def get_duration_matrix(self, J, inscope_orders, cr_times, ps_times):
        dur = []  # Initialize the duration matrix
        for i, job in enumerate(J):  # Iterate over each job
            job_dur = []  # Initialize the duration list for this job
            for task in job:  # Iterate over each task in this job
                # Determine whether to use cr_times or ps_times
                times = cr_times if inscope_orders.iloc[i]['Type'] == 'CR' else ps_times
                # Look up the duration using the task number and size
                duration = round(times.loc[task, inscope_orders.iloc[i]['Size']] * inscope_orders.iloc[i]['Order Qty'],
                                 1)
                job_dur.append(duration)
            dur.append(job_dur)
        return dur

    def get_due_date(self, inscope_orders, date='2024-03-18'):
        # Initialize empty list of due dates
        due = []

        for due_date in inscope_orders['Due Date ']:
            # Check if date is later than due_date
            if pd.Timestamp(date) > due_date:
                # If so, generate range in opposite direction and multiply by -1
                working_days = -len(pd.bdate_range(due_date, date)) * 480  # 480 minutes in an 8 hour working day
            else:
                # Otherwise, generate range as before
                working_days = len(pd.bdate_range(date, due_date)) * 480

            # append to list
            due.append(working_days)

        return due


class JobShop(Job, Shop):
    def preprocess_orders(self, croom_open_orders):
        # Add your preprocessing logic here
        inscope_data = (
            self.filter_in_scope_op1(croom_open_orders)
                .pipe(self.extract_info)
                .pipe(self.check_part_id_consistency)
        )

        return inscope_data

    def build_ga_representation(self, inscope_data, machine_qty_dict, cycle_times, task_to_machines):
        # Create correct representation of required variables for GA
        J = self.create_jobs_op1(inscope_data)
        M = self.create_machines(machine_qty_dict)
        compat = self.get_compatibility(J, task_to_machines)
        ps_times, cr_times = self.preprocess_cycle_times(cycle_times)
        dur = self.get_duration_matrix(J, inscope_data, cr_times, ps_times)
        due = self.get_due_date(inscope_data)

        # Put all variables in the input dictionary
        input_repr_dict = {
            'J': J,
            'M': M,
            'compat': compat,
            'dur': dur,
            'due': due,
        }

        return input_repr_dict


def create_chart(self, schedule: pd.DataFrame):
    # Add your chart creation logic here
    pass


def save_chart_to_html(self, gantt_chart):
    # Add your chart saving logic here
    pass
