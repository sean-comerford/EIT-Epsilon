import pytest
import pandas as pd
from collections import deque
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager


@pytest.fixture
def config_loader():
    return OmegaConfigLoader(conf_source=str(Path.cwd()))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="eit_epsilon",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
        env=None,
    )


class TestOutputSchedule:
    def test_project_path(self, project_context):
        """The project path should be the current working directory."""
        assert project_context.project_path == Path.cwd()

    def test_load_parameters_and_data(self, project_context):
        """Test loading of parameters and data from Kedro context."""
        # Load parameters
        parameters = project_context.catalog.load("params:scheduling_options")

        # Load a specific parameter
        change_over_time_op1 = parameters["change_over_time_op1"]

        # Load data
        data = project_context.catalog.load("croom_processed_orders")

        # Assertions to verify the loaded data
        assert isinstance(change_over_time_op1, int)
        assert data is not None

    def test_final_schedule_not_empty(self, project_context):
        """The output schedule should not be empty"""
        final_schedule = project_context.catalog.load("final_schedule")
        assert not final_schedule.empty

    def test_chronological_order(self, project_context):
        """Tasks of the same job should be scheduled in chronological order with no overlap."""
        final_schedule = project_context.catalog.load("final_schedule")
        scheduling_options = project_context.catalog.load("params:scheduling_options")
        for job_id in final_schedule["Order"].unique():
            job_schedule = final_schedule[final_schedule["Order"] == job_id]
            for i in range(1, len(job_schedule)):
                assert job_schedule.iloc[i]["Start_time"] >= job_schedule.iloc[i - 1][
                    "End_time"
                ] + pd.Timedelta(minutes=scheduling_options["task_time_buffer"] - 1), (
                    f"The start time for job {job_id}, task {job_schedule.iloc[i]['task']} "
                    f"is earlier than the completion time of the previous task!"
                    f"Start time: {job_schedule.iloc[i]['Start_time']}, "
                    f"End time: {job_schedule.iloc[i - 1]['End_time']}"
                )

    def test_machine_task_order(self, project_context):
        """Tasks on the same machine should not overlap in time."""
        final_schedule = project_context.catalog.load("final_schedule")
        scheduling_options = project_context.catalog.load("params:scheduling_options")
        for machine in final_schedule["Machine"].unique():
            machine_schedule = final_schedule[final_schedule["Machine"] == machine].sort_values(
                "Start_time"
            )
            for i in range(1, len(machine_schedule)):
                assert machine_schedule.iloc[i]["Start_time"] >= machine_schedule.iloc[i - 1][
                    "End_time"
                ] + pd.Timedelta(minutes=scheduling_options["task_time_buffer"]), (
                    f"The start time for job {machine_schedule.iloc[i]['Order']}, task {machine_schedule.iloc[i]['task']} "
                    f"in machine {machine} is earlier than the completion time of the previous task!"
                )

    def test_end_time_calculation(self, project_context):
        """The end time of a task should be the start time plus the duration."""
        final_schedule = project_context.catalog.load("final_schedule")
        for index, row in final_schedule.iterrows():
            assert (
                row["duration"] >= 0
            ), f"Negative duration found for job {row['Job']}: {row['duration']} minutes."
            calculated_end_time = (row["Start_time"] + pd.Timedelta(minutes=row["duration"])).round(
                "min"
            )
            actual_end_time = row["End_time"].round("min")
            assert calculated_end_time == actual_end_time, (
                f"Mismatch in end time calculation for job {row['Job']}: "
                f"expected {calculated_end_time}, found {actual_end_time}."
            )

    def test_no_start_times_on_weekends(self, project_context):
        """HAAS machines can run on the weekends, but only on Saturdays. Other machines cannot run in weekends."""
        final_schedule = project_context.catalog.load("final_schedule")
        for index, row in final_schedule.iterrows():
            start_time = row["Start_time"]
            machine = row["Machine"]
            assert (
                start_time.weekday() != 6
            ), f"Start time on Sunday found for job {row['Job']}: {start_time}."
            if "HAAS" not in machine:
                assert start_time.weekday() != 5, (
                    f"Start time on Saturday found for job {row['Job']} on machine {machine}: "
                    f"{start_time}."
                )

    def test_non_haas_machine_start_time_range(self, project_context):
        """The start time for non-HAAS machines should be between 06:30 and 14:30."""
        final_schedule = project_context.catalog.load("final_schedule")
        schedule_options = project_context.catalog.load("params:scheduling_options")
        start_time = pd.Timestamp(schedule_options["start_date"])
        end_time = start_time + pd.Timedelta(minutes=schedule_options["working_minutes_per_day"])
        for index, row in final_schedule.iterrows():
            machine = row["Machine"]
            if "HAAS" not in machine:
                task_start_time = row["Start_time"].time()
                start_time_range_valid = start_time.time() <= task_start_time <= end_time.time()
                assert start_time_range_valid, (
                    f"Start time out of range for non-HAAS machine for job {row['Job']}: "
                    f"{task_start_time}."
                )

    def test_start_and_end_machines_present(self, project_context):
        """
        Test that for every unique order number in the final_schedule DataFrame,
        at least one row has a task 0, 1, or 10 (starting machines: HAAS or Ceramic Drag),
        and at least one row has a task 7 or 19 (Final inspection for OP1 and OP2 respectively).
        """
        final_schedule = project_context.catalog.load("final_schedule")
        unique_orders = final_schedule["Order"].unique()

        for order in unique_orders:
            order_rows = final_schedule[final_schedule["Order"] == order]

            has_initial_task = order_rows["task"].isin([1, 10, 30]).any()
            assert has_initial_task, f"Order {order} does not have any task 0, 1, or 10."

            has_final_task = order_rows["task"].isin([7, 20, 44]).any()
            assert has_final_task, f"Order {order} does not have any task 7 or 19."

    def test_no_simultaneous_changeovers(self, project_context):
        """No two changeovers can happen simultaneously for the HAAS machines"""
        final_changeovers = project_context.catalog.load("final_changeovers")
        for machine in final_changeovers["Machine"].unique():
            machine_changeovers = final_changeovers[final_changeovers["Machine"] == machine].sort_values(
                "Start_time"
            )
            for i in range(1, len(machine_changeovers)):
                assert (
                    machine_changeovers.iloc[i]["Start_time"]
                    >= machine_changeovers.iloc[i - 1]["End_time"]
                ), (
                    f"Overlapping changeovers found on machine {machine} between changeovers starting at "
                    f"{machine_changeovers.iloc[i - 1]['Start_time']} and {machine_changeovers.iloc[i]['Start_time']}."
                )

    def test_max_two_changeovers_per_day(self, project_context):
        """Changeovers take 3 hours, and there are only 8 hours in a workday.
        Therefore, there should be at most two changeovers per day."""
        final_changeovers = project_context.catalog.load("final_changeovers")
        # Ensure Start_time is in datetime format
        final_changeovers["Start_time"] = pd.to_datetime(final_changeovers["Start_time"])

        # Convert Start_time to date format
        final_changeovers["Start_date"] = final_changeovers["Start_time"].dt.date

        # Group by the date part of Start_time and count the number of changeovers
        changeovers_per_day = final_changeovers.groupby("Start_date").size()

        # Load schedule options
        schedule_options = project_context.catalog.load("params:scheduling_options")

        # Calculate max_count
        max_count = (
            schedule_options["working_minutes_per_day"] // schedule_options["change_over_time_op1"]
        )

        # Assert that no more than two changeovers start on the same day
        for date, count in changeovers_per_day.items():
            assert count <= max_count, f"More than two changeovers start on {date}: {count} changeovers."

    def test_changeover_start_time_range(self, project_context):
        """Checks if all changeovers are completed before the end of the working day. Latest allowed starting time
        is defined as: start time of the working day + working minutes per day - changeover time for OP1.
        """
        final_changeovers = project_context.catalog.load("final_changeovers")
        schedule_options = project_context.catalog.load("params:scheduling_options")
        start_time = pd.Timestamp(schedule_options["start_date"])
        end_time = (
            start_time
            + pd.Timedelta(minutes=schedule_options["working_minutes_per_day"])
            - pd.Timedelta(minutes=schedule_options["change_over_time_op1"])
        )

        for index, row in final_changeovers.iterrows():
            changeover_start_time = row["Start_time"].time()
            start_time_range_valid = start_time.time() <= changeover_start_time <= end_time.time()
            assert (
                start_time_range_valid
            ), f"Changeover start time out of range for job {row['Job']}: {changeover_start_time}."

    def test_best_scores_improvement(self, project_context):
        """The best score per generation should never become worse from one iteration to the next."""
        best_scores = project_context.catalog.load("best_scores")

        assert isinstance(best_scores, deque), "The best_scores.pkl file does not contain a deque."

        previous_score = float("-inf")
        for score in best_scores:
            assert score >= previous_score, f"Score decreased from {previous_score} to {score}."
            previous_score = score
