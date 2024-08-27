import pytest
import pandas as pd
import pickle
import re
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


@pytest.fixture
def final_schedule_df():
    return pd.read_excel("data/08_reporting/final_schedule.xlsx")


@pytest.fixture
def final_changeovers_df():
    return pd.read_excel("data/08_reporting/final_changeovers.xlsx")


class TestOutputSchedule:
    def test_project_path(self, project_context):
        """The project path should be the current working directory."""
        assert project_context.project_path == Path.cwd()

    def test_final_schedule_not_empty(self, final_schedule_df):
        """The output schedule should not be empty"""
        assert not final_schedule_df.empty

    def test_chronological_order(self, final_schedule_df):
        """Tasks of the same job should be scheduled in chronological order with no overlap."""
        for job_id in final_schedule_df["Order"].unique():
            job_schedule = final_schedule_df[final_schedule_df["Order"] == job_id]
            for i in range(1, len(job_schedule)):
                assert job_schedule.iloc[i]["Start_time"] >= job_schedule.iloc[i - 1]["End_time"], (
                    f"The start time for job {job_id}, task {job_schedule.iloc[i]['task']} "
                    f"is earlier than the completion time of the previous task!"
                )

    def test_machine_task_order(self, final_schedule_df):
        """Tasks on the same machine should not overlap in time."""
        for machine in final_schedule_df["Machine"].unique():
            machine_schedule = final_schedule_df[final_schedule_df["Machine"] == machine].sort_values(
                "Start_time"
            )
            for i in range(1, len(machine_schedule)):
                assert (
                    machine_schedule.iloc[i]["Start_time"] >= machine_schedule.iloc[i - 1]["End_time"]
                ), (
                    f"The start time for job {machine_schedule.iloc[i]['Job']}, task {machine_schedule.iloc[i]['task']} "
                    f"in machine {machine} is earlier than the completion time of the previous task!"
                )

    def test_custom_part_id_format(self, final_schedule_df):
        pattern = re.compile(r"^(LEFT|RIGHT)-(PS|CR)-([1-9]|10)N?-(CLS|CTD)-(OP1|OP2)$")
        for index, row in final_schedule_df.iterrows():
            custom_part_id = row["Custom Part ID"]
            assert isinstance(
                custom_part_id, str
            ), f"Custom Part ID is not a string for job {row['Job']}: {custom_part_id}."
            assert pattern.match(
                custom_part_id
            ), f"Invalid Custom Part ID format for job {row['Job']}: {custom_part_id}."

    def test_part_id_custom_part_id_mapping(self, final_schedule_df):
        """A unique Part ID should always correspond to a unique Custom Part ID."""
        part_id_to_custom_part_id = {}
        for index, row in final_schedule_df.iterrows():
            part_id = row["Part ID"]
            custom_part_id = row["Custom Part ID"]
            if part_id in part_id_to_custom_part_id:
                assert part_id_to_custom_part_id[part_id] == custom_part_id, (
                    f"Mismatch found for Part ID {part_id} for job {row['Job']}: "
                    f"expected {part_id_to_custom_part_id[part_id]}, found {custom_part_id}."
                )
            else:
                part_id_to_custom_part_id[part_id] = custom_part_id

    def test_end_time_calculation(self, final_schedule_df):
        """The end time of a task should be the start time plus the duration."""
        for index, row in final_schedule_df.iterrows():
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

    def test_batch_size_limit(self, final_schedule_df):
        """The batch size should always be between 0-12."""
        for index, row in final_schedule_df.iterrows():
            assert (
                row["Production Qty"] <= 12
            ), f"Production Qty exceeds limit for job {row['Job']}: {row['Production Qty']}."
            assert (
                row["Production Qty"] >= 0
            ), f"Negative Production Qty found for job {row['Job']}: {row['Production Qty']}."

    def test_in_scope_orders(self, final_schedule_df):
        """All scheduled products should be in scope; they contain the substrings 'OP 1', 'OP 2', or 'ATT Primary'."""
        valid_substrings = ["OP 1", "OP 2", "ATT Primary"]
        for index, row in final_schedule_df.iterrows():
            assert any(
                substring in row["Product"] for substring in valid_substrings
            ), f"Invalid product value found for job {row['Job']}: {row['Product']}."

    def test_no_products_on_hold(self, final_schedule_df):
        """No schedule products should be on hold."""
        for index, row in final_schedule_df.iterrows():
            assert (
                row["On Hold?"] is False
            ), f"On Hold? is not False for job {row['Job']}: {row['On Hold?']}."

    def test_no_start_times_on_weekends(self, final_schedule_df):
        """HAAS machines can run on the weekends, but only on Saturdays. Other machines cannot run in weekends."""
        for index, row in final_schedule_df.iterrows():
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

    def test_non_haas_machine_start_time_range(self, final_schedule_df):
        """The start time for non-HAAS machines should be between 06:30 and 14:30."""
        for index, row in final_schedule_df.iterrows():
            machine = row["Machine"]
            if "HAAS" not in machine:
                start_time = row["Start_time"].time()
                start_time_range_valid = (
                    pd.Timestamp("06:30").time() <= start_time <= pd.Timestamp("14:30").time()
                )
                assert start_time_range_valid, (
                    f"Start time out of range for non-HAAS machine for job {row['Job']}: "
                    f"{start_time}."
                )

    def test_start_and_end_machines_present(self, final_schedule_df):
        """
        Test that for every unique order number in the final_schedule_df DataFrame,
        at least one row has a task 0, 1, or 10 (starting machines: HAAS or Ceramic Drag),
        and at least one row has a task 7 or 19 (Final inspection for OP1 and OP2 respectively).
        """
        unique_orders = final_schedule_df["Order"].unique()

        for order in unique_orders:
            order_rows = final_schedule_df[final_schedule_df["Order"] == order]

            has_initial_task = order_rows["task"].isin([0, 1, 10]).any()
            assert has_initial_task, f"Order {order} does not have any task 0, 1, or 10."

            has_final_task = order_rows["task"].isin([7, 19]).any()
            assert has_final_task, f"Order {order} does not have any task 7 or 19."

    def test_no_simultaneous_changeovers(self, final_changeovers_df):
        """No two changeovers can happen simultaneously for the HAAS machines"""
        for machine in final_changeovers_df["Machine"].unique():
            machine_changeovers = final_changeovers_df[
                final_changeovers_df["Machine"] == machine
            ].sort_values("Start_time")
            for i in range(1, len(machine_changeovers)):
                assert (
                    machine_changeovers.iloc[i]["Start_time"]
                    >= machine_changeovers.iloc[i - 1]["End_time"]
                ), (
                    f"Overlapping changeovers found on machine {machine} between changeovers starting at "
                    f"{machine_changeovers.iloc[i - 1]['Start_time']} and {machine_changeovers.iloc[i]['Start_time']}."
                )

    def test_max_two_changeovers_per_day(self, final_changeovers_df):
        """Changeovers take 3 hours, and there are only 8 hours in a workday.
        Therefore, there should be at most two changeovers per day."""
        # Ensure Start_time is in datetime format
        final_changeovers_df["Start_time"] = pd.to_datetime(final_changeovers_df["Start_time"])

        # Convert Start_time to date format
        final_changeovers_df["Start_date"] = final_changeovers_df["Start_time"].dt.date

        # Group by the date part of Start_time and count the number of changeovers
        changeovers_per_day = final_changeovers_df.groupby("Start_date").size()

        # Assert that no more than two changeovers start on the same day
        for date, count in changeovers_per_day.items():
            assert count <= 2, f"More than two changeovers start on {date}: {count} changeovers."

    def test_changeover_start_time_range(self, final_changeovers_df):
        """The start time of all changeovers should be between 06:30 and 11:30."""
        for index, row in final_changeovers_df.iterrows():
            start_time = row["Start_time"].time()
            start_time_range_valid = (
                pd.Timestamp("06:30").time() <= start_time <= pd.Timestamp("11:30").time()
            )
            assert (
                start_time_range_valid
            ), f"Changeover start time out of range for job {row['Job']}: {start_time}."

    def test_best_scores_improvement(self):
        """The best score per generation should never become worse from one iteration to the next."""
        with open("data/07_model_output/best_scores.pkl", "rb") as file:
            best_scores = pickle.load(file)

        assert isinstance(best_scores, deque), "The best_scores.pkl file does not contain a deque."

        previous_score = float("-inf")
        for score in best_scores:
            assert score >= previous_score, f"Score decreased from {previous_score} to {score}."
            previous_score = score


# TODO: Minimum 15 minutes between Roslers
# TODO: Depending on operation and Cementless, check if all tasks have been scheduled.
