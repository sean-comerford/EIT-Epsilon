from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager


def config_loader():
    # Update the path to point to the correct directory
    return OmegaConfigLoader(conf_source=str(Path.cwd().parent.parent.parent.parent))


def project_context(config_loader):
    return KedroContext(
        package_name="eit_epsilon",
        project_path=Path.cwd().parent.parent.parent.parent,
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
        env=None,
    )


# Define task sequences for each type of product
cementless_op1_tasks = [1, 2, 3, 4, 5, 6, 7]
cementless_op2_tasks = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
cemented_tasks = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]


def find_partial_tasks(project_context, user_date: str = "2024-03-13") -> None:
    """
    Find the remaining tasks for each order based on the first occurrence of each order.

    Parameters:
    project_context (KedroContext): The Kedro project context.
    user_date (str): The date to filter the dataframe for. Default is '2024-03-08'.

    Returns:
    None: The output is saved in place.

    Notes:
    This function assumes that the input dataframe has columns 'End_time', 'Order', and 'task'.
    The function filters the dataframe for a specific date (hardcoded as '2024-03-08' for now) and then finds the first occurrence of each order.
    The remaining tasks are determined based on the task sequence of each product type (cementless_op1, cementless_op2, and cemented).
    """
    # Set up: Load final schedule
    df = project_context.catalog.load("final_schedule")

    # Step 1: Sort the dataframe by 'End_time' ascending
    df = df.sort_values(by="End_time")

    # Step 2: Filter the dataframe for a user-specified date
    df_filtered = df[df["End_time"] >= user_date]
    print("Data filtered!")

    # Step 3: Find the first occurrence of every unique 'Order' and log the 'task'
    first_occurrences = df_filtered.groupby("Order").head(1)[["Order", "task"]]
    print(f"First occurences found! Length: {len(first_occurrences)}")

    # Step 4: Create a dictionary of 'Order' as keys and the remaining task sequence as values
    task_sequences = {}
    for index, row in first_occurrences.iterrows():
        order = row["Order"]
        task = row["task"]
        if task in cementless_op1_tasks:
            task_sequence = cementless_op1_tasks
        elif task in cementless_op2_tasks:
            task_sequence = cementless_op2_tasks
        elif task in cemented_tasks:
            task_sequence = cemented_tasks
        else:
            continue
        remaining_tasks = task_sequence[task_sequence.index(task) :]
        task_sequences[order] = remaining_tasks
        print(f"Added remaining tasks for order {order}!")

    # Step 5: Save the dictionary or partial_tasks
    project_context.catalog.save("custom_tasks_dict", task_sequences)
    print(f"Saved dataset!")


def main():
    loader = config_loader()
    context = project_context(loader)
    find_partial_tasks(context)


if __name__ == "__main__":
    main()
