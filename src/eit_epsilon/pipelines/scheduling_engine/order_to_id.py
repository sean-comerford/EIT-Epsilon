import numpy as np
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.io.core import DatasetError


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


# Function to handle assigning IDs to orders
def order_to_id(project_context):
    # Step 1: Load the mapping dictionary from the file or start with an empty one
    try:
        mapping_dict = project_context.catalog.load("mapping_dict_read")
    except (DatasetError, FileNotFoundError):
        mapping_dict = {}

    # Step 2: Load final schedule
    schedule = project_context.catalog.load("manual_order_to_id_run")

    # Step 3: Remove unused keys from the dictionary
    valid_orders = set(schedule["Job ID"])
    updated_mapping_dict = {k: v for k, v in mapping_dict.items() if k in valid_orders}

    # Step 4: Find unused numbers between 1 and 250
    unused_numbers = set(np.arange(1, 251)) - set(updated_mapping_dict.values())

    # Step 5: Assign new IDs to new orders
    for order in schedule["Job ID"]:
        if order not in updated_mapping_dict:
            new_id = unused_numbers.pop()  # Assign a new unique ID from the unused set
            updated_mapping_dict[order] = new_id  # Map the order to the new ID

    # Step 6: Update the schedule DataFrame with the IDs
    def find_mapping(row):
        id_value = updated_mapping_dict.get(row["Job ID"], None)
        row["ID"] = id_value
        return row

    # Apply the function row-wise
    schedule = schedule.apply(find_mapping, axis=1)

    # Check if everything worked as expected
    print(schedule)

    # Step 7: Save the updated mapping dictionary and schedule back to the catalog
    project_context.catalog.save("mapping_dict_write", updated_mapping_dict)
    project_context.catalog.save("manual_order_to_id_run", schedule)


def split_and_save_schedule(project_context):
    # Load the final schedule
    schedule = project_context.catalog.load("final_schedule")

    # Filter for the required columns and keep unique rows
    filtered_schedule = schedule[["Custom Part ID", "Order", "ID"]].drop_duplicates()

    # Split the dataframe based on 'Custom Part ID'
    ctd_df = filtered_schedule[filtered_schedule["Custom Part ID"].str.contains("CTD")]
    op1_df = filtered_schedule[
        ~filtered_schedule["Custom Part ID"].str.contains("CTD")
        & filtered_schedule["Custom Part ID"].str.contains("OP1")
    ]
    op2_df = filtered_schedule[
        ~filtered_schedule["Custom Part ID"].str.contains("CTD")
        & filtered_schedule["Custom Part ID"].str.contains("OP2")
    ]

    # Drop the 'Custom Part ID' column
    ctd_df = ctd_df.drop(columns=["Custom Part ID"])
    op1_df = op1_df.drop(columns=["Custom Part ID"])
    op2_df = op2_df.drop(columns=["Custom Part ID"])

    # Save the dataframes back to the catalog
    project_context.catalog.save("ctd_mapping", ctd_df)
    project_context.catalog.save("op1_mapping", op1_df)
    project_context.catalog.save("op2_mapping", op2_df)


def main():
    loader = config_loader()
    context = project_context(loader)
    order_to_id(context)
    # split_and_save_schedule(context)


if __name__ == "__main__":
    main()
