import pandas as pd
import numpy as np
import re
import random
from typing import List


def add_random_op(description: str) -> str:
    """
    Adds a random operation ('OP 1' or 'OP 2') to the description if 'CEM' is not present.
    We currently don't know if the forecast cementless orders are OP1 or OP2, so for the sake of simulation
    we will randomly assign one of the two.

    Args:
        description (str): The product description.

    Returns:
        str: The updated product description.
    """
    if "CEM" not in description.upper():
        return description + " OP " + str(random.choice([1, 2]))
    return description


def apply_add_random_op(new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the add_random_op function to the 'Product Description' column of the DataFrame.

    Args:
        new_df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with updated 'Product Description'.
    """
    new_df["Product Description"] = new_df["Product Description"].apply(add_random_op)
    return new_df


def split_quantity(quantity: int) -> List[int]:
    """
    Splits a quantity into a list of quantities of 12 and the remainder.
    We get the product quantity in absolute numbers, but we need a format where each row represents a batch.

    Args:
        quantity (int): The quantity to split.

    Returns:
        List[int]: A list of quantities.
    """
    num_twelves = quantity // 12
    remaining = quantity % 12
    quantities = [12] * num_twelves
    if remaining > 0:
        quantities.append(remaining)
    return quantities


def apply_split_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the split_quantity function to the 'Week 39' column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with a new 'Quantities' column.
    """
    df["Quantities"] = df["Week 39"].apply(split_quantity)
    return df


def explode_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explodes the 'Quantities' column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with exploded 'Quantities'.
    """
    df = df.explode("Quantities")
    return df


def rename_and_select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the 'Quantities' column to 'Production Qty' and selects specific columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with renamed and selected columns.
    """
    df = df.rename(columns={"Quantities": "Production Qty"})
    new_df = df[["Product Description", "Production Qty"]]
    return new_df


def process_product_description(new_df: pd.DataFrame, preprocess_options: dict) -> pd.DataFrame:
    """
    Processes the 'Product Description' column to extract and assign new columns.

    Args:
        new_df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new columns added.
    """
    new_df = new_df.assign(
        Type=lambda x: x["Product Description"].apply(
            lambda y: "CR" if "CR" in y else ("PS" if "PS" in y else "")
        ),
        Size=lambda x: x["Product Description"].apply(
            lambda y: re.sub(r" NAR", "N", re.search(r"(\d(?: NAR)?)", y).group(1))
            if re.search(r"(\d(?: NAR)?)", y)
            else ""
        ),
        Orientation=lambda x: x["Product Description"].apply(
            lambda y: ("LEFT" if "LT" in y.upper() else ("RIGHT" if "RT" in y.upper() else ""))
        ),
        Cementless=lambda x: x["Product Description"].apply(
            lambda y: "CLS" if "CEM" not in y.upper() else "CTD"
        ),
        operation=lambda x: x["Product Description"].apply(
            lambda y: "OP2" if "OP 2" in y.upper() else "OP1"
        ),
    )

    # Create Custom Part ID
    new_df["Custom Part ID"] = (
        new_df["Orientation"]
        + "-"
        + new_df["Type"]
        + "-"
        + new_df["Size"]
        + "-"
        + new_df["Cementless"]
        + "-"
        + new_df["operation"]
    )

    # Add Job ID (currently unknown, so we generate this ourselves)
    new_df["Job ID"] = np.arange(1, len(new_df) + 1)

    # Add Due Date (currently unknown, so we pick something)
    new_df["Due Date "] = preprocess_options["due_date"]
    new_df["Due Date "] = pd.to_datetime(new_df["Due Date "], format="%Y-%m-%d")

    # Add Created Date (currently unknown, so we pick something)
    new_df["Created Date"] = preprocess_options["created_date"]
    new_df["Created Date"] = pd.to_datetime(new_df["Created Date"], format="%Y-%m-%d")

    # Reset index and rename columns
    new_df.reset_index(inplace=True, drop=True)
    new_df.rename(columns={"Product Description": "Part Description"}, inplace=True)

    return new_df
