import logging
import pandas as pd

from typing import List, Dict, Tuple, Union

from .Shop import Shop

# Instantiate logger
logger = logging.getLogger(__name__)


class Job:
    """
    The Job class contains methods for preprocessing and extracting information from open orders that need
    to be processed in a manufacturing workshop.
    """

    @staticmethod
    def filter_in_scope(data: pd.DataFrame, operation: str = "OP 1") -> pd.DataFrame:
        """
        Filters the data to include only in-scope operations for OP 1.

        Args:
            data (pd.DataFrame): The input data.
            operation (str, optional): The operation for which to filter data. Defaults to 'OP 1'.

        Returns:
            pd.DataFrame: The filtered data.
        """
        # Debug statement
        logger.info(f"Total order data: {data.shape}")

        # Apply the filter
        if operation == "OP 1":
            in_scope_data = data[
                (
                    data["Part Description"].str.contains("OP 1")
                    | data["Part Description"].str.contains("ATT ")
                )
                & (~data["On Hold?"])
                & (~data["Part Description"].str.contains("OP 2"))
            ]

        elif operation == "OP 2":
            in_scope_data = data[
                (data["Part Description"].str.contains("OP 2"))
                & (~data["On Hold?"])
                & (~data["Part Description"].str.contains("OP 1"))
            ]

        else:
            logger.error(f"Invalid operation: {operation} - Only 'OP 1' and 'OP 2' are supported")
            raise ValueError("Invalid operation")

        # Debug statement
        logger.info(f"In-scope data for {operation}: {in_scope_data.shape}")

        return in_scope_data

    @staticmethod
    def extract_info(data: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts type, size, and orientation from the part description.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The data with extracted information.
        """
        data = data.assign(
            # CR: Cruciate retaining, PS: Posterior stabilizing
            Type=lambda x: x["Part Description"].apply(
                lambda y: "CR" if "CR" in y else ("PS" if "PS" in y else "")
            ),
            # Range 1-10 with optional 'N' for some sizes; e.g. '5N' (Not sure what this stands for)
            Size=lambda x: x["Part Description"].apply(
                lambda y: (re.search(r"Sz (\d+N?)", y).group(1) if re.search(r"Sz (\d+N?)", y) else "")
            ),
            # LEFT or RIGHT orientation
            Orientation=lambda x: x["Part Description"].apply(
                lambda y: ("LEFT" if "LEFT" in y.upper() else ("RIGHT" if "RIGHT" in y.upper() else ""))
            ),
            # CLS: Cementless, CTD: Cemented
            Cementless=lambda x: x["Part Description"].apply(
                lambda y: "CLS" if "CLS" in y.upper() else "CTD"
            ),
        )

        # Create custom Part ID
        data["Custom Part ID"] = (
            data["Orientation"] + "-" + data["Type"] + "-" + data["Size"] + "-" + data["Cementless"]
        )

        # Debug statement
        if data[["Type", "Size", "Orientation"]].isna().sum().sum() > 0:
            logger.warning(
                f"Data with extracted information: {data[['Type', 'Size', 'Orientation']].isna().sum()}"
            )
        else:
            logger.info(f"No missing values in Type, Size, and Orientation columns")

        return data

    @staticmethod
    def check_part_id_consistency(data: pd.DataFrame) -> None:
        """
        Checks the consistency of Part IDs.

        Args:
            data (pd.DataFrame): The input data.

        Raises:
            LoggerError: If Part ID is not unique for every combination of Type, Size, and Orientation.
        """
        grouped = data.groupby("Part ID")[["Type", "Size", "Orientation"]].nunique()

        if (grouped > 1).any().any():
            logger.error(
                "[bold red blink]Part ID not unique for every combination of Type, Size, and Orientation[/]",
                extra={"markup": True},
            )
        else:
            logger.info(f"Part ID consistency check passed")

    @staticmethod
    def create_jobs(data: pd.DataFrame, operation: str = "OP 1") -> Dict[int, Tuple[str, int]]:
        """ Extract the Job ID and corresponding Part ID from the data, calcuate the due date for each job
            and store the result in a dict object

        Args:
            data (pd.DataFrame): The input data i.e. the list of jobs

        Returns:
            Dict[int, Tuple[str, int]]: A dict, each entry of which contains a job ID, part ID and due time
            e.g. {
                4421322: ('MP0389', 2400)
                4421321: ('MP0389', 2400)            
                4420709: ('MP0442', 1440)
            }
        """
        if operation == "OP 1":
            data = data[~data["Part Description"].str.contains("OP 2")]
        
        return dict(zip(data['Job ID'], 
                        zip(data['Custom Part ID'], Shop.get_due_date(data))
                        ))

    # @staticmethod
    # def get_part_id(data: pd.DataFrame) -> List[str]:
    #     part_id = data["Custom Part ID"]

    #     # Convert the series to a list
    #     part_id = part_id.tolist()

    #     # Show snippet of Part IDs
    #     logger.info(f"Snippet of Part IDs: {part_id[:5]}")

    #     return part_id

    @staticmethod
    def create_partID_to_task_seq(data: pd.DataFrame) -> Dict[str, List[int]]:
        """ Create a dictionary which maps from a part ID to the list of tasks for that part

        Args:
            data (pd.DataFrame): _description_

        Returns:
            Dict[str, List[int]]: _description_
        """
        
        d = data[['Custom Part ID', 'Part Description', 'Cementless']].drop_duplicates()
        d = d.reset_index()
        
        dPartsOnly = data[['Custom Part ID']].drop_duplicates()
        
        if len(d) != len(dPartsOnly):
            print('Combination of part/description/cementless is not unique')

        #result = {id: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] if desc.str.contains("OP 2") for id, desc, cls in zip(d['Part ID'], d['Part Description'], d['Cementless'])
                        
        result = {}                       
        for _, row in d.iterrows():
            if 'OP 2' in row['Part Description']:
                result[row['Custom Part ID']] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] if row['Cementless'] == "CLS" else [10, 11, 12, 13, 14, 16, 17, 18, 19]
            else:
                # Operation 1
               # result[row['Custom Part ID']] = [1, 2, 3, 4, 5, 6, 7] if row['Cementless'] == "CLS" else [1, 2, 3, 6, 7]
                result[row['Custom Part ID']] = [1, 2, 3, 4, 5, 6, 7] if row['Cementless'] == "CLS" else [99, 2, 3, 6, 7]
        
        return result   