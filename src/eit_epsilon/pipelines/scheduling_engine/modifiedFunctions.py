import pandas as pd
from typing import List, Dict, Tuple, Union
from src.eit_epsilon.pipelines.scheduling_engine.nodes import JobShop

def create_partID_to_task_seq(data: pd.DataFrame) -> Dict[str, List[int]]:
    
    d = data[['Part ID', 'Part Description', 'Cementless']].drop_duplicates()
    d = d.reset_index()
    
    dPartsOnly = data[['Part ID']].drop_duplicates()
    
    if len(d) != len(dPartsOnly):
        print('Combination of part/description/cementless is not unique')
  
    #result = {id: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] if desc.str.contains("OP 2") for id, desc, cls in zip(d['Part ID'], d['Part Description'], d['Cementless'])
                     
    result = {}                       
    for _, row in d.iterrows():
        if 'OP 2' in row['Part Description']:
            result[row['Part ID']] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] if row['Cementless'] == "CLS" else [10, 11, 12, 13, 14, 16, 17, 18, 19]
        else:
            # Operation 1
            result[row['Part ID']] = [1, 2, 3, 4, 5, 6, 7] if row['Cementless'] == "CLS" else [1, 2, 3, 6, 7]
    
    return result     


def create_jobs2(data: pd.DataFrame, operation: str = "OP 1") -> Dict[int, Tuple[str, int]]:
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
                    zip(data['Part ID'], JobShop.get_due_date(data))
                    ))
   
    
def get_duration_matrix2(
        J: Dict[int, Tuple[str, int]],
        part_ID_to_task_seq: Dict[str, List[int]],
        in_scope_orders: pd.DataFrame,
        cr_times: pd.DataFrame,
        ps_times: pd.DataFrame,
        op2_times: pd.DataFrame,
    ) -> List[List[float]]:   
    
        dur = {}
        for job_id, (part_id, due_time) in J.items():
            # Find the corresponding row for the given job_id
            rows = in_scope_orders.loc[in_scope_orders['Job ID'] == job_id]
            
            if len(rows) > 1:
                print(f"Error: Multiple rows found for JobID {job_id}. Using the first row.")

            row = rows.iloc[0]
           
            for task in part_ID_to_task_seq[part_id]:
                if task < 10:  # Operation 1 tasks
                    times = cr_times if row["Type"] == "CR" else ps_times
                    duration = round(times.loc[task, row["Size"]] * row["Order Qty"], 1)
                else:  # Operation 2 tasks
                    #print(f'Job {job_id}\t Task: {task}\tTime: {op2_times.loc[task, "Actual "]} \tQty: {row["Order Qty"]}')
                    
                    duration = round(op2_times.loc[task, "Actual "] * row["Order Qty"], 1)
                # Store the duration in the dictionary with key (job_id, task)
                dur[(job_id, task)] = duration
            
            # for task in job:
            #     if (task < 10):  # Operation 1 tasks are in the range 1-10, Operation 2 tasks are in the range 10-20
            #         times = cr_times if in_scope_orders.iloc[i]["Type"] == "CR" else ps_times
            #         duration = round(
            #             times.loc[task, in_scope_orders.iloc[i]["Size"]]
            #             * in_scope_orders.iloc[i]["Order Qty"],
            #             1,
            #         )
            #     else:
            #         duration = round(
            #             op2_times.loc[task, "Actual "] * in_scope_orders.iloc[i]["Order Qty"],
            #             1,
            #         )
            #     job_dur.append(duration)
            # dur.append(job_dur)

        return dur