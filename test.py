import pandas as pd
import logging
import sys

from typing import List, Dict, Tuple

from src.eit_epsilon.pipelines.scheduling_engine.nodes import JobShop

from src.eit_epsilon.pipelines.scheduling_engine.Job import Job as Job2
from src.eit_epsilon.pipelines.scheduling_engine.Shop import Shop as Shop2
from src.eit_epsilon.pipelines.scheduling_engine.JobShop import JobShop as JobShop2

from src.eit_epsilon.pipelines.scheduling_engine.verification import *

#from src.eit_epsilon.pipelines.scheduling_engine.modifiedFunctions import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

task_to_machines= {
  # Operation 1
  1: [1, 2, 3, 4, 5, 6],  # HAAS
  2: [7, 8, 9, 10],  # Inspection
  3: [11],  # First Wash
  4: [12, 13, 14],  # Manual Prep
  5: [7, 8, 9, 10],  # Inspection (again)
  6: [15],  # Final Wash
  7: [16, 17],  # Final inspection

  # Operation 2
  10: [20],  # Ceramic drag
  11: [21],  # Wash 2
  12: [22],  # Plastic drag
  13: [21],  # Wash 2 (again)
  14: [7, 8, 9, 10],  # Inspection (shared with OP 1)
  15: [12, 13, 14],  # Manual Prep (shared with OP 1)
  16: [23],  # Nutshell drag
  17: [24, 25],  # Polishing
  18: [15],  # Final wash (shared with OP 1)
  19: [16, 17],  # Final inspection (shared with OP 1)

  # HAAS alternative
  99: [1, 2, 3]  # HAAS (only cemented)
}

scheduling_options={

  'start_date': "2024-03-18T09:00",  # Starting date used to calculate difference with due date
  'change_over_time': 180,  # Minutes required to do a changeover on the HAAS machines
  'n': 1200,  # Number of random schedules to generate
  'n_e': 0.1,  # Proportion of best schedules to automatically pass on to the next generation (e.g.: 10%)
  'n_c': 0.3,  # Proportion of children to generate in the offspring function
  'minutes_per_day': 480,  # Number of working minutes per day (8 hours * 60 minutes)
  'max_iterations': 10,  # Maximum number of iterations/generations to run the algorithm
  'urgent_multiplier': 3,  # Multiplier for (completion_time - due_date) for urgent_orders
  'urgent_orders': [], # List of job indices of urgent orders; starting from 1
  'column_mapping': {
    'Job ID': 'Order',
    'Created Date': 'Order_date',
    'Part Description': 'Product',
    'Due Date ': 'Due_date',
    'Order Qty': 'Order Qty'
    }
}

jobshop = JobShop()

df = pd.read_excel(r'C:\Dev\UCD Repos\EIT_Epsilon_UCD_Collaboration\data\01_raw\Week 9 open orders.xlsm', sheet_name='Data Week 9')

################################################################################################################################

croom_processed_orders = jobshop.preprocess_orders(df)

df2 = pd.read_excel(r'C:\Dev\UCD Repos\EIT_Epsilon_UCD_Collaboration\data\01_raw\Monza cycle times for op1 + 2 2021-09-14.xlsx', sheet_name='Monza OP1 Cycle times')
df3 = pd.read_excel(r'C:\Dev\UCD Repos\EIT_Epsilon_UCD_Collaboration\data\01_raw\Monza cycle times for op1 + 2 2021-09-14.xlsx', sheet_name='Monza OP2  Cycle times')

ps_cycle_times, cr_cycle_times, op2_cycle_times = JobShop.preprocess_cycle_times(df2, df3)

################# Original #################

# JOld1 = JobShop.create_jobs(croom_processed_orders, operation="OP 1")
# JOld2 = JobShop.create_jobs(croom_processed_orders, operation="OP 2")
# JOld = JOld1 + JOld2

jShop = JobShop()
jShop2 = JobShop2()

gaRep1 = jShop.build_ga_representation(croom_processed_orders, cr_cycle_times, ps_cycle_times, op2_cycle_times, task_to_machines, scheduling_options)
gaRep2 = jShop2.build_ga_representation(croom_processed_orders, cr_cycle_times, ps_cycle_times, op2_cycle_times, task_to_machines, scheduling_options)

verifyJobsMatch(gaRep1['J'], gaRep2['J'])
verifyTasksMatch(gaRep1['J'], gaRep2['J'], gaRep2['part_to_tasks'])

# print(str(gaRep2['J']))

# for i, (k, v) in enumerate(gaRep2['J'].items()):
#   print(f"{gaRep1['J'][i]}  {k} {v}")


################# New #####################

# JNew = Job2.create_jobs(croom_processed_orders, operation="OP 1")
# JNew2 = Job2.create_jobs(croom_processed_orders, operation="OP 2")
# JNew.update(JNew2)

# partToTasks = Job2.create_partID_to_task_seq(croom_processed_orders)

# # print(f"Length 2: {len(jobs2)}")
# # for jobID, (partID, due) in jobs2.items():
# #     print(f"{jobID} {partID} {partToTasks[partID]}")
    
# #**************************************************************************************

# duration1 = JobShop.get_duration_matrix(JOld, croom_processed_orders, cr_cycle_times, ps_cycle_times, op2_cycle_times)
# duration2 = Shop2.get_duration_matrix(JNew, partToTasks, croom_processed_orders, cr_cycle_times, ps_cycle_times, op2_cycle_times)


# orders = croom_processed_orders  #[croom_processed_orders['operation'] == 'OP2']
# due1 = JobShop.get_due_date(orders)

# print(f"Length of cpo: {len(orders)}   Length of dur1: {len(duration1)}")

# # Check if due times are durations are the same between the two data structures
# equalDD = True
# for i, jobID in enumerate(orders['Job ID']):    
#     durations = [d for (j, task), d in duration2.items() if j == jobID]
#     if due1[i] != JNew[jobID][1] or duration1[i] != durations:
#       equalDD = False
#       print(f"Not equal J: {jobID}\tDue1: {due1[i]}\tDue2: {JNew[jobID][1]}\tDur1: {duration1[i]} Dur2: {durations}")
# if equalDD: print("Due Times and durations are equal")
    


