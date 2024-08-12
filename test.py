import pandas as pd
import logging
import sys

import random

from typing import List, Dict, Tuple

from src.eit_epsilon.pipelines.scheduling_engine.nodes import JobShop, GeneticAlgorithmScheduler


from src.eit_epsilon.pipelines.scheduling_engine.Job import Job as Job2
from src.eit_epsilon.pipelines.scheduling_engine.Shop import Shop as Shop2
from src.eit_epsilon.pipelines.scheduling_engine.JobShop import JobShop as JobShop2

from src.eit_epsilon.pipelines.scheduling_engine.GASch import GeneticAlgorithmScheduler as GeneticAlgorithmScheduler2

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
  'change_over_time_op1': 180,  # Minutes required to do a changeover on the HAAS machines (Only OP 1)
  'change_over_time_op2': 20,  # Minutes required to do a changeover on the HAAS machines (Only OP 1)
  'change_over_machines_op2': [20, 22, 23],  # Machines that require a changeover for OP2 (key in machine dict above)
  'change_over_time': 180,  # Minutes required to do a changeover on the HAAS machines
  'n': 25,  # Number of random schedules to generate
  'n_e': 0.1,  # Proportion of best schedules to automatically pass on to the next generation (e.g.: 10%)
  'n_c': 0.3,  # Proportion of children to generate in the offspring function
  'minutes_per_day': 480,  # Number of working minutes per day (8 hours * 60 minutes)
  'max_iterations': 10,  # Maximum number of iterations/generations to run the algorithm
  'urgent_multiplier': 3,  # Multiplier for (completion_time - due_date) for urgent_orders
  'urgent_orders': [], # List of job IDs (urgency in reverse order)
  'column_mapping': {
    'Job ID': 'Order',
    'Created Date': 'Order_date',
    'Part Description': 'Product',
    'Due Date ': 'Due_date',
    'Order Qty': 'Order Qty'
    }
}

# Define OP2 size categories
size_categories_op2 = {
    "small": {"1", "2", "3N"},
    "medium": {"3", "4N", "4", "5N", "5", "6N"},
    "large": {"6", "7", "8", "9", "10"}
}

jobshop = JobShop()

#df = pd.read_excel(r'C:\Dev\UCD Repos\EIT_Epsilon_UCD_Collaboration\data\01_raw\Week 9 open orders.xlsm', sheet_name='Data Week 9')
df = pd.read_excel(r'C:\Dev\UCD Repos\EIT_Epsilon_UCD_Collaboration\data\01_raw\orders_with_custom_part_id.xlsx', sheet_name='orders_with_custom_part_id')

#df = pd.read_csv(r'C:\Dev\UCD Repos\EIT_Epsilon_UCD_Collaboration\data\01_raw\orders_with_custom_part_id.csv')

################################################################################################################################

croom_processed_orders = jobshop.preprocess_orders(df)

df2 = pd.read_excel(r'C:\Dev\UCD Repos\EIT_Epsilon_UCD_Collaboration\data\01_raw\Monza cycle times for op1 + 2 2021-09-14.xlsx', sheet_name='Monza OP1 Cycle times')
df3 = pd.read_excel(r'C:\Dev\UCD Repos\EIT_Epsilon_UCD_Collaboration\data\01_raw\Monza cycle times for op1 + 2 2021-09-14.xlsx', sheet_name='Monza OP2  Cycle times')

ps_cycle_times, cr_cycle_times, op2_cycle_times = JobShop.preprocess_cycle_times(df2, df3)

jShop = JobShop()
jShop2 = JobShop2()

gaRep1 = jShop.build_ga_representation(croom_processed_orders, cr_cycle_times, ps_cycle_times, op2_cycle_times, task_to_machines, scheduling_options)
gaRep2 = jShop2.build_ga_representation(croom_processed_orders, cr_cycle_times, ps_cycle_times, op2_cycle_times, task_to_machines, scheduling_options)

verifyJobsMatch(gaRep1['J'], gaRep2['J'])
verifyTasksMatch(gaRep1['J'], gaRep2['J'], gaRep2['part_to_tasks'])
verifyMachinesMatch(gaRep1['M'], gaRep2['M'])
verifyDurationsMatch(gaRep1['dur'], gaRep2['J'], gaRep2['dur'])


gas1 = GeneticAlgorithmScheduler()
gas2 = GeneticAlgorithmScheduler2()

comp = JobShop.build_changeover_compatibility(croom_processed_orders, size_categories_op2)
comp2 = JobShop2.build_changeover_compatibility(croom_processed_orders, size_categories_op2)

#seedNum = 2
seedNum = random.randint(0, 1000)

if seedNum is not None: random.seed(seedNum)  
pop1 = gas1.run(gaRep1, scheduling_options, comp)

if seedNum is not None: random.seed(seedNum)  
pop2 = gas2.run(gaRep2, scheduling_options, comp)

# Compare initial populations
popsMatch = verifyPopulationsMatch(pop1, pop2, croom_processed_orders)
print("Populations " + ("match" if popsMatch else "do not match") + "\n")

# gas1.evalPopTest()
# gas2.evalPopTest()

# Compare results of evaluating the inital populations
scores1 = []
gas1.evaluate_population(scores1)
scores1 = gas1.scores

scores2 = []
gas2.evaluate_population(scores2)
scores2 = gas2.scores

print("Population scores " + ("match" if scores1 == scores2 else "do not match") +"\n")

# Compare best schedules
best_sch1 = gas1.find_best_schedules()
best_sch2 = gas2.find_best_schedules()
print("Best schedules " + ("match" if verifyPopulationsMatch(best_sch1, best_sch2, croom_processed_orders) else "do not match") + "\n")

# Compare results of a single call of offspring()
if seedNum is not None: random.seed(seedNum)  
gas1.offspring()
if seedNum is not None: random.seed(seedNum)  
gas2.offspring()

print("Snippet of populations after offspring: ")
for i in range(5):
  print(f"{gas1.P[0][i]} {gas2.P[0][i]}")

popsMatch = verifyPopulationsMatch(gas1.P, gas2.P, croom_processed_orders)
print("Offspring populations " + ("match" if popsMatch else "do not match") + "\n")

# Compare results of a single call of mutate()
if seedNum is not None: random.seed(seedNum)  
gas1.mutate()
if seedNum is not None: random.seed(seedNum)  
gas2.mutate()

print("Snippet of populations after mutate: ")
for i in range(5):
  print(f"{gas1.P[0][i]} {gas2.P[0][i]}")

popsMatch = verifyPopulationsMatch(gas1.P, gas2.P, croom_processed_orders)
print("Mutated populations " + ("match" if popsMatch else "do not match"))