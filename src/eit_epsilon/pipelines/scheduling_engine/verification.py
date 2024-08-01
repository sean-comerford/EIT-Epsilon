def verifyJobsMatch(j1, j2):
    print(f"Number of jobs J1: {len(j1)} J2: {len(j2)}" +  " Equal" if len(j1) == len(j2) else " Not equal")

def verifyTasksMatch(j1, j2 , part_to_tasks):
    # Create a dict: key = Length of task seq, value = number of seqs of this length
    # e.g. tasks1[3] gives the number of task sequences of length 3
    tasks1 = {}
    for j in j1:
        l = len(j)
        tasks1[l] = tasks1.get(l, 0) + 1
        
    tasks2= {}
    for jobID, (partID, dueTime) in j2.items():
        l = len(part_to_tasks[partID])
        tasks2[l] = tasks2.get(l, 0) + 1
        
    print(tasks1)
    print(tasks2)

        